"""Pipeline orchestration.

`run_pipeline` is the single entry point everything else calls: the local CLI,
the Modal wrapper, a notebook. It takes a local video path and a config, and
writes its outputs to a directory.

Deliberately knows nothing about R2, Modal, or Supabase. The caller is
responsible for getting the video onto local disk and for doing something with
the output directory afterwards.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

from pipeline import appearance
from pipeline.config import PipelineConfig
from pipeline.court_projection import compute_frame_homography, parse_court_keypoints, project_points
from pipeline.detection import PrefetchingDetector, build_detector, detections_from_result
from pipeline.number_evidence import (
    NumberPredictionRecord,
    associate_number_rows,
    write_number_debug_artifacts,
    write_number_confirmation_artifacts,
    write_number_prediction_artifacts,
)
from pipeline.number_predictor import build_number_predictor, normalize_number_prediction
from pipeline.projected_shots import build_projected_shots, write_projected_shots_image
from pipeline.reid import Embedder, build_embedder
from pipeline.results import RESULTS_SCHEMA_VERSION, PipelineResults, RunSummary, TrackWriter
from pipeline.shot_events import (
    flush_open_shot_event,
    shot_frame_signals,
    shot_raw_prediction_record,
    shot_signal_trace_record,
    update_shot_event_candidate,
    write_shot_event_record,
    write_shot_raw_prediction_record,
    write_shot_signal_record,
)
from pipeline.team_evidence import TeamEvidenceSample, select_top_k_evidence, summarize_track_evidence, write_debug_artifacts
from pipeline.team_embedder import TeamEmbeddingSample, build_team_embedder
from pipeline.team_id import (
    KMeansTeamClusterer,
    TeamObservation,
    identify_teams,
)
from pipeline.tracking import (
    Sam2SegmentTracker,
    assign_track_ids,
    carry_unmatched_tracks,
    detection_rows,
)
from pipeline.video import AnnotatedVideoWriter, frame_step_for, iter_segments, probe

logger = logging.getLogger(__name__)

TRACKS_FILENAME = "tracks.jsonl"
RESULTS_FILENAME = "results.json"
ANNOTATED_FILENAME = "annotated.mp4"
SWAPS_FILENAME = "suspected_swaps.jsonl"
TEAMS_FILENAME = "teams.json"
NUMBERS_FILENAME = "numbers.json"
SHOT_EVENTS_FILENAME = "shot_events.jsonl"
SHOT_SIGNALS_FILENAME = "shot_signals.jsonl"
SHOT_RAW_PREDICTIONS_FILENAME = "shot_raw_predictions.jsonl"
PROJECTED_TRACKS_FILENAME = "projected_tracks.jsonl"
PROJECTED_SHOTS_FILENAME = "projected_shots.json"
PROJECTED_SHOTS_IMAGE_FILENAME = "projected_shots.png"


def _build_detector_for_model(config: PipelineConfig, *, model_id: str | None = None):
    try:
        return build_detector(config, model_id=model_id)
    except TypeError as error:
        if "model_id" not in str(error):
            raise
        return build_detector(config)


def _frame_timestamp_seconds(frame_index: int, fps: float) -> float | None:
    return None if fps <= 0 else frame_index / fps


def _split_detection_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    player_rows = [row for row in rows if str(row.get("class_name", "player")) == "player"]
    number_rows = [row for row in rows if str(row.get("class_name", "player")) == "number"]
    return player_rows, number_rows


def _with_team_ids(rows: list[dict], team_by_track_id: dict[int, int]) -> list[dict]:
    enriched = []
    for row in rows:
        copy = dict(row)
        track_id = copy.get("track_id")
        copy["team_id"] = None if track_id is None else team_by_track_id.get(int(track_id))
        enriched.append(copy)
    return enriched


def _iter_number_evidence_samples(number_evidence_by_track_id: dict[int, list]):
    for track_id in sorted(number_evidence_by_track_id):
        yield from sorted(number_evidence_by_track_id[track_id], key=lambda sample: (sample.frame_index, sample.track_id))


def _identify_track_teams(
    selected_evidence_by_track_id: dict[int, list[TeamEvidenceSample]],
    *,
    backend: str,
    team_count: int,
    minimum_samples: int,
    assignment_margin: float,
):
    observations = []
    detection_id = 1
    for track_id, samples in sorted(selected_evidence_by_track_id.items()):
        for sample in samples:
            observations.append(
                TeamObservation(
                    detection_id=detection_id,
                    track_id=track_id,
                    color_signature=sample.color_signature,
                    embedding=sample.embedding,
                )
            )
            detection_id += 1
    return identify_teams(
        observations,
        clusterer=KMeansTeamClusterer(),
        team_count=team_count,
        backend=backend,
        minimum_samples=minimum_samples,
        assignment_margin=assignment_margin,
    )


def _collect_team_evidence(
    frame,
    rows: list[dict],
    *,
    frame_index: int,
    team_embedder,
    evidence_by_track_id: dict[int, list[TeamEvidenceSample]],
    min_quality: float,
) -> None:
    samples = [
        TeamEmbeddingSample(
            detection_id=index,
            track_id=None if row.get("track_id") is None else int(row["track_id"]),
            xyxy=tuple(float(value) for value in row["xyxy"]),
            player_mask=row.get("player_mask"),
        )
        for index, row in enumerate(rows)
    ]
    features_by_row = team_embedder.embed(frame, samples)
    for row, features in zip(rows, features_by_row):
        if row.get("track_id") is None:
            continue
        track_id = int(row["track_id"])
        accepted = (
            (features.color_signature is not None or features.embedding is not None)
            and features.quality_score >= min_quality
            and not features.rejection_reasons
        )
        reasons = list(features.rejection_reasons)
        if not accepted and features.quality_score < min_quality:
            reasons.append("below_quality_threshold")
        evidence_by_track_id.setdefault(track_id, []).append(
            TeamEvidenceSample(
                track_id=track_id,
                frame_index=frame_index,
                track_state=str(row.get("track_state", "unknown")),
                quality_score=features.quality_score,
                accepted=accepted,
                rejection_reasons=reasons,
                color_signature=features.color_signature,
                embedding=features.embedding,
                used_mask=features.used_mask,
                uniform_xyxy=features.uniform_xyxy,
                uniform_pixel_count=features.uniform_pixel_count,
                total_pixel_count=features.total_pixel_count,
                quality_details=features.quality_details,
                crop_image=features.debug_crop,
                mask_image=features.debug_mask,
            )
        )


def _rewrite_track_team_ids(tracks_path: Path, team_by_track_id: dict[int, int]) -> None:
    temporary_path = tracks_path.with_suffix(".tmp")
    with tracks_path.open("r", encoding="utf-8") as source, temporary_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            record = json.loads(line)
            track_id = record.get("track_id")
            record["team_id"] = None if track_id is None else team_by_track_id.get(int(track_id))
            target.write(json.dumps(record) + "\n")
    temporary_path.replace(tracks_path)


def _iter_track_rows_by_frame(tracks_path: Path):
    current_frame_index = None
    current_rows: list[dict] = []
    with tracks_path.open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            frame_index = int(record["frame_index"])
            if current_frame_index is None:
                current_frame_index = frame_index
            if frame_index != current_frame_index:
                yield current_frame_index, current_rows
                current_frame_index = frame_index
                current_rows = []
            current_rows.append(record)
    if current_frame_index is not None:
        yield current_frame_index, current_rows


def _load_confirmed_numbers_by_track_id(numbers_path: Path) -> dict[int, str]:
    if not numbers_path.exists():
        return {}
    payload = json.loads(numbers_path.read_text(encoding="utf-8"))
    confirmed_numbers = {}
    for track in payload.get("tracks", []):
        if track.get("status") != "confirmed":
            continue
        number = track.get("number")
        track_id = track.get("track_id")
        if number in (None, "", "unknown") or track_id is None:
            continue
        confirmed_numbers[int(track_id)] = str(number)
    return confirmed_numbers


def _load_team_ids_by_track_id(teams_path: Path) -> dict[int, int]:
    if not teams_path.exists():
        return {}
    payload = json.loads(teams_path.read_text(encoding="utf-8"))
    assignments = payload.get("assignments_by_track_id", {})
    return {int(track_id): int(team_id) for track_id, team_id in assignments.items()}


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _foot_point_for_bbox(xyxy) -> list[float]:
    x1, _, x2, y2 = xyxy
    return [float((x1 + x2) / 2.0), float(y2)]


def _write_projected_track_rows(handle, *, frame_index: int, rows: list[dict], homography_result) -> int:
    projected_points = project_points(
        homography_result.matrix,
        [tuple(_foot_point_for_bbox(row["xyxy"])) for row in rows],
    )
    written = 0
    for row, projected_point in zip(rows, projected_points):
        image_foot_xy = _foot_point_for_bbox(row["xyxy"])
        handle.write(
            json.dumps(
                {
                    "frame_index": int(frame_index),
                    "track_id": None if row.get("track_id") is None else int(row["track_id"]),
                    "track_state": row.get("track_state"),
                    "class_name": str(row.get("class_name", "player")),
                    "confidence": None if row.get("confidence") is None else float(row["confidence"]),
                    "xyxy": [float(value) for value in row["xyxy"]],
                    "image_foot_xy": image_foot_xy,
                    "court_xy_m": [float(projected_point[0]), float(projected_point[1])],
                    "homography_frame_status": "ok",
                    "homography_inliers_count": int(homography_result.inliers_count),
                    "homography_reprojection_mean_m": float(homography_result.reprojection_mean_m),
                }
            )
            + "\n"
        )
        written += 1
    return written


def _maybe_write_projected_shots(output_dir: Path) -> tuple[str | None, str | None]:
    projected_tracks_path = output_dir / PROJECTED_TRACKS_FILENAME
    if not projected_tracks_path.exists():
        return None, None

    shot_events_path = output_dir / SHOT_EVENTS_FILENAME
    projected_shots_path = output_dir / PROJECTED_SHOTS_FILENAME
    projected_shots_image_path = output_dir / PROJECTED_SHOTS_IMAGE_FILENAME
    artifact = build_projected_shots(
        _load_jsonl_rows(shot_events_path),
        _load_jsonl_rows(projected_tracks_path),
        team_ids_by_track_id=_load_team_ids_by_track_id(output_dir / TEAMS_FILENAME),
        confirmed_numbers_by_track_id=_load_confirmed_numbers_by_track_id(output_dir / NUMBERS_FILENAME),
    )
    projected_shots_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    write_projected_shots_image(artifact, projected_shots_image_path)
    return PROJECTED_SHOTS_FILENAME, PROJECTED_SHOTS_IMAGE_FILENAME


def _render_annotated_video(
    video_path: Path,
    tracks_path: Path,
    numbers_path: Path,
    output_path: Path,
    *,
    info,
    frame_stride: int,
    max_frames: int | None,
    frame_step: int,
    fps: float,
) -> None:
    from pipeline.annotate import draw_tracks

    confirmed_numbers_by_track_id = _load_confirmed_numbers_by_track_id(numbers_path)
    rows_by_frame = _iter_track_rows_by_frame(tracks_path)
    pending = next(rows_by_frame, None)

    with AnnotatedVideoWriter(output_path, info, fps=fps) as writer:
        for segment in iter_segments(
            video_path,
            frame_stride=frame_stride,
            max_frames=max_frames,
            frame_step=frame_step,
        ):
            last_offset = len(segment.frames) - 1 if segment.is_last else len(segment.frames) - 2
            for offset in range(last_offset + 1):
                frame_index = segment.frame_indices[offset]
                rows = []
                if pending is not None and pending[0] == frame_index:
                    rows = [
                        {
                            **row,
                            **(
                                {"confirmed_number": confirmed_numbers_by_track_id[int(row["track_id"])]}
                                if row.get("track_id") is not None and int(row["track_id"]) in confirmed_numbers_by_track_id
                                else {}
                            ),
                        }
                        for row in pending[1]
                    ]
                    pending = next(rows_by_frame, None)
                elif pending is not None and pending[0] < frame_index:
                    raise RuntimeError(
                        f"tracks.jsonl is out of sync with processed frames at frame {frame_index}"
                    )
                writer.write(draw_tracks(segment.frames[offset], rows))

    if pending is not None:
        raise RuntimeError(
            f"tracks.jsonl has extra frame rows starting at frame {pending[0]}"
        )


def _write_shot_events(video_path: Path, output_path: Path, *, info, step: int, config: PipelineConfig) -> None:
    detector = _build_detector_for_model(config, model_id=config.shot_model_id)
    active_shot_event = None
    next_shot_event_id = 1
    last_emitted_frame_index = 0
    signals_path = output_path.with_name(SHOT_SIGNALS_FILENAME)
    raw_predictions_path = output_path.with_name(SHOT_RAW_PREDICTIONS_FILENAME)

    with output_path.open("w", encoding="utf-8") as shot_events_file, signals_path.open(
        "w", encoding="utf-8"
    ) as shot_signals_file, raw_predictions_path.open("w", encoding="utf-8") as shot_raw_predictions_file:
        for segment in iter_segments(
            video_path,
            frame_stride=config.shot_frame_stride,
            max_frames=config.max_frames,
            frame_step=step,
        ):
            frame_index = segment.frame_indices[0]
            result = detector.detect(segment.frames[0])
            timestamp_seconds = _frame_timestamp_seconds(frame_index, info.fps)
            write_shot_raw_prediction_record(
                shot_raw_predictions_file,
                shot_raw_prediction_record(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    model_id=config.shot_model_id,
                    result=result,
                ),
            )
            signals = shot_frame_signals(result)
            active_shot_event, resolved_shot_event, next_shot_event_id = update_shot_event_candidate(
                active_shot_event,
                next_event_id=next_shot_event_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                signals=signals,
            )
            write_shot_signal_record(
                shot_signals_file,
                shot_signal_trace_record(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    signals=signals,
                    candidate=active_shot_event,
                    resolved_event=resolved_shot_event,
                ),
            )
            if resolved_shot_event is not None:
                write_shot_event_record(shot_events_file, resolved_shot_event)
            last_emitted_frame_index = frame_index

        final_shot_event = flush_open_shot_event(
            active_shot_event,
            frame_index=last_emitted_frame_index,
            timestamp_seconds=_frame_timestamp_seconds(last_emitted_frame_index, info.fps),
        )
        if final_shot_event is not None:
            write_shot_event_record(shot_events_file, final_shot_event)


def run_pipeline(
    video_path: Path,
    config: PipelineConfig,
    output_dir: Path,
    *,
    tracker: Sam2SegmentTracker | None = None,
    embedder: Embedder | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> PipelineResults:
    """Run detection + SAM2 tracking over `video_path`, writing into `output_dir`.

    `tracker` can be passed in so an already-loaded SAM2 is reused across jobs.
    On Modal this is loaded once per container in `@modal.enter()`; loading it
    per job would pay the model download and init on every single request.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = probe(video_path)
    step = frame_step_for(info.fps, config.target_fps)
    effective_fps = info.fps / step if info.fps else 0.0
    # Progress is measured against frames we will actually PROCESS (after
    # downsampling), not source frames — otherwise it caps at target_fps/fps
    # (e.g. ~25% for 15fps on 60fps footage) and never approaches 100%.
    available_frames = (info.frame_count + step - 1) // step if (info.frame_count and step) else 0
    expected_frames = min(available_frames, config.max_frames) if config.max_frames else available_frames
    shot_step = step if config.shot_target_fps is None else frame_step_for(info.fps, config.shot_target_fps)
    shot_effective_fps = info.fps / shot_step if info.fps else 0.0
    logger.info(
        "video: %sx%s @ %.2ffps, %s frames -> processing every %s frame(s) (~%.1ffps)",
        info.width,
        info.height,
        info.fps,
        info.frame_count or "unknown",
        step,
        effective_fps,
    )

    tracker = tracker or Sam2SegmentTracker(config.sam2_model_id)
    detector = _build_detector_for_model(config)
    court_detector = (
        _build_detector_for_model(config, model_id=config.court_model_id) if config.court_model_id else None
    )
    started = time.monotonic()

    summary = RunSummary()
    next_track_id = 1
    # Rows SAM2 was carrying into the current segment's first frame. Used only to
    # match fresh detector boxes back to existing track IDs.
    active_rows: list[dict] = []
    # track_id -> consecutive detector calls it has gone unmatched.
    missed_counts: dict[int, int] = {}
    # track_id -> torso colour signature, averaged over clean observations.
    appearance_by_track_id: dict[int, list[float]] = {}
    embedder = embedder or build_embedder(config.reid_backend)
    # Detections that resemble a different track than the one they were given.
    suspected_swaps: list[dict] = []
    team_embedder = build_team_embedder(
        config.team_embedder_backend,
        model_id=config.team_embedder_model_id,
    )
    team_evidence_by_track_id: dict[int, list[TeamEvidenceSample]] = {}
    number_evidence_by_track_id: dict[int, list] = {}
    number_predictor = build_number_predictor(config)

    annotated_path = output_dir / ANNOTATED_FILENAME if config.write_annotated_video else None
    projected_tracks_path = output_dir / PROJECTED_TRACKS_FILENAME
    projected_track_records = 0

    prefetch = PrefetchingDetector(detector) if config.prefetch_detections else None

    try:
        with TrackWriter(output_dir / TRACKS_FILENAME) as tracks:
            projected_context = (
                projected_tracks_path.open("w", encoding="utf-8") if court_detector is not None else nullcontext()
            )
            with projected_context as projected_tracks_file:
                for segment in iter_segments(
                    video_path,
                    frame_stride=config.frame_stride,
                    max_frames=config.max_frames,
                    frame_step=step,
                ):
                    frames = segment.frames
                    indices = segment.frame_indices

                    # --- detect on the segment's first frame -------------------
                    result = prefetch.take(frames[0]) if prefetch else detector.detect(frames[0])
                    detections = detections_from_result(
                        result,
                        min_confidence=config.confidence,
                        allowed_classes=config.allowed_classes,
                    )
                    player_rows, number_rows = _split_detection_rows(detection_rows(detections))

                # Describe each detection before matching. The embedder returns
                # None for players too occluded to describe, so crossings degrade
                # to geometry rather than to a confident wrong answer.
                    if config.appearance_weight > 0.0:
                        signatures = embedder.embed(frames[0], [row["xyxy"] for row in player_rows])
                        for row, signature in zip(player_rows, signatures):
                            row["appearance"] = signature

                    seed_rows, next_track_id = assign_track_ids(
                        player_rows,
                        active_rows,
                        next_track_id,
                        config.sam2_match_iou,
                        appearance_by_track_id=appearance_by_track_id,
                        appearance_weight=config.appearance_weight,
                        min_appearance_match=config.min_appearance_match,
                    )

                # Check for swaps BEFORE updating signatures, so a swap is caught
                # before it teaches itself the wrong colour. See
                # `appearance.find_better_match` for why this is the only
                # automatic signal available.
                    for row in seed_rows:
                        signature = row.get("appearance")
                        if signature is None:
                            continue
                        better = appearance.find_better_match(
                            signature, int(row["track_id"]), appearance_by_track_id
                        )
                        if better is None:
                            continue
                        better_id, own_score, other_score = better
                        suspected_swaps.append(
                            {
                                "frame_index": indices[0],
                                "track_id": int(row["track_id"]),
                                "looks_like_track_id": better_id,
                                "own_similarity": round(own_score, 4),
                                "other_similarity": round(other_score, 4),
                            }
                        )

                # Update each track's stored signature from this observation.
                # Averaged over time so a single marginal crop cannot redefine a
                # player's colour.
                    for row in seed_rows:
                        signature = row.get("appearance")
                        if signature is None:
                            continue
                        track_id = int(row["track_id"])
                        appearance_by_track_id[track_id] = appearance.blend(
                            appearance_by_track_id.get(track_id), signature
                        )

                    for sample in associate_number_rows(
                        frames[0],
                        seed_rows,
                        number_rows,
                        frame_index=indices[0],
                    ):
                        number_evidence_by_track_id.setdefault(sample.track_id, []).append(sample)

                    _collect_team_evidence(
                        frames[0],
                        seed_rows,
                        frame_index=indices[0],
                        team_embedder=team_embedder,
                        evidence_by_track_id=team_evidence_by_track_id,
                        min_quality=config.min_team_sample_quality,
                    )

                # Re-seed SAM2 with tracks the detector missed as well, so a
                # one-frame miss does not cost a player their identity.
                    seed_rows = carry_unmatched_tracks(
                        seed_rows, active_rows, missed_counts, config.max_missed_detections
                    )
                    seed_rows_with_teams = _with_team_ids(seed_rows, {})

                    if court_detector is not None and projected_tracks_file is not None and seed_rows:
                        try:
                            court_result = court_detector.detect(frames[0])
                            keypoints = parse_court_keypoints(
                                court_result,
                                min_confidence=config.court_keypoint_confidence,
                                width=info.width,
                                height=info.height,
                            )
                            homography_result = compute_frame_homography(
                                keypoints,
                                min_confidence=config.court_keypoint_confidence,
                            )
                        except (RuntimeError, ValueError):
                            homography_result = None
                        if homography_result is not None:
                            projected_track_records += _write_projected_track_rows(
                                projected_tracks_file,
                                frame_index=indices[0],
                                rows=seed_rows,
                                homography_result=homography_result,
                            )

                # Fire the next segment's detector call now, so the HTTP wait
                # overlaps with SAM2 compute instead of idling a billed GPU.
                # The next detector frame is this segment's last frame.
                    if prefetch and not segment.is_last:
                        prefetch.submit(frames[-1])

                # --- propagate through the segment -------------------------
                    propagated = tracker.propagate_segment(frames, seed_rows)
                    for offset in range(1, len(frames)):
                        if offset % config.team_sample_stride != 0:
                            continue
                        propagated_rows = propagated.get(offset, [])
                        if not propagated_rows:
                            continue
                        _collect_team_evidence(
                            frames[offset],
                            propagated_rows,
                            frame_index=indices[offset],
                            team_embedder=team_embedder,
                            evidence_by_track_id=team_evidence_by_track_id,
                            min_quality=config.min_team_sample_quality,
                        )

                # --- emit ---------------------------------------------------
                # A shared boundary frame belongs to the next segment, which will
                # run the detector on it. Only the final segment emits its last frame.
                    last_offset = len(frames) - 1 if segment.is_last else len(frames) - 2

                # Track records carry ORIGINAL video frame indices, so
                # downsampling never changes what a record refers to.
                # When the detector finds nothing, the frame keeps whatever SAM2
                # was already carrying, matching the original script's behaviour.
                    tracks.write_frame(indices[0], seed_rows_with_teams or _with_team_ids(active_rows, {}))
                    for offset in range(1, last_offset + 1):
                        tracks.write_frame(indices[offset], _with_team_ids(propagated.get(offset, []), {}))

                    active_rows = [] if segment.is_last else propagated.get(len(frames) - 1, [])

                    summary.segments_processed += 1
                    summary.frames_processed += last_offset + 1
                    if on_progress is not None:
                        on_progress(summary.frames_processed, expected_frames)

            summary.track_records = tracks.record_count
            summary.distinct_track_ids = len(tracks.track_ids)
            summary.suspected_swaps = len(suspected_swaps)
    finally:
        if prefetch is not None:
            prefetch.close()

    summary.detector_calls = detector.call_count
    summary.wall_seconds = round(time.monotonic() - started, 2)

    team_debug_summary: dict[str, dict] = {}
    selected_team_evidence_by_track_id: dict[int, list[TeamEvidenceSample]] = {}
    for track_id, samples in sorted(team_evidence_by_track_id.items()):
        selected = select_top_k_evidence(samples, config.max_team_samples_per_track)
        if selected:
            selected_team_evidence_by_track_id[track_id] = selected
        team_debug_summary[str(track_id)] = summarize_track_evidence(samples)

    team_result = _identify_track_teams(
        selected_team_evidence_by_track_id,
        backend=team_embedder.name,
        team_count=config.team_count,
        minimum_samples=config.min_team_samples_per_track,
        assignment_margin=config.team_assignment_margin,
    )
    for track_id, diagnostics in team_result.track_diagnostics.items():
        team_debug_summary.setdefault(str(track_id), {}).update(diagnostics)
    team_by_track_id = {
        int(assignment.track_id): int(assignment.team_id)
        for assignment in team_result.assignments
        if assignment.track_id is not None and assignment.team_id is not None
    }
    tracks_path = output_dir / TRACKS_FILENAME
    _rewrite_track_team_ids(tracks_path, team_by_track_id)

    results = PipelineResults(
        schema_version=RESULTS_SCHEMA_VERSION,
        video={
            "filename": video_path.name,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "width": info.width,
            "height": info.height,
            "duration_seconds": info.duration_seconds,
        },
        config={
            "target_fps": config.target_fps,
            "frame_step": step,
            "effective_fps": round(effective_fps, 2),
            "shot_target_fps": config.shot_target_fps,
            "shot_frame_step": shot_step,
            "shot_effective_fps": round(shot_effective_fps, 2),
            "frame_stride": config.frame_stride,
            "shot_frame_stride": config.shot_frame_stride,
            "confidence": config.confidence,
            "player_model_id": config.player_model_id,
            "shot_model_id": config.shot_model_id,
            "court_model_id": config.court_model_id,
            "court_keypoint_confidence": config.court_keypoint_confidence,
            "sam2_model_id": config.sam2_model_id,
            "sam2_match_iou": config.sam2_match_iou,
            "max_frames": config.max_frames,
            "max_missed_detections": config.max_missed_detections,
            "reid_backend": getattr(embedder, "name", config.reid_backend),
            "appearance_weight": config.appearance_weight,
            "team_embedder_backend": team_embedder.name,
            "team_embedder_model_id": config.team_embedder_model_id,
            "team_count": config.team_count,
            "min_team_samples_per_track": config.min_team_samples_per_track,
            "team_sample_stride": config.team_sample_stride,
            "min_team_sample_quality": config.min_team_sample_quality,
            "max_team_samples_per_track": config.max_team_samples_per_track,
            "team_assignment_margin": config.team_assignment_margin,
            "team_debug_enabled": config.team_debug_enabled,
            "number_predictor_backend": number_predictor.name,
        },
        summary=summary,
        device=tracker.device,
        teams_file=TEAMS_FILENAME,
        numbers_file=NUMBERS_FILENAME,
        shot_events_file=SHOT_EVENTS_FILENAME,
        projected_tracks_file=PROJECTED_TRACKS_FILENAME if projected_track_records > 0 else None,
        annotated_video_file=ANNOTATED_FILENAME if annotated_path else None,
    )

    _write_shot_events(
        video_path,
        output_dir / SHOT_EVENTS_FILENAME,
        info=info,
        step=shot_step,
        config=config,
    )

    number_prediction_records: list[NumberPredictionRecord] = []
    for sample in _iter_number_evidence_samples(number_evidence_by_track_id):
        prediction = number_predictor.predict(sample)
        tentative_number, accepted = normalize_number_prediction(
            prediction,
            threshold=config.number_predictor_score_threshold,
        )
        number_prediction_records.append(
            NumberPredictionRecord(
                frame_index=sample.frame_index,
                track_id=sample.track_id,
                track_state=sample.track_state,
                raw_text=prediction.raw_text,
                tentative_number=tentative_number,
                accepted=accepted,
                score=prediction.score,
                raw_response=prediction.raw_response,
            )
        )
    write_number_prediction_artifacts(output_dir, number_prediction_records)
    write_number_confirmation_artifacts(output_dir, number_prediction_records)
    if annotated_path is not None:
        _render_annotated_video(
            video_path,
            tracks_path,
            output_dir / NUMBERS_FILENAME,
            annotated_path,
            info=info,
            frame_stride=config.frame_stride,
            max_frames=config.max_frames,
            frame_step=step,
            fps=effective_fps,
        )

    (output_dir / TEAMS_FILENAME).write_text(
        json.dumps(
            {
                **team_result.to_dict(),
                "assignments_by_track_id": {
                    str(assignment.track_id): int(assignment.team_id)
                    for assignment in team_result.assignments
                    if assignment.track_id is not None and assignment.team_id is not None
                },
                "samples_by_track_id": {
                    str(track_id): len(selected_team_evidence_by_track_id.get(track_id, []))
                    for track_id in sorted(selected_team_evidence_by_track_id)
                },
                "track_evidence": team_debug_summary,
                "unassigned_track_ids": sorted(
                    int(track_id)
                    for track_id in tracks.track_ids
                    if track_id not in team_by_track_id
                ),
                "minimum_samples_per_track": config.min_team_samples_per_track,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if config.team_debug_enabled:
        write_debug_artifacts(output_dir, team_evidence_by_track_id)
        write_number_debug_artifacts(output_dir, number_evidence_by_track_id)

    # One line per suspected swap, so a run can be reviewed without watching the
    # video. Written even when empty, so its absence means "not computed" rather
    # than "nothing found".
    with (output_dir / SWAPS_FILENAME).open("w", encoding="utf-8") as handle:
        for record in suspected_swaps:
            handle.write(json.dumps(record) + "\n")

    if projected_track_records == 0 and projected_tracks_path.exists():
        projected_tracks_path.unlink()
    results.projected_shots_file, results.projected_shots_image_file = _maybe_write_projected_shots(output_dir)
    results.write(output_dir / RESULTS_FILENAME)

    logger.info(
        "done: %s frames, %s segments, %s detector calls, %s track records in %.1fs",
        summary.frames_processed,
        summary.segments_processed,
        summary.detector_calls,
        summary.track_records,
        summary.wall_seconds,
    )
    return results
