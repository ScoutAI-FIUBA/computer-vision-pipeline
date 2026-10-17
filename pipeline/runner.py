"""Pipeline orchestration.

`run_pipeline` is the single entry point everything else calls: the local CLI,
the Modal wrapper, a notebook. It takes a local video path and a config, and
writes its outputs to a directory.

Deliberately knows nothing about R2, Modal, or Supabase. The caller is
responsible for getting the video onto local disk and for doing something with
the output directory afterwards.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import tarfile
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from numbers import Real
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable

import numpy as np
import cv2

from pipeline import appearance
from pipeline.ball import (
    ball_track_record,
    ball_rows_from_result,
    build_tracknet_detector,
    candidate_to_ball_row,
    filter_ball_rows,
    reconcile_ball_track_id,
    select_seed_ball_rows,
    TRACKNET_CONTEXT_SIZE,
    write_ball_track_rows,
)
from pipeline.ball_contract import attach_shot_evidence, load_ball_track_rows, validate_ball_track_rows
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
from pipeline.v2_numbers import _remove_artifacts, run_v2_number_evidence
from pipeline.possession import resolve_possession_events
from pipeline.possession_transitions import mark_terminal_loss_rows, resolve_possession_transitions
from pipeline.possession_v3 import (
    PossessionResolver, derive_ball_primitives, derive_ball_transitions, classify_transition,
    normalize_ball_rows, score_candidates, score_candidates_v4, debounce_reverted_switches, V3_CONTRACT_VERSION, _point,
)
from pipeline.projected_shots import (
    _load_confirmed_numbers_by_stable_player_id,
    build_projected_shots,
    write_projected_shots_image,
)
from pipeline.reid import Embedder, build_embedder
from pipeline.results import (
    RESULTS_SCHEMA_VERSION, PipelineResults, RunSummary, TrackWriter, write_v3_artifacts,
    validate_stage_generation,
)
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
from pipeline.shot_outcome import refine_shot_events_file
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
from scripts.ball.tracknet import padded_window

logger = logging.getLogger(__name__)


def _possession_result_config(config: PipelineConfig, base: dict) -> dict:
    """Keep legacy result metadata byte-for-byte compatible; add V3 only when enabled."""
    if not config.possession_v3_enabled:
        return base
    return {
        **base,
        "possession_engine": config.possession_engine,
        "possession_v3_schema_version": V3_CONTRACT_VERSION,
        "possession_v3_reverted_switch_window_seconds": config.possession_v3_reverted_switch_window_seconds,
        "possession_v3_recovery_window_seconds": config.possession_v3_recovery_window_seconds,
    }

TRACKS_FILENAME = "tracks.jsonl"
RESULTS_FILENAME = "results.json"
ANNOTATED_FILENAME = "annotated.mp4"
SWAPS_FILENAME = "suspected_swaps.jsonl"
TEAMS_FILENAME = "teams.json"
NUMBERS_FILENAME = "numbers.json"
SHOT_EVENTS_FILENAME = "shot_events.jsonl"
SHOT_EVIDENCE_FILENAME = "shot_evidence.jsonl"
SHOT_SIGNALS_FILENAME = "shot_signals.jsonl"
SHOT_RAW_PREDICTIONS_FILENAME = "shot_raw_predictions.jsonl"
BALL_TRACKS_FILENAME = "ball_tracks.jsonl"
POSSESSION_EVENTS_FILENAME = "possession_events.jsonl"
POSSESSION_TRANSITIONS_FILENAME = "possession_transitions.jsonl"
POSSESSION_V3_FILENAME = "possession_v3.jsonl"
BALL_TRANSITIONS_V3_FILENAME = "ball_transitions_v3.jsonl"
POSSESSION_EVENTS_V3_FILENAME = "possession_events_v3.jsonl"
POSSESSION_V3_MANIFEST_FILENAME = "possession_v3_manifest.json"
PROJECTED_TRACKS_FILENAME = "projected_tracks.jsonl"
PROJECTED_SHOTS_FILENAME = "projected_shots.json"
PROJECTED_SHOTS_IMAGE_FILENAME = "projected_shots.png"
RELEASE_START_FIELDS = ("start_frame_index", "start_timestamp_seconds", "start_image_xy", "start_shot_xyxy")


def _build_detector_for_model(config: PipelineConfig, *, model_id: str | None = None):
    try:
        return build_detector(config, model_id=model_id)
    except TypeError as error:
        if "model_id" not in str(error):
            raise
        return build_detector(config)


def _frame_timestamp_seconds(frame_index: int, fps: float) -> float | None:
    return None if fps <= 0 else frame_index / fps


def _v3_ball_row_invalid(ball: object) -> bool:
    """Keep malformed evidence fail-closed without treating authoritative loss as invalid."""
    if not isinstance(ball, dict):
        return True
    state = ball.get("track_state")
    if state == "lost":
        geometry_fields = ("center_xy", "center", "ball_center", "x", "y", "center_x", "center_y", "xyxy")
        return ball.get("confidence") is not None or any(ball.get(field) is not None for field in geometry_fields)
    if state not in {"detected", "fallback"} or _point(ball) is None:
        return True
    confidence = ball.get("confidence", ball.get("score"))
    return type(confidence) not in {int, float} or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1


def _mask_probe(mask, box, *, frame_width=None, frame_height=None):
    try:
        shape = tuple(int(value) for value in mask.shape)
        if len(shape) != 2 or not all(shape):
            return None
        if getattr(mask.dtype, "kind", None) not in "biuf" or not np.isfinite(mask).all():
            return None
        values = mask.astype(bool, copy=False)
        if not values.any():
            return None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    x1, y1, x2, y2 = (float(value) for value in box)
    height, width = shape
    crop_width = round(x2 - x1)
    crop_height = round(y2 - y1)
    is_frame_mask = (
        type(frame_width) is int and type(frame_height) is int
        and width == frame_width and height == frame_height
    )
    if not is_frame_mask and (width != crop_width or height != crop_height):
        return None

    def location(x, y):
        if width == round(x2 - x1) and height == round(y2 - y1):
            return int(math.floor(x - x1)), int(math.floor(y - y1))
        return int(math.floor(x)), int(math.floor(y))

    def contains(x, y):
        px, py = location(x, y)
        return 0 <= px < width and 0 <= py < height and bool(values[py, px])

    foreground = [(int(x), int(y)) for y, x in zip(*values.nonzero())]

    def distance(x, y):
        px, py = location(x, y)
        return min(math.hypot(px - fx, py - fy) for fx, fy in foreground)

    return contains, distance


def _attach_v3_mask_evidence(rows, *, masks_by_observation=None, clean_masks_by_observation=None,
                             frame_width=None, frame_height=None):
    clean_masks_by_observation = clean_masks_by_observation or {}
    masks_by_observation = masks_by_observation or {}
    for row in rows:
        key = str(row.get("observation_key", f"{row.get('frame_index')}:{row.get('track_id')}"))
        mask = clean_masks_by_observation.get(key)
        clean = mask is not None
        box = row.get("xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        probe = _mask_probe(mask, box, frame_width=frame_width, frame_height=frame_height) if mask is not None else None
        if probe is None:
            mask = masks_by_observation.get(key)
            clean = False
            probe = _mask_probe(mask, box, frame_width=frame_width, frame_height=frame_height) if mask is not None else None
        if probe is not None:
            row["mask_contains"], row["mask_distance"] = probe
            row["mask_valid"] = True
            row["clean_mask"] = clean


def _build_v3_artifacts(output_dir: Path, rows: list[dict], ball_rows: list[dict], *, fps: float, config: PipelineConfig,
                        masks_by_observation=None, clean_masks_by_observation=None,
                        frame_width=None, frame_height=None) -> dict:
    """Run the additive V3 path; legacy files are never read or rewritten."""
    if not config.possession_v3_enabled:
        return {}
    by_frame: dict[int, list[dict]] = {}
    for row in rows:
        item = dict(row)
        item["xyxy"] = item.get("xyxy", item.get("bbox", item.get("box")))
        by_frame.setdefault(int(item["frame_index"]), []).append(item)
    _attach_v3_mask_evidence(
        [item for frame_rows in by_frame.values() for item in frame_rows],
        masks_by_observation=masks_by_observation,
        clean_masks_by_observation=clean_masks_by_observation,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    balls = {int(row["frame_index"]): row for row in ball_rows if isinstance(row, dict) and type(row.get("frame_index")) is int}
    # V5 is the a4-compatible profile: the a4 diff predates both the V4
    # scorer and resolver gates, so it uses score_candidates plus v4=False.
    # Keep this branch shared with V3 and explicit so V4 cannot leak diagnostics.
    resolver = PossessionResolver(real_fps=fps, grace_seconds=config.possession_v3_grace_seconds,
                                  recovery_window_seconds=config.possession_v3_recovery_window_seconds,
                                  hysteresis_margin=(config.possession_v4_ambiguity_margin
                                                     if config.possession_v4_enabled else config.possession_v3_hysteresis_margin),
                                  ambiguity_margin=(config.possession_v4_ambiguity_margin
                                                    if config.possession_v4_enabled else None),
                                   persistence_seconds=config.possession_v3_persistence_seconds,
                                   v4=config.possession_v4_enabled,
                                   holder_control_score=(config.holder_control_score
                                                         if config.possession_v4_enabled else .55),
                                   holder_release_score=(config.holder_release_score
                                                         if config.possession_v4_enabled else .35),
                                   challenger_acquisition_score=(config.challenger_acquisition_score
                                                                 if config.possession_v4_enabled else .6),
                                   release_persistence_seconds=(config.possession_v4_release_persistence_seconds
                                                                if config.possession_v4_enabled else 0.0),
                                   acquisition_persistence_seconds=(config.possession_v4_acquisition_persistence_seconds
                                                                   if config.possession_v4_enabled else 0.0),
                                   profile=config.possession_v3_calibration_profile)
    observations = []
    previous_ball = None
    previous_players = None
    for frame in sorted(set(by_frame) | set(balls)):
        ball = balls.get(frame)
        scorer = score_candidates_v4 if config.possession_v4_enabled else score_candidates
        candidates = scorer(by_frame.get(frame, []), ball, previous_holder=resolver.holder_stable,
                            ambiguity_margin=(config.possession_v4_ambiguity_margin
                                              if config.possession_v4_enabled else config.possession_v3_hysteresis_margin),
                             **({"disagreement_score": config.possession_v4_spatial_disagreement_score,
                                 "previous_ball": previous_ball,
                                 "previous_players": previous_players}
                                if config.possession_v4_enabled else {}))
        observation = resolver.resolve(frame, candidates, missing=ball is None,
                                       invalid=_v3_ball_row_invalid(ball),
                                       ball_lost=ball is not None and ball.get("track_state") == "lost",
                                       timestamp=ball.get("timestamp_seconds") if ball else None)
        observations.append(observation)
        if ball is not None:
            previous_ball = ball
        if frame in by_frame:
            previous_players = by_frame[frame]
    observations = debounce_reverted_switches(
        observations, real_fps=fps,
        window_seconds=config.possession_v3_reverted_switch_window_seconds,
    )
    normalized = normalize_ball_rows(ball_rows, real_fps=fps, config=config)
    primitives = derive_ball_primitives(normalized, observations, real_fps=fps)
    transitions = derive_ball_transitions(normalized, observations, real_fps=fps,
                                          event_window_seconds=config.possession_v3_event_window_seconds)
    events = [event for transition in transitions for event in classify_transition(transition)]
    manifest = write_v3_artifacts(output_dir, observations, primitives, transitions, events,
                                   {"profile": config.possession_v3_calibration_profile,
                                     "input_contract_version": config.possession_v3_input_contract_version,
                                      "event_window_seconds": config.possession_v3_event_window_seconds,
                                      "recovery_window_seconds": config.possession_v3_recovery_window_seconds,
                                     "reverted_switch_window_seconds": config.possession_v3_reverted_switch_window_seconds,
                                      "engine": config.possession_engine})
    manifest["observability"] = {"status": "complete", "phase": "artifact_write",
                                  "counts": {"observations": len(observations), "primitives": len(primitives),
                                             "transitions": len(transitions), "events": len(events)},
                                  "elapsed_seconds": 0.0, "error": None}
    return manifest


def _write_v3_artifacts(output_dir: Path, rows: list[dict], ball_rows: list[dict], *, fps: float, config: PipelineConfig,
                        masks_by_observation=None, clean_masks_by_observation=None,
                        frame_width=None, frame_height=None) -> dict:
    """Keep optional V3 diagnostics non-authoritative when they fail."""
    if not config.possession_v3_enabled:
        return {}
    started = time.monotonic()
    try:
        manifest = _build_v3_artifacts(
            output_dir, rows, ball_rows, fps=fps, config=config,
            masks_by_observation=masks_by_observation,
            clean_masks_by_observation=clean_masks_by_observation,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        manifest["observability"]["elapsed_seconds"] = round(time.monotonic() - started, 6)
        try:
            (output_dir / "possession_v3_failure.json").unlink(missing_ok=True)
        except OSError:
            pass
        try:
            logger.info("v3_observability %s", json.dumps(manifest["observability"], sort_keys=True))
        except Exception:
            pass
        return manifest
    except Exception as error:
        failure = {"schema": "possession-engine-v3", "schema_version": V3_CONTRACT_VERSION,
                   "status": "failed", "error": f"{type(error).__name__}: {error}",
                   "observability": {"status": "failed", "phase": "artifact_write",
                                      "counts": {}, "elapsed_seconds": round(time.monotonic() - started, 6),
                                      "error": f"{type(error).__name__}: {error}"}}
        try:
            (output_dir / "possession_v3_failure.json").write_text(json.dumps(failure, sort_keys=True), encoding="utf-8")
        except OSError:
            pass
        try:
            logger.error("v3_observability %s", json.dumps(failure["observability"], sort_keys=True))
        except Exception:
            pass
        return failure


def _v3_artifact_path(manifest: dict, name: str) -> str | None:
    artifact = manifest.get("artifacts", {}).get(name)
    return artifact.get("path") if isinstance(artifact, dict) else None


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
) -> int:
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


def _annotated_frame_rows(
    track_rows: list[dict],
    *,
    confirmed_numbers_by_track_id: dict[int, str],
    ball_rows: list[dict] | None = None,
) -> list[dict]:
    rows = [
        {
            **row,
            **(
                {"confirmed_number": confirmed_numbers_by_track_id[int(row["track_id"])]}
                if row.get("track_id") is not None
                and int(row["track_id"]) in confirmed_numbers_by_track_id
                else {}
            ),
        }
        for row in track_rows
    ]
    if ball_rows:
        rows.extend(dict(row) for row in ball_rows)
    return rows


def _cap_ball_fallback_rows(rows: list[dict], remaining: int) -> tuple[list[dict], int]:
    capped: list[dict] = []
    emitted = 0
    for row in rows:
        if row.get("track_state") != "fallback" or emitted < remaining:
            capped.append(row)
            if row.get("track_state") == "fallback":
                emitted += 1
            continue
        capped.append({**row, "track_state": "lost", "confidence": None, "xyxy": None})
    return capped, emitted


def _collect_tracknet_rows(
    video_path: Path,
    *,
    info,
    config: PipelineConfig,
    detector,
    tracker=None,
    player_step: int,
    output_path: Path,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> None:
    """Run TrackNet plus bounded ball-only SAM2 fallback on a dense stream."""
    from scripts.ball.candidates import extract_candidates

    ball_step = 1 if config.ball_all_frames else frame_step_for(info.fps, config.ball_target_fps)
    ball_max_frames = None
    if config.max_frames is not None:
        ball_max_frames = (config.max_frames * player_step + ball_step - 1) // ball_step

    # Keep only the temporal window and the immediately previous frame. The
    # JSONL is the merge boundary; it avoids retaining a whole match in RAM.
    frames_by_index: dict[int, object] = {}
    indices: list[int] = []
    previous_frame = None
    active_rows: list[dict] = []
    next_track_id = 1
    fallback_remaining = 0
    records = 0
    if on_progress is not None:
        on_progress(0, info.frame_count if info.frame_count > 0 else None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for segment in iter_segments(
            video_path, frame_stride=1, max_frames=ball_max_frames, frame_step=ball_step
        ):
            for frame, frame_index in zip(segment.frames, segment.frame_indices):
                if frame_index in frames_by_index:
                    continue
                frames_by_index[frame_index] = frame
                indices.append(frame_index)
                window_indices = padded_window(indices, len(indices) - 1, TRACKNET_CONTEXT_SIZE)
                prediction = detector.predict(
                    [frames_by_index[index] for index in window_indices], window_indices
                )
                candidates = extract_candidates(
                    prediction.heatmap,
                    original_frame_size=prediction.source_frame_size,
                    min_score=config.tracknet_heatmap_threshold,
                    min_distance=config.tracknet_seed_radius,
                    max_candidates=1,
                )
                rows = (
                    [
                        candidate_to_ball_row(
                            candidates[0],
                            frame_size=prediction.source_frame_size,
                            radius=config.tracknet_seed_radius,
                        )
                    ]
                    if candidates else []
                )
                if rows:
                    rows, next_track_id = reconcile_ball_track_id(rows, active_rows, next_track_id)
                    active_rows = rows
                    fallback_remaining = config.tracknet_fallback_budget
                elif tracker is not None and active_rows and previous_frame is not None and fallback_remaining > 0:
                    fallback_seed = {**active_rows[0], "track_state": "fallback"}
                    propagated = tracker.propagate_segment([previous_frame, frame], [fallback_seed])
                    rows = filter_ball_rows(propagated.get(1, []))
                    if rows:
                        rows[0]["track_state"] = "fallback"
                        active_rows = rows
                        fallback_remaining -= 1
                if not rows:
                    rows = [{
                        "track_id": active_rows[0].get("track_id") if active_rows else None,
                        "track_state": "lost",
                        "class_name": "ball",
                        "confidence": None,
                        "xyxy": None,
                    }]
                record = ball_track_record(
                    frame_index,
                    rows[0],
                    timestamp_seconds=_frame_timestamp_seconds(frame_index, info.fps),
                )
                output.write(json.dumps(record) + "\n")
                records += 1
                if on_progress is not None:
                    on_progress(records, info.frame_count if info.frame_count > 0 else None)

                # The temporal window is causal; discard frames older than it.
                previous_frame = frame
                while len(indices) > TRACKNET_CONTEXT_SIZE:
                    old_index = indices.pop(0)
                    frames_by_index.pop(old_index, None)
    return records


class _BallRecordReader:
    """Forward-only lookup for dense TrackNet records during player playback."""

    def __init__(self, path: Path) -> None:
        self._source = path.open("r", encoding="utf-8")
        self._pending = self._next()

    def _next(self) -> dict | None:
        line = self._source.readline()
        return json.loads(line) if line else None

    def take(self, frame_index: int) -> tuple[bool, list[dict]]:
        while self._pending is not None and self._pending["frame_index"] < frame_index:
            self._pending = self._next()
        if self._pending is None or self._pending["frame_index"] != frame_index:
            return False, []
        row = self._pending
        self._pending = self._next()
        # Presence is separate from drawable rows: an authoritative lost row
        # must suppress SAM2 propagation at this exact source frame.
        return True, ([] if row.get("xyxy") is None else [row])

    def close(self) -> None:
        self._source.close()


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


def run_possession_only_from_v2_artifacts(
    output_dir: Path,
    observations: list[dict],
    ball_rows: list[dict],
    *,
    fps: float,
    config: PipelineConfig,
    masks_by_observation=None,
    clean_masks_by_observation=None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict[str, Any]:
    """Generate possession artifacts only from already-persisted V2 evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = config.possession_engine
    if frame_width and frame_height:
        observations = _normalize_v2_stats_rows(observations, width=frame_width, height=frame_height)
    if engine in {"v3", "v4", "v5"}:
        manifest = _build_v3_artifacts(
            output_dir, observations, ball_rows, fps=fps, config=config,
            masks_by_observation=masks_by_observation,
            clean_masks_by_observation=clean_masks_by_observation,
        )
        files = [path for path in output_dir.rglob("*") if path.is_file()]
    elif engine == "legacy":
        team_ids = {
            int(row["track_id"]): int(row["team_id"])
            for row in observations
            if row.get("track_id") is not None and type(row.get("team_id")) is int
        }
        possession, transitions = _write_possession_artifacts(
            output_dir, observations, ball_rows, fps=fps, frame_step=1,
            team_ids_by_track_id=team_ids, max_frame_gap=config.possession_transition_max_gap,
        )
        manifest = {"engine": "legacy", "artifacts": {"possession": possession.name, "transitions": transitions.name}}
        files = [possession, transitions]
    else:
        raise ValueError(f"unsupported possession engine: {engine}")
    summary = {
        "schema": "possession-only-v1",
        "engine": engine,
        "observation_count": len(observations),
        "ball_row_count": len(ball_rows),
        "artifact_count": len(files),
        "possession": manifest,
    }
    (output_dir / "results.json").write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    return summary


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


def _write_possession_events(
    output_dir: Path,
    tracks: list[dict],
    ball_rows: list[dict],
    *,
    fps: float | None,
    frame_step: int = 1,
    team_ids_by_track_id: dict[int, int] | None = None,
) -> Path:
    """Write possession after final team IDs are available."""
    events = resolve_possession_events(
        tracks,
        ball_rows,
        fps=fps,
        frame_step=frame_step,
        team_ids_by_track_id=team_ids_by_track_id,
    )
    path = output_dir / POSSESSION_EVENTS_FILENAME
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return path


def _write_possession_transitions(output_dir: Path, *, max_frame_gap: int) -> Path:
    source_path = output_dir / POSSESSION_EVENTS_FILENAME
    path = output_dir / POSSESSION_TRANSITIONS_FILENAME
    possession_rows = mark_terminal_loss_rows(_load_jsonl_rows(source_path), max_frame_gap=max_frame_gap)
    source_path.write_text("".join(json.dumps(row) + "\n" for row in possession_rows), encoding="utf-8")
    transitions = resolve_possession_transitions(possession_rows, max_frame_gap=max_frame_gap)
    path.write_text("".join(json.dumps(event) + "\n" for event in transitions), encoding="utf-8")
    return path


def _write_possession_artifacts(
    output_dir: Path,
    tracks: list[dict],
    ball_rows: list[dict],
    *,
    fps: float | None,
    frame_step: int = 1,
    team_ids_by_track_id: dict[int, int] | None = None,
    max_frame_gap: int = 15,
) -> tuple[Path, Path]:
    possession_path = _write_possession_events(
        output_dir, tracks, ball_rows, fps=fps, frame_step=frame_step,
        team_ids_by_track_id=team_ids_by_track_id,
    )
    transitions_path = _write_possession_transitions(output_dir, max_frame_gap=max_frame_gap)
    return possession_path, transitions_path


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
                    **({"stable_player_id": row["stable_player_id"]}
                       if row.get("stable_player_id") is not None else {}),
                    **({"team_id": int(row["team_id"])} if row.get("team_id") is not None else {}),
                }
            )
            + "\n"
        )
        written += 1
    return written


def _read_video_frames(video_path: Path, frame_indices: Sequence[int]) -> dict[int, object]:
    """Read only the frames needed by an optional court projection pass."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    frames: dict[int, object] = {}
    try:
        for frame_index in sorted(set(int(value) for value in frame_indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok:
                frames[frame_index] = frame
    finally:
        capture.release()
    return frames


def _project_v2_rows(
    video_path: Path,
    output_path: Path,
    rows: list[dict],
    *,
    info,
    config: PipelineConfig,
    emit: Callable[[str], None],
    counters: dict[str, int] | None = None,
) -> int:
    """Project authoritative V2 rows without creating a second player pipeline."""
    if counters is not None:
        counters["court_detection_frames"] = 0
    if not config.court_model_id or not rows:
        return 0
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["frame_index"]), []).append(row)
    try:
        detector = _build_detector_for_model(config, model_id=config.court_model_id)
        frames = _read_video_frames(video_path, grouped)
        written = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for frame_index in sorted(grouped):
                frame = frames.get(frame_index)
                if frame is None:
                    continue
                try:
                    if counters is not None:
                        counters["court_detection_frames"] += 1
                    keypoints = parse_court_keypoints(
                        detector.detect(frame),
                        min_confidence=config.court_keypoint_confidence,
                        width=info.width,
                        height=info.height,
                    )
                    homography = compute_frame_homography(
                        keypoints, min_confidence=config.court_keypoint_confidence
                    )
                except (RuntimeError, ValueError) as error:
                    emit(f"court projection frame {frame_index} unavailable: {error}")
                    continue
                written += _write_projected_track_rows(
                    handle, frame_index=frame_index, rows=grouped[frame_index], homography_result=homography
                )
        if written:
            emit(f"court projection complete: {written} projected rows")
        else:
            output_path.unlink(missing_ok=True)
            emit("court projection unavailable: no valid homography")
        return written
    except Exception as error:
        output_path.unlink(missing_ok=True)
        emit(f"court projection failed: {type(error).__name__}: {error}")
        return 0


def _maybe_write_projected_shots(output_dir: Path) -> tuple[str | None, str | None]:
    projected_tracks_path = output_dir / PROJECTED_TRACKS_FILENAME
    projected_shots_path = output_dir / PROJECTED_SHOTS_FILENAME
    projected_shots_image_path = output_dir / PROJECTED_SHOTS_IMAGE_FILENAME
    if not projected_tracks_path.exists():
        projected_shots_path.unlink(missing_ok=True)
        projected_shots_image_path.unlink(missing_ok=True)
        return None, None

    shot_events_path = output_dir / SHOT_EVENTS_FILENAME
    projected_tracks = _load_jsonl_rows(projected_tracks_path)
    if not projected_tracks:
        projected_tracks_path.unlink(missing_ok=True)
        projected_shots_path.unlink(missing_ok=True)
        projected_shots_image_path.unlink(missing_ok=True)
        return None, None
    artifact = build_projected_shots(
        _load_jsonl_rows(shot_events_path),
        projected_tracks,
        team_ids_by_track_id=_load_team_ids_by_track_id(output_dir / TEAMS_FILENAME),
        confirmed_numbers_by_track_id=_load_confirmed_numbers_by_track_id(output_dir / NUMBERS_FILENAME),
        confirmed_numbers_by_stable_player_id=_load_confirmed_numbers_by_stable_player_id(output_dir / NUMBERS_FILENAME),
        raw_prediction_rows=(
            _load_jsonl_rows(output_dir / SHOT_RAW_PREDICTIONS_FILENAME)
            if (output_dir / SHOT_RAW_PREDICTIONS_FILENAME).exists()
            else None
        ),
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
    ball_tracks_path: Path | None = None,
) -> None:
    from pipeline.annotate import draw_tracks

    confirmed_numbers_by_track_id = _load_confirmed_numbers_by_track_id(numbers_path)
    rows_by_frame = _iter_track_rows_by_frame(tracks_path)
    ball_rows = (
        _iter_track_rows_by_frame(ball_tracks_path)
        if ball_tracks_path is not None and ball_tracks_path.exists()
        else iter(())
    )
    pending = next(rows_by_frame, None)
    pending_ball = next(ball_rows, None)

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
                track_rows = []
                if pending is not None and pending[0] == frame_index:
                    track_rows = pending[1]
                    pending = next(rows_by_frame, None)
                elif pending is not None and pending[0] < frame_index:
                    raise RuntimeError(
                        f"tracks.jsonl is out of sync with processed frames at frame {frame_index}"
                    )
                while pending_ball is not None and pending_ball[0] < frame_index:
                    pending_ball = next(ball_rows, None)
                current_ball_rows = []
                if pending_ball is not None and pending_ball[0] == frame_index:
                    current_ball_rows = pending_ball[1]
                    pending_ball = next(ball_rows, None)
                rows = _annotated_frame_rows(
                    track_rows,
                    confirmed_numbers_by_track_id=confirmed_numbers_by_track_id,
                    ball_rows=current_ball_rows,
                )
                writer.write(draw_tracks(segment.frames[offset], rows))

    if pending is not None:
        raise RuntimeError(
            f"tracks.jsonl has extra frame rows starting at frame {pending[0]}"
        )
def _write_shot_events(
    video_path: Path,
    output_path: Path,
    *,
    info,
    step: int,
    config: PipelineConfig,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> None:
    detector = _build_detector_for_model(config, model_id=config.shot_model_id)
    active_shot_event = None
    next_shot_event_id = 1
    last_emitted_frame_index = 0
    signals_path = output_path.with_name(SHOT_SIGNALS_FILENAME)
    raw_predictions_path = output_path.with_name(SHOT_RAW_PREDICTIONS_FILENAME)
    processed = 0
    if on_progress is not None:
        on_progress(0, info.frame_count if info.frame_count > 0 else None)

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
            processed += 1
            if on_progress is not None:
                on_progress(processed, info.frame_count if info.frame_count > 0 else None)

        final_shot_event = flush_open_shot_event(
            active_shot_event,
            frame_index=last_emitted_frame_index,
            timestamp_seconds=_frame_timestamp_seconds(last_emitted_frame_index, info.fps),
        )
        if final_shot_event is not None:
            write_shot_event_record(shot_events_file, final_shot_event)


def _refine_shot_outcomes(output_dir: Path, config: PipelineConfig) -> None:
    if not getattr(config, "shot_rim_trajectory_outcome", True):
        return
    refine_shot_events_file(
        output_dir / SHOT_EVENTS_FILENAME,
        output_dir / SHOT_RAW_PREDICTIONS_FILENAME,
        output_dir / BALL_TRACKS_FILENAME,
    )


def _attach_ball_evidence(
    output_dir: Path, *, expected_frame_indices, team_ids_by_track_id: dict[int, int] | None = None,
    projected_shots: dict | None = None,
) -> None:
    events = _load_jsonl_rows(output_dir / SHOT_EVENTS_FILENAME)
    evidence_path = output_dir / SHOT_EVIDENCE_FILENAME
    tracks = _load_jsonl_rows(output_dir / TRACKS_FILENAME)
    if team_ids_by_track_id is None:
        team_ids_by_track_id = _load_team_ids_by_track_id(output_dir / TEAMS_FILENAME)
    if projected_shots is None:
        projected_shots_path = output_dir / PROJECTED_SHOTS_FILENAME
        if projected_shots_path.exists():
            projected_shots = json.loads(projected_shots_path.read_text(encoding="utf-8"))
    projected_by_event_id = None
    if projected_shots is not None:
        projected_by_event_id = {
            shot.get("event_id"): shot
            for shot in projected_shots.get("shots", [])
            if isinstance(shot, dict) and shot.get("event_id") is not None
        }
        # Evidence windows follow the corrected release frame; shot_events.jsonl itself stays as detected.
        events = [
            {**event, **{key: projected[key] for key in RELEASE_START_FIELDS}}
            if (projected := projected_by_event_id.get(event.get("event_id")))
            and projected.get("start_frame_source") == "action_box_release"
            else event
            for event in events
        ]
    enriched = attach_shot_evidence(
        events,
        load_ball_track_rows(output_dir / BALL_TRACKS_FILENAME),
        expected_frame_indices=expected_frame_indices,
        player_rows=tracks,
        team_ids_by_track_id=team_ids_by_track_id,
    )
    if projected_by_event_id is not None:
        for event in enriched:
            identity = event.get("player_identity", {})
            projected = projected_by_event_id.get(event.get("event_id"))
            stable_player_id = projected.get("shooter_stable_player_id") if projected else None
            if identity.get("status") != "unknown" or not isinstance(stable_player_id, str) or not stable_player_id.strip():
                continue
            event["player_identity"] = {
                "status": "inferred",
                "track_id": projected.get("shooter_track_id"),
                "stable_player_id": stable_player_id.strip(),
                "team_id": projected.get("shooter_team_id"),
                "attribution_method": "projected_shot_track",
                "authority": "non_authoritative",
            }
    evidence_path.write_text("".join(json.dumps(event) + "\n" for event in enriched), encoding="utf-8")


def _normalize_v2_stats_rows(observations, *, width: int, height: int) -> list[dict]:
    rows = []
    for observation in observations:
        row = dict(observation)
        track_id = row.get("raw_track_id", row.get("track_id"))
        bbox = row.get("bbox_xyxy", row.get("xyxy"))
        if track_id is None or bbox is None:
            continue
        if (
            isinstance(bbox, (str, bytes, bytearray))
            or not isinstance(bbox, Sequence)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, Real) for value in bbox)
        ):
            continue
        coordinate_space = row.get("bbox_coordinate_space")
        if coordinate_space not in (None, "pixel", "normalized"):
            continue
        try:
            coordinates = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
            continue
        if coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
            continue
        if coordinate_space == "normalized":
            if not all(0.0 <= value <= 1.0 for value in coordinates):
                continue
            coordinates = [
                coordinates[0] * width,
                coordinates[1] * height,
                coordinates[2] * width,
                coordinates[3] * height,
            ]
        rows.append({
            "frame_index": int(row["frame_index"]),
            "track_id": int(track_id),
            "observation_key": row.get("observation_key"),
            "mask_ref": row.get("mask_ref"),
            "clean_mask_ref": row.get("clean_mask_ref"),
            "stable_player_id": row.get("stable_player_id"),
            "team_id": row.get("team_id"),
            "track_state": row.get("track_state", "unknown"),
            "class_name": "player",
            "confidence": row.get("confidence"),
            "xyxy": coordinates,
        })
    return sorted(rows, key=lambda row: (row["frame_index"], row["track_id"]))


class _BoundedProgress:
    """Turn frame counts into sparse progress messages without division hazards."""

    def __init__(self, emit: Callable[[str], None], label: str) -> None:
        self.emit = emit
        self.label = label
        self.next_percent = 0
        self.processed = 0

    def __call__(self, processed: int, total: int | None) -> None:
        self.processed = processed
        if total is None or total <= 0:
            if processed == 0 or processed == total:
                self.emit(f"{self.label} progress: {processed} frames")
            return
        percent = min(100, int(processed * 100 / total))
        while percent >= self.next_percent and self.next_percent <= 100:
            self.emit(f"{self.label} progress: {processed}/{total} frames ({self.next_percent}%)")
            self.next_percent += 25

    def complete(self, total: int | None) -> None:
        if total is None or total <= 0:
            self.emit(f"{self.label} progress: phase complete (100%; processed {self.processed} frames)")
        elif self.processed == total:
            self.emit(f"{self.label} progress: phase complete (100%; processed {self.processed}/{total} frames)")
        else:
            self.emit(
                f"{self.label} progress: phase complete (100%; processed {self.processed} frames; "
                f"source progress {self.processed}/{total})"
            )


def _emit_file_batch_progress(
    emit: Callable[[str], None], label: str, processed: int, total: int, *, batch_size: int = 100
) -> None:
    """Keep large artifact loads observable without logging every small-file read."""
    if processed == total or processed % batch_size == 0:
        emit(f"{label}: {processed}/{total} files")


_STATS_PHASE_METRIC_KEYS = {
    "court projection": "court_projection_seconds",
    "TrackNet": "tracknet_seconds",
    "shot detection": "shot_detection_seconds",
    "possession": "possession_seconds",
    "possession V3": "possession_seconds",
    "V2 jersey numbers": "number_evidence_seconds",
    "artifact writing": "artifact_write_seconds",
}


def run_stats_from_v2_observations(
    video_path: Path,
    config: PipelineConfig,
    output_dir: Path,
    observations,
    *,
    on_progress: Callable[[str], None] | None = None,
    number_predictor=None,
    masks_by_observation=None,
    clean_masks_by_observation=None,
    number_detector=None,
    metrics=None,
) -> PipelineResults:
    """Run stats stages from authoritative V2 player rows without player tracking."""
    emit = on_progress or (lambda message: print(f"[v2-stats] {message}", flush=True))

    def phase(name: str, operation):
        started = time.perf_counter()
        try:
            result = operation()
        except Exception as error:
            emit(f"{name} failed: {type(error).__name__}: {error}")
            raise
        if metrics is not None and name in _STATS_PHASE_METRIC_KEYS:
            metrics.add_phase(_STATS_PHASE_METRIC_KEYS[name], time.perf_counter() - started)
        return result

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emit("stats start")
    info = phase("probe", lambda: probe(video_path))
    rows = phase(
        "normalize observations",
        lambda: _normalize_v2_stats_rows(observations, width=info.width, height=info.height),
    )
    frame_indices = sorted({row["frame_index"] for row in rows})
    tracks_path = output_dir / TRACKS_FILENAME
    def write_tracks():
        with tracks_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                record = {key: value for key, value in row.items() if value is not None}
                handle.write(json.dumps(record) + "\n")

    phase("TrackNet/player artifact writing", write_tracks)

    projected_tracks_path = output_dir / PROJECTED_TRACKS_FILENAME
    court_counters: dict[str, int] = {}
    projected_track_records = phase(
        "court projection",
        lambda: _project_v2_rows(
            video_path, projected_tracks_path, rows, info=info, config=config, emit=emit,
            **({"counters": court_counters} if metrics is not None else {}),
        ),
    )

    team_by_track_id: dict[int, int] = {}
    team_values: dict[int, set[int]] = {}
    for row in rows:
        if row.get("team_id") is not None:
            team_values.setdefault(row["track_id"], set()).add(int(row["team_id"]))
    team_by_track_id = {
        track_id: next(iter(values)) for track_id, values in team_values.items() if len(values) == 1
    }
    phase("team artifact writing", lambda: (output_dir / TEAMS_FILENAME).write_text(
        json.dumps({
            "assignments_by_track_id": {str(key): value for key, value in team_by_track_id.items()},
            "unassigned_track_ids": sorted({row["track_id"] for row in rows} - set(team_by_track_id)),
            "source": "sam31_v2_observations",
        }, indent=2), encoding="utf-8"
    ))

    ball_path = output_dir / BALL_TRACKS_FILENAME
    if config.tracknet_enabled:
        if not config.tracknet_checkpoint:
            error = RuntimeError("TrackNet is enabled but TRACKNET_CHECKPOINT is not configured")
            emit(f"TrackNet failed: {type(error).__name__}: {error}")
            raise error
        emit("TrackNet start")
        tracknet_progress = _BoundedProgress(emit, "TrackNet")

        def collect_tracknet():
            _collect_tracknet_rows(
                video_path, info=info, config=config,
                detector=build_tracknet_detector(config.tracknet_checkpoint),
                tracker=None, player_step=1, output_path=ball_path,
                on_progress=tracknet_progress,
            )

        phase("TrackNet", collect_tracknet)
        tracknet_progress.complete(info.frame_count if info.frame_count > 0 else None)
        ball_rows = phase("TrackNet artifact filtering", lambda: [
            row for row in load_ball_track_rows(ball_path) if row["frame_index"] in frame_indices
        ])
        phase("TrackNet artifact writing", lambda: ball_path.write_text(
            "".join(json.dumps(row) + "\n" for row in ball_rows), encoding="utf-8"
        ))
    else:
        # No ball model is silently substituted: exact V2 frames remain explicit lost evidence.
        ball_rows = [ball_track_record(frame, {"track_state": "lost", "class_name": "ball"},
                                       timestamp_seconds=_frame_timestamp_seconds(frame, info.fps))
                     for frame in frame_indices]
        phase("ball artifact writing", lambda: ball_path.write_text(
            "".join(json.dumps(row) + "\n" for row in ball_rows), encoding="utf-8"
        ))

    emit("shot evidence start")
    shot_progress = _BoundedProgress(emit, "shot detection")
    shot_frames: list[int] = []

    def track_shot_progress(processed: int, total: int | None) -> None:
        shot_progress(processed, total)
        shot_frames[:] = [processed]

    phase("shot detection", lambda: _write_shot_events(
        video_path, output_dir / SHOT_EVENTS_FILENAME, info=info, step=1, config=config,
        on_progress=track_shot_progress if metrics is not None else shot_progress,
    ))
    shot_progress.complete(info.frame_count if info.frame_count > 0 else None)
    phase("shot outcome refinement", lambda: _refine_shot_outcomes(output_dir, config))
    projected_shots_file, projected_shots_image_file = phase(
        "projected shot artifacts", lambda: _maybe_write_projected_shots(output_dir)
    )
    phase("shot evidence", lambda: _attach_ball_evidence(
        output_dir, expected_frame_indices=frame_indices, team_ids_by_track_id=team_by_track_id
    ))
    emit("possession start")
    possession_path, transitions_path = phase("possession", lambda: _write_possession_artifacts(
        output_dir, rows, ball_rows, fps=info.fps, frame_step=1,
        team_ids_by_track_id=team_by_track_id,
        max_frame_gap=config.possession_transition_max_gap,
    ))
    v3_manifest = phase("possession V3", lambda: _write_v3_artifacts(
        output_dir, rows, ball_rows, fps=info.fps, config=config,
        masks_by_observation=masks_by_observation,
        clean_masks_by_observation=clean_masks_by_observation,
        frame_width=info.width,
        frame_height=info.height,
    ))
    _remove_artifacts(output_dir)
    numbers_available = False
    number_counters: dict[str, int] = {"number_detector_frames": 0, "ocr_crops": 0}
    if getattr(config, "number_evidence_enabled", True) and number_predictor is not None:
        numbers_available = phase("V2 jersey numbers", lambda: run_v2_number_evidence(
            video_path, rows, config, output_dir, predictor=number_predictor,
            masks_by_observation=masks_by_observation,
            clean_masks_by_observation=clean_masks_by_observation,
            detector=number_detector,
            **({"counters": number_counters} if metrics is not None else {}),
        ))
    if metrics is not None:
        # None stays "unknown": a replaced helper that never reported must not look like zero calls.
        metrics.set_api_calls(
            roboflow_shot_detection_frames=shot_frames[0] if shot_frames else None,
            roboflow_court_detection_frames=court_counters.get("court_detection_frames"),
            roboflow_number_detector_frames=number_counters["number_detector_frames"],
            # The dummy predictor makes no hosted call.
            roboflow_ocr_crops=(number_counters["ocr_crops"]
                                if config.number_predictor_backend == "roboflow-hosted" else 0),
        )
    results = PipelineResults(
        schema_version=RESULTS_SCHEMA_VERSION,
        video={"filename": video_path.name, "fps": info.fps, "frame_count": info.frame_count,
               "width": info.width, "height": info.height, "duration_seconds": info.duration_seconds},
         config=_possession_result_config(
             config, {"source": "sam31_v2_observations", "tracknet_enabled": config.tracknet_enabled,
                       "court_model_id": config.court_model_id,
                       "court_keypoint_confidence": config.court_keypoint_confidence,
                       "number_predictor_backend": config.number_predictor_backend,
                      "number_predictor_score_threshold": config.number_predictor_score_threshold}
         ),
        summary=RunSummary(frames_processed=len(frame_indices), track_records=len(rows),
                           distinct_track_ids=len({row["track_id"] for row in rows})),
         device="cpu", teams_file=TEAMS_FILENAME, shot_events_file=SHOT_EVENTS_FILENAME,
         numbers_file=NUMBERS_FILENAME if numbers_available else None,
         shot_evidence_file=SHOT_EVIDENCE_FILENAME, ball_tracks_file=BALL_TRACKS_FILENAME,
         possession_events_file=POSSESSION_EVENTS_FILENAME,
          possession_transitions_file=POSSESSION_TRANSITIONS_FILENAME,
           projected_tracks_file=PROJECTED_TRACKS_FILENAME if projected_track_records else None,
          projected_shots_file=projected_shots_file,
          projected_shots_image_file=projected_shots_image_file,
         possession_v3_file=_v3_artifact_path(v3_manifest, "possession") if v3_manifest and v3_manifest.get("status") != "failed" else None,
         possession_v3_transitions_file=_v3_artifact_path(v3_manifest, "transitions") if v3_manifest and v3_manifest.get("status") != "failed" else None,
         possession_v3_events_file=_v3_artifact_path(v3_manifest, "events") if v3_manifest and v3_manifest.get("status") != "failed" else None,
         possession_v3_manifest_file=POSSESSION_V3_MANIFEST_FILENAME if v3_manifest and v3_manifest.get("status") != "failed" else None,
           possession_v3_schema_version=V3_CONTRACT_VERSION if v3_manifest and v3_manifest.get("status") != "failed" else None,
           possession_v3_observability=v3_manifest.get("observability") if v3_manifest else None,
           possession_v4_diagnostics_file=_v3_artifact_path(v3_manifest, "v4_diagnostics")
           if v3_manifest and v3_manifest.get("v4", {}).get("enabled") else None,
           possession_v4_schema_version=1 if v3_manifest and v3_manifest.get("v4", {}).get("enabled") else None,
          artifact_references={
            "possession_events": {"file": possession_path.name, "schema_version": 1},
             "possession_transitions": {
                "file": transitions_path.name, "schema_version": 1,
                "event_types": ["pass", "steal", "turnover", "lost_possession"],
             },
             **({"numbers": {"file": NUMBERS_FILENAME, "manifest": "number_manifest.json",
                               "source": "sam31_v2_observations",
                               "artifacts": json.loads((output_dir / "number_manifest.json").read_text(encoding="utf-8")).get("artifacts", {})}}
                if numbers_available else {}),
              **({"possession_v3": {"manifest": POSSESSION_V3_MANIFEST_FILENAME,
                                     "artifacts": v3_manifest.get("artifacts", {})}} if v3_manifest and v3_manifest.get("status") != "failed" else {}),
              **({"possession_v4": {"feature_flag": "v4", "diagnostics": _v3_artifact_path(v3_manifest, "v4_diagnostics")}}
                 if v3_manifest and v3_manifest.get("v4", {}).get("enabled") else {}),
             **({"court_projection": {
                 "projected_tracks": PROJECTED_TRACKS_FILENAME,
                 "projected_shots": PROJECTED_SHOTS_FILENAME,
                 "projected_shots_image": PROJECTED_SHOTS_IMAGE_FILENAME,
             }} if projected_track_records else {}),
         },
    )
    emit("artifact writing start")
    phase("artifact writing", lambda: results.write(output_dir / RESULTS_FILENAME))
    emit("stats complete")
    return results


def run_stats_from_v2_generation(
    video_path: Path, config: PipelineConfig, generation_root: Path, output_dir: Path,
    manifest: dict, *, masks_by_observation=None, clean_masks_by_observation=None,
    number_predictor=None, number_detector=None, validated: bool = False,
    tracking_payload: Mapping[str, Any] | None = None, metrics=None,
) -> PipelineResults:
    """Validate and consume one immutable tracking generation without re-tracking."""
    emit = lambda message: print(f"[v2-stats] {message}", flush=True)
    if not validated:
        validation_started = time.perf_counter()
        emit("stage generation validation/hash checks start")
        validate_stage_generation(Path(generation_root), manifest)
        emit(
            "stage generation validation/hash checks complete: "
            f"elapsed={time.perf_counter() - validation_started:.2f}s"
        )
    generation = Path(generation_root) / "generations" / manifest["generation"]
    if tracking_payload is not None:
        observations = list(tracking_payload["observations"])
        masks_by_observation = tracking_payload.get("masks_by_observation", {})
        clean_masks_by_observation = tracking_payload.get("clean_masks_by_observation", {})
        emit(f"in-memory tracking payload: {len(observations)} rows")
        return run_stats_from_v2_observations(
            video_path, config, output_dir, observations,
            masks_by_observation=masks_by_observation,
            clean_masks_by_observation=clean_masks_by_observation,
            number_predictor=number_predictor, number_detector=number_detector,
            **({"metrics": metrics} if metrics is not None else {}),
        )
    load_started = time.perf_counter()
    package_manifest_path = generation / "tracking-manifest.json"
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    package_artifacts = package_manifest.get("artifacts", {})
    tracks_ref = package_artifacts.get("tracks")
    if not isinstance(tracks_ref, Mapping) or not isinstance(tracks_ref.get("key"), str):
        raise ValueError("tracking contract does not declare tracks artifact")
    tracks_path = generation / Path(tracks_ref["key"])
    if not tracks_path.is_file():
        raise ValueError("declared tracks artifact is missing")
    tracks_started = time.perf_counter()
    emit("tracks load start")
    observations = [json.loads(line) for line in tracks_path.read_text(encoding="utf-8").splitlines() if line]
    emit(
        f"tracks load complete: {len(observations)} rows, "
        f"elapsed={time.perf_counter() - tracks_started:.2f}s"
    )

    def load_masks(name: str) -> dict[str, np.ndarray]:
        group = package_artifacts.get(name, {})
        references = group.get("references", {}) if isinstance(group, Mapping) else {}
        loaded = {}
        total = len(references)
        started = time.perf_counter()
        bundle = group.get("bundle") if isinstance(group, Mapping) else None
        if isinstance(bundle, Mapping) and isinstance(bundle.get("key"), str):
            bundle_key = bundle["key"]
            bundle_parts = PurePosixPath(bundle_key)
            if (bundle_parts.is_absolute() or "\\" in bundle_key
                    or any(part in {"", ".", ".."} for part in bundle_parts.parts)):
                raise ValueError(f"unsafe {name} bundle path")
            bundle_path = generation / Path(bundle_key)
            if not bundle_path.is_file():
                raise ValueError(f"declared {name} bundle is missing")
            emit(f"{name.replace('_', ' ')} bundle load start: {total} files")
            try:
                bundle_data = bundle_path.read_bytes()
                with tarfile.open(fileobj=io.BytesIO(bundle_data), mode="r") as archive:
                    members = archive.getmembers()
                    member_names = set()
                    for member in members:
                        member_path = PurePosixPath(member.name)
                        if (not member.isfile() or member_path.is_absolute() or "\\" in member.name
                                or any(part in {"", ".", ".."} for part in member_path.parts)):
                            raise ValueError(f"unsafe {name} bundle member")
                        if member.name in member_names:
                            raise ValueError(f"duplicate {name} bundle member")
                        member_names.add(member.name)
                    if (type(bundle.get("size_bytes")) is not int
                            or bundle["size_bytes"] != len(bundle_data)
                            or bundle.get("sha256") != hashlib.sha256(bundle_data).hexdigest()):
                        raise ValueError(f"{name} bundle hash or size mismatch")
                    expected_members = set()
                    for index, (observation_key, reference) in enumerate(references.items(), 1):
                        if not isinstance(reference, Mapping) or not isinstance(reference.get("key"), str):
                            raise ValueError(f"invalid {name} artifact reference")
                        member_name = Path(reference["key"]).name
                        if member_name in expected_members:
                            raise ValueError(f"duplicate {name} artifact reference")
                        expected_members.add(member_name)
                        try:
                            member = archive.getmember(member_name)
                        except KeyError as error:
                            raise ValueError(f"missing {name} bundle member") from error
                        handle = archive.extractfile(member)
                        data = handle.read() if handle is not None else b""
                        if (type(reference.get("size_bytes")) is not int
                                or reference["size_bytes"] != len(data)
                                or reference.get("sha256") != hashlib.sha256(data).hexdigest()):
                            raise ValueError(f"{name} bundle member hash or size mismatch")
                        mask = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        if mask is None or mask.ndim != 2:
                            raise ValueError(f"invalid {name} artifact")
                        loaded[str(observation_key)] = mask
                        _emit_file_batch_progress(emit, f"{name.replace('_', ' ')} bundle load", index, total)
                    if member_names != expected_members:
                        raise ValueError(f"{name} bundle members do not match references")
            except (OSError, tarfile.TarError) as error:
                raise ValueError(f"invalid {name} bundle") from error
            emit(
                f"{name.replace('_', ' ')} bundle load complete: {total} files, "
                f"elapsed={time.perf_counter() - started:.2f}s"
            )
            return loaded
        emit(f"{name.replace('_', ' ')} load start: {total} files")
        for index, (observation_key, reference) in enumerate(references.items(), 1):
            if not isinstance(reference, Mapping) or not isinstance(reference.get("key"), str):
                raise ValueError(f"invalid {name} artifact reference")
            path = generation / Path(reference["key"])
            if not path.is_file():
                raise ValueError(f"declared {name} artifact is missing")
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2:
                raise ValueError(f"invalid {name} artifact")
            loaded[str(observation_key)] = mask
            _emit_file_batch_progress(emit, f"{name.replace('_', ' ')} load", index, total)
        if total == 0:
            emit(f"{name.replace('_', ' ')} load: 0/0 files")
        emit(
            f"{name.replace('_', ' ')} load complete: {total} files, "
            f"elapsed={time.perf_counter() - started:.2f}s"
        )
        return loaded

    masks_by_observation = load_masks("masks")
    clean_masks_by_observation = load_masks("clean_masks")
    emit("stats start")
    if metrics is not None:
        metrics.add_phase("tracking_load_seconds", time.perf_counter() - load_started)
    return run_stats_from_v2_observations(
        video_path, config, output_dir, observations,
        masks_by_observation=masks_by_observation,
        clean_masks_by_observation=clean_masks_by_observation,
        number_predictor=number_predictor, number_detector=number_detector,
        **({"metrics": metrics} if metrics is not None else {}),
    )


def run_pipeline(
    video_path: Path,
    config: PipelineConfig,
    output_dir: Path,
    *,
    tracker: Sam2SegmentTracker | None = None,
    embedder: Embedder | None = None,
    tracknet_detector=None,
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

    if config.tracknet_enabled:
        if tracknet_detector is None and not config.tracknet_checkpoint:
            raise RuntimeError("TrackNet is enabled but TRACKNET_CHECKPOINT is not configured")
        if tracknet_detector is None:
            tracknet_detector = build_tracknet_detector(config.tracknet_checkpoint)
    else:
        tracknet_detector = None
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
    active_ball_rows: list[dict] = []
    next_ball_track_id = 1
    ball_missed_counts: dict[int, int] = {}
    ball_fallback_frames_remaining = 0
    ball_track_records = 0
    tracknet_output_path = output_dir / BALL_TRACKS_FILENAME
    tracknet_reader = None
    ball_rows_by_frame: dict[int, list[dict]] = {}

    if tracknet_detector is not None:
        ball_track_records = _collect_tracknet_rows(
            video_path,
            info=info,
            config=config,
            detector=tracknet_detector,
            tracker=tracker,
            player_step=step,
            output_path=tracknet_output_path,
        )
        tracknet_reader = _BallRecordReader(tracknet_output_path)

    annotated_path = output_dir / ANNOTATED_FILENAME if config.write_annotated_video else None
    projected_tracks_path = output_dir / PROJECTED_TRACKS_FILENAME
    projected_track_records = 0

    prefetch = PrefetchingDetector(detector) if config.prefetch_detections else None

    try:
        with TrackWriter(output_dir / TRACKS_FILENAME) as tracks:
            projected_context = (
                projected_tracks_path.open("w", encoding="utf-8") if court_detector is not None else nullcontext()
            )
            with projected_context as projected_tracks_file, (output_dir / BALL_TRACKS_FILENAME).open(
                "a" if tracknet_detector is not None else "w", encoding="utf-8"
            ) as ball_tracks_file:
                for segment in iter_segments(
                    video_path,
                    frame_stride=config.frame_stride,
                    max_frames=config.max_frames,
                    frame_step=step,
                ):
                    frames = segment.frames
                    indices = segment.frame_indices

                    tracknet_present, tracknet_rows = (
                        tracknet_reader.take(indices[0]) if tracknet_reader else (False, [])
                    )

                    # --- detect on the segment's first frame -------------------
                    result = prefetch.take(frames[0]) if prefetch else detector.detect(frames[0])
                    detections = detections_from_result(
                        result,
                        min_confidence=config.confidence,
                        allowed_classes=config.allowed_classes,
                    )
                    player_rows, number_rows = _split_detection_rows(detection_rows(detections))
                    detected_ball_rows = (
                        tracknet_rows
                        if tracknet_detector is not None
                        else ball_rows_from_result(
                            result,
                            min_confidence=config.confidence,
                            detections_from_result=detections_from_result,
                            detection_rows=detection_rows,
                        )
                    )
                    seed_ball_rows = select_seed_ball_rows(
                        detected_ball_rows,
                        active_rows=active_ball_rows,
                        min_match_iou=config.sam2_match_iou,
                    )
                    tracknet_detected = bool(tracknet_rows) if tracknet_detector is not None else False
                    lost_ball_rows: list[dict] = []
                    if seed_ball_rows:
                        if tracknet_detector is not None:
                            seed_ball_rows, next_ball_track_id = reconcile_ball_track_id(
                                seed_ball_rows, active_ball_rows, next_ball_track_id
                            )
                        else:
                            seed_ball_rows, next_ball_track_id = assign_track_ids(
                                seed_ball_rows, active_ball_rows, next_ball_track_id, config.sam2_match_iou
                            )
                        ball_missed_counts.clear()
                        if tracknet_detector is not None:
                            ball_fallback_frames_remaining = config.tracknet_fallback_budget
                    elif tracknet_detector is None:
                        seed_ball_rows = carry_unmatched_tracks(
                            [], active_ball_rows, ball_missed_counts, config.max_missed_detections
                        )
                    elif tracknet_present:
                        # An authoritative lost row suppresses SAM2 at this
                        # exact source frame; fallback resumes only when the
                        # dense reader has no row for a frame.
                        seed_ball_rows = []
                    elif active_ball_rows and ball_fallback_frames_remaining > 0:
                        seed_ball_rows = [{**active_ball_rows[0], "track_state": "fallback"}]
                    elif tracknet_detector is not None and active_ball_rows:
                        lost_ball_rows = [
                            {
                                "track_id": active_ball_rows[0].get("track_id"),
                                "track_state": "lost",
                                "class_name": "ball",
                                "confidence": None,
                                "xyxy": None,
                            }
                        ]

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
                    combined_seed_rows = seed_rows + seed_ball_rows

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
                    propagated = tracker.propagate_segment(frames, combined_seed_rows)
                    for offset in range(1, len(frames)):
                        if offset % config.team_sample_stride != 0:
                            continue
                        propagated_rows = [
                            row
                            for row in propagated.get(offset, [])
                            if str(row.get("class_name", "player")) == "player"
                        ]
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
                    seed_ball_rows, emitted_fallback_frames = _cap_ball_fallback_rows(
                        seed_ball_rows, ball_fallback_frames_remaining
                    )
                    ball_fallback_frames_remaining -= emitted_fallback_frames
                    if tracknet_detector is None:
                        ball_rows_by_frame[indices[0]] = seed_ball_rows + lost_ball_rows
                    tracks.write_frame(indices[0], seed_rows_with_teams or _with_team_ids(active_rows, {}))
                    for offset in range(1, last_offset + 1):
                        frame_index = indices[offset]
                        authoritative_present, authoritative_rows = (
                            tracknet_reader.take(frame_index) if tracknet_reader else (False, [])
                        )
                        propagated_ball_rows = filter_ball_rows(propagated.get(offset, []))
                        if authoritative_present:
                            ball_rows = [dict(authoritative_rows[0])] if authoritative_rows else []
                            if not ball_rows:
                                # TrackNet explicitly reported lost; do not let
                                # the propagated ball replace that source row.
                                propagated_ball_rows = []
                            else:
                                prior_ball_rows = seed_ball_rows or active_ball_rows
                                if prior_ball_rows and prior_ball_rows[0].get("track_id") is not None:
                                    ball_rows[0]["track_id"] = prior_ball_rows[0]["track_id"]
                                else:
                                    ball_rows, next_ball_track_id = reconcile_ball_track_id(
                                        ball_rows, [], next_ball_track_id
                                    )
                                ball_fallback_frames_remaining = config.tracknet_fallback_budget
                        else:
                            ball_rows = [
                                {**row, "track_state": "fallback"}
                                for row in propagated_ball_rows
                            ] if tracknet_detector is not None else propagated_ball_rows
                            if tracknet_detector is not None and not ball_rows and active_ball_rows:
                                ball_rows = [
                                    {
                                        "track_id": active_ball_rows[0].get("track_id"),
                                        "track_state": "lost",
                                        "class_name": "ball",
                                        "confidence": None,
                                        "xyxy": None,
                                    }
                                ]
                        ball_rows, emitted_fallback = _cap_ball_fallback_rows(
                            ball_rows, ball_fallback_frames_remaining
                        )
                        ball_fallback_frames_remaining -= emitted_fallback
                        if tracknet_detector is None:
                            ball_rows_by_frame[frame_index] = ball_rows
                        tracks.write_frame(
                            indices[offset],
                            _with_team_ids(
                                [
                                    row
                                    for row in propagated.get(offset, [])
                                    if str(row.get("class_name", "player")) == "player"
                                ],
                                {},
                            ),
                        )

                    active_rows = [] if segment.is_last else propagated.get(len(frames) - 1, [])
                    active_ball_rows = [] if segment.is_last else filter_ball_rows(propagated.get(len(frames) - 1, []))
                    active_rows = [
                        row for row in active_rows if str(row.get("class_name", "player")) == "player"
                    ]

                    summary.segments_processed += 1
                    summary.frames_processed += last_offset + 1
                    if on_progress is not None:
                        on_progress(summary.frames_processed, expected_frames)

                if tracknet_detector is None:
                    for frame_index in sorted(ball_rows_by_frame):
                        ball_track_records += write_ball_track_rows(
                            ball_tracks_file,
                            frame_index=frame_index,
                            rows=ball_rows_by_frame[frame_index],
                            timestamp_seconds=_frame_timestamp_seconds(frame_index, info.fps),
                        )

            summary.track_records = tracks.record_count
            summary.distinct_track_ids = len(tracks.track_ids)
            summary.suspected_swaps = len(suspected_swaps)
    finally:
        if prefetch is not None:
            prefetch.close()
        if tracknet_reader is not None:
            tracknet_reader.close()

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
            "ball_target_fps": config.ball_target_fps,
            "ball_all_frames": config.ball_all_frames,
            "ball_frame_step": 1 if config.ball_all_frames else frame_step_for(info.fps, config.ball_target_fps),
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
            "tracknet_enabled": config.tracknet_enabled,
            "tracknet_checkpoint": config.tracknet_checkpoint,
            "tracknet_context_size": TRACKNET_CONTEXT_SIZE if tracknet_detector is not None else None,
            "tracknet_input_size": [512, 288] if tracknet_detector is not None else None,
            "tracknet_seed_radius": config.tracknet_seed_radius,
            "tracknet_heatmap_threshold": config.tracknet_heatmap_threshold,
            "tracknet_fallback_budget": config.tracknet_fallback_budget,
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
        shot_evidence_file=SHOT_EVIDENCE_FILENAME,
        ball_tracks_file=BALL_TRACKS_FILENAME if ball_track_records > 0 else None,
        possession_events_file=POSSESSION_EVENTS_FILENAME,
        possession_transitions_file=POSSESSION_TRANSITIONS_FILENAME,
        projected_tracks_file=PROJECTED_TRACKS_FILENAME if projected_track_records > 0 else None,
        annotated_video_file=ANNOTATED_FILENAME if annotated_path else None,
    )
    results.config = _possession_result_config(config, results.config)

    _write_shot_events(
        video_path,
        output_dir / SHOT_EVENTS_FILENAME,
        info=info,
        step=shot_step,
        config=config,
    )
    _refine_shot_outcomes(output_dir, config)
    if projected_track_records == 0 and projected_tracks_path.exists():
        projected_tracks_path.unlink()
    results.projected_shots_file, results.projected_shots_image_file = _maybe_write_projected_shots(output_dir)
    ball_rows = load_ball_track_rows(output_dir / BALL_TRACKS_FILENAME)
    ball_step = 1 if config.ball_all_frames else frame_step_for(info.fps, config.ball_target_fps)
    expected_ball_count = info.frame_count
    if config.max_frames is not None:
        expected_ball_count = min(expected_ball_count, config.max_frames * step)
    expected_ball_frames = range(0, expected_ball_count, step if tracknet_detector is None else ball_step)
    _attach_ball_evidence(
        output_dir,
        expected_frame_indices=expected_ball_frames,
        team_ids_by_track_id=team_by_track_id,
    )
    ball_validation = validate_ball_track_rows(
        ball_rows,
        expected_frame_indices=expected_ball_frames,
        frame_step=ball_step if tracknet_detector is not None else step,
        frame_width=info.width,
        frame_height=info.height,
    )
    results.ball_provenance = {
        "backend": "tracknet" if tracknet_detector is not None else "hosted_detector",
        "contract_version": ball_validation.metadata["contract_version"],
        "validation": ball_validation.to_dict(),
        "evidence_policy": "no_interpolation; missing evidence remains unknown",
    }

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
            ball_tracks_path=output_dir / BALL_TRACKS_FILENAME,
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

    possession_path, transitions_path = _write_possession_artifacts(
        output_dir, _load_jsonl_rows(tracks_path), ball_rows, fps=info.fps,
        frame_step=step, team_ids_by_track_id=team_by_track_id,
        max_frame_gap=config.possession_transition_max_gap,
    )
    v3_manifest = _write_v3_artifacts(output_dir, _load_jsonl_rows(tracks_path), ball_rows, fps=info.fps, config=config)
    if v3_manifest and v3_manifest.get("status") != "failed":
        results.possession_v3_file = _v3_artifact_path(v3_manifest, "possession")
        results.possession_v3_transitions_file = _v3_artifact_path(v3_manifest, "transitions")
        results.possession_v3_events_file = _v3_artifact_path(v3_manifest, "events")
        results.possession_v3_manifest_file = POSSESSION_V3_MANIFEST_FILENAME
        results.possession_v3_schema_version = V3_CONTRACT_VERSION
        if v3_manifest.get("v4", {}).get("enabled"):
            results.possession_v4_diagnostics_file = _v3_artifact_path(v3_manifest, "v4_diagnostics")
            results.possession_v4_schema_version = 1
    results.possession_v3_observability = v3_manifest.get("observability") if v3_manifest else None
    results.artifact_references = {
        "possession_events": {
            "file": possession_path.name,
            "schema_version": 1,
            "status_vocabulary": ["confirmed", "unknown", "lost", "unassigned"],
        },
        "possession_transitions": {
            "file": transitions_path.name,
            "schema_version": 1,
            "event_types": ["pass", "steal", "turnover", "lost_possession"],
        },
        **({"possession_v3": {"manifest": POSSESSION_V3_MANIFEST_FILENAME,
                              "artifacts": v3_manifest.get("artifacts", {})}} if v3_manifest and v3_manifest.get("status") != "failed" else {}),
        **({"possession_v4": {"feature_flag": "v4", "diagnostics": _v3_artifact_path(v3_manifest, "v4_diagnostics")}}
           if v3_manifest and v3_manifest.get("v4", {}).get("enabled") else {}),
    }

    # One line per suspected swap, so a run can be reviewed without watching the
    # video. Written even when empty, so its absence means "not computed" rather
    # than "nothing found".
    with (output_dir / SWAPS_FILENAME).open("w", encoding="utf-8") as handle:
        for record in suspected_swaps:
            handle.write(json.dumps(record) + "\n")

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
