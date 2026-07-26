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
from pathlib import Path
from typing import Callable

from pipeline import appearance
from pipeline.config import PipelineConfig
from pipeline.detection import PrefetchingDetector, build_detector, detections_from_result
from pipeline.reid import Embedder, build_embedder
from pipeline.results import RESULTS_SCHEMA_VERSION, PipelineResults, RunSummary, TrackWriter
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
    detector = build_detector(config)
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

    annotated_path = output_dir / ANNOTATED_FILENAME if config.write_annotated_video else None
    # The annotated video only contains processed frames, so it plays at the
    # downsampled rate to stay in sync with real time.
    writer = AnnotatedVideoWriter(annotated_path, info, fps=effective_fps) if annotated_path else None
    if writer is not None:
        # Imported only when annotating so the main path never pulls it in.
        from pipeline.annotate import draw_tracks

    prefetch = PrefetchingDetector(detector) if config.prefetch_detections else None

    try:
        with TrackWriter(output_dir / TRACKS_FILENAME) as tracks:
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
                new_rows = detection_rows(detections)

                # Describe each detection before matching. The embedder returns
                # None for players too occluded to describe, so crossings degrade
                # to geometry rather than to a confident wrong answer.
                if config.appearance_weight > 0.0:
                    signatures = embedder.embed(frames[0], [row["xyxy"] for row in new_rows])
                    for row, signature in zip(new_rows, signatures):
                        row["appearance"] = signature

                seed_rows, next_track_id = assign_track_ids(
                    new_rows,
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
                # Re-seed SAM2 with tracks the detector missed as well, so a
                # one-frame miss does not cost a player their identity.
                seed_rows = carry_unmatched_tracks(
                    seed_rows, active_rows, missed_counts, config.max_missed_detections
                )

                # Fire the next segment's detector call now, so the HTTP wait
                # overlaps with SAM2 compute instead of idling a billed GPU.
                # The next detector frame is this segment's last frame.
                if prefetch and not segment.is_last:
                    prefetch.submit(frames[-1])

                # --- propagate through the segment -------------------------
                propagated = tracker.propagate_segment(frames, seed_rows)

                # --- emit ---------------------------------------------------
                # A shared boundary frame belongs to the next segment, which will
                # run the detector on it. Only the final segment emits its last frame.
                last_offset = len(frames) - 1 if segment.is_last else len(frames) - 2

                # Track records carry ORIGINAL video frame indices, so
                # downsampling never changes what a record refers to.
                # When the detector finds nothing, the frame keeps whatever SAM2
                # was already carrying, matching the original script's behaviour.
                tracks.write_frame(indices[0], seed_rows or active_rows)
                for offset in range(1, last_offset + 1):
                    tracks.write_frame(indices[offset], propagated.get(offset, []))

                if writer is not None:
                    writer.write(draw_tracks(frames[0], seed_rows or active_rows))
                    for offset in range(1, last_offset + 1):
                        writer.write(draw_tracks(frames[offset], propagated.get(offset, [])))

                active_rows = [] if segment.is_last else propagated.get(len(frames) - 1, [])

                summary.segments_processed += 1
                summary.frames_processed += last_offset + 1
                if on_progress is not None:
                    on_progress(summary.frames_processed, info.frame_count)

            summary.track_records = tracks.record_count
            summary.distinct_track_ids = len(tracks.track_ids)
            summary.suspected_swaps = len(suspected_swaps)
    finally:
        if prefetch is not None:
            prefetch.close()
        if writer is not None:
            writer.release()

    summary.detector_calls = detector.call_count
    summary.wall_seconds = round(time.monotonic() - started, 2)

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
            "frame_stride": config.frame_stride,
            "confidence": config.confidence,
            "player_model_id": config.player_model_id,
            "sam2_model_id": config.sam2_model_id,
            "sam2_match_iou": config.sam2_match_iou,
            "max_frames": config.max_frames,
            "max_missed_detections": config.max_missed_detections,
            "reid_backend": getattr(embedder, "name", config.reid_backend),
            "appearance_weight": config.appearance_weight,
        },
        summary=summary,
        device=tracker.device,
        annotated_video_file=ANNOTATED_FILENAME if annotated_path else None,
    )
    results.write(output_dir / RESULTS_FILENAME)

    # One line per suspected swap, so a run can be reviewed without watching the
    # video. Written even when empty, so its absence means "not computed" rather
    # than "nothing found".
    with (output_dir / SWAPS_FILENAME).open("w", encoding="utf-8") as handle:
        for record in suspected_swaps:
            handle.write(json.dumps(record) + "\n")

    logger.info(
        "done: %s frames, %s segments, %s detector calls, %s track records in %.1fs",
        summary.frames_processed,
        summary.segments_processed,
        summary.detector_calls,
        summary.track_records,
        summary.wall_seconds,
    )
    return results
