"""Dedicated player-position homography pipeline.

Primary output: `player_positions.jsonl`
Optional debug artifact: `player_positions_debug.mp4` when rows are emitted.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

from pipeline import appearance
from pipeline.config import PipelineConfig
from pipeline.court_projection import compute_frame_homography, parse_court_keypoints, project_points
from pipeline.detection import PrefetchingDetector, build_detector, detections_from_result
from pipeline.projected_shots import COURT_LENGTH_M, COURT_WIDTH_M, draw_fiba_court
from pipeline.reid import build_embedder
from pipeline.runner import _collect_team_evidence, _identify_track_teams
from pipeline.team_evidence import TeamEvidenceSample, select_top_k_evidence
from pipeline.team_embedder import build_team_embedder
from pipeline.tracking import Sam2SegmentTracker, assign_track_ids, carry_unmatched_tracks, detection_rows
from pipeline.video import frame_step_for, iter_segments, probe

PLAYER_POSITIONS_FILENAME = "player_positions.jsonl"
PLAYER_POSITIONS_DEBUG_VIDEO_FILENAME = "player_positions_debug.mp4"
DEFAULT_PLAYER_POSITION_FPS = 5.0
DEFAULT_PLAYER_POSITION_FRAME_STRIDE = 5
DEFAULT_PLAYER_POSITION_CONFIDENCE = 0.8


def _frame_timestamp_seconds(frame_index: int, fps: float) -> float | None:
    return None if fps <= 0 else frame_index / fps


def _foot_point_for_bbox(xyxy) -> list[float]:
    x1, _, x2, y2 = xyxy
    return [float((x1 + x2) / 2.0), float(y2)]


def build_player_position_records(
    *,
    frame_index: int,
    fps: float,
    rows: list[dict],
    homography_result,
    homography_status: str = "ok",
    project_points_fn: Callable | None = None,
) -> list[dict]:
    project_points_fn = project_points if project_points_fn is None else project_points_fn
    anchors = [_foot_point_for_bbox(row["xyxy"]) for row in rows]
    projected_points = None
    if homography_result is not None and anchors:
        projected_points = project_points_fn(homography_result.matrix, [tuple(anchor) for anchor in anchors])

    records = []
    for index, row in enumerate(rows):
        record = {
            "frame_index": int(frame_index),
            "timestamp_seconds": _frame_timestamp_seconds(frame_index, fps),
            "player_id": None if row.get("track_id") is None else int(row["track_id"]),
            "team_id": None if row.get("team_id") is None else int(row["team_id"]),
            "track_state": row.get("track_state"),
            "class_name": str(row.get("class_name", "player")),
            "confidence": None if row.get("confidence") is None else float(row["confidence"]),
            "bbox_xyxy": [float(value) for value in row["xyxy"]],
            "image_anchor_xy": anchors[index],
            "court_xy_m": None,
            "homography_frame_status": homography_status,
            "homography_inliers_count": None,
            "homography_reprojection_mean_m": None,
        }
        if projected_points is not None:
            point = projected_points[index]
            record["court_xy_m"] = [float(point[0]), float(point[1])]
            record["homography_frame_status"] = "ok"
            record["homography_inliers_count"] = int(homography_result.inliers_count)
            record["homography_reprojection_mean_m"] = float(homography_result.reprojection_mean_m)
        records.append(record)
    return records


def _append_jsonl_records(handle, records: list[dict]) -> int:
    for record in records:
        handle.write(json.dumps(record) + "\n")
    return len(records)


def _rewrite_position_team_ids(path: Path, team_by_player_id: dict[int, int]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with path.open("r", encoding="utf-8") as source, tmp_path.open("w", encoding="utf-8") as target:
        for line in source:
            record = json.loads(line)
            player_id = record.get("player_id")
            record["team_id"] = None if player_id is None else team_by_player_id.get(int(player_id))
            target.write(json.dumps(record) + "\n")
    tmp_path.replace(path)


def identify_track_teams(
    selected_evidence_by_track_id: dict[int, list[TeamEvidenceSample]],
    *,
    team_embedder_name: str,
    config: PipelineConfig,
):
    return _identify_track_teams(
        selected_evidence_by_track_id,
        backend=team_embedder_name,
        team_count=config.team_count,
        minimum_samples=config.min_team_samples_per_track,
        assignment_margin=config.team_assignment_margin,
    )


def _track_color(player_id: int | None) -> tuple[int, int, int]:
    if player_id is None:
        return (90, 90, 90)
    palette = (
        (255, 99, 71),
        (65, 105, 225),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (231, 76, 60),
    )
    r, g, b = palette[player_id % len(palette)]
    return (b, g, r)


def _debug_point_color(*, player_id: int | None, team_id: int | None) -> tuple[int, int, int]:
    return _track_color(team_id if team_id is not None else player_id)


def _meter_to_px(x_m: float, y_m: float, pixels_per_meter: int) -> tuple[int, int]:
    return int(round(x_m * pixels_per_meter)), int(round(y_m * pixels_per_meter))


def _point_in_bounds(court_xy_m: list[float] | None) -> bool:
    return court_xy_m is not None and len(court_xy_m) >= 2 and 0.0 <= court_xy_m[0] <= COURT_LENGTH_M and 0.0 <= court_xy_m[1] <= COURT_WIDTH_M


def write_player_position_debug_video(
    positions_path: Path,
    output_path: Path,
    *,
    output_fps: float = DEFAULT_PLAYER_POSITION_FPS,
    pixels_per_meter: int = 40,
    point_radius: int = 5,
    trail_length: int = 12,
) -> int:
    import cv2

    grouped: dict[int, list[dict]] = defaultdict(list)
    with positions_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            grouped[int(record["frame_index"])].append(record)

    if not grouped:
        raise ValueError("No player-position rows available for debug video")

    first_frame = draw_fiba_court(pixels_per_meter=pixels_per_meter)
    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not write debug video: {output_path}")

    history_by_player_id: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_length))
    frames_written = 0
    try:
        for frame_index in sorted(grouped):
            frame = draw_fiba_court(pixels_per_meter=pixels_per_meter)
            rows = grouped[frame_index]
            for row in rows:
                if not _point_in_bounds(row.get("court_xy_m")):
                    continue
                x_px, y_px = _meter_to_px(float(row["court_xy_m"][0]), float(row["court_xy_m"][1]), pixels_per_meter)
                player_id = None if row.get("player_id") is None else int(row["player_id"])
                team_id = None if row.get("team_id") is None else int(row["team_id"])
                color = _debug_point_color(player_id=player_id, team_id=team_id)
                if player_id is not None:
                    history = history_by_player_id[player_id]
                    history.append((x_px, y_px))
                    if len(history) > 1:
                        import numpy as np

                        cv2.polylines(frame, [np.asarray(list(history), dtype="int32")], False, color, 2, cv2.LINE_AA)
                cv2.circle(frame, (x_px, y_px), point_radius, color, -1)
                label = "?" if player_id is None else str(player_id)
                cv2.putText(frame, label, (x_px + point_radius + 2, y_px - point_radius - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"frame={frame_index}", (12, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
            writer.write(frame)
            frames_written += 1
    finally:
        writer.release()
    return frames_written


def run_player_position_pipeline(
    video_path: Path,
    config: PipelineConfig,
    output_dir: Path,
    *,
    tracker: Sam2SegmentTracker | None = None,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config.court_model_id:
        raise ValueError("court_model_id is required for the player-position pipeline")

    info = probe(video_path)
    step = frame_step_for(info.fps, config.target_fps)
    tracker = tracker or Sam2SegmentTracker(config.sam2_model_id)
    detector = build_detector(config)
    court_detector = build_detector(config, model_id=config.court_model_id)
    embedder = build_embedder(config.reid_backend)
    team_embedder = build_team_embedder(config.team_embedder_backend, model_id=config.team_embedder_model_id)
    prefetch = PrefetchingDetector(detector) if config.prefetch_detections else None

    next_track_id = 1
    active_rows: list[dict] = []
    missed_counts: dict[int, int] = {}
    appearance_by_track_id: dict[int, list[float]] = {}
    team_evidence_by_track_id: dict[int, list[TeamEvidenceSample]] = {}
    positions_path = output_dir / PLAYER_POSITIONS_FILENAME
    rows_written = 0

    try:
        with positions_path.open("w", encoding="utf-8") as positions_file:
            for segment in iter_segments(
                video_path,
                frame_stride=config.frame_stride,
                max_frames=config.max_frames,
                frame_step=step,
            ):
                frames = segment.frames
                indices = segment.frame_indices
                result = prefetch.take(frames[0]) if prefetch else detector.detect(frames[0])
                detections = detections_from_result(
                    result,
                    min_confidence=config.confidence,
                    allowed_classes=("player",),
                )
                player_rows = [row for row in detection_rows(detections) if str(row.get("class_name", "player")) == "player"]

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

                for row in seed_rows:
                    signature = row.get("appearance")
                    if signature is None:
                        continue
                    track_id = int(row["track_id"])
                    appearance_by_track_id[track_id] = appearance.blend(appearance_by_track_id.get(track_id), signature)

                if team_embedder.name != "none":
                    _collect_team_evidence(
                        frames[0],
                        seed_rows,
                        frame_index=indices[0],
                        team_embedder=team_embedder,
                        evidence_by_track_id=team_evidence_by_track_id,
                        min_quality=config.min_team_sample_quality,
                    )

                seed_rows = carry_unmatched_tracks(seed_rows, active_rows, missed_counts, config.max_missed_detections)

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
                    homography_status = "ok"
                except (RuntimeError, ValueError) as error:
                    homography_result = None
                    homography_status = str(error)

                if prefetch and not segment.is_last:
                    prefetch.submit(frames[-1])

                propagated = tracker.propagate_segment(frames, seed_rows)
                if team_embedder.name != "none":
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

                last_offset = len(frames) - 1 if segment.is_last else len(frames) - 2
                rows_written += _append_jsonl_records(
                    positions_file,
                    build_player_position_records(
                        frame_index=indices[0],
                        fps=info.fps,
                        rows=seed_rows or active_rows,
                        homography_result=homography_result,
                        homography_status=homography_status,
                    ),
                )
                for offset in range(1, last_offset + 1):
                    rows_written += _append_jsonl_records(
                        positions_file,
                        build_player_position_records(
                            frame_index=indices[offset],
                            fps=info.fps,
                            rows=propagated.get(offset, []),
                            homography_result=homography_result,
                            homography_status=homography_status,
                        ),
                    )

                active_rows = [] if segment.is_last else propagated.get(len(frames) - 1, [])
    finally:
        if prefetch is not None:
            prefetch.close()

    selected_team_evidence_by_track_id: dict[int, list[TeamEvidenceSample]] = {}
    for track_id, samples in sorted(team_evidence_by_track_id.items()):
        selected = select_top_k_evidence(samples, config.max_team_samples_per_track)
        if selected:
            selected_team_evidence_by_track_id[track_id] = selected
    team_result = identify_track_teams(
        selected_team_evidence_by_track_id,
        team_embedder_name=team_embedder.name,
        config=config,
    )
    team_by_player_id = {
        int(assignment.track_id): int(assignment.team_id)
        for assignment in getattr(team_result, "assignments", [])
        if assignment.track_id is not None and assignment.team_id is not None
    }
    _rewrite_position_team_ids(positions_path, team_by_player_id)

    debug_video_path = output_dir / PLAYER_POSITIONS_DEBUG_VIDEO_FILENAME
    debug_frames = 0
    if rows_written > 0:
        debug_frames = write_player_position_debug_video(
            positions_path,
            debug_video_path,
            output_fps=config.target_fps or DEFAULT_PLAYER_POSITION_FPS,
        )
    return {
        "positions_file": PLAYER_POSITIONS_FILENAME,
        "debug_video_file": PLAYER_POSITIONS_DEBUG_VIDEO_FILENAME if rows_written > 0 else None,
        "rows_written": rows_written,
        "debug_frames": debug_frames,
        "team_assignments": len(team_by_player_id),
        "effective_fps": info.fps / step if info.fps else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the dedicated player-position homography pipeline.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output-dir", type=Path, help="Directory for player_positions.jsonl and player_positions_debug.mp4")
    parser.add_argument("--max-frames", type=int, help="Stop after N processed frames")
    parser.add_argument(
        "--confidence",
        type=float,
        help=f"Minimum detector confidence (default: {DEFAULT_PLAYER_POSITION_CONFIDENCE} for this pipeline)",
    )
    parser.add_argument("--sam2-model-id", help="SAM2 checkpoint override")
    parser.add_argument("--team-embedder-backend", choices=("siglip", "colour", "none"), help="Team ID embedder backend")
    parser.add_argument("--team-embedder-model-id", help="SigLIP model id for team embeddings")
    parser.add_argument("--team-count", type=int, help="Number of teams to cluster")
    parser.add_argument("--min-team-samples-per-track", type=int, help="Minimum team samples before assignment")
    parser.add_argument("--team-assignment-margin", type=float, help="Leave a track unassigned when cluster votes are too close")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig.from_env(
        max_frames=args.max_frames,
        confidence=DEFAULT_PLAYER_POSITION_CONFIDENCE if args.confidence is None else args.confidence,
        sam2_model_id=args.sam2_model_id,
        team_embedder_backend=args.team_embedder_backend,
        team_embedder_model_id=args.team_embedder_model_id,
        team_count=args.team_count,
        min_team_samples_per_track=args.min_team_samples_per_track,
        team_assignment_margin=args.team_assignment_margin,
        target_fps=DEFAULT_PLAYER_POSITION_FPS,
        frame_stride=DEFAULT_PLAYER_POSITION_FRAME_STRIDE,
        allowed_classes=("player",),
        write_annotated_video=False,
    )
    output_dir = args.output_dir or Path("outputs") / f"{args.video.stem}-player-positions"
    artifact = run_player_position_pipeline(args.video, config, output_dir)
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
