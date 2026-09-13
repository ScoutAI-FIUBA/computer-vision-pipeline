#!/usr/bin/env python3
"""Run the local TrackNet checkpoint against a video and nothing else."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ball.candidates import BallCandidate, extract_candidates
from scripts.ball.tracknet import build_detector, inspect_tracknet_checkpoint, padded_window


INPUT_SIZE = (512, 288)
BALL_RADIUS = 8


def ball_box(candidate: BallCandidate, width: int, height: int) -> list[float]:
    radius = BALL_RADIUS
    return [
        float(max(0, candidate.x - radius)),
        float(max(0, candidate.y - radius)),
        float(min(width - 1, candidate.x + radius)),
        float(min(height - 1, candidate.y + radius)),
    ]


def frame_record(
    frame_index: int,
    source_frame_index: int,
    fps: float,
    window_indices: tuple[int, ...],
    candidates: list[BallCandidate],
    width: int,
    height: int,
) -> dict[str, object]:
    selected = candidates[0] if candidates else None
    return {
        "frame_index": frame_index,
        "source_frame_index": source_frame_index,
        "source_time_seconds": source_frame_index / fps if fps > 0 else None,
        "tracknet_window_source_frames": list(window_indices),
        "tracknet_candidates": [candidate.to_dict() for candidate in candidates],
        "selected_center_xy": [selected.x, selected.y] if selected else None,
        "selected_box_xyxy": ball_box(selected, width, height) if selected else None,
        "confidence": selected.score if selected else None,
        "score": selected.score if selected else None,
        "track_state": "detected" if selected else "lost",
    }


def annotate(frame, record: dict[str, object]):
    import cv2

    result = frame.copy()
    for candidate in record["tracknet_candidates"]:
        center = (round(candidate["x"]), round(candidate["y"]))
        cv2.circle(result, center, 7, (0, 255, 255), 2, cv2.LINE_AA)
    box = record["selected_box_xyxy"]
    if box is not None:
        x1, y1, x2, y2 = (round(value) for value in box)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, f"ball {record['confidence']:.3f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(result, "ball lost", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--model", type=Path, required=True, help="TrackNet checkpoint path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--frame-stride", type=int, default=1, help="Process every Nth source frame")
    sampling.add_argument("--target-fps", type=float, help="Approximate processed FPS")
    parser.add_argument("--threshold", "--min-score", dest="threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=12.0)
    parser.add_argument("--max-candidates", type=int, default=5)
    args = parser.parse_args(argv)
    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    if not args.model.is_file():
        parser.error(f"TrackNet checkpoint not found: {args.model}")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.frame_stride < 1:
        parser.error("--frame-stride must be at least 1")
    if args.target_fps is not None and args.target_fps <= 0:
        parser.error("--target-fps must be positive")
    if args.threshold < 0:
        parser.error("--threshold must be non-negative")
    if args.min_distance < 0 or args.max_candidates < 1:
        parser.error("--min-distance must be non-negative and --max-candidates at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    import cv2

    args = parse_args(argv)
    info = inspect_tracknet_checkpoint(args.model)
    seq_len = info.get("seq_len")
    if not isinstance(seq_len, int) or seq_len < 1:
        raise SystemExit("TrackNet checkpoint does not contain a valid seq_len")
    try:
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = args.frame_stride if args.target_fps is None else max(1, round(fps / args.target_fps))
        frames: list = []
        source_indices: list[int] = []
        source_index = 0
        while args.max_frames is None or len(frames) < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % step == 0:
                frames.append(frame)
                source_indices.append(source_index)
            source_index += 1
        capture.release()
    except Exception:
        capture.release()
        raise
    if not frames:
        raise SystemExit(f"Video contains no readable frames: {args.video}")

    try:
        detector = build_detector(args.model, backend="auto", allow_stub=False, context_size=seq_len, input_size=INPUT_SIZE)
    except (RuntimeError, ModuleNotFoundError) as error:
        raise SystemExit(f"TrackNet inference unavailable: {error}") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = args.output_dir / "ball_tracks.jsonl"
    video_path = args.output_dir / "ball_only.mp4"
    output_fps = fps / step if fps > 0 else 30.0
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open output video: {video_path}")
    records = []
    frame_by_source_index = dict(zip(source_indices, frames))
    try:
        for frame_index, source_frame_index in enumerate(source_indices):
            window_indices = padded_window(source_indices, frame_index, detector.context_size)
            window_frames = [frame_by_source_index[index] for index in window_indices]
            prediction = detector.predict(window_frames, window_indices)
            candidates = extract_candidates(
                prediction.heatmap,
                original_frame_size=(width, height),
                min_score=args.threshold,
                min_distance=args.min_distance,
                max_candidates=args.max_candidates,
            )
            record = frame_record(frame_index, source_frame_index, fps, window_indices, candidates, width, height)
            records.append(record)
            writer.write(annotate(frames[frame_index], record))
    finally:
        writer.release()
    with tracks_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(json.dumps({"tracks": str(tracks_path), "video": str(video_path), "frames": len(records), "backend": detector.backend_name}))


if __name__ == "__main__":
    main()
