#!/usr/bin/env python3
"""Standalone TrackNet-style ball heatmap experiment scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ball.candidates import extract_candidates
from scripts.ball.debug import annotate_candidates, append_candidates_jsonl, overlay_heatmap, save_debug_frame, write_summary_json
from scripts.ball.tracknet import TemporalFrameBuffer, build_detector, default_tracknet_checkpoint_path, inspect_tracknet_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal standalone TrackNet-style ball experiment.")
    parser.add_argument("video", type=Path, nargs="?", help="Path to the input video")
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to a TrackNet-style checkpoint. Defaults to scripts/ball/TrackNet_best.pt when present.",
    )
    parser.add_argument("--backend", choices=("auto", "onnx", "pytorch"), default="auto", help="Backend to use when a model is provided.")
    parser.add_argument(
        "--inspect-model",
        action="store_true",
        help="Inspect the checkpoint structure and exit without opening a video.",
    )
    parser.add_argument(
        "--onnx-input-layout",
        choices=("auto", "temporal_5d", "stacked_4d"),
        default="auto",
        help="ONNX input tensor layout. Generic checkpoints may need trying both explicit layouts. Default: auto",
    )
    parser.add_argument(
        "--onnx-color-order",
        choices=("rgb", "bgr"),
        default="rgb",
        help="Color order for ONNX preprocessing. Default: rgb",
    )
    parser.add_argument(
        "--onnx-normalization",
        choices=("zero_one", "neg_one_one"),
        default="zero_one",
        help="Normalization for ONNX preprocessing. Default: zero_one",
    )
    parser.add_argument("--allow-stub", action="store_true", help="Use an empty-heatmap stub backend when no model is available.")
    parser.add_argument("--output-dir", type=Path, help="Directory for debug artifacts and annotated video")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to read")
    parser.add_argument("--buffer-size", type=int, default=3, help="Temporal context window size. Default: 3")
    parser.add_argument("--input-width", type=int, default=640, help="Detector input width. Default: 640")
    parser.add_argument("--input-height", type=int, default=360, help="Detector input height. Default: 360")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum heatmap score for a candidate. Default: 0.5")
    parser.add_argument("--min-distance", type=float, default=12.0, help="Minimum candidate spacing in output pixels. Default: 12")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum candidates to keep per frame. Default: 5")
    parser.add_argument("--debug-every", type=int, default=30, help="Save a debug frame every N processed frames. Use 0 to disable. Default: 30")
    return parser.parse_args()


def open_video(video_path: Path):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def default_output_dir(video_path: Path) -> Path:
    return Path("outputs") / "ball" / video_path.stem


def main() -> None:
    import cv2

    args = parse_args()
    resolved_model = args.model or default_tracknet_checkpoint_path()

    if args.inspect_model:
        if resolved_model is None:
            raise SystemExit("No TrackNet checkpoint was provided, and scripts/ball/TrackNet_best.pt was not found.")
        print(json.dumps(inspect_tracknet_checkpoint(resolved_model), indent=2, sort_keys=True))
        return

    if args.video is None:
        raise SystemExit("A video path is required unless --inspect-model is used.")
    if args.buffer_size < 1:
        raise SystemExit("--buffer-size must be at least 1")
    if args.input_width < 1 or args.input_height < 1:
        raise SystemExit("--input-width and --input-height must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates must be at least 1")
    if args.debug_every is not None and args.debug_every < 0:
        raise SystemExit("--debug-every must be 0 or greater")
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    output_dir = args.output_dir or default_output_dir(args.video)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{args.video.stem}_tracknet_annotated.mp4"
    candidates_path = output_dir / "candidates.jsonl"
    summary_path = output_dir / "summary.json"
    debug_dir = output_dir / "debug_frames"
    candidates_path.write_text("", encoding="utf-8")

    try:
        detector = build_detector(
            resolved_model,
            backend=args.backend,
            allow_stub=args.allow_stub,
            context_size=args.buffer_size,
            input_size=(args.input_width, args.input_height),
            onnx_input_layout=args.onnx_input_layout,
            onnx_color_order=args.onnx_color_order,
            onnx_normalization=args.onnx_normalization,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    capture = open_video(args.video)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not write output video: {output_video_path}")

    buffer = TemporalFrameBuffer(detector.context_size)
    frames_read = 0
    processed_frames = 0
    frames_with_candidates = 0
    total_candidates = 0
    best_score = 0.0

    try:
        while args.max_frames is None or frames_read < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            buffer.push(frames_read, frame)
            frames_read += 1
            if not buffer.ready:
                continue

            prediction = detector.predict(buffer.frames(), buffer.frame_indices())
            candidates = extract_candidates(
                prediction.heatmap,
                original_frame_size=(width, height),
                min_score=args.min_score,
                min_distance=args.min_distance,
                max_candidates=args.max_candidates,
            )
            processed_frames += 1
            if candidates:
                frames_with_candidates += 1
                total_candidates += len(candidates)
                best_score = max(best_score, max(candidate.score for candidate in candidates))

            latest_frame = buffer.latest()
            annotated = annotate_candidates(overlay_heatmap(latest_frame, prediction.heatmap), candidates, frame_index=buffer.frame_indices()[-1])
            writer.write(annotated)
            append_candidates_jsonl(
                candidates_path,
                frame_index=buffer.frame_indices()[-1],
                candidates=candidates,
                backend_name=prediction.backend_name,
                frame_indices=prediction.frame_indices,
            )

            if args.debug_every and processed_frames % args.debug_every == 0:
                save_debug_frame(
                    debug_dir / f"frame_{buffer.frame_indices()[-1]:06d}.jpg",
                    latest_frame,
                    prediction.heatmap,
                    candidates,
                    frame_index=buffer.frame_indices()[-1],
                )
    finally:
        capture.release()
        writer.release()

    practical_command_parts = [
        "python",
        "scripts/ball/test_tracknet.py",
        str(args.video),
        "--output-dir",
        str(output_dir),
    ]
    if args.allow_stub:
        practical_command_parts.append("--allow-stub")
    elif resolved_model:
        practical_command_parts.extend(
            [
                "--model",
                str(resolved_model),
                "--backend",
                args.backend,
                "--buffer-size",
                str(detector.context_size),
                "--onnx-input-layout",
                args.onnx_input_layout,
                "--onnx-color-order",
                args.onnx_color_order,
                "--onnx-normalization",
                args.onnx_normalization,
            ]
        )

    summary = {
        "video": str(args.video),
        "model": str(resolved_model) if resolved_model else None,
        "backend": detector.backend_name,
        "allow_stub": bool(args.allow_stub),
        "buffer_size": detector.context_size,
        "input_size": [args.input_width, args.input_height],
        "onnx_options": {
            "input_layout": args.onnx_input_layout,
            "color_order": args.onnx_color_order,
            "normalization": args.onnx_normalization,
        },
        "frame_count": frame_count or None,
        "frames_read": frames_read,
        "processed_frames": processed_frames,
        "frames_with_candidates": frames_with_candidates,
        "total_candidates": total_candidates,
        "best_score": best_score,
        "output_video": str(output_video_path),
        "candidates_jsonl": str(candidates_path),
        "debug_dir": str(debug_dir),
        "summary_path": str(summary_path),
        "practical_command": " ".join(practical_command_parts),
        "note": "Generic public TrackNet checkpoints may still require trying different ONNX preprocessing settings; the PyTorch path is an explicit TrackNet-style experimental adapter.",
    }
    write_summary_json(summary_path, summary)

    print(f"Backend: {summary['backend']}")
    print(f"Frames read: {frames_read}/{frame_count or 'unknown'}")
    print(f"Processed frames: {processed_frames}")
    print(f"Frames with candidates: {frames_with_candidates}")
    print(f"Candidates JSONL: {candidates_path}")
    print(f"Annotated video: {output_video_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
