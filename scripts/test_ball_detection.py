#!/usr/bin/env python3
"""Run the player detector model and keep only ball detections."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.shared.annotation_helpers import annotate_detections, detections_from_inference
from scripts.shared.env_config import ROBOFLOW_API_URL, get_roboflow_api_key

MODEL_ID = "basketball-player-detection-3-ycjdo/16"
API_URL = ROBOFLOW_API_URL
DEFAULT_BALL_CLASSES = ("basketball", "ball", "basketball-ball")


def roboflow_client(api_key: str):
    from inference_sdk import InferenceHTTPClient

    return InferenceHTTPClient(api_url=API_URL, api_key=api_key)


def infer_frame(client, frame, suffix: str) -> dict:
    import cv2

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        if not cv2.imwrite(str(temp_path), frame):
            raise RuntimeError("Failed to write temporary frame image")
        return client.infer(str(temp_path), model_id=MODEL_ID)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def open_video(video_path: Path):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def filter_ball_predictions(result: dict, class_names: tuple[str, ...]) -> dict:
    allowed = {name.strip().lower() for name in class_names if name.strip()}
    predictions = result.get("predictions", []) if isinstance(result, dict) else []
    return {
        **(result if isinstance(result, dict) else {}),
        "predictions": [
            prediction
            for prediction in predictions
            if str(prediction.get("class", "")).strip().lower() in allowed
        ],
    }


def draw_ball_detections(frame, result: dict, min_confidence: float):
    detections = detections_from_inference(result, min_confidence=min_confidence)
    return annotate_detections(frame, detections)


def process_video(
    video_path: Path,
    output_path: Path,
    api_key: str,
    *,
    frame_stride: int,
    max_frames: int | None,
    confidence: float,
    class_names: tuple[str, ...],
) -> dict:
    import cv2

    capture = open_video(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / frame_stride), (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    frame_index = 0
    roboflow_calls = 0
    frames_with_ball = 0
    total_ball_predictions = 0
    seen_classes: set[str] = set()

    try:
        client = roboflow_client(api_key)
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            raw_result = infer_frame(client, frame, f"_{frame_index}.jpg")
            roboflow_calls += 1
            seen_classes.update(
                str(prediction.get("class", "")).strip()
                for prediction in raw_result.get("predictions", [])
                if str(prediction.get("class", "")).strip()
            )

            ball_result = filter_ball_predictions(raw_result, class_names)
            ball_count = len(ball_result.get("predictions", []))
            total_ball_predictions += ball_count
            if ball_count:
                frames_with_ball += 1

            writer.write(draw_ball_detections(frame, ball_result, confidence))
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {
        "model_id": MODEL_ID,
        "frame_count": frame_count or None,
        "frames_read": frame_index,
        "roboflow_calls": roboflow_calls,
        "frames_with_ball": frames_with_ball,
        "total_ball_predictions": total_ball_predictions,
        "seen_classes": sorted(seen_classes),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect only the ball using the player detector model.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Path for the annotated sampled video")
    parser.add_argument("--frame-stride", type=int, default=10, help="Infer every N frames. Default: 10")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to read")
    parser.add_argument("--confidence", type=float, default=0.4, help="Minimum confidence to draw. Default: 0.4")
    parser.add_argument(
        "--ball-class",
        dest="ball_classes",
        action="append",
        default=list(DEFAULT_BALL_CLASSES),
        help="Allowed class name for the ball. Repeat to add more aliases.",
    )
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")

    args.output = args.output or Path("outputs") / f"{args.video.stem}_ball_detections.mp4"

    api_key = get_roboflow_api_key()
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    try:
        stats = process_video(
            args.video,
            args.output,
            api_key,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            confidence=args.confidence,
            class_names=tuple(args.ball_classes),
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(f"Model: {stats['model_id']}")
    print(f"Frames read: {stats['frames_read']}/{stats['frame_count'] or 'unknown'}")
    print(f"Roboflow calls: {stats['roboflow_calls']}")
    print(f"Frames with ball: {stats['frames_with_ball']}")
    print(f"Total ball detections: {stats['total_ball_predictions']}")
    print(f"Seen classes: {', '.join(stats['seen_classes']) or 'none'}")
    print(f"Annotated sampled video saved to: {stats['output_path']}")


if __name__ == "__main__":
    main()
