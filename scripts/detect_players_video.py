#!/usr/bin/env python3
"""Run the Roboflow basketball player detection model on a video."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


MODEL_ID = "basketball-player-detection-3-ycjdo/16"
API_URL = "https://serverless.roboflow.com"


def roboflow_client(api_key: str):
    from inference_sdk import InferenceHTTPClient

    return InferenceHTTPClient(api_url=API_URL, api_key=api_key)


def estimate_calls(frames_to_read: int, frame_stride: int) -> int:
    return 0 if frames_to_read <= 0 else (frames_to_read + frame_stride - 1) // frame_stride


def draw_detections(frame, predictions: list[dict], min_confidence: float) -> None:
    import cv2

    height, width = frame.shape[:2]

    for prediction in predictions:
        confidence = prediction.get("confidence")
        if confidence is not None and float(confidence) < min_confidence:
            continue

        x = float(prediction["x"])
        y = float(prediction["y"])
        box_width = float(prediction["width"])
        box_height = float(prediction["height"])

        left = max(0, int(x - box_width / 2))
        top = max(0, int(y - box_height / 2))
        right = min(width - 1, int(x + box_width / 2))
        bottom = min(height - 1, int(y + box_height / 2))

        label = prediction.get("class", "object")
        if confidence is not None:
            label = f"{label} {float(confidence):.2f}"

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, max(0, top - 20)), (left + 8 * len(label), top), (0, 0, 255), -1)
        cv2.putText(frame, label, (left + 2, max(12, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


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
        raise SystemExit(f"Could not open video: {video_path}")
    return capture


def video_info(capture, max_frames: int | None) -> tuple[float, int, int]:
    import cv2

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_to_read = frame_count if max_frames is None else min(frame_count, max_frames)
    return fps, frame_count, frames_to_read


def process_video(args: argparse.Namespace, api_key: str) -> None:
    import cv2

    capture = open_video(args.video)
    fps, frame_count, frames_to_read = video_info(capture, args.max_frames)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_fps = args.output_fps
    if output_fps is None:
        output_fps = max(1.0, (fps or 30) / args.frame_stride) if args.sampled_output_only else fps or 30

    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not write output video: {args.output}")

    client = roboflow_client(api_key)
    last_predictions: list[dict] = []
    frame_index = 0
    calls = 0

    try:
        while args.max_frames is None or frame_index < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            predictions: list[dict] = []
            inferred = False
            if frame_index % args.frame_stride == 0:
                result = infer_frame(client, frame, f"_{frame_index}.jpg")
                predictions = result.get("predictions", [])
                last_predictions = predictions
                calls += 1
                inferred = True
            elif args.reuse_last_detections:
                predictions = last_predictions

            draw_detections(frame, predictions, args.confidence)
            if inferred or not args.sampled_output_only:
                writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    print(f"Frames read: {frame_index}/{frame_count or 'unknown'}")
    print(f"Roboflow calls: {calls}")
    print(f"Output FPS: {output_fps}")
    print(f"Annotated video saved to: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect basketball players in a video with cost controls.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Path for the annotated video")
    parser.add_argument("--frame-stride", type=int, default=30, help="Infer every N frames. Default: 30")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to read")
    parser.add_argument("--dry-run", action="store_true", help="Print video info and estimated calls without inference")
    parser.add_argument("--confidence", type=float, default=0.0, help="Minimum confidence to draw, when provided")
    parser.add_argument(
        "--sampled-output-only",
        action="store_true",
        help="Write only frames that were sent to Roboflow. Best for debugging sampled inference.",
    )
    parser.add_argument("--output-fps", type=float, help="Output video FPS. Defaults to input FPS / frame stride for sampled output.")
    parser.add_argument(
        "--reuse-last-detections",
        action="store_true",
        help="Draw the last inferred detections on skipped frames",
    )
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    if args.output_fps is not None and args.output_fps <= 0:
        raise SystemExit("--output-fps must be greater than 0")

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing dependency: python-dotenv. Run `python -m pip install -r requirements.txt`."
        ) from error

    load_dotenv()
    args.output = args.output or Path("outputs") / f"{args.video.stem}_detections.mp4"

    capture = open_video(args.video)
    fps, frame_count, frames_to_read = video_info(capture, args.max_frames)
    capture.release()

    if args.dry_run:
        print(f"Video: {args.video}")
        print(f"FPS: {fps or 'unknown'}")
        print(f"Frame count: {frame_count or 'unknown'}")
        print(f"Frames to read: {frames_to_read or 'unknown'}")
        print(f"Estimated Roboflow calls: {estimate_calls(frames_to_read, args.frame_stride)}")
        return

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")

    process_video(args, api_key)


if __name__ == "__main__":
    main()
