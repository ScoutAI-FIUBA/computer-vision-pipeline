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


def draw_detections(frame, result: dict, min_confidence: float):
    import supervision as sv

    detections = sv.Detections.from_inference(result)
    if detections.confidence is not None:
        detections = detections[detections.confidence >= min_confidence]

    class_names = detections.data.get("class_name", ["object"] * len(detections))
    confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
    labels = [
        f"{class_name} {confidence:.2f}" if confidence is not None else str(class_name)
        for class_name, confidence in zip(class_names, confidences)
    ]

    annotated = sv.BoxAnnotator().annotate(scene=frame, detections=detections)
    return sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=labels)


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


def video_info(capture, max_frames: int | None) -> tuple[float, int, int]:
    import cv2

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_to_read = frame_count if max_frames is None else min(frame_count, max_frames)
    return fps, frame_count, frames_to_read


def inspect_video(video_path: Path, frame_stride: int, max_frames: int | None) -> dict:
    capture = open_video(video_path)
    try:
        fps, frame_count, frames_to_read = video_info(capture, max_frames)
    finally:
        capture.release()

    return {
        "fps": fps or None,
        "frame_count": frame_count or None,
        "frames_to_read": frames_to_read,
        "estimated_roboflow_calls": estimate_calls(frames_to_read, frame_stride),
    }


def default_output_fps(fps: float, frame_stride: int, sampled_output_only: bool, output_fps: float | None) -> float:
    if output_fps is not None:
        return output_fps
    return max(1.0, (fps or 30) / frame_stride) if sampled_output_only else fps or 30


def process_video(
    video_path: Path,
    output_path: Path,
    api_key: str,
    frame_stride: int = 30,
    max_frames: int | None = None,
    sampled_output_only: bool = True,
    reuse_last_detections: bool = False,
    confidence: float = 0.0,
    output_fps: float | None = None,
) -> dict:
    import cv2

    capture = open_video(video_path)
    fps, frame_count, frames_to_read = video_info(capture, max_frames)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fps = default_output_fps(fps, frame_stride, sampled_output_only, output_fps)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    empty_result = {"predictions": []}
    last_result = empty_result
    frame_index = 0
    calls = 0

    try:
        client = roboflow_client(api_key)
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            result = empty_result
            inferred = False
            if frame_index % frame_stride == 0:
                result = infer_frame(client, frame, f"_{frame_index}.jpg")
                last_result = result
                calls += 1
                inferred = True
            elif reuse_last_detections:
                result = last_result

            frame = draw_detections(frame, result, confidence)
            if inferred or not sampled_output_only:
                writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {
        "model_id": MODEL_ID,
        "frames_read": frame_index,
        "frame_count": frame_count or None,
        "frames_to_read": frames_to_read,
        "roboflow_calls": calls,
        "output_fps": output_fps,
    }


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

    try:
        info = inspect_video(args.video, args.frame_stride, args.max_frames)

        if args.dry_run:
            print(f"Video: {args.video}")
            print(f"FPS: {info['fps'] or 'unknown'}")
            print(f"Frame count: {info['frame_count'] or 'unknown'}")
            print(f"Frames to read: {info['frames_to_read'] or 'unknown'}")
            print(f"Estimated Roboflow calls: {info['estimated_roboflow_calls']}")
            return
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")

    try:
        stats = process_video(
            args.video,
            args.output,
            api_key,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            sampled_output_only=args.sampled_output_only,
            reuse_last_detections=args.reuse_last_detections,
            confidence=args.confidence,
            output_fps=args.output_fps,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(f"Frames read: {stats['frames_read']}/{stats['frame_count'] or 'unknown'}")
    print(f"Roboflow calls: {stats['roboflow_calls']}")
    print(f"Output FPS: {stats['output_fps']}")
    print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
