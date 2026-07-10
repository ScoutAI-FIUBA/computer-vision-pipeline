#!/usr/bin/env python3
"""Run the Roboflow basketball court keypoint model on a video."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from env_config import COURT_MODEL_ID, ROBOFLOW_API_URL, get_roboflow_api_key  # noqa: E402

API_URL = ROBOFLOW_API_URL
DEFAULT_MODEL_ID = COURT_MODEL_ID

from detect_court_keypoints import draw_with_annotators, parse_keypoints, to_keypoints  # noqa: E402  (sys.path tweak above)
from detect_players_video import (  # noqa: E402
    default_output_fps,
    inspect_video,
    open_video,
    video_info,
)


def roboflow_client(api_key: str):
    from inference_sdk import InferenceHTTPClient

    return InferenceHTTPClient(api_url=API_URL, api_key=api_key)


def infer_frame(client, frame, suffix: str, model_id: str) -> dict:
    import cv2

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        if not cv2.imwrite(str(temp_path), frame):
            raise RuntimeError("Failed to write temporary frame image")
        return client.infer(str(temp_path), model_id=model_id)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def draw_keypoints(frame, keypoints: list[dict]):
    return draw_with_annotators(frame, to_keypoints(keypoints))


def filter_keypoints(result: Any, min_confidence: float, width: int, height: int) -> list[dict]:
    return parse_keypoints(result, min_confidence, width, height)


def process_video(
    video_path: Path,
    output_path: Path,
    api_key: str,
    model_id: str,
    frame_stride: int = 30,
    max_frames: int | None = None,
    min_confidence: float = 0.7,
    sampled_output_only: bool = True,
    output_fps: float | None = None,
) -> dict:
    import cv2

    capture = open_video(video_path)
    fps, frame_count, frames_to_read = video_info(capture, max_frames)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    effective_output_fps = default_output_fps(fps, frame_stride, sampled_output_only, output_fps)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), effective_output_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    frame_index = 0
    calls = 0
    frames_written = 0
    keypoints_total = 0

    try:
        client = roboflow_client(api_key)
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            inferred = False
            if frame_index % frame_stride == 0:
                result = infer_frame(client, frame, f"_{frame_index}.jpg", model_id)
                calls += 1
                inferred = True
                keypoints = filter_keypoints(result, min_confidence, width, height)
                frame = draw_keypoints(frame, keypoints)
                keypoints_total += len(keypoints)

            if inferred or not sampled_output_only:
                writer.write(frame)
                frames_written += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {
        "model_id": model_id,
        "frames_read": frame_index,
        "frame_count": frame_count or None,
        "frames_to_read": frames_to_read,
        "roboflow_calls": calls,
        "frames_written": frames_written,
        "keypoints_total": keypoints_total,
        "output_fps": effective_output_fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect basketball court keypoints in a video with cost controls.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Path for the annotated video")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Roboflow model ID. Defaults to COURT_MODEL_ID or basketball-court-detection-2/22",
    )
    parser.add_argument("--frame-stride", type=int, default=30, help="Infer every N frames. Default: 30")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to read")
    parser.add_argument("--dry-run", action="store_true", help="Print video info and estimated calls without inference")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Minimum keypoint confidence to draw. Default: 0.7 (filters model noise; 0.0 disables)",
    )
    parser.add_argument(
        "--sampled-output-only",
        dest="sampled_output_only",
        action="store_true",
        default=True,
        help="Write only frames sent to Roboflow. Default: true",
    )
    parser.add_argument(
        "--no-sampled-output-only",
        dest="sampled_output_only",
        action="store_false",
        help="Write every frame; only inferred frames get keypoints drawn",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        help="Output video FPS. Defaults to input FPS / frame stride for sampled output",
    )
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    if args.confidence < 0 or args.confidence > 1:
        raise SystemExit("--confidence must be between 0 and 1")
    if args.output_fps is not None and args.output_fps <= 0:
        raise SystemExit("--output-fps must be greater than 0")

    api_key = get_roboflow_api_key()
    if not api_key and not args.dry_run:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")
    model_id = args.model_id or DEFAULT_MODEL_ID
    args.output = args.output or Path("outputs") / f"{args.video.stem}_court_keypoints.mp4"

    try:
        info = inspect_video(args.video, args.frame_stride, args.max_frames)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    if args.dry_run:
        print(f"Video: {args.video}")
        print(f"FPS: {info['fps'] or 'unknown'}")
        print(f"Frame count: {info['frame_count'] or 'unknown'}")
        print(f"Frames to read: {info['frames_to_read'] or 'unknown'}")
        print(f"Estimated Roboflow calls: {info['estimated_roboflow_calls']}")
        print(f"Model ID: {model_id}")
        return

    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")

    try:
        stats = process_video(
            args.video,
            args.output,
            api_key,
            model_id=model_id,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            min_confidence=args.confidence,
            sampled_output_only=args.sampled_output_only,
            output_fps=args.output_fps,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(f"Model ID: {stats['model_id']}")
    print(f"Frames read: {stats['frames_read']}/{stats['frame_count'] or 'unknown'}")
    print(f"Roboflow calls: {stats['roboflow_calls']}")
    print(f"Keypoints drawn (post-filter): {stats['keypoints_total']}")
    print(f"Output FPS: {stats['output_fps']}")
    print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
