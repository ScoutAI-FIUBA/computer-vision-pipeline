#!/usr/bin/env python3
"""Render only jump-shot detections on a video for debugging."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from urllib import error, parse, request


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.shared.annotation_helpers import annotate_detections, detections_from_inference
from scripts.shared.env_config import ROBOFLOW_API_URL, SHOT_MODEL_ID, get_roboflow_api_key

MODEL_ID = SHOT_MODEL_ID
JUMP_SHOT_CLASS_ID = 5
JUMP_SHOT_CLASS_NAMES = {"jump_shot"}


def roboflow_client(api_key: str):
    try:
        from inference_sdk import InferenceHTTPClient

        return InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=api_key)
    except ModuleNotFoundError:
        return DirectRoboflowClient(api_url=ROBOFLOW_API_URL, api_key=api_key)


class DirectRoboflowClient:
    """Tiny stdlib fallback for hosted Roboflow inference."""

    def __init__(self, *, api_url: str, api_key: str) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key

    def infer(self, inference_input, *, model_id: str) -> dict:
        payload = self._read_bytes(inference_input)
        boundary = "----scoutai-jump-shot-boundary"
        body = self._multipart_body(boundary, payload)
        url = f"{self._api_url}/{model_id}?{parse.urlencode({'api_key': self._api_key})}"
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Roboflow request failed ({exc.code}): {detail}") from exc

    @staticmethod
    def _read_bytes(inference_input) -> bytes:
        path = Path(inference_input)
        return path.read_bytes()

    @staticmethod
    def _multipart_body(boundary: str, payload: bytes) -> bytes:
        prefix = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8")
        suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
        return prefix + payload + suffix


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


def jump_shot_only(result: dict) -> dict:
    filtered = []
    for prediction in result.get("predictions", []):
        class_name = str(prediction.get("class", "")).strip().lower().replace("-", "_")
        class_id = prediction.get("class_id")
        try:
            class_id = None if class_id is None else int(class_id)
        except (TypeError, ValueError):
            class_id = None
        if class_id == JUMP_SHOT_CLASS_ID or class_name in JUMP_SHOT_CLASS_NAMES:
            filtered.append(prediction)
    return {**result, "predictions": filtered}


def draw_jump_shots(frame, result: dict, min_confidence: float):
    detections = detections_from_inference(jump_shot_only(result), min_confidence=min_confidence)
    return annotate_detections(frame, detections)


def process_video(
    video_path: Path,
    output_path: Path,
    api_key: str,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    confidence: float = 0.0,
    sampled_output_only: bool = False,
) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fps = max(1.0, fps / frame_stride) if sampled_output_only else fps
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    client = roboflow_client(api_key)
    frame_index = 0
    calls = 0

    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            inferred = frame_index % frame_stride == 0
            result = {"predictions": []}
            if inferred:
                result = infer_frame(client, frame, f"_{frame_index}.jpg")
                calls += 1

            annotated = draw_jump_shots(frame, result, confidence)
            if inferred or not sampled_output_only:
                writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {
        "model_id": MODEL_ID,
        "frames_read": frame_index,
        "frame_count": frame_count or None,
        "roboflow_calls": calls,
        "output_fps": output_fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render jump-shot boxes on a video for debugging.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Path for the annotated video")
    parser.add_argument("--frame-stride", type=int, default=1, help="Infer every N frames. Default: 1")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to read")
    parser.add_argument("--confidence", type=float, default=0.0, help="Minimum confidence to draw")
    parser.add_argument(
        "--sampled-output-only",
        action="store_true",
        help="Write only frames that were sent to the detector.",
    )
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")

    args.output = args.output or Path("outputs") / f"{args.video.stem}_jump_shot.mp4"

    api_key = get_roboflow_api_key()
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")

    stats = process_video(
        args.video,
        args.output,
        api_key,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        confidence=args.confidence,
        sampled_output_only=args.sampled_output_only,
    )

    print(f"Model: {stats['model_id']}")
    print(f"Frames read: {stats['frames_read']}/{stats['frame_count'] or 'unknown'}")
    print(f"Roboflow calls: {stats['roboflow_calls']}")
    print(f"Output FPS: {stats['output_fps']}")
    print(f"Jump-shot debug video saved to: {args.output}")


if __name__ == "__main__":
    main()
