#!/usr/bin/env python3
"""Render shot-model detections for only the first frame of a video."""

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


def roboflow_client(api_key: str):
    try:
        from inference_sdk import InferenceHTTPClient

        return InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=api_key)
    except ModuleNotFoundError:
        return DirectRoboflowClient(api_url=ROBOFLOW_API_URL, api_key=api_key)


class DirectRoboflowClient:
    def __init__(self, *, api_url: str, api_key: str) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key

    def infer(self, inference_input, *, model_id: str) -> dict:
        payload = Path(inference_input).read_bytes()
        boundary = "----scoutai-first-shot-frame-boundary"
        body = (
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n'
                "Content-Type: image/jpeg\r\n\r\n"
            ).encode("utf-8")
            + payload
            + f"\r\n--{boundary}--\r\n".encode("utf-8")
        )
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


def infer_frame(client, frame) -> dict:
    import cv2

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix="_frame0.jpg", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        if not cv2.imwrite(str(temp_path), frame):
            raise RuntimeError("Failed to write temporary frame image")
        return client.infer(str(temp_path), model_id=MODEL_ID)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Render shot-model detections on the first frame only.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Path for the annotated first-frame image")
    parser.add_argument("--confidence", type=float, default=0.0, help="Minimum confidence to draw")
    args = parser.parse_args()

    output = args.output or Path("outputs") / f"{args.video.stem}_shot_frame0.jpg"
    api_key = get_roboflow_api_key()
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"Could not read first frame from: {args.video}")

    result = infer_frame(roboflow_client(api_key), frame)
    detections = detections_from_inference(result, min_confidence=args.confidence)
    annotated = annotate_detections(frame, detections)

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated):
        raise SystemExit(f"Could not write output image: {output}")

    print(f"Model: {MODEL_ID}")
    print(f"Saved first-frame shot detections to: {output}")


if __name__ == "__main__":
    main()
