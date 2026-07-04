#!/usr/bin/env python3
"""Run the Roboflow basketball player detection model on one image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


MODEL_ID = "basketball-player-detection-3-ycjdo/16"
API_URL = "https://serverless.roboflow.com"


def detect_players(image_path: Path, api_key: str) -> dict:
    from inference_sdk import InferenceHTTPClient

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    client = InferenceHTTPClient(api_url=API_URL, api_key=api_key)
    return client.infer(str(image_path), model_id=MODEL_ID)


def annotate_image(image_path: Path, result: dict):
    try:
        import cv2
        import supervision as sv
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from error

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    detections = sv.Detections.from_inference(result)
    class_names = detections.data.get("class_name", ["object"] * len(detections))
    confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
    labels = [
        f"{class_name} {confidence:.2f}" if confidence is not None else str(class_name)
        for class_name, confidence in zip(class_names, confidences)
    ]

    annotated = sv.BoxAnnotator().annotate(scene=image, detections=detections)
    return sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=labels)


def draw_detections(image_path: Path, result: dict, output_path: Path) -> None:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing dependency. Run `python -m pip install -r requirements.txt`."
        ) from error

    try:
        annotated = annotate_image(image_path, result)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise SystemExit(f"Could not write annotated image: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect basketball players in one frame.")
    parser.add_argument("image", type=Path, help="Path to an image/frame")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the annotated image. Defaults to outputs/<image-stem>_detections.jpg",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing dependency: python-dotenv. Run `python -m pip install -r requirements.txt`."
        ) from error

    load_dotenv()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")

    result = detect_players(args.image, api_key)
    output_path = args.output or Path("outputs") / f"{args.image.stem}_detections.jpg"
    draw_detections(args.image, result, output_path)

    print(json.dumps(result, indent=2))
    print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
