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


def draw_detections(image_path: Path, result: dict, output_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing dependency: Pillow. Run `python -m pip install -r requirements.txt`."
        ) from error

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for prediction in result.get("predictions", []):
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])

        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        label = prediction.get("class", "object")
        confidence = prediction.get("confidence")
        if confidence is not None:
            label = f"{label} {float(confidence):.2f}"

        draw.rectangle((left, top, right, bottom), outline="red", width=3)

        text_left, text_top, text_right, text_bottom = draw.textbbox((left, top), label, font=font)
        draw.rectangle((text_left, text_top, text_right + 4, text_bottom + 4), fill="red")
        draw.text((text_left + 2, text_top + 2), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


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
