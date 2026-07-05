#!/usr/bin/env python3
"""Run the Roboflow basketball court keypoint model on one image."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

try:
    from scripts.court_template import (
        edges_for_labels,
        label_class_mapping as template_label_class_mapping,
        labeled_edges_for_labels,
        sort_keypoints_by_template,
    )
    from scripts.env_config import COURT_MODEL_ID, ROBOFLOW_API_URL, get_roboflow_api_key
except ModuleNotFoundError:
    from court_template import (
        edges_for_labels,
        label_class_mapping as template_label_class_mapping,
        labeled_edges_for_labels,
        sort_keypoints_by_template,
    )
    from env_config import COURT_MODEL_ID, ROBOFLOW_API_URL, get_roboflow_api_key

API_URL = ROBOFLOW_API_URL
DEFAULT_MODEL_ID = COURT_MODEL_ID

def detect_keypoints(image_path: Path, api_key: str, model_id: str) -> dict:
    from inference_sdk import InferenceHTTPClient

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    client = InferenceHTTPClient(api_url=API_URL, api_key=api_key)
    return client.infer(str(image_path), model_id=model_id)


def numeric(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def label_for(point: dict) -> str:
    return str(
        point.get("class")
        or point.get("class_name")
        or point.get("name")
        or point.get("label")
        or point.get("keypoint")
        or "keypoint"
    )


def prediction_items(result: Any) -> list[Any]:
    if isinstance(result, dict):
        predictions = result.get("predictions", result.get("prediction", []))
        return predictions if isinstance(predictions, list) else [predictions]
    return result if isinstance(result, list) else []


def keypoint_dicts_from_keypoints(result: Any) -> list[dict]:
    items: list[dict] = []
    candidates = [result, *prediction_items(result)]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        keypoints = candidate.get("keypoints")
        if isinstance(keypoints, list):
            items.extend(point for point in keypoints if isinstance(point, dict))
    return items


def direct_keypoint_dicts(result: Any) -> list[dict]:
    candidates = [result, *prediction_items(result)]
    return [candidate for candidate in candidates if isinstance(candidate, dict) and "x" in candidate and "y" in candidate]


def parse_keypoints(result: dict, min_confidence: float, width: int, height: int) -> list[dict]:
    raw_points = keypoint_dicts_from_keypoints(result) or direct_keypoint_dicts(result)
    keypoints = []
    for point in raw_points:
        x = numeric(point.get("x"))
        y = numeric(point.get("y"))
        confidence = numeric(point.get("confidence"))
        if x is None or y is None:
            continue
        if confidence is not None and confidence < min_confidence:
            continue
        keypoints.append(
            {
                "label": label_for(point),
                "x": x,
                "y": y,
                "x_normalized": x / width if width else None,
                "y_normalized": y / height if height else None,
                "confidence": confidence,
            }
        )
    return sort_keypoints_by_template(keypoints)


def label_class_mapping(parsed_keypoints: list[dict]) -> dict[str, int]:
    return template_label_class_mapping([str(point["label"]) for point in parsed_keypoints])


def to_keypoints(parsed_keypoints: list[dict]):
    """Build a Supervision ``sv.KeyPoints`` from the parsed list of dicts.

    Returns ``None`` (or a zero-keypoint KeyPoints, depending on caller) when
    there is nothing to draw.
    """
    import numpy as np
    import supervision as sv

    if not parsed_keypoints:
        return sv.KeyPoints(
            xy=np.zeros((1, 0, 2), dtype=np.float32),
            class_id=np.array([0]),
            keypoint_confidence=None,
        )

    class_lookup = label_class_mapping(parsed_keypoints)
    labels = [point["label"] for point in parsed_keypoints]
    per_keypoint_class_ids = [class_lookup[label] for label in labels]
    xy = np.array(
        [[(float(point["x"]), float(point["y"])) for point in parsed_keypoints]],
        dtype=np.float32,
    )
    confidences = [point.get("confidence") for point in parsed_keypoints]
    has_confidence = all(value is not None for value in confidences)
    keypoint_confidence = (
        np.array([confidences], dtype=np.float32) if has_confidence else None
    )
    # ponytail: sv.KeyPoints.data lists must have length n (instances, here 1),
    # so per-keypoint label info is wrapped in a length-1 outer list. Keep the
    # per-keypoint class ids alongside so callers can recover them too.
    return sv.KeyPoints(
        xy=xy,
        class_id=np.array([0]),
        keypoint_confidence=keypoint_confidence,
        data={"labels": [labels], "class_ids": [per_keypoint_class_ids]},
    )


def draw_with_annotators(image, key_points) -> Any:
    """Annotate ``image`` with keypoint vertices and labels.

    Returns the annotated image. Label text is still drawn with cv2 since the
    Supervision VertexAnnotator only renders circles.
    """
    import cv2
    import supervision as sv

    keypoint_count = len(key_points.xy[0]) if len(key_points) > 0 else 0
    annotated = image
    if keypoint_count > 0:
        vertex = sv.VertexAnnotator(color=sv.Color.YELLOW, radius=6)
        annotated = vertex.annotate(annotated, key_points)

    nested_labels = (key_points.data or {}).get("labels") if key_points.data else None
    labels = nested_labels[0] if nested_labels else []
    for point, label in zip(key_points.xy[0], labels):
        if hasattr(point, "__len__") and len(point) >= 2:
            x, y = int(round(float(point[0]))), int(round(float(point[1])))
        else:
            continue
        cv2.putText(
            annotated,
            str(label),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    if not labels:
        cv2.putText(
            annotated,
            "No court keypoints found",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
    return annotated


def visible_edges_for_keypoints(key_points) -> list[tuple[int, int]]:
    labels_data = (key_points.data or {}).get("labels") if key_points.data else None
    labels = labels_data[0] if labels_data else []
    if len(labels) < 2 or len(key_points.xy) == 0:
        return []

    candidate_edges = labeled_edges_for_labels(labels)
    if len(candidate_edges) <= 1:
        return edges_for_labels(labels)

    points = key_points.xy[0]
    distances: list[float] = []
    edge_distances: dict[tuple[int, int], float] = {}
    for _, _, start_index, end_index in candidate_edges:
        start = points[start_index]
        end = points[end_index]
        distance = math.hypot(float(start[0]) - float(end[0]), float(start[1]) - float(end[1]))
        edge_key = (start_index, end_index)
        edge_distances[edge_key] = distance
        if distance > 0:
            distances.append(distance)

    if not distances:
        return []

    distance_limit = median(distances) * 1.8
    plausible_edges = [
        (start_index, end_index)
        for _, _, start_index, end_index in candidate_edges
        if edge_distances[(start_index, end_index)] <= distance_limit
    ]
    if not plausible_edges:
        return []

    adjacency: dict[int, set[int]] = {}
    for start_index, end_index in plausible_edges:
        adjacency.setdefault(start_index, set()).add(end_index)
        adjacency.setdefault(end_index, set()).add(start_index)

    best_component: set[int] = set()
    best_edge_count = -1
    visited: set[int] = set()
    for node in adjacency:
        if node in visited:
            continue
        stack = [node]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(neighbor for neighbor in adjacency[current] if neighbor not in visited)

        component_edge_count = sum(
            1 for start_index, end_index in plausible_edges if start_index in component and end_index in component
        )
        if component_edge_count > best_edge_count or (
            component_edge_count == best_edge_count and len(component) > len(best_component)
        ):
            best_component = component
            best_edge_count = component_edge_count

    return [
        (start_index, end_index)
        for start_index, end_index in plausible_edges
        if start_index in best_component and end_index in best_component
    ]


def draw_keypoints(image_path: Path, keypoints: list[dict], output_path: Path) -> None:
    try:
        import cv2
        import supervision as sv
    except ModuleNotFoundError as error:
        raise SystemExit("Missing dependency. Run `python -m pip install -r requirements.txt`.") from error

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    key_points = to_keypoints(keypoints)
    annotated = draw_with_annotators(image, key_points)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise SystemExit(f"Could not write annotated image: {output_path}")


def image_size(image_path: Path) -> tuple[int, int]:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise SystemExit("Missing dependency. Run `python -m pip install -r requirements.txt`.") from error

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect basketball court keypoints in one frame.")
    parser.add_argument("image", type=Path, help="Path to an image/frame")
    parser.add_argument("--output", type=Path, help="Path for the annotated image")
    parser.add_argument("--json-output", type=Path, help="Path for raw and parsed keypoint JSON")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Roboflow model ID. Defaults to COURT_MODEL_ID or basketball-court-detection-2/22",
    )
    parser.add_argument("--confidence", type=float, default=0.0, help="Minimum keypoint confidence to draw, when provided")
    args = parser.parse_args()

    api_key = get_roboflow_api_key()
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")
    model_id = args.model_id or DEFAULT_MODEL_ID

    output_path = args.output or Path("outputs") / f"{args.image.stem}_court_keypoints.jpg"
    json_output = args.json_output or Path("outputs") / f"{args.image.stem}_court_keypoints.json"
    width, height = image_size(args.image)

    try:
        result = detect_keypoints(args.image, api_key, model_id)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    keypoints = parse_keypoints(result, args.confidence, width, height)
    draw_keypoints(args.image, keypoints, output_path)

    json_output.parent.mkdir(parents=True, exist_ok=True)
    with json_output.open("w", encoding="utf-8") as json_file:
        json.dump(
            {
                "model_id": model_id,
                "raw_result": result,
                "keypoints": keypoints,
                "keypoints_sv": _supervision_summary(keypoints),
            },
            json_file,
            indent=2,
            default=str,
        )

    if not keypoints:
        print("Warning: no court keypoints found")
    print(f"Keypoints: {len(keypoints)}")
    print(f"Annotated image saved to: {output_path}")
    print(f"Keypoint JSON saved to: {json_output}")


def _supervision_summary(parsed_keypoints: list[dict]) -> dict[str, Any] | None:
    """Compact, JSON-friendly summary of the Supervision KeyPoints view."""
    try:
        import numpy as np
    except ModuleNotFoundError:
        return None

    if not parsed_keypoints:
        return {
            "xy": [],
            "labels": [],
            "class_ids": [],
            "keypoint_confidence": [],
        }

    class_lookup = label_class_mapping(parsed_keypoints)
    xy = [[float(point["x"]), float(point["y"])] for point in parsed_keypoints]
    labels = [point["label"] for point in parsed_keypoints]
    class_ids = [class_lookup[label] for label in labels]
    confidences = [point.get("confidence") for point in parsed_keypoints]
    return {
        "xy": xy,
        "labels": labels,
        "class_ids": class_ids,
        "keypoint_confidence": [
            float(confidence) if confidence is not None else None
            for confidence in confidences
        ],
    }


if __name__ == "__main__":
    main()
