"""Minimal court-keypoint parsing and homography helpers for the active pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_COURT_MODEL_ID = "basketball-court-detection-2/22"

_COURT_TEMPLATE_INDEX = {
    "01": 0,
    "02": 1,
    "03": 2,
    "04": 3,
    "05": 4,
    "06": 5,
    "07": 6,
    "08": 7,
    "09": 8,
    "10": 9,
    "11": 10,
    "12": 11,
    "13": 12,
    "14": 13,
    "15": 14,
    "16": 15,
    "17": 16,
    "18": 17,
    "19": 18,
    "20": 19,
    "21": 20,
    "22": 21,
    "23": 22,
    "24": 23,
    "25": 24,
    "26": 25,
    "27": 26,
    "28": 27,
    "29": 28,
    "30": 29,
    "31": 30,
    "32": 31,
    "33": 32,
    "34": 33,
    "35": 34,
    "36": 35,
    "37": 36,
    "38": 37,
    "39": 38,
    "40": 39,
    "41": 40,
}

_CANONICAL_POINT_MAP = {
    "01": (0.0, 0.0),
    "02": (0.0, 0.9),
    "04": (0.0, 5.05),
    "05": (0.0, 9.95),
    "07": (0.0, 14.1),
    "08": (0.0, 15.0),
    "09": (1.575, 7.5),
    "10": (2.99, 0.9),
    "11": (2.99, 14.1),
    "12": (5.8, 5.05),
    "13": (5.8, 7.5),
    "14": (5.8, 9.95),
    "15": (8.325, 0.0),
    "16": (8.325, 7.5),
    "17": (8.325, 15.0),
    "18": (14.0, 5.7),
    "19": (14.0, 0.0),
    "20": (14.0, 7.5),
    "21": (14.0, 7.5),
    "22": (14.0, 9.3),
    "23": (14.0, 15.0),
    "25": (19.675, 0.0),
    "26": (19.675, 7.5),
    "27": (19.675, 15.0),
    "28": (22.2, 5.05),
    "29": (22.2, 7.5),
    "30": (22.2, 9.95),
    "31": (25.01, 0.9),
    "32": (25.01, 14.1),
    "33": (26.425, 7.5),
    "34": (28.0, 0.0),
    "35": (28.0, 0.9),
    "37": (28.0, 5.05),
    "38": (28.0, 9.95),
    "40": (28.0, 14.1),
    "41": (28.0, 15.0),
}


@dataclass
class HomographyResult:
    matrix: Any
    inliers_count: int
    reprojection_mean_m: float


def _numeric(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _label_for(point: dict) -> str:
    return str(
        point.get("class")
        or point.get("class_name")
        or point.get("name")
        or point.get("label")
        or point.get("keypoint")
        or "keypoint"
    )


def _prediction_items(result: Any) -> list[Any]:
    if isinstance(result, dict):
        predictions = result.get("predictions", result.get("prediction", []))
        return predictions if isinstance(predictions, list) else [predictions]
    return result if isinstance(result, list) else []


def _keypoint_dicts_from_keypoints(result: Any) -> list[dict]:
    items: list[dict] = []
    candidates = [result, *_prediction_items(result)]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        keypoints = candidate.get("keypoints")
        if isinstance(keypoints, list):
            items.extend(point for point in keypoints if isinstance(point, dict))
    return items


def _direct_keypoint_dicts(result: Any) -> list[dict]:
    candidates = [result, *_prediction_items(result)]
    return [candidate for candidate in candidates if isinstance(candidate, dict) and "x" in candidate and "y" in candidate]


def parse_court_keypoints(result: dict, *, min_confidence: float, width: int, height: int) -> list[dict]:
    raw_points = _keypoint_dicts_from_keypoints(result) or _direct_keypoint_dicts(result)
    keypoints = []
    for point in raw_points:
        x = _numeric(point.get("x"))
        y = _numeric(point.get("y"))
        confidence = _numeric(point.get("confidence"))
        if x is None or y is None:
            continue
        if confidence is not None and confidence < min_confidence:
            continue
        keypoints.append(
            {
                "label": _label_for(point),
                "x": x,
                "y": y,
                "x_normalized": x / width if width else None,
                "y_normalized": y / height if height else None,
                "confidence": confidence,
            }
        )
    fallback_index = len(_COURT_TEMPLATE_INDEX)
    return sorted(
        keypoints,
        key=lambda point: (
            _COURT_TEMPLATE_INDEX.get(str(point.get("label", "")), fallback_index),
            str(point.get("label", "")),
        ),
    )


def compute_frame_homography(keypoints: list[dict], *, min_confidence: float) -> HomographyResult:
    import cv2
    import numpy as np

    matched = []
    src_pts: list[tuple[float, float]] = []
    dst_pts: list[tuple[float, float]] = []
    for point in keypoints:
        label = str(point.get("label", ""))
        confidence = point.get("confidence")
        if label not in _CANONICAL_POINT_MAP:
            continue
        if confidence is not None and float(confidence) < min_confidence:
            continue
        src_pts.append((float(point["x"]), float(point["y"])))
        dst_pts.append(_CANONICAL_POINT_MAP[label])
        matched.append((float(point["x"]), float(point["y"])))

    if len(src_pts) < 4 or len({(x, y) for x, y in dst_pts}) < 4:
        raise ValueError("insufficient_points")

    image_points = np.asarray(matched, dtype=np.float32)
    if float(np.ptp(image_points[:, 0])) < 40.0 or float(np.ptp(image_points[:, 1])) < 40.0:
        raise ValueError("insufficient_spread")

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
    if homography is None or mask is None:
        raise RuntimeError("cv2.findHomography returned no result")

    inlier_mask = mask.reshape(-1).astype(bool)
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    inlier_errors = errors[inlier_mask]
    reprojection_mean = float(inlier_errors.mean()) if inlier_errors.size else 0.0
    return HomographyResult(
        matrix=homography,
        inliers_count=int(inlier_mask.sum()),
        reprojection_mean_m=reprojection_mean,
    )


def project_points(homography: Any, points: list[tuple[float, float]]):
    import cv2
    import numpy as np

    arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, homography).reshape(-1, 2)
