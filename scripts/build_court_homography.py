#!/usr/bin/env python3
"""Build a local FIBA court homography from detected keypoints.

WARNING / TODO: this script uses a best-guess ``COURT_POINT_MAP`` that maps
detected keypoint labels to FIBA (28m x 15m) court coordinates. The mapping
was inferred from a single broadcast-angle test image and is explicitly an
APPROXIMATION, not ground truth. Before relying on the resulting homography
for production stats, replace the mapping with a measurement taken on a real
FIBA court (or the target broadcast camera angle). Every consumer of the
output JSON must treat the homography as approximate.

The script:

- Reads parsed keypoints (output of ``detect_court_keypoints.py``).
- Filters them by ``--min-confidence`` and by membership in ``COURT_POINT_MAP``.
- Calls ``cv2.findHomography`` with RANSAC and a configurable reprojection
  threshold (pixel space).
- Returns the homography, the inlier mask, and per-inlier reprojection error
  in meters.
- Persists everything to a JSON file and optionally renders a 2D top-down
  FIBA court visualization with the projected keypoints.

This script does NOT call Roboflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# FIBA court dimensions: 28m long x 15m wide. The mapping is best-guess only
# and needs real measurement before production use. See module docstring.
#
# Coordinate system: x_m along court length (0 = near baseline, 28 = far
# baseline), y_m across court width (0 = one sideline, 15 = the other). Near
# basket rim is at (0, 7.5); far basket rim is at (28, 7.5).
#
# The numeric labels below come from the Roboflow basketball-court-detection-2
# keypoint model. The mapping was inferred from a single broadcast-angle
# frame and is not validated against a measured court.
COURT_POINT_MAP: dict[str, tuple[float, float]] = {
    # Paint / free-throw rectangle corners and free-throw line edges.
    "01": (5.80, 9.95),
    "02": (5.80, 11.10),
    "04": (5.80, 5.05),
    "11": (0.00, 7.50),
    # Free-throw circle: 3.6m radius centered at (5.8, 7.5).
    "09": (2.20, 7.50),
    # 3-point line: 6.75m arc from rim, corner at 1.575m from baseline.
    "05": (1.575, 0.90),
    "10": (6.75, 7.50),
    "12": (5.00, 3.90),
    "13": (4.50, 4.20),
    "14": (1.575, 14.10),
    # Half-court landmarks (labels 15/16 sit past the near half in the test
    # image; treat as approximate).
    "15": (14.00, 11.00),
    "16": (14.00, 4.00),
}


@dataclass
class HomographyResult:
    matrix: np.ndarray
    mask: np.ndarray
    inliers_count: int
    reprojection_mean_m: float
    reprojection_median_m: float
    matched: list[dict[str, Any]]


def load_keypoints_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_keypoints(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the parsed keypoint list out of a detect_court_keypoints.py JSON.

    The image script writes the parsed list under ``keypoints``. Older
    fixtures may store it under ``keypoints_sv.labels`` with coords in
    ``keypoints_sv.xy``; support both.
    """
    keypoints = blob.get("keypoints")
    if isinstance(keypoints, list) and keypoints and {"label", "x", "y"} <= set(keypoints[0].keys()):
        return keypoints

    summary = blob.get("keypoints_sv") or {}
    labels = summary.get("labels") or []
    xy = summary.get("xy") or []
    confidences = summary.get("keypoint_confidence") or []
    out: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        if index >= len(xy):
            break
        point = xy[index]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        confidence = confidences[index] if index < len(confidences) else None
        out.append({"label": str(label), "x": float(point[0]), "y": float(point[1]), "confidence": confidence})
    return out


def compute_homography(
    keypoints: list[dict[str, Any]],
    *,
    min_confidence: float,
    min_points: int,
    reproj_threshold: float,
    point_map: dict[str, tuple[float, float]] | None = None,
) -> HomographyResult:
    mapping = point_map if point_map is not None else COURT_POINT_MAP

    matched: list[dict[str, Any]] = []
    src_pts: list[tuple[float, float]] = []
    dst_pts: list[tuple[float, float]] = []
    skipped_unknown: list[str] = []
    skipped_low_conf: list[str] = []
    for point in keypoints:
        label = str(point.get("label", ""))
        confidence = point.get("confidence")
        if label not in mapping:
            skipped_unknown.append(label)
            continue
        if confidence is not None and confidence < min_confidence:
            skipped_low_conf.append(label)
            continue
        src_pts.append((float(point["x"]), float(point["y"])))
        dst_pts.append(mapping[label])
        matched.append(
            {
                "label": label,
                "image_xy": [float(point["x"]), float(point["y"])],
                "court_meters": list(mapping[label]),
                "confidence": confidence,
            }
        )

    if len(matched) < min_points:
        raise ValueError(
            f"Need at least {min_points} matched points to fit a homography; got {len(matched)} "
            f"(unknown labels: {len(skipped_unknown)}, below confidence: {len(skipped_low_conf)})."
        )

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)

    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj_threshold)
    if homography is None or mask is None:
        raise RuntimeError("cv2.findHomography returned no result (degenerate inputs?)")

    inlier_mask = mask.reshape(-1).astype(bool)
    inliers_count = int(inlier_mask.sum())

    # Reproject all source points through the homography and measure error in
    # FIBA meters for the inlier subset.
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    inlier_errors = errors[inlier_mask]
    if inlier_errors.size:
        reprojection_mean = float(inlier_errors.mean())
        reprojection_median = float(np.median(inlier_errors))
    else:
        reprojection_mean = 0.0
        reprojection_median = 0.0

    for index, entry in enumerate(matched):
        entry["inlier"] = bool(inlier_mask[index])
        entry["reprojection_error_m"] = float(errors[index])

    return HomographyResult(
        matrix=homography,
        mask=mask,
        inliers_count=inliers_count,
        reprojection_mean_m=reprojection_mean,
        reprojection_median_m=reprojection_median,
        matched=matched,
    )


def project_points(homography: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, homography).reshape(-1, 2)


def draw_fiba_court(width_m: float = 28.0, height_m: float = 15.0, pixels_per_meter: int = 40) -> np.ndarray:
    """Render a top-down 2D FIBA court with the standard lines.

    This is a sanity-check visualization, not a precise blueprint. Hoops are
    drawn as small circles; the 3-point line is a single arc per half plus the
    straight corner segments.
    """
    width_px = int(width_m * pixels_per_meter)
    height_px = int(height_m * pixels_per_meter)
    image = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    color_court = (40, 40, 40)
    color_paint = (60, 60, 60)

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        return int(round(x_m * pixels_per_meter)), int(round(y_m * pixels_per_meter))

    # Outer boundary.
    cv2.rectangle(image, to_px(0, 0), to_px(width_m, height_m), color_court, 2)

    # Half-court line.
    cv2.line(image, to_px(width_m / 2, 0), to_px(width_m / 2, height_m), color_court, 2)
    # Center circle (FIBA 3.6m radius around center court).
    cv2.circle(image, to_px(width_m / 2, height_m / 2), int(3.6 * pixels_per_meter), color_court, 2)

    for basket_x in (0.0, width_m):
        sign = 1 if basket_x == 0.0 else -1
        rim_y = height_m / 2
        paint_half_width = 2.45
        three_pt_inset_from_sideline = 0.9
        corner_x = basket_x + sign * 1.575
        corner_y_top = rim_y - (paint_half_width + three_pt_inset_from_sideline)
        corner_y_bot = rim_y + (paint_half_width + three_pt_inset_from_sideline)
        backboard_y_top = rim_y - 0.9
        backboard_y_bot = rim_y + 0.9

        # Backboard + rim.
        cv2.line(
            image,
            to_px(corner_x, backboard_y_top),
            to_px(corner_x, backboard_y_bot),
            color_court,
            2,
        )
        cv2.circle(image, to_px(basket_x, rim_y), int(0.225 * pixels_per_meter), color_court, 2)

        # Paint / key (4.9m wide x 5.8m deep).
        cv2.rectangle(
            image,
            to_px(basket_x, rim_y - paint_half_width),
            to_px(basket_x + sign * 5.8, rim_y + paint_half_width),
            color_paint,
            2,
        )

        # Free-throw circle.
        cv2.circle(
            image,
            to_px(basket_x + sign * 5.8, rim_y),
            int(3.6 * pixels_per_meter),
            color_court,
            2,
        )

        # 3-point line: corner straight segments + arc of 6.75m from rim.
        cv2.line(image, to_px(corner_x, corner_y_top), to_px(corner_x, corner_y_bot), color_court, 2)
        arc_center = (int(round(basket_x * pixels_per_meter)), int(round(rim_y * pixels_per_meter)))
        arc_radius = int(6.75 * pixels_per_meter)
        # Draw the arc on the side facing the basket. The arc goes from
        # (corner_x, corner_y_top) to (corner_x, corner_y_bot) sweeping away
        # from the baseline.
        if sign == 1:
            start_angle = 0.0
            end_angle = 180.0
        else:
            start_angle = 180.0
            end_angle = 360.0
        cv2.ellipse(
            image,
            arc_center,
            (arc_radius, arc_radius),
            0,
            start_angle,
            end_angle,
            color=color_court,
            thickness=2,
        )

    return image


def draw_projected_keypoints(court_image: np.ndarray, homography: np.ndarray, matched: list[dict[str, Any]], pixels_per_meter: int = 40) -> np.ndarray:
    annotated = court_image.copy()
    for entry in matched:
        x_m, y_m = entry["court_meters"]
        x_px, y_px = int(round(x_m * pixels_per_meter)), int(round(y_m * pixels_per_meter))
        if not (0 <= x_px < annotated.shape[1] and 0 <= y_px < annotated.shape[0]):
            continue
        is_inlier = bool(entry.get("inlier", False))
        color = (0, 180, 0) if is_inlier else (0, 0, 220)
        cv2.circle(annotated, (x_px, y_px), 8, color, 2)
        cv2.putText(
            annotated,
            str(entry["label"]),
            (x_px + 8, y_px - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def write_court_visualization(
    output_path: Path,
    homography: np.ndarray,
    matched: list[dict[str, Any]],
    *,
    pixels_per_meter: int = 40,
) -> None:
    court = draw_fiba_court(pixels_per_meter=pixels_per_meter)
    annotated = draw_projected_keypoints(court, homography, matched, pixels_per_meter=pixels_per_meter)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not write court visualization: {output_path}")


def homography_result_to_dict(result: HomographyResult, *, blob: dict[str, Any], min_confidence: float, reproj_threshold: float) -> dict[str, Any]:
    return {
        "model_id": blob.get("model_id"),
        "min_confidence": min_confidence,
        "reproj_threshold": reproj_threshold,
        "matched_points": result.matched,
        "homography_matrix": result.matrix.tolist(),
        "inliers_count": result.inliers_count,
        "reprojection_error_meters": {
            "mean": result.reprojection_mean_m,
            "median": result.reprojection_median_m,
        },
        "court_dimensions_m": {"length": 28.0, "width": 15.0},
        "court_point_map_used": {label: list(coords) for label, coords in COURT_POINT_MAP.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "Build FIBA court homography from keypoints.")
    parser.add_argument("keypoints_json", type=Path, nargs="?", help="Path to keypoints JSON from detect_court_keypoints.py")
    parser.add_argument("--output", type=Path, help="Path for the homography JSON (default outputs/<stem>_homography.json)")
    parser.add_argument(
        "--court-image",
        action="store_true",
        help="Also write a 2D FIBA court PNG with projected keypoints to outputs/<stem>_court_2d.png",
    )
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum keypoint confidence (default 0.7)")
    parser.add_argument("--min-points", type=int, default=4, help="Minimum matched points to attempt homography (default 4)")
    parser.add_argument("--reproj-threshold", type=float, default=8.0, help="RANSAC reprojection threshold in pixels (default 8.0)")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a synthetic 4-point sanity check before processing the input JSON, then exit.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.keypoints_json is None:
        raise SystemExit("keypoints_json is required (or pass --self-test)")

    if not args.keypoints_json.exists():
        raise SystemExit(f"Keypoints JSON not found: {args.keypoints_json}")

    blob = load_keypoints_json(args.keypoints_json)
    keypoints = extract_keypoints(blob)
    if not keypoints:
        raise SystemExit("No keypoints found in input JSON")

    result = compute_homography(
        keypoints,
        min_confidence=args.min_confidence,
        min_points=args.min_points,
        reproj_threshold=args.reproj_threshold,
    )

    output_path = args.output or Path("outputs") / f"{args.keypoints_json.stem}_homography.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = homography_result_to_dict(
        result,
        blob=blob,
        min_confidence=args.min_confidence,
        reproj_threshold=args.reproj_threshold,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if args.court_image:
        court_image_path = output_path.with_name(output_path.stem.replace("_homography", "") + "_court_2d.png")
        if court_image_path == output_path:
            court_image_path = output_path.with_name(output_path.stem + "_court_2d.png")
        write_court_visualization(court_image_path, result.matrix, result.matched)

    print(f"Matched points: {len(result.matched)}")
    print(f"Inliers: {result.inliers_count}/{len(result.matched)}")
    print(f"Reprojection error (m): mean={result.reprojection_mean_m:.3f} median={result.reprojection_median_m:.3f}")
    print(f"Homography JSON: {output_path}")
    if args.court_image:
        print(f"Court visualization: {court_image_path}")


def run_self_test() -> None:
    # Synthetic 4-point square: image pts form a perfect 1m x 1m square in
    # pixel space; FIBA points form a perfect 1m x 1m square. After fitting,
    # the homography should map every image point back to its source within
    # sub-pixel tolerance.
    image_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    court_pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    label_lookup = {label: court_pts[index] for index, label in enumerate(["A", "B", "C", "D"])}
    fake_keypoints = [
        {"label": "A", "x": image_pts[0][0], "y": image_pts[0][1], "confidence": 0.99},
        {"label": "B", "x": image_pts[1][0], "y": image_pts[1][1], "confidence": 0.99},
        {"label": "C", "x": image_pts[2][0], "y": image_pts[2][1], "confidence": 0.99},
        {"label": "D", "x": image_pts[3][0], "y": image_pts[3][1], "confidence": 0.99},
    ]
    try:
        result = compute_homography(
            fake_keypoints,
            min_confidence=0.0,
            min_points=4,
            reproj_threshold=8.0,
            point_map=label_lookup,
        )
    except Exception as error:  # pragma: no cover - failure path
        print(f"SELF-TEST FAIL: {error}")
        sys.exit(1)

    # Project the FIBA points back through the inverse homography; they should
    # land on the original image points within a pixel.
    inverse = np.linalg.inv(result.matrix)
    projected_image = cv2.perspectiveTransform(
        np.asarray(court_pts, dtype=np.float32).reshape(-1, 1, 2),
        inverse,
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected_image - np.asarray(image_pts, dtype=np.float32), axis=1)
    if float(errors.max()) > 1.0:
        print(f"SELF-TEST FAIL: round-trip error too high: {errors.tolist()}")
        sys.exit(1)
    print(f"SELF-TEST PASS: max round-trip pixel error = {float(errors.max()):.4f}, inliers = {result.inliers_count}")


if __name__ == "__main__":
    main()
