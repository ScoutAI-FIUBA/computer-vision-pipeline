#!/usr/bin/env python3
"""Build a local FIBA court homography from detected keypoints.

This script uses the shared labeled court template from
``scripts/court/court_template.py``. That template is the authoritative label ->
court-coordinate map for this repo.

The script:

- Reads parsed keypoints (output of ``detect_court_keypoints.py``).
- Filters them by ``--min-confidence`` and by canonical geometry allowed in the
  selected ``--geometry-mode``.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.court.court_template import (
    COURT_TEMPLATE_LOOKUP,
    GEOMETRY_MODE_STATUSES,
    canonical_point_map,
    geometry_statuses_for_mode,
    template_points,
)

# FIBA court dimensions: 28m long x 15m wide. Coordinate system: x_m along
# court length and y_m across court width, with (0, 0) at the top-left corner.
COURT_POINT_MAP: dict[str, tuple[float, float]] = canonical_point_map()


@dataclass
class HomographyResult:
    matrix: Any
    mask: Any
    inliers_count: int
    reprojection_mean_m: float
    reprojection_median_m: float
    geometry_mode: str
    geometry_statuses_allowed: tuple[str, ...]
    point_map_used: dict[str, tuple[float, float]]
    validation: dict[str, Any]
    matched: list[dict[str, Any]]
    skipped: list[dict[str, Any]]


@dataclass
class KeypointGeometryValidation:
    ok: bool
    reasons: list[str]
    stats: dict[str, Any]


class GeometryValidationError(ValueError):
    def __init__(self, validation: KeypointGeometryValidation):
        self.validation = validation
        reason_text = ", ".join(validation.reasons) if validation.reasons else "unknown"
        super().__init__(f"Keypoint geometry rejected before homography: {reason_text}")


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


def validate_keypoint_geometry(
    matched: list[dict[str, Any]],
    *,
    min_points: int,
    min_image_span_px: float = 40.0,
    min_hull_area_px2: float = 1200.0,
) -> KeypointGeometryValidation:
    import numpy as np

    image_points = np.asarray([entry["image_xy"] for entry in matched], dtype=np.float32) if matched else np.empty((0, 2), dtype=np.float32)
    distinct_labels = len({entry["label"] for entry in matched})
    x_span = float(np.ptp(image_points[:, 0])) if len(image_points) else 0.0
    y_span = float(np.ptp(image_points[:, 1])) if len(image_points) else 0.0

    hull_area = 0.0
    if len(image_points) >= 3:
        order = np.lexsort((image_points[:, 1], image_points[:, 0]))
        sorted_points = image_points[order]

        def cross(origin, a, b) -> float:
            return float((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))

        lower: list[Any] = []
        for point in sorted_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)

        upper: list[Any] = []
        for point in reversed(sorted_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)

        hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)
        if len(hull) >= 3:
            x = hull[:, 0]
            y = hull[:, 1]
            hull_area = float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)

    stats = {
        "matched_points": len(matched),
        "distinct_labels": distinct_labels,
        "image_x_span_px": x_span,
        "image_y_span_px": y_span,
        "image_hull_area_px2": hull_area,
    }
    reasons: list[str] = []
    if len(matched) < min_points:
        reasons.append("insufficient_points")
    if distinct_labels < min_points:
        reasons.append("insufficient_distinct_labels")
    if x_span < min_image_span_px:
        reasons.append("insufficient_x_spread")
    if y_span < min_image_span_px:
        reasons.append("insufficient_y_spread")
    if len(matched) >= 3 and hull_area < min_hull_area_px2:
        reasons.append("degenerate_area")
    return KeypointGeometryValidation(ok=not reasons, reasons=reasons, stats=stats)


def compute_homography(
    keypoints: list[dict[str, Any]],
    *,
    min_confidence: float,
    min_points: int,
    reproj_threshold: float,
    geometry_mode: str = "strict",
    point_map: dict[str, tuple[float, float]] | None = None,
) -> HomographyResult:
    import cv2
    import numpy as np

    mapping = point_map if point_map is not None else canonical_point_map(geometry_mode)
    allowed_statuses = geometry_statuses_for_mode(geometry_mode)

    matched: list[dict[str, Any]] = []
    src_pts: list[tuple[float, float]] = []
    dst_pts: list[tuple[float, float]] = []
    skipped: list[dict[str, Any]] = []
    for point in keypoints:
        label = str(point.get("label", ""))
        confidence = point.get("confidence")
        if label not in mapping:
            template_point = COURT_TEMPLATE_LOOKUP.get(label)
            if template_point is None:
                reason = "label_not_in_template"
                geometry_status = None
            elif template_point.court_xy is None:
                reason = "geometry_unknown"
                geometry_status = template_point.geometry_status
            else:
                reason = "geometry_status_excluded"
                geometry_status = template_point.geometry_status
            skipped.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "reason": reason,
                    "geometry_status": geometry_status,
                }
            )
            continue
        if confidence is not None and confidence < min_confidence:
            skipped.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "reason": "below_confidence",
                    "geometry_status": COURT_TEMPLATE_LOOKUP.get(label).geometry_status if label in COURT_TEMPLATE_LOOKUP else None,
                }
            )
            continue
        src_pts.append((float(point["x"]), float(point["y"])))
        dst_pts.append(mapping[label])
        matched.append(
            {
                "label": label,
                "image_xy": [float(point["x"]), float(point["y"])],
                "court_meters": list(mapping[label]),
                "confidence": confidence,
                "geometry_status": COURT_TEMPLATE_LOOKUP.get(label).geometry_status if label in COURT_TEMPLATE_LOOKUP else None,
            }
        )

    validation = validate_keypoint_geometry(matched, min_points=min_points)
    if not validation.ok:
        raise GeometryValidationError(validation)

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
        geometry_mode=geometry_mode,
        geometry_statuses_allowed=allowed_statuses,
        point_map_used=mapping,
        validation={"ok": validation.ok, "reasons": validation.reasons, "stats": validation.stats},
        matched=matched,
        skipped=skipped,
    )


def project_points(homography: Any, points: list[tuple[float, float]]) -> Any:
    import cv2
    import numpy as np

    arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, homography).reshape(-1, 2)


def draw_fiba_court(width_m: float = 28.0, height_m: float = 15.0, pixels_per_meter: int = 40) -> Any:
    """Render a top-down 2D FIBA court with the standard lines.

    This is a sanity-check visualization using the same court geometry as the
    canonical point map.
    """
    import math

    import cv2
    import numpy as np

    width_px = int(width_m * pixels_per_meter)
    height_px = int(height_m * pixels_per_meter)
    image = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    color_court = (40, 40, 40)
    color_paint = (60, 60, 60)
    line_thickness = 2

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        return int(round(x_m * pixels_per_meter)), int(round(y_m * pixels_per_meter))

    def pt(label: str) -> tuple[float, float]:
        return COURT_POINT_MAP[label]

    def radius_m(center_label: str, edge_label: str) -> float:
        center_x, center_y = pt(center_label)
        edge_x, edge_y = pt(edge_label)
        return math.hypot(edge_x - center_x, edge_y - center_y)

    def angle_deg(center_label: str, edge_label: str) -> float:
        center_x, center_y = pt(center_label)
        edge_x, edge_y = pt(edge_label)
        return math.degrees(math.atan2(edge_y - center_y, edge_x - center_x)) % 360

    def arc_angles(start_deg: float, end_deg: float, through_deg: float) -> tuple[int, int]:
        start_deg %= 360
        end_deg %= 360
        through_deg %= 360
        if (through_deg - start_deg) % 360 <= (end_deg - start_deg) % 360:
            if start_deg <= end_deg:
                return int(round(start_deg)), int(round(end_deg))
            return int(round(start_deg - 360)), int(round(end_deg))
        if end_deg <= start_deg:
            return int(round(end_deg)), int(round(start_deg))
        return int(round(end_deg - 360)), int(round(start_deg))

    # Outer boundary.
    cv2.rectangle(image, to_px(0, 0), to_px(width_m, height_m), color_court, line_thickness)

    # Half-court line.
    cv2.line(image, to_px(*pt("19")), to_px(*pt("23")), color_court, line_thickness)
    cv2.circle(
        image,
        to_px(*pt("21")),
        int(round(radius_m("21", "18") * pixels_per_meter)),
        color_court,
        line_thickness,
    )

    for hoop_label in ("09", "33"):
        cv2.circle(image, to_px(*pt(hoop_label)), int(round(0.225 * pixels_per_meter)), color_court, line_thickness)

    cv2.rectangle(image, to_px(*pt("04")), to_px(*pt("14")), color_paint, line_thickness)
    cv2.rectangle(image, to_px(*pt("28")), to_px(*pt("38")), color_paint, line_thickness)

    cv2.circle(
        image,
        to_px(*pt("13")),
        int(round(radius_m("13", "12") * pixels_per_meter)),
        color_court,
        line_thickness,
    )
    cv2.circle(
        image,
        to_px(*pt("29")),
        int(round(radius_m("29", "28") * pixels_per_meter)),
        color_court,
        line_thickness,
    )

    cv2.line(image, to_px(*pt("10")), to_px(*pt("11")), color_court, line_thickness)
    cv2.line(image, to_px(*pt("31")), to_px(*pt("32")), color_court, line_thickness)

    left_arc_start, left_arc_end = arc_angles(
        angle_deg("09", "10"),
        angle_deg("09", "11"),
        angle_deg("09", "16"),
    )
    right_arc_start, right_arc_end = arc_angles(
        angle_deg("33", "31"),
        angle_deg("33", "32"),
        angle_deg("33", "26"),
    )

    cv2.ellipse(
        image,
        to_px(*pt("09")),
        tuple([int(round(radius_m("09", "10") * pixels_per_meter))] * 2),
        0,
        left_arc_start,
        left_arc_end,
        color_court,
        line_thickness,
    )
    cv2.ellipse(
        image,
        to_px(*pt("33")),
        tuple([int(round(radius_m("33", "31") * pixels_per_meter))] * 2),
        0,
        right_arc_start,
        right_arc_end,
        color_court,
        line_thickness,
    )

    return image


def draw_projected_keypoints(court_image: Any, homography: Any, matched: list[dict[str, Any]], pixels_per_meter: int = 40) -> Any:
    import cv2

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
    homography: Any,
    matched: list[dict[str, Any]],
    *,
    pixels_per_meter: int = 40,
) -> None:
    import cv2

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
        "geometry_mode": result.geometry_mode,
        "geometry_statuses_allowed": list(result.geometry_statuses_allowed),
        "geometry_validation": result.validation,
        "matched_points": result.matched,
        "skipped_points": result.skipped,
        "homography_matrix": result.matrix.tolist(),
        "inliers_count": result.inliers_count,
        "reprojection_error_meters": {
            "mean": result.reprojection_mean_m,
            "median": result.reprojection_median_m,
        },
        "court_dimensions_m": {"length": 28.0, "width": 15.0},
        "court_template": template_points(),
        "court_point_map_used": {label: list(coords) for label, coords in result.point_map_used.items()},
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
        "--geometry-mode",
        choices=sorted(GEOMETRY_MODE_STATUSES),
        default="strict",
        help="Canonical geometry filter mode. Both modes currently use the same authoritative mapped labels.",
    )
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
        geometry_mode=args.geometry_mode,
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
    print(f"Geometry mode: {result.geometry_mode} ({', '.join(result.geometry_statuses_allowed)})")
    print(f"Inliers: {result.inliers_count}/{len(result.matched)}")
    print(f"Reprojection error (m): mean={result.reprojection_mean_m:.3f} median={result.reprojection_median_m:.3f}")
    print(f"Used labels: {[entry['label'] for entry in result.matched]}")
    print(
        "Skipped labels: "
        f"{[{ 'label': entry['label'], 'reason': entry['reason'], 'geometry_status': entry['geometry_status'] } for entry in result.skipped]}"
    )
    print(f"Homography JSON: {output_path}")
    if args.court_image:
        print(f"Court visualization: {court_image_path}")


def run_self_test() -> None:
    import cv2
    import numpy as np

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
