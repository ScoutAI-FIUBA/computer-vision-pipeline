"""Conservative offline temporal sanitization for SAM3.1 V2 boxes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class BBoxSanitizerConfig:
    enabled: bool = False
    history_size: int = 15
    max_interpolation_gap: int = 2
    signal_threshold: float = 3.0
    minimum_signals: int = 2
    neighbor_displacement_factor: float = 4.0

    def __post_init__(self) -> None:
        if self.history_size < 2:
            raise ValueError("history_size must be at least 2")
        if self.max_interpolation_gap < 1:
            raise ValueError("max_interpolation_gap must be at least 1")
        if self.signal_threshold <= 0:
            raise ValueError("signal_threshold must be positive")
        if self.minimum_signals < 2 or self.minimum_signals > 5:
            raise ValueError("minimum_signals must be between 2 and 5")
        if self.neighbor_displacement_factor <= 0:
            raise ValueError("neighbor_displacement_factor must be positive")


@dataclass(frozen=True)
class TemporalSanitizerResult:
    rows: list[dict[str, Any]]
    corrected_count: int = 0
    unresolved_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    cleaned_masks_by_observation: dict[str, Any] = field(default_factory=dict)


def _box(value: Any, coordinate_space: str | None) -> tuple[float, float, float, float] | None:
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        return None
    if coordinate_space not in (None, "pixel", "normalized"):
        return None
    if coordinate_space == "normalized" and not all(0 <= item <= 1 for item in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return values


def _raw_value(row: Mapping[str, Any]) -> Any:
    value = row.get("bbox_xyxy")
    return row.get("bbox") if value is None else value


def _preserve_raw_provenance(row: dict[str, Any]) -> None:
    if "raw_bbox" not in row:
        raw = _raw_value(row)
        row["raw_bbox"] = list(raw) if isinstance(raw, (list, tuple)) else raw
    if "raw_bbox_coordinate_space" not in row and row.get("bbox_coordinate_space") is not None:
        row["raw_bbox_coordinate_space"] = row["bbox_coordinate_space"]


def _features(box: tuple[float, float, float, float], previous: tuple[float, float, float, float] | None) -> tuple[float, ...]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    displacement = 0.0
    if previous is not None:
        px1, py1, px2, py2 = previous
        displacement = math.hypot(center_x - (px1 + px2) / 2, center_y - (py1 + py2) / 2)
    return (width * height, width, height, width / height, displacement)


def _robust_score(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    baseline = median(values)
    if baseline <= 0:
        return 0.0
    scale = median([abs(item - baseline) for item in values])
    # Relative floors stop a perfectly stable track from making harmless integer
    # movement look anomalous while retaining sensitivity to isolated spikes.
    floor = max(abs(baseline) * 0.05, 1e-6)
    return abs(value - baseline) / max(1.4826 * scale, floor)


def _interpolate(first: tuple[float, ...], second: tuple[float, ...], fraction: float) -> list[float]:
    return [round(left + (right - left) * fraction, 6) for left, right in zip(first, second)]


def _mask_bbox(component: tuple[int, int, int, int], shape: tuple[int, int], coordinate_space: str | None) -> list[float]:
    x, y, width, height = component
    x2, y2 = x + width, y + height
    if coordinate_space == "normalized":
        frame_height, frame_width = shape
        return [round(x / frame_width, 6), round(y / frame_height, 6), round(x2 / frame_width, 6), round(y2 / frame_height, 6)]
    return [float(x), float(y), float(x2), float(y2)]


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _mask_cleanup(
    rows: list[dict[str, Any]],
    masks_by_observation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Keep a component only when geometry gives a conservative winner."""
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(int(row.get("canonical_raw_track_id", row["raw_track_id"])), []).append(index)
    cleaned: dict[str, Any] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    for indices in groups.values():
        indices.sort(key=lambda index: int(rows[index]["frame_index"]))
        for position, index in enumerate(indices):
            row = rows[index]
            key = str(row.get("observation_key", ""))
            if key not in masks_by_observation:
                continue
            try:
                raw_mask = np.asarray(masks_by_observation[key])
                if raw_mask.ndim != 2 or not np.isfinite(raw_mask).all():
                    raise ValueError
                binary = raw_mask.astype(bool)
            except (TypeError, ValueError, OverflowError):
                diagnostics[index] = {"reason": "invalid_mask"}
                continue
            if not binary.any():
                diagnostics[index] = {"reason": "empty_mask"}
                continue
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
            components = [
                (int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP]), int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT]), int(stats[label, cv2.CC_STAT_AREA]), label)
                for label in range(1, count)
            ]
            if not components:
                diagnostics[index] = {"reason": "empty_mask"}
                continue
            coordinate_space = row.get("bbox_coordinate_space")
            previous = _box(_raw_value(rows[indices[position - 1]]), coordinate_space) if position else None
            following = _box(_raw_value(rows[indices[position + 1]]), coordinate_space) if position + 1 < len(indices) else None
            neighbor_boxes = [box for box in (previous, following) if box is not None]
            neighbor_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in neighbor_boxes]
            if coordinate_space == "normalized":
                neighbor_areas = [area * binary.shape[0] * binary.shape[1] for area in neighbor_areas]
            total_area = float(binary.sum())
            scored: list[tuple[float, tuple[int, int, int, int, int, int], dict[str, float]]] = []
            for component in components:
                component_box = tuple(_mask_bbox(component[:4], binary.shape, coordinate_space))
                area = float(component[4])
                overlap = sum(_iou(component_box, box) for box in neighbor_boxes) / len(neighbor_boxes) if neighbor_boxes else 0.0
                proximity = 0.0
                if neighbor_boxes:
                    cx = (component_box[0] + component_box[2]) / 2
                    cy = (component_box[1] + component_box[3]) / 2
                    distances = []
                    for box in neighbor_boxes:
                        diagonal = max(math.hypot(box[2] - box[0], box[3] - box[1]), 1e-6)
                        distances.append(math.hypot(cx - (box[0] + box[2]) / 2, cy - (box[1] + box[3]) / 2) / diagonal)
                    proximity = 1.0 / (1.0 + sum(distances) / len(distances))
                size_consistency = 0.0
                if neighbor_areas:
                    size_consistency = 1.0 / (1.0 + abs(math.log(max(area, 1.0) / max(median(neighbor_areas), 1.0))))
                relative_area = area / total_area
                score_parts = {"overlap": overlap, "proximity": proximity, "size_consistency": size_consistency, "relative_area": relative_area}
                score = 3.0 * overlap + 2.0 * proximity + size_consistency + relative_area
                scored.append((score, component, score_parts))
            scored.sort(key=lambda item: (-item[0], -item[1][4], item[1][0], item[1][1]))
            best_score, best, best_parts = scored[0]
            raw_box = _box(_raw_value(row), coordinate_space)
            reference_area = median(neighbor_areas) if neighbor_areas else None
            raw_area = None if raw_box is None else (raw_box[2] - raw_box[0]) * (raw_box[3] - raw_box[1])
            if coordinate_space == "normalized" and raw_area is not None:
                raw_area *= binary.shape[0] * binary.shape[1]
            mask_is_implausibly_broad = (
                raw_area is not None
                and (best[4] > raw_area * 2.0
                     or reference_area is not None and best[4] > reference_area * 2.0)
            )
            both_temporally_supported = len(scored) > 1 and best_parts["overlap"] > 0 and scored[1][2]["overlap"] > 0
            ambiguous = (
                len(scored) > 1
                and scored[1][1][4] >= best[4] * 0.8
                and best_score - scored[1][0] < 3.0
                and (both_temporally_supported or best_score - scored[1][0] < 0.45)
            )
            if not neighbor_boxes and len(scored) > 1:
                raw_iou = 0.0 if raw_box is None else _iou(tuple(_mask_bbox(best[:4], binary.shape, coordinate_space)), raw_box)
                dominant = best[4] >= scored[1][1][4] * 4.0 and best_parts["relative_area"] >= 0.7 and raw_iou >= 0.5
                ambiguous = not dominant
            if ambiguous:
                reason = "ambiguous_components_no_temporal_evidence" if not neighbor_boxes else "ambiguous_components"
                diagnostics[index] = {
                    "reason": reason,
                    "component_count": len(scored),
                    "scores": [round(item[0], 6) for item in scored],
                }
                continue
            if mask_is_implausibly_broad:
                diagnostics[index] = {"reason": "implausibly_broad_component"}
                continue
            clean = np.zeros_like(binary, dtype=bool)
            clean[labels == best[5]] = True
            cleaned[key] = clean
            new_bbox = _mask_bbox(best[:4], binary.shape, coordinate_space)
            mask_changed = not np.array_equal(clean, binary)
            if mask_changed:
                row["mask_cleanup_corrected"] = True
            if _box(_raw_value(row), coordinate_space) != tuple(new_bbox):
                _preserve_raw_provenance(row)
                if "mask_ref" in row:
                    row.setdefault("raw_mask_ref", row["mask_ref"])
                row["bbox_xyxy"] = new_bbox
                row.update({"bbox_corrected": True, "bbox_correction_method": "mask_cleanup", "bbox_unresolved": False})
                row["bbox_outlier_score"] = round(best_score, 6)
                row["bbox_outlier_diagnostics"] = {"mask_cleanup": {**best_parts, "component_count": len(scored)}}
            else:
                # Keep the cleaned component even when its bbox is unchanged;
                # ReID and debug rendering must not silently fall back to blobs.
                pass
    return cleaned, diagnostics


def sanitize_temporal_bboxes(
    observations: Iterable[Mapping[str, Any]],
    config: BBoxSanitizerConfig | Any,
    masks_by_observation: Mapping[str, Any] | None = None,
) -> TemporalSanitizerResult:
    """Detect robust multi-signal outliers and interpolate only bounded gaps.

    All measurements stay in each row's declared coordinate space. No identity or
    team field is read or written by this module.
    """
    enabled = bool(getattr(config, "enabled", getattr(config, "bbox_sanitizer_enabled", False)))
    if not enabled:
        return TemporalSanitizerResult([dict(row) for row in observations])

    cfg = config if isinstance(config, BBoxSanitizerConfig) else BBoxSanitizerConfig(
        enabled=True,
        history_size=int(getattr(config, "bbox_history_size", 15)),
        max_interpolation_gap=int(getattr(config, "max_interpolation_gap", getattr(config, "bbox_max_interpolation_gap", 2))),
        signal_threshold=float(getattr(config, "bbox_signal_threshold", 3.0)),
        minimum_signals=int(getattr(config, "bbox_minimum_signals", 2)),
        neighbor_displacement_factor=float(getattr(config, "bbox_neighbor_displacement_factor", 4.0)),
    )
    rows = [dict(row) for row in observations]
    for row in rows:
        if "bbox_xyxy" not in row and "bbox" in row:
            value = row["bbox"]
            row["bbox_xyxy"] = list(value) if isinstance(value, (list, tuple)) else value
    cleaned_masks: dict[str, Any] = {}
    mask_diagnostics: dict[int, dict[str, Any]] = {}
    if masks_by_observation is not None:
        cleaned_masks, mask_diagnostics = _mask_cleanup(rows, masks_by_observation)
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(int(row.get("canonical_raw_track_id", row["raw_track_id"])), []).append(index)
    candidates: set[int] = set()
    scores: dict[int, tuple[float, dict[str, float]]] = {}
    for indices in groups.values():
        indices.sort(key=lambda index: int(rows[index]["frame_index"]))
        valid: dict[int, tuple[float, float, float, float]] = {}
        for index in indices:
            box = _box(_raw_value(rows[index]), rows[index].get("bbox_coordinate_space"))
            if box is not None:
                valid[index] = box
        feature_by_index = {
            index: _features(box, valid.get(indices[position - 1]))
            for position, (index, box) in enumerate((item, valid[item]) for item in indices if item in valid)
        }
        first_valid = next(iter(valid), None)
        last_valid = next(reversed(valid), None) if valid else None
        for position, index in enumerate(indices):
            row = rows[index]
            current = valid.get(index)
            if current is None:
                continue
            nearby = sorted(
                (item for item in valid if item != index),
                key=lambda item: abs(int(rows[item]["frame_index"]) - int(row["frame_index"])),
            )[: cfg.history_size]
            features = feature_by_index[index]
            signal_names = ("area", "width", "height", "aspect_ratio", "center_displacement")
            signal_scores = {
                name: _robust_score(features[offset], [feature_by_index[item][offset] for item in nearby])
                for offset, (name, value) in enumerate(zip(signal_names, features))
            }
            score = sum(signal_scores.values()) / len(signal_scores)
            # Endpoints cannot be corrected and are never candidates. Comparing
            # geometry with both offline endpoints catches a short run of bad
            # boxes even when the run itself pollutes a local median baseline.
            endpoint_shape_signals = 0
            if position > 0 and position < len(indices) - 1 and first_valid is not None and last_valid is not None:
                endpoint_shape_signals = sum(
                    abs(features[offset] - feature_by_index[endpoint][offset])
                    > max(abs(feature_by_index[endpoint][offset]) * 0.45, 1e-6)
                    for offset in range(4)
                    for endpoint in (first_valid, last_valid)
                ) // 2
            robust_signals = sum(value >= cfg.signal_threshold for value in signal_scores.values())
            if position > 0 and position < len(indices) - 1 and (
                robust_signals >= cfg.minimum_signals or endpoint_shape_signals >= cfg.minimum_signals
            ):
                candidates.add(index)
                scores[index] = (score, signal_scores)

    mask_cleanup_count = sum(
        row.get("bbox_correction_method") == "mask_cleanup"
        or row.get("mask_cleanup_corrected", False)
        for row in rows
    )
    corrected = mask_cleanup_count
    unresolved = 0
    for indices in groups.values():
        indices.sort(key=lambda index: int(rows[index]["frame_index"]))
        position = 0
        while position < len(indices):
            if indices[position] not in candidates:
                position += 1
                continue
            start = position
            while position + 1 < len(indices) and indices[position + 1] in candidates:
                position += 1
            end = position
            previous_index = start - 1
            next_index = end + 1
            gap = end - start + 1
            previous_box = None if previous_index < 0 else _box(_raw_value(rows[indices[previous_index]]), rows[indices[previous_index]].get("bbox_coordinate_space"))
            next_box = None if next_index >= len(indices) else _box(_raw_value(rows[indices[next_index]]), rows[indices[next_index]].get("bbox_coordinate_space"))
            frames = int(rows[indices[next_index]]["frame_index"]) - int(rows[indices[previous_index]]["frame_index"]) if previous_box and next_box else 0
            space = rows[indices[start]].get("bbox_coordinate_space")
            same_space = previous_index >= 0 and next_index < len(indices) and rows[indices[previous_index]].get("bbox_coordinate_space") == space == rows[indices[next_index]].get("bbox_coordinate_space")
            mask_blocks_interpolation = any(
                mask_diagnostics.get(indices[offset], {}).get("reason") == "ambiguous_components"
                for offset in range(start, end + 1)
            )
            if previous_box and next_box and same_space and not mask_blocks_interpolation and gap <= cfg.max_interpolation_gap and frames - 1 <= cfg.max_interpolation_gap:
                endpoint_distance = _features(next_box, previous_box)[4]
                baseline_distances = [
                    _features(valid[item], valid[indices[position - 1]])[4]
                    for position, item in enumerate(indices)
                    if position > 0 and item in valid and indices[position - 1] in valid
                    and item not in candidates and indices[position - 1] not in candidates
                ]
                if baseline_distances:
                    typical_distance = median(baseline_distances)
                else:
                    typical_distance = max(
                        previous_box[2] - previous_box[0],
                        previous_box[3] - previous_box[1],
                        next_box[2] - next_box[0],
                        next_box[3] - next_box[1],
                    )
                distance_limit = max(typical_distance * cfg.neighbor_displacement_factor, 1e-6)
                if endpoint_distance <= distance_limit:
                    first_frame = int(rows[indices[previous_index]]["frame_index"])
                    last_frame = int(rows[indices[next_index]]["frame_index"])
                    for offset in range(gap):
                        index = indices[start + offset]
                        fraction = (int(rows[index]["frame_index"]) - first_frame) / (last_frame - first_frame)
                        row = rows[index]
                        _preserve_raw_provenance(row)
                        row["bbox_xyxy"] = _interpolate(previous_box, next_box, fraction)
                        row.update({"bbox_corrected": True, "bbox_correction_method": "linear_interpolation", "bbox_outlier": True, "bbox_unresolved": False})
                        row["bbox_outlier_score"] = round(scores[index][0], 6)
                        row["bbox_outlier_diagnostics"] = {"signals": scores[index][1], "reason": "short_temporal_glitch"}
                        corrected += 1
                    position += 1
                    continue
            for offset in range(start, end + 1):
                row = rows[indices[offset]]
                _preserve_raw_provenance(row)
                row.update({"bbox_corrected": False, "bbox_correction_method": "unresolved_outlier", "bbox_outlier": True, "bbox_unresolved": True})
                row["bbox_outlier_score"] = round(scores[indices[offset]][0], 6)
                diagnostics = {"signals": scores[indices[offset]][1], "reason": "ambiguous_or_long_gap"}
                if indices[offset] in mask_diagnostics:
                    diagnostics["mask_cleanup"] = mask_diagnostics[indices[offset]]
                row["bbox_outlier_diagnostics"] = diagnostics
                unresolved += 1
            position += 1

    for index, mask_diagnostic in mask_diagnostics.items():
        if not mask_diagnostic.get("reason", "").startswith("ambiguous_components"):
            continue
        row = rows[index]
        if row.get("bbox_unresolved"):
            continue
        _preserve_raw_provenance(row)
        row.update({"bbox_corrected": False, "bbox_correction_method": "unresolved_mask", "bbox_outlier": True, "bbox_unresolved": True})
        row["bbox_outlier_score"] = 0.0
        row["bbox_outlier_diagnostics"] = {"mask_cleanup": mask_diagnostic}
        unresolved += 1

    for row in rows:
        if "bbox_corrected" not in row:
            _preserve_raw_provenance(row)
            row.update({"bbox_corrected": False, "bbox_correction_method": "none", "bbox_outlier": False, "bbox_unresolved": False, "bbox_outlier_score": 0.0})
    diagnostics = {"corrected_count": corrected, "unresolved_count": unresolved}
    if masks_by_observation is not None:
        diagnostics["mask_cleanup_count"] = mask_cleanup_count
    return TemporalSanitizerResult(
        rows,
        corrected,
        unresolved,
        diagnostics,
        cleaned_masks,
    )
