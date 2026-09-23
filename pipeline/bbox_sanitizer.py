"""Conservative offline temporal sanitization for SAM3.1 V2 boxes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable, Mapping


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


def sanitize_temporal_bboxes(
    observations: Iterable[Mapping[str, Any]],
    config: BBoxSanitizerConfig | Any,
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
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(int(row["raw_track_id"]), []).append(index)
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

    corrected = 0
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
            if previous_box and next_box and same_space and gap <= cfg.max_interpolation_gap and frames - 1 <= cfg.max_interpolation_gap:
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
                row["bbox_outlier_diagnostics"] = {"signals": scores[indices[offset]][1], "reason": "ambiguous_or_long_gap"}
                unresolved += 1
            position += 1

    for row in rows:
        if "bbox_corrected" not in row:
            _preserve_raw_provenance(row)
            row.update({"bbox_corrected": False, "bbox_correction_method": "none", "bbox_outlier": False, "bbox_unresolved": False, "bbox_outlier_score": 0.0})
    return TemporalSanitizerResult(rows, corrected, unresolved, {"corrected_count": corrected, "unresolved_count": unresolved})
