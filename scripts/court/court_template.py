#!/usr/bin/env python3
"""Shared labeled basketball court keypoint template.

This module is the single source of truth for the numbered court-label layout
used by the detection and homography scripts.

Coordinate convention:
- (0, 0) is the top-left court corner
- x runs along court length
- y runs along court width
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CourtTemplatePoint:
    label: str
    name: str
    court_xy: tuple[float, float] | None = None
    geometry_status: str = "authoritative"
    note: str = ""


COURT_LENGTH_M = 28.0
COURT_WIDTH_M = 15.0


COURT_TEMPLATE: tuple[CourtTemplatePoint, ...] = (
    CourtTemplatePoint("01", "top_left_corner", (0.0, 0.0)),
    CourtTemplatePoint("02", "left_3pt_corner_top", (0.0, 0.9)),
    CourtTemplatePoint("03", "template_label_03", None, "unknown", "Present in the model/template, but not mapped by the provided source of truth."),
    CourtTemplatePoint("04", "left_paint_baseline_top", (0.0, 5.05)),
    CourtTemplatePoint("05", "left_paint_baseline_bottom", (0.0, 9.95)),
    CourtTemplatePoint("06", "template_label_06", None, "unknown", "Present in the model/template, but not mapped by the provided source of truth."),
    CourtTemplatePoint("07", "left_3pt_corner_bottom", (0.0, 14.1)),
    CourtTemplatePoint("08", "bottom_left_corner", (0.0, 15.0)),
    CourtTemplatePoint("09", "left_hoop_center", (1.575, 7.5)),
    CourtTemplatePoint("10", "left_3pt_line_top_start", (2.99, 0.9)),
    CourtTemplatePoint("11", "left_3pt_line_bottom_start", (2.99, 14.1)),
    CourtTemplatePoint("12", "left_free_throw_line_top", (5.8, 5.05)),
    CourtTemplatePoint("13", "left_free_throw_center", (5.8, 7.5)),
    CourtTemplatePoint("14", "left_free_throw_line_bottom", (5.8, 9.95)),
    CourtTemplatePoint("15", "left_3pt_apex_projected_top", (8.325, 0.0)),
    CourtTemplatePoint("16", "left_3pt_arc_apex", (8.325, 7.5)),
    CourtTemplatePoint("17", "left_3pt_apex_projected_bottom", (8.325, 15.0)),
    CourtTemplatePoint("18", "center_circle_top", (14.0, 5.7)),
    CourtTemplatePoint("19", "center_line_top", (14.0, 0.0)),
    CourtTemplatePoint("20", "center_court", (14.0, 7.5)),
    CourtTemplatePoint("21", "center_circle_center", (14.0, 7.5)),
    CourtTemplatePoint("22", "center_circle_bottom", (14.0, 9.3)),
    CourtTemplatePoint("23", "center_line_bottom", (14.0, 15.0)),
    CourtTemplatePoint("24", "template_label_24", None, "unknown", "Present in the model/template, but not mapped by the provided source of truth."),
    CourtTemplatePoint("25", "right_3pt_apex_projected_top", (19.675, 0.0)),
    CourtTemplatePoint("26", "right_3pt_arc_apex", (19.675, 7.5)),
    CourtTemplatePoint("27", "right_3pt_apex_projected_bottom", (19.675, 15.0)),
    CourtTemplatePoint("28", "right_free_throw_line_top", (22.2, 5.05)),
    CourtTemplatePoint("29", "right_free_throw_center", (22.2, 7.5)),
    CourtTemplatePoint("30", "right_free_throw_line_bottom", (22.2, 9.95)),
    CourtTemplatePoint("31", "right_3pt_line_top_start", (25.01, 0.9)),
    CourtTemplatePoint("32", "right_3pt_line_bottom_start", (25.01, 14.1)),
    CourtTemplatePoint("33", "right_hoop_center", (26.425, 7.5)),
    CourtTemplatePoint("34", "top_right_corner", (28.0, 0.0)),
    CourtTemplatePoint("35", "right_3pt_corner_top", (28.0, 0.9)),
    CourtTemplatePoint("36", "template_label_36", None, "unknown", "Present in the model/template, but not mapped by the provided source of truth."),
    CourtTemplatePoint("37", "right_paint_baseline_top", (28.0, 5.05)),
    CourtTemplatePoint("38", "right_paint_baseline_bottom", (28.0, 9.95)),
    CourtTemplatePoint("39", "template_label_39", None, "unknown", "Present in the model/template, but not mapped by the provided source of truth."),
    CourtTemplatePoint("40", "right_3pt_corner_bottom", (28.0, 14.1)),
    CourtTemplatePoint("41", "bottom_right_corner", (28.0, 15.0)),
)


# Visualization skeleton follows the labeled template the user provided.
# Keep this focused on straight segments that should be visible in the video
# overlay; do not fake circles/arcs by connecting unrelated anchors.
COURT_TEMPLATE_POLYLINES: tuple[tuple[str, ...], ...] = (
    ("01", "15", "19", "25", "34"),
    ("01", "02", "04", "05", "07", "08"),
    ("08", "17", "23", "27", "41"),
    ("34", "35", "37", "38", "40", "41"),
    ("04", "12", "13", "14", "05"),
    ("37", "28", "29", "30", "38"),
    ("10", "16", "11"),
    ("31", "26", "32"),
    ("19", "21", "23"),
)


COURT_TEMPLATE_INDEX = {point.label: index for index, point in enumerate(COURT_TEMPLATE)}
COURT_TEMPLATE_LOOKUP = {point.label: point for point in COURT_TEMPLATE}
GEOMETRY_MODE_STATUSES: dict[str, tuple[str, ...]] = {
    "strict": ("authoritative",),
    "debug": ("authoritative",),
}


def geometry_statuses_for_mode(geometry_mode: str = "strict") -> tuple[str, ...]:
    try:
        return GEOMETRY_MODE_STATUSES[geometry_mode]
    except KeyError as error:
        valid = ", ".join(sorted(GEOMETRY_MODE_STATUSES))
        raise ValueError(f"Unknown geometry mode {geometry_mode!r}. Expected one of: {valid}") from error


def canonical_point_map(geometry_mode: str = "strict") -> dict[str, tuple[float, float]]:
    allowed_statuses = set(geometry_statuses_for_mode(geometry_mode))
    return {
        point.label: point.court_xy
        for point in COURT_TEMPLATE
        if point.court_xy is not None and point.geometry_status in allowed_statuses
    }


def template_points() -> list[dict[str, object]]:
    return [
        {
            "label": point.label,
            "name": point.name,
            "court_xy": list(point.court_xy) if point.court_xy is not None else None,
            "geometry_status": point.geometry_status,
            "note": point.note or None,
        }
        for point in COURT_TEMPLATE
    ]


def sort_keypoints_by_template(keypoints: list[dict]) -> list[dict]:
    fallback_index = len(COURT_TEMPLATE_INDEX)
    return sorted(
        keypoints,
        key=lambda point: (
            COURT_TEMPLATE_INDEX.get(str(point.get("label", "")), fallback_index),
            str(point.get("label", "")),
        ),
    )


def label_class_mapping(labels: list[str]) -> dict[str, int]:
    fallback_index = len(COURT_TEMPLATE_INDEX)
    ordered_labels = sorted(
        {str(label) for label in labels},
        key=lambda label: (COURT_TEMPLATE_INDEX.get(label, fallback_index), label),
    )
    return {label: index for index, label in enumerate(ordered_labels)}


def edges_for_labels(labels: list[str]) -> list[tuple[int, int]]:
    index_by_label = {str(label): index for index, label in enumerate(labels)}
    edges: list[tuple[int, int]] = []
    for polyline in COURT_TEMPLATE_POLYLINES:
        for start_label, end_label in zip(polyline, polyline[1:]):
            start_index = index_by_label.get(start_label)
            end_index = index_by_label.get(end_label)
            if start_index is None or end_index is None:
                continue
            edges.append((start_index, end_index))
    return edges


def labeled_edges_for_labels(labels: list[str]) -> list[tuple[str, str, int, int]]:
    index_by_label = {str(label): index for index, label in enumerate(labels)}
    edges: list[tuple[str, str, int, int]] = []
    for polyline in COURT_TEMPLATE_POLYLINES:
        for start_label, end_label in zip(polyline, polyline[1:]):
            start_index = index_by_label.get(start_label)
            end_index = index_by_label.get(end_label)
            if start_index is None or end_index is None:
                continue
            edges.append((start_label, end_label, start_index, end_index))
    return edges
