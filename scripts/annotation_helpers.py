"""Small shared helpers for annotation formatting and filtering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

DEFAULT_CLASS_NAME = "object"
DEFAULT_MIN_CONFIDENCE = 0.0
DEFAULT_ANNOTATION_COLOR = (0, 255, 255)
ANNOTATION_COLORS = {
    "default": DEFAULT_ANNOTATION_COLOR,
    "accent": (255, 255, 0),
    "muted": (160, 160, 160),
}


def annotation_color(color: str | Sequence[int] | None = None) -> tuple[int, int, int]:
    if color is None:
        return DEFAULT_ANNOTATION_COLOR
    if isinstance(color, str):
        return ANNOTATION_COLORS.get(color, DEFAULT_ANNOTATION_COLOR)

    values = tuple(int(value) for value in color)
    if len(values) != 3:
        raise ValueError("Annotation color must contain exactly three BGR values")
    return values


def label_prefix(value: object | None, marker: str = "#") -> str:
    return "" if value is None else f"{marker}{value} "


def format_label(
    name: object,
    confidence: float | None = None,
    prefix: str = "",
    metadata: Iterable[object] | None = None,
) -> str:
    parts = [f"{prefix}{name}"]
    if metadata is not None:
        parts.extend(str(value) for value in metadata if value is not None and str(value) != "")
    label = " ".join(part for part in parts if part)
    return f"{label} {confidence:.2f}" if confidence is not None else label


def class_names_for(detections, default: str = DEFAULT_CLASS_NAME) -> list[str]:
    raw_names = detections.data.get("class_name", [default] * len(detections))
    return [str(name) for name in raw_names]


def filter_detections(detections, min_confidence: float = DEFAULT_MIN_CONFIDENCE, allowed_classes: Iterable[str] | None = None):
    if detections.confidence is not None and min_confidence > DEFAULT_MIN_CONFIDENCE:
        detections = detections[detections.confidence >= min_confidence]

    if allowed_classes is None:
        return detections

    allowed = {str(name) for name in allowed_classes}
    if not allowed:
        return detections

    class_names = class_names_for(detections)
    keep = [class_name in allowed for class_name in class_names]
    return detections[keep]


def detections_from_inference(result: dict, min_confidence: float = DEFAULT_MIN_CONFIDENCE, allowed_classes: Iterable[str] | None = None):
    import supervision as sv

    detections = sv.Detections.from_inference(result)
    return filter_detections(detections, min_confidence=min_confidence, allowed_classes=allowed_classes)


def labels_for_detections(
    detections,
    prefixes: Sequence[str] | None = None,
    metadata_items: Sequence[Iterable[object] | None] | None = None,
) -> list[str]:
    confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
    labels = []
    for index, (class_name, confidence) in enumerate(zip(class_names_for(detections), confidences)):
        prefix = "" if prefixes is None else prefixes[index]
        metadata = None if metadata_items is None else metadata_items[index]
        labels.append(format_label(class_name, confidence, prefix=prefix, metadata=metadata))
    return labels


def annotate_detections(scene, detections, labels: Sequence[str] | None = None):
    import supervision as sv

    color_lookup = sv.ColorLookup.CLASS
    tracker_ids = getattr(detections, "tracker_id", None)
    if tracker_ids is None:
        tracker_ids = detections.data.get("tracker_id")
    if tracker_ids is not None:
        color_lookup = sv.ColorLookup.TRACK
    elif detections.class_id is None:
        color_lookup = sv.ColorLookup.INDEX

    labels = labels if labels is not None else labels_for_detections(detections)
    annotated = sv.BoxAnnotator(color_lookup=color_lookup).annotate(scene=scene, detections=detections)
    return sv.LabelAnnotator(color_lookup=color_lookup).annotate(scene=annotated, detections=detections, labels=labels)


def labels_for_points(points: Sequence[dict], name_key: str = "label") -> list[str]:
    return [format_label(point.get(name_key, DEFAULT_CLASS_NAME), point.get("confidence")) for point in points]


def annotate_points(
    scene,
    points: Sequence[dict],
    labels: Sequence[str] | None = None,
    color: str | Sequence[int] | None = None,
    empty_label: str | None = None,
    draw_markers: bool = True,
    radius: int = 5,
):
    import cv2

    draw_color = annotation_color(color)
    labels = labels if labels is not None else labels_for_points(points)

    for point, label in zip(points, labels):
        x = int(round(float(point["x"])))
        y = int(round(float(point["y"])))
        if draw_markers:
            cv2.circle(scene, (x, y), radius, draw_color, -1)
        cv2.putText(scene, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

    if empty_label and not points:
        cv2.putText(scene, empty_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, draw_color, 2)

    return scene
