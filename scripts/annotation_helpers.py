"""Small shared helpers for annotation formatting and filtering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

DEFAULT_CLASS_NAME = "object"
DEFAULT_MIN_CONFIDENCE = 0.0
DEFAULT_ANNOTATION_COLOR = (0, 255, 255)


def format_label(name: object, confidence: float | None = None, prefix: str = "") -> str:
    label = f"{prefix}{name}"
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


def labels_for_detections(detections, prefixes: Sequence[str] | None = None) -> list[str]:
    confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
    labels = []
    for index, (class_name, confidence) in enumerate(zip(class_names_for(detections), confidences)):
        prefix = "" if prefixes is None else prefixes[index]
        labels.append(format_label(class_name, confidence, prefix=prefix))
    return labels


def annotate_detections(scene, detections, labels: Sequence[str] | None = None):
    import supervision as sv

    labels = labels if labels is not None else labels_for_detections(detections)
    annotated = sv.BoxAnnotator().annotate(scene=scene, detections=detections)
    return sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=labels)
