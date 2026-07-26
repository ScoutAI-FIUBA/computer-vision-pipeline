"""Optional debug annotation.

Only used when `PipelineConfig.write_annotated_video` is on. Kept separate so the
main path never imports it, and so `pipeline/` has no dependency on the older
`scripts/` helpers.
"""

from __future__ import annotations


def rows_to_detections(rows: list[dict]):
    """Convert pipeline row dicts into a supervision Detections for drawing."""

    import numpy as np
    import supervision as sv

    if not rows:
        return sv.Detections.empty()

    confidences = [0.0 if row["confidence"] is None else float(row["confidence"]) for row in rows]
    track_ids = [int(row["track_id"]) for row in rows]
    return sv.Detections(
        xyxy=np.asarray([row["xyxy"] for row in rows], dtype=float),
        confidence=np.asarray(confidences, dtype=float),
        class_id=np.zeros(len(rows), dtype=int),
        tracker_id=np.asarray(track_ids, dtype=int),
        data={
            "class_name": [row["class_name"] for row in rows],
            "track_state": [row["track_state"] for row in rows],
        },
    )


def draw_tracks(frame, rows: list[dict]):
    """Draw boxes and `#id name conf` labels onto a copy-safe frame."""

    import supervision as sv

    detections = rows_to_detections(rows)
    if len(detections) == 0:
        return frame

    labels = [
        f"#{row['track_id']} {row['class_name']}"
        + ("" if row["confidence"] is None else f" {float(row['confidence']):.2f}")
        for row in rows
    ]
    annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK).annotate(
        scene=frame, detections=detections
    )
    return sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK).annotate(
        scene=annotated, detections=detections, labels=labels
    )
