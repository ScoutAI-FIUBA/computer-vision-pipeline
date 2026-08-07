"""Optional debug annotation.

Only used when `PipelineConfig.write_annotated_video` is on. Kept separate so the
main path never imports it, and so `pipeline/` has no dependency on the older
`scripts/` helpers.
"""

from __future__ import annotations

TEAM_COLORS = {
    0: (0, 90, 255),
    1: (255, 140, 0),
}


def team_color(team_id: int | None) -> tuple[int, int, int]:
    return TEAM_COLORS.get(team_id, (160, 255, 160))


def track_label_id(row: dict) -> str | None:
    confirmed_number = row.get("confirmed_number")
    if confirmed_number not in (None, "", "unknown"):
        return str(confirmed_number)
    return None


def track_label(row: dict) -> str:
    label_id = track_label_id(row)
    if label_id is None:
        return str(row["class_name"])
    return f"#{label_id} {row['class_name']}"


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

    if any(row.get("team_id") is not None for row in rows):
        import cv2

        annotated = frame.copy()
        for row in rows:
            x1, y1, x2, y2 = [int(round(value)) for value in row["xyxy"]]
            team_id = row.get("team_id")
            label = track_label(row)
            if team_id is not None:
                label += f" T{int(team_id)}"
            if row["confidence"] is not None:
                label += f" {float(row['confidence']):.2f}"
            color = team_color(None if team_id is None else int(team_id))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return annotated

    import supervision as sv

    detections = rows_to_detections(rows)
    if len(detections) == 0:
        return frame

    labels = [
        track_label(row)
        + ("" if row["confidence"] is None else f" {float(row['confidence']):.2f}")
        for row in rows
    ]
    annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK).annotate(
        scene=frame, detections=detections
    )
    return sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK).annotate(
        scene=annotated, detections=detections, labels=labels
    )
