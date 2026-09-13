"""TrackNet-first ball rows with bounded SAM2 fallback seed helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.tracking import xyxy_iou

DEFAULT_BALL_CLASSES = ("basketball", "ball", "basketball-ball")
DEFAULT_BALL_TRACK_MAX_DISTANCE = 80.0
TRACKNET_CONTEXT_SIZE = 8
TRACKNET_INPUT_SIZE = (512, 288)


def _normalized_class_names(class_names: tuple[str, ...]) -> set[str]:
    return {name.strip().lower() for name in class_names if name.strip()}


def filter_ball_predictions(result: dict, class_names: tuple[str, ...] = DEFAULT_BALL_CLASSES) -> dict:
    allowed = _normalized_class_names(class_names)
    predictions = result.get("predictions", []) if isinstance(result, dict) else []
    return {
        **(result if isinstance(result, dict) else {}),
        "predictions": [
            prediction
            for prediction in predictions
            if str(prediction.get("class", "")).strip().lower() in allowed
        ],
    }


def filter_ball_rows(rows: list[dict], class_names: tuple[str, ...] = DEFAULT_BALL_CLASSES) -> list[dict]:
    allowed = _normalized_class_names(class_names)
    return [
        dict(row)
        for row in rows
        if str(row.get("class_name", row.get("class", ""))).strip().lower() in allowed
    ]


def ball_rows_from_result(
    result,
    *,
    min_confidence: float,
    detections_from_result,
    detection_rows,
    class_names: tuple[str, ...] = DEFAULT_BALL_CLASSES,
) -> list[dict]:
    if isinstance(result, list):
        rows = filter_ball_rows(result, class_names)
        return [dict(row) for row in rows if float(row.get("confidence") or 0.0) >= min_confidence]

    return detection_rows(
        detections_from_result(
            filter_ball_predictions(result, class_names),
            min_confidence=min_confidence,
        )
    )


def bbox_center(xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))


def candidate_to_ball_row(candidate, *, frame_size: tuple[int, int], radius: float) -> dict:
    """Convert one accepted TrackNet peak into a conservative SAM2 seed box."""
    width, height = frame_size
    x = float(candidate.x)
    y = float(candidate.y)
    return {
        "xyxy": [
            max(0.0, x - radius),
            max(0.0, y - radius),
            min(float(width), x + radius),
            min(float(height), y + radius),
        ],
        "confidence": float(candidate.score),
        "class_name": "ball",
        "track_id": None,
        "track_state": "detected",
    }


def reconcile_ball_track_id(
    rows: list[dict], active_rows: list[dict], next_track_id: int, *, max_distance: float = DEFAULT_BALL_TRACK_MAX_DISTANCE
) -> tuple[list[dict], int]:
    """Keep the sole public ball ID across a TrackNet re-seed when nearby."""
    if not rows:
        return rows, next_track_id
    if active_rows and sum(
        (a - b) ** 2 for a, b in zip(bbox_center(rows[0]["xyxy"]), bbox_center(active_rows[0]["xyxy"]))
    ) ** 0.5 <= max_distance:
        rows[0]["track_id"] = int(active_rows[0]["track_id"])
        return rows, next_track_id
    rows[0]["track_id"] = next_track_id
    return rows, next_track_id + 1


def build_tracknet_detector(checkpoint: str | Path):
    from scripts.ball.tracknet import build_detector, inspect_tracknet_checkpoint

    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.exists():
        raise RuntimeError(f"TrackNet checkpoint not found: {checkpoint}")
    inspection = inspect_tracknet_checkpoint(checkpoint)
    if (
        Path(checkpoint).suffix.lower() == ".pt"
        and (inspection.get("seq_len") != TRACKNET_CONTEXT_SIZE or inspection.get("bg_mode") != "concat")
    ):
        raise RuntimeError("TrackNet checkpoint must use seq_len=8 and bg_mode=concat for the local pipeline")

    return build_detector(
        checkpoint,
        backend="auto",
        allow_stub=False,
        context_size=TRACKNET_CONTEXT_SIZE,
        input_size=TRACKNET_INPUT_SIZE,
    )


def select_seed_ball_rows(
    rows: list[dict],
    *,
    active_rows: list[dict],
    min_match_iou: float,
) -> list[dict]:
    if not rows:
        return []

    if active_rows:
        active_row = active_rows[0]
        overlapping = [
            row for row in rows if xyxy_iou(row["xyxy"], active_row["xyxy"]) >= min_match_iou
        ]
        if overlapping:
            return [
                max(
                    overlapping,
                    key=lambda row: (
                        xyxy_iou(row["xyxy"], active_row["xyxy"]),
                        float(row.get("confidence") or 0.0),
                    ),
                )
            ]

    return [
        max(
            rows,
            key=lambda row: (
                float(row.get("confidence") or 0.0),
                -bbox_center(row["xyxy"])[0],
                -bbox_center(row["xyxy"])[1],
            )
        )
    ]


def ball_track_record(frame_index: int, row: dict, *, timestamp_seconds: float | None = None) -> dict:
    xyxy = row.get("xyxy")
    center = bbox_center(xyxy) if xyxy is not None else None
    return {
        "frame_index": int(frame_index),
        "timestamp_seconds": timestamp_seconds,
        "track_id": None if row.get("track_id") is None else int(row["track_id"]),
        "track_state": row.get("track_state"),
        "class_name": str(row.get("class_name", "ball")),
        "confidence": None if row.get("confidence") is None else float(row["confidence"]),
        "xyxy": None if xyxy is None else [float(value) for value in xyxy],
        "center_xy": None if center is None else [center[0], center[1]],
    }


def write_ball_track_record(handle, record: dict) -> None:
    handle.write(json.dumps(record) + "\n")


def write_ball_track_rows(
    handle, *, frame_index: int, rows: list[dict], timestamp_seconds: float | None = None
) -> int:
    written = 0
    for row in rows:
        write_ball_track_record(
            handle, ball_track_record(frame_index, row, timestamp_seconds=timestamp_seconds)
        )
        written += 1
    return written
