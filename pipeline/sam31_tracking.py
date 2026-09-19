"""Small boundary around the SAM 3.1 multiplex tracking session.

The adapter deliberately knows nothing about model packages.  A session object
is injected so the contract can be tested without CUDA and the V2 runner can
bind the official predictor later.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def observation_key(frame_index: int, raw_track_id: int) -> str:
    """Return the stable key for one raw observation."""

    return f"{int(frame_index)}:{int(raw_track_id)}"


def _xyxy(bbox: list[float], bbox_format: str) -> list[float]:
    values = [float(value) for value in bbox]
    if len(values) != 4:
        raise ValueError("bbox must contain four coordinates")
    if bbox_format == "xywh":
        values[2] += values[0]
        values[3] += values[1]
    elif bbox_format != "xyxy":
        raise ValueError(f"unsupported bbox_format: {bbox_format!r}")
    return values


def normalize_multiplex_output(
    frame_index: int,
    timestamp_seconds: float,
    objects: list[Mapping[str, Any]],
    stable_player_ids: Mapping[int, str] | None = None,
    *,
    track_state: str = "tracked",
) -> list[dict[str, Any]]:
    """Convert session objects into the V2 observation contract."""

    stable_player_ids = stable_player_ids or {}
    rows = []
    for obj in objects:
        raw_track_id = int(obj["object_id"])
        rows.append(
            {
                "observation_key": observation_key(frame_index, raw_track_id),
                "frame_index": int(frame_index),
                "timestamp_seconds": float(timestamp_seconds),
                "raw_track_id": raw_track_id,
                "stable_player_id": stable_player_ids.get(raw_track_id),
                "bbox_xyxy": _xyxy(list(obj["bbox"]), str(obj.get("bbox_format", "xyxy"))),
                "confidence": None if obj.get("confidence") is None else float(obj["confidence"]),
                "probability": None if obj.get("probability") is None else float(obj["probability"]),
                "track_state": track_state,
            }
        )
    return rows


class Sam31MultiplexTracker:
    """Manage session lifecycle and bounded retention around a multiplex API."""

    def __init__(self, session: Any, retention_frames: int = 2) -> None:
        if retention_frames < 0:
            raise ValueError("retention_frames must be at least 0")
        self._session = session
        self._retention_frames = retention_frames
        self._last_rows: list[dict[str, Any]] = []
        self._missed_frames = 0
        self._started = False

    def start(self, video: Any) -> None:
        self._session.start(video)
        self._started = True

    def add_prompt(self, prompt: str) -> None:
        if not self._started:
            raise RuntimeError("tracker must be started before adding a prompt")
        self._session.add_prompt(prompt)

    def propagate(self, frame_index: int, timestamp_seconds: float) -> list[dict[str, Any]]:
        if not self._started:
            raise RuntimeError("tracker must be started before propagation")
        objects = list(self._session.propagate(frame_index))
        if objects:
            rows = normalize_multiplex_output(frame_index, timestamp_seconds, objects)
            self._last_rows = rows
            self._missed_frames = 0
            return rows
        if not self._last_rows or self._missed_frames >= self._retention_frames:
            self._last_rows = []
            return []
        self._missed_frames += 1
        return [{**row, "track_state": "coasted"} for row in self._last_rows]

    def close(self) -> None:
        self._session.close()
        self._started = False
        self._last_rows = []
        self._missed_frames = 0
