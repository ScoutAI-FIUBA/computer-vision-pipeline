"""Pipeline output format.

Two files come out of a run:

  tracks.jsonl  one JSON object per track per frame, written incrementally so a
                long video never has to fit in memory
  results.json  run metadata and summary counts, small enough to read directly
                or store in a database row

Keeping the bulk data in JSONL matters: a 10-minute clip with 10 players on
court produces roughly 180k track records, which is fine as a stream and
unpleasant as one in-memory list.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

RESULTS_SCHEMA_VERSION = 5


@dataclass
class RunSummary:
    """Counts and timings describing one pipeline run."""

    frames_processed: int = 0
    segments_processed: int = 0
    detector_calls: int = 0
    track_records: int = 0
    distinct_track_ids: int = 0
    # Detections whose colours matched a different track better than their own.
    # This is the ID-swap signal: `distinct_track_ids` cannot see a swap, because
    # swapping two labels neither creates nor destroys an ID.
    suspected_swaps: int = 0
    wall_seconds: float = 0.0


@dataclass
class PipelineResults:
    """Everything about a run except the per-frame track rows."""

    schema_version: int
    video: dict
    config: dict
    summary: RunSummary
    device: str
    tracks_file: str = "tracks.jsonl"
    teams_file: str | None = None
    numbers_file: str | None = None
    shot_events_file: str | None = None
    projected_tracks_file: str | None = None
    projected_shots_file: str | None = None
    projected_shots_image_file: str | None = None
    annotated_video_file: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["summary"] = asdict(self.summary)
        if payload.get("projected_shots_file") is None:
            payload.pop("projected_shots_file", None)
        if payload.get("projected_shots_image_file") is None:
            payload.pop("projected_shots_image_file", None)
        if payload.get("projected_tracks_file") is None:
            payload.pop("projected_tracks_file", None)
        return payload

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class TrackWriter:
    """Appends track rows to a JSONL file, one line per track per frame."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")
        self.record_count = 0
        self.track_ids: set[int] = set()

    def write_frame(self, frame_index: int, rows: list[dict]) -> None:
        for row in rows:
            track_id = row.get("track_id")
            if track_id is not None:
                self.track_ids.add(int(track_id))
            self._handle.write(
                json.dumps(
                    {
                        "frame_index": frame_index,
                        "track_id": None if track_id is None else int(track_id),
                        "team_id": None if row.get("team_id") is None else int(row["team_id"]),
                        "track_state": row.get("track_state"),
                        "class_name": str(row.get("class_name", "player")),
                        "confidence": None if row.get("confidence") is None else float(row["confidence"]),
                        "xyxy": [float(value) for value in row["xyxy"]],
                    }
                )
                + "\n"
            )
            self.record_count += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "TrackWriter":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
