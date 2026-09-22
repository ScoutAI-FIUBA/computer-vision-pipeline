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

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

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
    ball_tracks_file: str | None = None
    projected_tracks_file: str | None = None
    projected_shots_file: str | None = None
    projected_shots_image_file: str | None = None
    annotated_video_file: str | None = None
    artifact_manifest_file: str | None = None
    artifact_references: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["summary"] = asdict(self.summary)
        if payload.get("projected_shots_file") is None:
            payload.pop("projected_shots_file", None)
        if payload.get("projected_shots_image_file") is None:
            payload.pop("projected_shots_image_file", None)
        if payload.get("projected_tracks_file") is None:
            payload.pop("projected_tracks_file", None)
        if payload.get("ball_tracks_file") is None:
            payload.pop("ball_tracks_file", None)
        if payload.get("artifact_manifest_file") is None:
            payload.pop("artifact_manifest_file", None)
        if payload.get("artifact_references") is None:
            payload.pop("artifact_references", None)
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


def _artifact_ref(path: Path, key: str) -> dict[str, Any]:
    data = path.read_bytes()
    return {"key": key, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _mask_filename(observation_key: str) -> str:
    return f"{hashlib.sha256(observation_key.encode('utf-8')).hexdigest()}.png"


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(manifest))
    payload["artifacts"]["manifest"].pop("sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def package_artifacts(
    output_dir: Path,
    observations: Iterable[Mapping[str, Any]],
    masks: Mapping[str, Any],
    *,
    input_fingerprint: str | None = None,
    video_metadata: Mapping[str, Any] | None = None,
    frame_bounds: list[int] | None = None,
    prompts: Iterable[str] = (),
    config: Mapping[str, Any] | None = None,
    benchmark: Mapping[str, Any] | None = None,
    acceptance: Mapping[str, Any] | None = None,
    acceptance_reasons: Iterable[str] = (),
    output_prefix: str = "",
    transfer_errors: Iterable[str] = (),
    truncated: bool = False,
) -> dict[str, Any]:
    """Write deterministic V2 evidence files and return a retrieval manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(observations)
    keys = [str(row["observation_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate observation_key")

    tracks_path = output_dir / "tracks.jsonl"
    tracks_path.write_text(
        "".join(json.dumps({key: value for key, value in row.items() if key != "bbox_coordinate_space"}, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    missing_masks: list[str] = []
    mask_refs: dict[str, dict[str, Any]] = {}
    for key in keys:
        if key not in masks:
            missing_masks.append(key)
            continue
        mask = np.asarray(masks[key])
        if mask.ndim != 2:
            raise ValueError(f"mask for {key!r} must be a 2-D array")
        if mask.dtype == bool:
            mask = mask.astype(np.uint8)
        if mask.dtype != np.uint8:
            raise ValueError(f"mask for {key!r} must use lossless uint8 data")
        filename = _mask_filename(key)
        mask_path = mask_dir / filename
        if not cv2.imwrite(str(mask_path), mask):
            raise OSError(f"could not write mask for {key!r}")
        mask_refs[key] = _artifact_ref(mask_path, f"{output_prefix}/masks/{filename}".lstrip("/"))

    prefix = output_prefix.strip("/")
    tracks_key = f"{prefix}/tracks.jsonl" if prefix else "tracks.jsonl"
    transfer_errors_list = list(transfer_errors)
    incomplete = bool(truncated or missing_masks or transfer_errors_list)
    manifest: dict[str, Any] = {
        "schema": "sam31-artifacts",
        "schema_version": 1,
        "input_fingerprint": input_fingerprint,
        "video": dict(video_metadata or {}),
        "frame_bounds": frame_bounds,
        "prompts": list(prompts),
        "config": dict(config or {}),
        "benchmark": dict(benchmark or {}),
        "acceptance": dict(acceptance or {}),
        "acceptance_reasons": list(acceptance_reasons),
        "observation_count": len(rows),
        "mask_encoding": "PNG uint8 lossless",
        "artifacts": {
            "manifest": {"key": f"{prefix}/manifest.json" if prefix else "manifest.json"},
            "tracks": {**_artifact_ref(tracks_path, tracks_key)},
            "masks": {
                "key": f"{prefix}/masks/" if prefix else "masks/",
                "count": len(mask_refs),
                "missing": missing_masks,
                "references": mask_refs,
            },
        },
        "transfer_errors": transfer_errors_list,
        "truncated": incomplete,
        "complete": not incomplete,
    }
    manifest_path = output_dir / "manifest.json"
    manifest["artifacts"]["manifest"]["sha256"] = _manifest_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
