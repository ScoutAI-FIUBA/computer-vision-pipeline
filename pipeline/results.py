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
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from pipeline.possession_v3 import V3_CONTRACT_VERSION

RESULTS_SCHEMA_VERSION = 5
V3_GENERATION_RETENTION = 2
V3_STAGING_MAX_AGE_SECONDS = 3600


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
    shot_evidence_file: str | None = None
    ball_provenance: dict[str, Any] | None = None
    possession_events_file: str | None = None
    possession_transitions_file: str | None = None
    possession_v3_file: str | None = None
    possession_v3_schema_version: int | None = None
    possession_v3_transitions_file: str | None = None
    possession_v3_events_file: str | None = None
    possession_v3_manifest_file: str | None = None
    possession_v3_observability: dict[str, Any] | None = None

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
        if payload.get("shot_evidence_file") is None:
            payload.pop("shot_evidence_file", None)
        if payload.get("artifact_manifest_file") is None:
            payload.pop("artifact_manifest_file", None)
        if payload.get("artifact_references") is None:
            payload.pop("artifact_references", None)
        if payload.get("ball_provenance") is None:
            payload.pop("ball_provenance", None)
        if payload.get("possession_events_file") is None:
            payload.pop("possession_events_file", None)
        if payload.get("possession_transitions_file") is None:
            payload.pop("possession_transitions_file", None)
        if payload.get("possession_v3_file") is None:
            payload.pop("possession_v3_file", None)
        for key in ("possession_v3_transitions_file", "possession_v3_events_file", "possession_v3_manifest_file"):
            if payload.get(key) is None:
                payload.pop(key, None)
        if payload.get("possession_v3_schema_version") is None:
            payload.pop("possession_v3_schema_version", None)
        if payload.get("possession_v3_observability") is None:
            payload.pop("possession_v3_observability", None)
        return payload

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


_V4_DIAGNOSTIC_FIELDS = {"spatial_score", "mask_score", "bbox_score", "mask_bbox_agreement", "source"}


def _without_v4_diagnostics(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_v4_diagnostics(item) for key, item in value.items()
                if key not in _V4_DIAGNOSTIC_FIELDS}
    if isinstance(value, list):
        return [_without_v4_diagnostics(item) for item in value]
    return value


def _json_record(value: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if config.get("engine", config.get("possession_engine")) != "v4":
        value = _without_v4_diagnostics(value)
    record = dict(value)
    record.setdefault("schema", "possession-engine-v3")
    record.setdefault("schema_version", V3_CONTRACT_VERSION)
    frame = record.get("frame_index")
    if record.get("frame_start") is None:
        record["frame_start"] = frame
    if record.get("frame_end") is None:
        record["frame_end"] = frame
    timestamp = record.get("timestamp_seconds")
    if record.get("timestamp_start") is None:
        record["timestamp_start"] = timestamp
    if record.get("timestamp_end") is None:
        record["timestamp_end"] = timestamp
    if config:
        record.setdefault("config", dict(config))
    return record


def write_v3_artifacts(
    output_dir: Path,
    possession: Iterable[Any],
    primitives: Iterable[Mapping[str, Any]],
    transitions: Iterable[Any],
    events: Iterable[Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write deterministic, additive V3 artifacts and a self-hashed manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config or {})
    datasets = {
        "possession": [_json_record(row, config) for row in possession],
        "primitives": [_json_record(row, config) for row in primitives],
        "transitions": [_json_record(row, config) for row in transitions],
        "events": [_json_record(row, config) for row in events],
    }
    filenames = {
        "possession": "possession_v3.jsonl",
        "primitives": "possession_primitives_v3.jsonl",
        "transitions": "ball_transitions_v3.jsonl",
        "events": "possession_events_v3.jsonl",
    }
    generation_id = hashlib.sha256(json.dumps({"datasets": datasets, "config": config}, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()[:16]
    generation_root = output_dir / "possession_v3" / "generations"
    generation_path = generation_root / generation_id
    manifest_path = output_dir / "possession_v3_manifest.json"
    old_root_files = [output_dir / filename for filename in (*filenames.values(),)]
    artifacts: dict[str, Any] = {}
    staging_path = Path(tempfile.mkdtemp(prefix=".possession-v3-", dir=output_dir))
    published_generation = False
    try:
        generation_stage = staging_path / "generation"
        generation_stage.mkdir()
        for name, rows in datasets.items():
            path = generation_stage / filenames[name]
            path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
            artifacts[name] = {"path": f"possession_v3/generations/{generation_id}/{filenames[name]}",
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "count": len(rows)}
        manifest: dict[str, Any] = {"schema": "possession-engine-v3", "schema_version": V3_CONTRACT_VERSION,
                                    "generation": generation_id, "config": config, "artifacts": artifacts}
        manifest["artifacts"]["manifest"] = {"path": "possession_v3_manifest.json"}
        manifest["artifacts"]["manifest"]["sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        pointer_stage = staging_path / "possession_v3_manifest.json"
        pointer_stage.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        generation_root.mkdir(parents=True, exist_ok=True)
        if generation_path.exists():
            shutil.rmtree(generation_stage)
        else:
            os.replace(generation_stage, generation_path)
            published_generation = True
        os.replace(pointer_stage, manifest_path)
        try:
            for path in old_root_files:
                path.unlink(missing_ok=True)
            _prune_v3_generations(generation_root, current=generation_id)
        except OSError:
            pass
        return manifest
    except Exception:
        if published_generation:
            shutil.rmtree(generation_path, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)


def _prune_v3_generations(root: Path, *, current: str) -> None:
    """Retain the current generation plus one rollback generation."""
    generations = sorted((path for path in root.iterdir() if path.is_dir()),
                         key=lambda path: (path.name == current, path.stat().st_mtime), reverse=True)
    for path in generations[V3_GENERATION_RETENTION:]:
        shutil.rmtree(path, ignore_errors=True)
    now = time.time()
    for path in root.parent.parent.glob(".possession-v3-*"):
        try:
            if path.is_dir() and now - path.stat().st_mtime > V3_STAGING_MAX_AGE_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


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
    clean_masks: Mapping[str, Any] | None = None,
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

    if clean_masks:
        for row in rows:
            key = str(row["observation_key"])
            if key in clean_masks:
                row["clean_mask_ref"] = f"masks_clean/{_mask_filename(key)}"

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

    clean_mask_refs: dict[str, dict[str, Any]] = {}
    if clean_masks:
        clean_mask_dir = output_dir / "masks_clean"
        clean_mask_dir.mkdir(exist_ok=True)
        for key, value in clean_masks.items():
            mask = np.asarray(value)
            if mask.ndim != 2:
                raise ValueError(f"clean mask for {key!r} must be a 2-D array")
            mask = mask.astype(np.uint8, copy=False)
            filename = _mask_filename(key)
            mask_path = clean_mask_dir / filename
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"could not write clean mask for {key!r}")
            clean_mask_refs[key] = _artifact_ref(mask_path, f"{output_prefix}/masks_clean/{filename}".lstrip("/"))

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
    if clean_mask_refs:
        manifest["artifacts"]["clean_masks"] = {
            "key": f"{prefix}/masks_clean/" if prefix else "masks_clean/",
            "count": len(clean_mask_refs),
            "references": clean_mask_refs,
        }
    manifest_path = output_dir / "manifest.json"
    manifest["artifacts"]["manifest"]["sha256"] = _manifest_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
