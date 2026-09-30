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
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from pipeline.possession_v3 import V3_CONTRACT_VERSION

RESULTS_SCHEMA_VERSION = 5
V3_GENERATION_RETENTION = 2
V3_STAGING_MAX_AGE_SECONDS = 3600
V2_STAGE_SCHEMA = "v2-stage-1"


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
    possession_v4_diagnostics_file: str | None = None
    possession_v4_schema_version: int | None = None

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
        if payload.get("possession_v4_diagnostics_file") is None:
            payload.pop("possession_v4_diagnostics_file", None)
        if payload.get("possession_v4_schema_version") is None:
            payload.pop("possession_v4_schema_version", None)
        return payload

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


_V4_DIAGNOSTIC_FIELDS = {
    "spatial_score", "mask_score", "bbox_score", "mask_bbox_agreement", "source",
    "control_evidence", "release_evidence", "acquisition_evidence", "ball_in_transit",
    "evidence_window_start", "evidence_window_end", "v4_schema_version", "feature_flag",
    "control_score", "release_score", "acquisition_score", "gate_decision",
    "gate_reason_codes", "spatial_contributions", "evidence_sources",
}


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
    if config.get("engine", config.get("possession_engine")) == "v4":
        audit = record.setdefault("audit", {})
        reasons = list(audit.get("provenance", {}).get("reason_codes", []))
        audit.setdefault("gate_reason_codes", reasons)
        audit.setdefault("gate_decision", "allow" if audit.get("switch_confirmed") else "hold")
        audit.setdefault("feature_flag", "v4")
        record["v4"] = {"feature_flag": True, "schema_version": 1}
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
    v4_enabled = config.get("engine", config.get("possession_engine")) == "v4"
    if v4_enabled:
        datasets["v4_diagnostics"] = datasets["possession"]
    filenames = {
        "possession": "possession_v3.jsonl",
        "primitives": "possession_primitives_v3.jsonl",
        "transitions": "ball_transitions_v3.jsonl",
        "events": "possession_events_v3.jsonl",
    }
    if v4_enabled:
        filenames["v4_diagnostics"] = "possession_v4_diagnostics.jsonl"
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
        if v4_enabled:
            manifest["v4"] = {"enabled": True, "schema_version": 1, "feature_flag": "v4",
                              "overlay_fields": ["holder_control", "holder_release", "challenger_acquisition",
                                                  "mask_bbox_agreement", "gate_reason_codes"]}
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


def _write_mask_bundle(path: Path, mask_paths: Mapping[str, Path]) -> None:
    """Write a small, flat tar archive without trusting observation keys as paths."""
    with tarfile.open(path, mode="w") as archive:
        for filename, mask_path in sorted(mask_paths.items()):
            data = mask_path.read_bytes()
            info = tarfile.TarInfo(filename)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(manifest))
    payload["artifacts"]["manifest"].pop("sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def stage_fingerprint(*values: Any) -> str:
    """Return a stable content fingerprint for effective stage inputs."""
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stage_relative_path(value: str) -> Path:
    path = Path(value)
    if not isinstance(value, str) or not value or path.is_absolute() or "\\" in value:
        raise ValueError(f"unsafe artifact path: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe artifact path: {value!r}")
    return path


def publish_stage_generation(
    root: Path,
    *,
    stage: str,
    input_fingerprint: str,
    request_fingerprint: str,
    artifacts: Mapping[str, bytes | Path],
    frame_bounds: list[int] | None = None,
    tracking_generation: str | None = None,
) -> dict[str, Any]:
    """Publish a complete immutable stage generation and atomically advance current.json."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generation = stage_fingerprint(stage, input_fingerprint, request_fingerprint,
                                   frame_bounds, tracking_generation,
                                   {key: (value.read_bytes() if isinstance(value, Path) else value)
                                    for key, value in sorted(artifacts.items())})
    generations = root / "generations"
    generations.mkdir(exist_ok=True)
    final = generations / generation
    if final.exists():
        manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        validate_stage_generation(root, manifest)
        return manifest
    staging = Path(tempfile.mkdtemp(prefix=".stage-", dir=root))
    published = False
    try:
        refs: dict[str, Any] = {}
        for name, value in artifacts.items():
            relative = _stage_relative_path(name)
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value.read_bytes() if isinstance(value, Path) else value)
            refs[name] = {"path": name, **_artifact_ref(path, name)}
        manifest = {
            "schema": V2_STAGE_SCHEMA, "stage": stage, "generation": generation,
            "input_fingerprint": input_fingerprint, "request_fingerprint": request_fingerprint,
            "frame_bounds": frame_bounds, "tracking_generation": tracking_generation,
            "complete": True, "artifacts": refs,
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        os.replace(staging, final)
        published = True
        pointer = root / "current.json"
        pointer_tmp = root / ".current.json.tmp"
        pointer_tmp.write_text(json.dumps({"generation": generation, "stage": stage}), encoding="utf-8")
        os.replace(pointer_tmp, pointer)
        return manifest
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def validate_stage_generation(
    root: Path, manifest: Mapping[str, Any], *, expected_generation: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless a published stage manifest and every hash are valid."""
    if manifest.get("schema") != V2_STAGE_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("incomplete stage manifest")
    for field in ("stage", "input_fingerprint", "request_fingerprint", "manifest_sha256"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"missing required stage field: {field}")
    if not isinstance(manifest.get("frame_bounds"), list) or len(manifest["frame_bounds"]) != 2:
        raise ValueError("invalid frame_bounds")
    expected_manifest_hash = hashlib.sha256(
        json.dumps({key: value for key, value in manifest.items() if key != "manifest_sha256"},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("manifest hash mismatch")
    generation = manifest.get("generation")
    if (not isinstance(generation, str) or not generation or generation != Path(generation).name
            or (expected_generation is not None and generation != expected_generation)):
        raise ValueError("invalid generation")
    directory = Path(root) / "generations" / generation
    if not directory.is_dir():
        raise ValueError("generation is missing")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("missing stage artifacts")
    required = {"tracks.jsonl", "tracking-manifest.json"} if manifest["stage"] == "tracking" else {"results.json"}
    if not required <= set(artifacts):
        raise ValueError("missing required tracking/stats artifact")
    for name, reference in artifacts.items():
        if not isinstance(reference, Mapping) or reference.get("path") != name:
            raise ValueError("invalid artifact reference")
        path = directory / _stage_relative_path(name)
        if not path.is_file():
            raise ValueError(f"missing artifact: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference.get("sha256"):
            raise ValueError(f"hash mismatch: {name}")
        if type(reference.get("size_bytes")) is not int or reference["size_bytes"] != path.stat().st_size:
            raise ValueError(f"size mismatch: {name}")
    expected_fingerprint = stage_fingerprint(
        manifest["stage"], manifest["input_fingerprint"], manifest["request_fingerprint"],
        manifest["frame_bounds"], manifest.get("tracking_generation"),
        {name: (directory / _stage_relative_path(name)).read_bytes() for name in sorted(artifacts)},
    )
    if generation != expected_fingerprint:
        raise ValueError("generation fingerprint mismatch")
    if manifest["stage"] == "tracking":
        _validate_embedded_tracking_manifest(directory, manifest)
    return dict(manifest)


def _validate_embedded_tracking_manifest(directory: Path, outer: Mapping[str, Any]) -> None:
    path = directory / "tracking-manifest.json"
    try:
        embedded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid embedded tracking package manifest") from error
    required = {
        "schema", "schema_version", "input_fingerprint", "frame_bounds", "observation_count",
        "mask_encoding", "artifacts", "transfer_errors", "truncated", "complete",
    }
    declared = embedded.get("artifacts") if isinstance(embedded, Mapping) else None
    if (not isinstance(embedded, Mapping) or not required <= set(embedded)
            or not isinstance(declared, Mapping)
            or embedded.get("schema") != "sam31-artifacts"
            or type(embedded.get("schema_version")) is not int
            or embedded.get("input_fingerprint") != outer.get("input_fingerprint")
            or embedded.get("frame_bounds") != outer.get("frame_bounds")
            or type(embedded.get("observation_count")) is not int
            or type(embedded.get("truncated")) is not bool or embedded["truncated"]
            or embedded.get("complete") is not True
            or not isinstance(embedded.get("transfer_errors"), list)
            or any(not isinstance(item, str) for item in embedded["transfer_errors"])):
        raise ValueError("invalid embedded tracking package manifest")
    if embedded.get("artifacts", {}).get("manifest", {}).get("sha256") != _manifest_hash(embedded):
        raise ValueError("embedded tracking manifest hash mismatch")

    if not isinstance(declared.get("manifest"), Mapping) or not isinstance(declared.get("tracks"), Mapping):
        raise ValueError("invalid embedded tracking artifact contract")
    manifest_reference = declared["manifest"]
    if not isinstance(manifest_reference.get("key"), str) or manifest_reference.get("sha256") != _manifest_hash(embedded):
        raise ValueError("invalid embedded manifest reference")
    if not isinstance(declared.get("masks"), Mapping):
        raise ValueError("invalid embedded tracking artifact contract")

    def check_reference(reference: Mapping[str, Any], fallback: str | None = None) -> None:
        key = reference.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("invalid embedded artifact reference")
        candidate = directory / _stage_relative_path(key)
        if not candidate.is_file() and fallback:
            candidate = directory / fallback
        if not candidate.is_file():
            raise ValueError("missing embedded artifact")
        if reference.get("sha256") != hashlib.sha256(candidate.read_bytes()).hexdigest():
            raise ValueError("embedded artifact hash mismatch")
        if "size_bytes" in reference and reference["size_bytes"] != candidate.stat().st_size:
            raise ValueError("embedded artifact size mismatch")

    check_reference(declared["tracks"], "tracks.jsonl")
    masks = declared["masks"]
    if not isinstance(masks.get("missing"), list) or any(not isinstance(item, str) for item in masks["missing"]):
        raise ValueError("invalid embedded mask contract")
    references = masks.get("references")
    if not isinstance(references, Mapping) or type(masks.get("count")) is not int or masks["count"] != len(references):
        raise ValueError("invalid embedded mask contract")
    for reference in references.values():
        if not isinstance(reference, Mapping):
            raise ValueError("invalid embedded mask reference")
        check_reference(reference)
    _validate_embedded_mask_bundle(directory, "masks", masks, check_reference)
    for name in ("clean_masks",):
        if name in declared:
            group = declared[name]
            refs = group.get("references") if isinstance(group, Mapping) else None
            if not isinstance(refs, Mapping) or type(group.get("count")) is not int or group["count"] != len(refs):
                raise ValueError("invalid embedded clean-mask contract")
            for reference in refs.values():
                if not isinstance(reference, Mapping):
                    raise ValueError("invalid embedded clean-mask reference")
                check_reference(reference)
            _validate_embedded_mask_bundle(directory, name, group, check_reference)


def _validate_embedded_mask_bundle(
    directory: Path,
    name: str,
    group: Mapping[str, Any],
    check_reference,
) -> None:
    """Validate bundle bytes and decoded content against individual references."""
    bundle = group.get("bundle")
    if bundle is None:
        return  # Legacy packages contain only individual mask files.
    if not isinstance(bundle, Mapping):
        raise ValueError(f"invalid embedded {name} bundle")
    check_reference(bundle)
    references = group.get("references")
    if not isinstance(references, Mapping):
        raise ValueError(f"invalid embedded {name} references")
    bundle_path = directory / _stage_relative_path(bundle["key"])
    bundle_data = bundle_path.read_bytes()
    try:
        with tarfile.open(fileobj=io.BytesIO(bundle_data), mode="r:") as archive:
            members = archive.getmembers()
            member_names: set[str] = set()
            member_data: dict[str, bytes] = {}
            for member in members:
                member_path = PurePosixPath(member.name)
                if (not member.isfile() or member_path.is_absolute() or "\\" in member.name
                        or any(part in {"", ".", ".."} for part in member_path.parts)):
                    raise ValueError(f"unsafe embedded {name} bundle member")
                if member.name in member_names:
                    raise ValueError(f"duplicate embedded {name} bundle member")
                member_names.add(member.name)
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"unreadable embedded {name} bundle member")
                member_data[member.name] = handle.read()
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"invalid embedded {name} bundle") from error

    expected_names: set[str] = set()
    for observation_key, reference in references.items():
        if not isinstance(reference, Mapping) or not isinstance(reference.get("key"), str):
            raise ValueError(f"invalid embedded {name} reference")
        member_name = Path(reference["key"]).name
        if member_name in expected_names:
            raise ValueError(f"duplicate embedded {name} reference")
        expected_names.add(member_name)
        data = member_data.get(member_name)
        if data is None:
            raise ValueError(f"missing embedded {name} bundle member")
        if (type(reference.get("size_bytes")) is not int
                or reference["size_bytes"] != len(data)
                or reference.get("sha256") != hashlib.sha256(data).hexdigest()):
            raise ValueError(f"embedded {name} bundle member hash or size mismatch")
        individual_path = directory / _stage_relative_path(reference["key"])
        individual_data = individual_path.read_bytes()
        bundle_mask = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        individual_mask = cv2.imdecode(np.frombuffer(individual_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if (bundle_mask is None or individual_mask is None or bundle_mask.ndim != 2
                or individual_mask.ndim != 2 or not np.array_equal(bundle_mask, individual_mask)):
            raise ValueError(f"embedded {name} bundle content mismatch: {observation_key}")
    if member_names != expected_names:
        raise ValueError(f"embedded {name} bundle members do not match references")


def load_stage_generation(root: Path, generation: str | None = None) -> dict[str, Any]:
    pointer = Path(root) / "current.json"
    if generation is None:
        if not pointer.is_file():
            raise ValueError("no published generation")
        generation = json.loads(pointer.read_text(encoding="utf-8"))["generation"]
    manifest_path = Path(root) / "generations" / generation / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"generation not found: {generation}")
    return validate_stage_generation(
        root, json.loads(manifest_path.read_text(encoding="utf-8")),
        expected_generation=generation,
    )


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
    mask_paths: dict[str, Path] = {}
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
        mask_paths[filename] = mask_path

    mask_bundle_path = output_dir / "masks.tar"
    if mask_paths:
        _write_mask_bundle(mask_bundle_path, mask_paths)

    clean_mask_refs: dict[str, dict[str, Any]] = {}
    clean_mask_paths: dict[str, Path] = {}
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
            clean_mask_paths[filename] = mask_path

    clean_mask_bundle_path = output_dir / "masks_clean.tar"
    if clean_mask_paths:
        _write_mask_bundle(clean_mask_bundle_path, clean_mask_paths)

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
    if mask_paths:
        manifest["artifacts"]["masks"]["bundle"] = _artifact_ref(
            mask_bundle_path, f"{prefix}/masks.tar" if prefix else "masks.tar"
        )
    if clean_mask_refs:
        manifest["artifacts"]["clean_masks"] = {
            "key": f"{prefix}/masks_clean/" if prefix else "masks_clean/",
            "count": len(clean_mask_refs),
            "references": clean_mask_refs,
        }
        manifest["artifacts"]["clean_masks"]["bundle"] = _artifact_ref(
            clean_mask_bundle_path, f"{prefix}/masks_clean.tar" if prefix else "masks_clean.tar"
        )
    manifest_path = output_dir / "manifest.json"
    manifest["artifacts"]["manifest"]["sha256"] = _manifest_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
