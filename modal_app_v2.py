"""Isolated SAM 3.1 Modal boundary with direct local artifact return."""

from __future__ import annotations

import hashlib
import fcntl
import importlib
import inspect
import json
import os
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

import modal

from pipeline.results import (
    _artifact_ref,
    _manifest_hash,
    _mask_filename,
    package_artifacts,
    load_stage_generation,
    publish_stage_generation,
    stage_fingerprint,
)
from pipeline.config import PipelineConfig
from pipeline.number_predictor import build_number_predictor
from pipeline.detection import build_detector
from pipeline.runner import (
    run_stats_from_v2_observations,
    run_stats_from_v2_generation,
    run_possession_only_from_v2_artifacts,
)
from pipeline.runner_v2 import run_v2
from pipeline.input_integrity import verify_sha256_file
from pipeline.progress import ProgressReporter
from pipeline.stage_metrics import (
    METRICS_FILENAME,
    StageMetrics,
    detect_gpu,
    format_job_line,
    format_stage_line,
    metrics_artifact,
)
from pipeline.sam31_tracking import normalize_multiplex_output
from pipeline.sam31_segments import (
    SAM31_MAX_NUM_OBJECTS,
    SAM31_SEGMENT_GRACE_MAX_OBJECTS,
    DEFAULT_SEAM_AMBIGUITY_MARGIN,
    DEFAULT_SEAM_MIN_SCORE,
    assert_segment_object_limit,
    prune_segment_object_overflow,
    deduplicate_segment_observations,
    extract_segment_video,
    plan_segment_windows,
    remap_segment_rows,
    stitch_segment_seams,
)

SAM31_REPOSITORY = "git+https://github.com/facebookresearch/sam3.git"
SAM31_COMMIT = "2345a4a"


def _deploy_git_sha() -> str | None:
    """Short git SHA of the pipeline checkout, evaluated where this module is loaded for deploy.

    Inside the container the image env already carries PIPELINE_GIT_SHA and there is no
    git checkout, so the env value wins. Never raises; None when unknown.
    """
    existing = os.environ.get("PIPELINE_GIT_SHA")
    if existing:
        return existing
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=False,
        )
    except Exception:
        return None
    sha = completed.stdout.strip() if completed.returncode == 0 else ""
    return sha or None


PIPELINE_GIT_SHA = _deploy_git_sha()
SAM31_SMOKE_MODULES = (
    "pycocotools.mask",
    "PIL",
    "psutil",
    "einops",
    "scipy",
    "cv2",
    "sam3.model_builder",
)
CACHE_DIR = "/cache"
TRACKNET_VOLUME_DIR = "/tracknet-data"
TRACKNET_CHECKPOINT = (
    f"{TRACKNET_VOLUME_DIR}/tracknet-finetune/runs/"
    "basketball-v1-20ep-r3/checkpoints/TrackNet_best.pt"
)

app = modal.App("scoutai-pipeline-v2")
model_cache = modal.Volume.from_name("scoutai-model-cache", create_if_missing=True)
tracknet_data = modal.Volume.from_name(
    "tracknet-data", environment_name="main", create_if_missing=False
)
V2_JOB_VOLUME_DIR = "/v2-job-data"
v2_job_data = modal.Volume.from_name("v2-job-data", create_if_missing=True)
V2_VOLUMES = {CACHE_DIR: model_cache, TRACKNET_VOLUME_DIR: tracknet_data, V2_JOB_VOLUME_DIR: v2_job_data}
JOB_SCHEMA = "v2-job-1"
JOB_LEASE_SECONDS = 1800
MAX_JOB_ERRORS = 8
# One value for the decorator and for stage_metrics.json, so the margin cannot drift from the real limit.
PIPELINE_TIMEOUT_SECONDS = 3600
# The class decorator declares no cpu/memory; keep in sync if it ever does.
PIPELINE_CONTAINER = {"cpu": None, "memory_mib": None}


def _job_path(root: Path, job_id: str) -> Path:
    if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("invalid job_id")
    return Path(root) / job_id


def _job_input_path(job: Path, input_fingerprint: str) -> Path:
    """Return the immutable input object for a content fingerprint."""
    digest = input_fingerprint.removeprefix("sha256:")
    if not digest or Path(digest).name != digest or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("invalid input fingerprint")
    return Path(job) / "inputs" / f"{digest}.mp4"


def _persist_job_input(job: Path, input_fingerprint: str, video_bytes: bytes) -> Path:
    """Persist input by fingerprint; never overwrite another generation's input."""
    path = _job_input_path(job, input_fingerprint)
    digest = input_fingerprint.removeprefix("sha256:")
    if hashlib.sha256(video_bytes).hexdigest() != digest:
        raise ValueError("job input fingerprint mismatch")

    job = Path(job)
    job.mkdir(parents=True, exist_ok=True)
    with (job / ".job.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("job input fingerprint collision")
            return path

        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(video_bytes)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def _committed_job_input(job: Path, input_fingerprint: str) -> Path:
    """Resolve immutable inputs, or legacy inputs for manifests without them."""
    if input_fingerprint.startswith("sha256:"):
        versioned = _job_input_path(job, input_fingerprint)
        return verify_sha256_file(versioned, input_fingerprint)
    legacy = job / "input.mp4"
    return legacy


def read_job_manifest(root: Path, job_id: str | None = None) -> dict[str, Any]:
    path = _job_path(root, job_id) / "job.json" if job_id else Path(root) / "job.json"
    if not path.is_file():
        raise ValueError(f"job manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema", JOB_SCHEMA) != JOB_SCHEMA:
        raise ValueError("unsupported job manifest schema")
    return payload


def write_job_manifest(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.setdefault("schema", JOB_SCHEMA)
    payload.setdefault("job_id", Path(root).name)
    payload.setdefault("state", "created")
    payload.setdefault("attempts", {})
    payload.setdefault("lease", None)
    payload.setdefault("errors", [])
    path = Path(root) / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.with_name(".job.lock")).open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _write_job_manifest_unlocked(path, payload)


def _write_job_manifest_unlocked(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return dict(payload)


@contextmanager
def _job_lock(root: Path, job_id: str):
    job = _job_path(root, job_id)
    job.mkdir(parents=True, exist_ok=True)
    with (job / ".job.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield job


def acquire_job_lease(root: Path, job_id: str, *, owner: str, now: float | None = None,
                      lease_seconds: int = JOB_LEASE_SECONDS, stage: str | None = None,
                      input_fingerprint: str | None = None,
                      request_fingerprint: str | None = None) -> dict[str, Any]:
    job = _job_path(root, job_id)
    job.mkdir(parents=True, exist_ok=True)
    path = job / "job.json"
    lock_path = job / ".job.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("schema", JOB_SCHEMA) != JOB_SCHEMA:
                raise ValueError("unsupported job manifest schema")
        else:
            manifest = {"schema": JOB_SCHEMA, "job_id": job_id, "state": "created",
                        "attempts": {}, "lease": None, "errors": []}
        now = time.time() if now is None else now
        current = manifest.get("lease") or {}
        if current.get("owner") and float(current.get("expires_at", 0)) > now and current.get("owner") != owner:
            raise RuntimeError("active lease exists")
        attempts = dict(manifest.get("attempts") or {})
        stage = stage or manifest.get("stage", "tracking")
        attempts[stage] = int(attempts.get(stage, 0)) + 1
        manifest.update({"state": stage, "stage": stage, "attempts": attempts,
                         "lease": {"owner": owner, "expires_at": now + lease_seconds}})
        if input_fingerprint is not None:
            manifest["input_fingerprint"] = input_fingerprint
        if request_fingerprint is not None:
            manifest["request_fingerprint"] = request_fingerprint
        return _write_job_manifest_unlocked(path, manifest)


def record_job_error(root: Path, job_id: str, stage: str, error: Exception) -> dict[str, Any]:
    with _job_lock(root, job_id) as job:
        path = job / "job.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        errors = list(manifest.get("errors") or [])
        errors.append({"stage": stage, "error": f"{type(error).__name__}: {error}", "time": time.time()})
        manifest["errors"] = errors[-MAX_JOB_ERRORS:]
        manifest["state"] = "failed"
        manifest["lease"] = None
        return _write_job_manifest_unlocked(path, manifest)


def _stage_owner(worker_id: str | None) -> str:
    if worker_id:
        return worker_id
    for name in ("MODAL_TASK_ID", "MODAL_FUNCTION_CALL_ID", "MODAL_INVOCATION_ID", "MODAL_SANDBOX_ID"):
        value = os.getenv(name)
        if value:
            return f"modal:{name}:{value}"
    raise RuntimeError(
        "unique worker identity is required; pass worker_id in tests or provide a Modal task/invocation identity"
    )


def _stats_progress(message: str) -> None:
    print(f"[v2-stats] {message}", flush=True)


def _assert_lease_owner(manifest: Mapping[str, Any], owner: str, *, now: float | None = None) -> None:
    lease = manifest.get("lease") or {}
    if lease.get("owner") != owner or float(lease.get("expires_at", 0)) <= (time.time() if now is None else now):
        raise RuntimeError("lease ownership lost before finalization")


def _assert_lease_available(root: Path, job_id: str, owner: str) -> None:
    _assert_lease_owner(read_job_manifest(root, job_id), owner)


def _finalize_stage(root: Path, job_id: str, *, owner: str, publish, manifest):
    """Publish and update the job pointer while the lease check is held."""
    with _job_lock(root, job_id) as job:
        path = job / "job.json"
        current = json.loads(path.read_text(encoding="utf-8"))
        _assert_lease_owner(current, owner)
        published = publish()
        updates = manifest(published) if callable(manifest) else manifest
        current.update(updates)
        _write_job_manifest_unlocked(path, current)
        return published


def _record_job_error_if_owner(root: Path, job_id: str, stage: str, error: Exception, owner: str) -> None:
    try:
        with _job_lock(root, job_id) as job:
            path = job / "job.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            _assert_lease_owner(manifest, owner)
            errors = list(manifest.get("errors") or [])
            errors.append({"stage": stage, "error": f"{type(error).__name__}: {error}", "time": time.time()})
            manifest.update({"errors": errors[-MAX_JOB_ERRORS:], "state": "failed", "lease": None})
            _write_job_manifest_unlocked(path, manifest)
    except RuntimeError:
        # A newer worker owns the job; never let a stale worker clear its lease.
        return


def _find_matching_generation(root: Path, *, input_fingerprint: str,
                              request_fingerprint: str, tracking_generation: str | None = None) -> dict[str, Any] | None:
    for path in sorted((Path(root) / "generations").glob("*/manifest.json")):
        try:
            manifest = load_stage_generation(Path(root), path.parent.name)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (manifest.get("input_fingerprint"), manifest.get("request_fingerprint"),
                manifest.get("tracking_generation")) == (input_fingerprint, request_fingerprint, tracking_generation):
            return manifest
    return None

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "setuptools<81",
        "einops",
        "pycocotools",
        "Pillow",
        "psutil",
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.0.0",
        "huggingface_hub==1.7.1",
        "opencv-python-headless==4.10.0.84",
        "numpy",
        "scipy==1.14.1",
        "umap-learn==0.5.9.post2",
        "boxmot==22.0.0",
        "open_clip_torch==2.32.0",
        "timm==1.0.22",
        "inference-sdk==1.3.3",
        "supervision==0.29.1",
        "python-dotenv==1.2.2",
    )
    .pip_install(f"{SAM31_REPOSITORY}@{SAM31_COMMIT}")
    .env({
        "HF_HOME": f"{CACHE_DIR}/huggingface",
        "HF_HUB_CACHE": f"{CACHE_DIR}/huggingface/hub",
        "TRACKNET_ENABLED": "1",
        "TRACKNET_CHECKPOINT": TRACKNET_CHECKPOINT,
        "BALL_ALL_FRAMES": "1",
        "BBOX_SANITIZER_ENABLED": "1",
        "POSSESSION_ENGINE": "v3",
        "NUMBER_PREDICTOR_BACKEND": "roboflow-hosted",
        **({"PIPELINE_GIT_SHA": PIPELINE_GIT_SHA} if PIPELINE_GIT_SHA else {}),
    })
    .add_local_python_source("pipeline")
    .add_local_file("scripts/ball/tracknet.py", remote_path="/root/scripts/ball/tracknet.py")
    .add_local_file("scripts/ball/candidates.py", remote_path="/root/scripts/ball/candidates.py")
)

hf_secret = modal.Secret.from_name("scoutai-huggingface", required_keys=["HF_TOKEN"])
roboflow_secret = modal.Secret.from_name("scoutai-roboflow", required_keys=["ROBOFLOW_API_KEY"])


def check_sam31_imports(import_module=importlib.import_module) -> dict[str, Any]:
    failures: dict[str, str] = {}
    imported: list[str] = []
    for module_name in SAM31_SMOKE_MODULES:
        try:
            module = import_module(module_name)
            if module_name == "sam3.model_builder":
                builder = getattr(module, "build_sam3_multiplex_video_predictor", None)
                if not callable(builder):
                    raise TypeError("build_sam3_multiplex_video_predictor is not callable")
            imported.append(module_name)
        except Exception as error:
            failures[module_name] = f"{type(error).__name__}: {error}"
    return {"ok": not failures, "imported": imported, "failures": failures}


def run_sam31_dependency_smoke(import_checker=check_sam31_imports) -> dict[str, Any]:
    report = import_checker()
    if report["failures"]:
        failed_modules = ", ".join(sorted(report["failures"]))
        raise RuntimeError(f"SAM3.1 dependency smoke failed: {failed_modules}")
    return report


@app.function(image=image, timeout=60)
def sam31_dependency_smoke_test() -> dict[str, Any]:
    """Check the official SAM3.1 import boundary without loading model weights."""
    return run_sam31_dependency_smoke()


def validate_entrypoint_args(
    *, video_bytes: bytes, prompt: str, frame_start: int, frame_end: int
) -> None:
    if type(video_bytes) is not bytes or not video_bytes:
        raise ValueError("video_bytes must be non-empty bytes")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-blank string")
    if type(frame_start) is not int or type(frame_end) is not int:
        raise ValueError("frame range must use integer values")
    if frame_start < 0 or frame_end < frame_start:
        raise ValueError("frame range must be non-negative and ordered")


DEFAULT_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


def _artifact_size_limit(max_bytes: int | None) -> int:
    if max_bytes is None:
        raw = os.getenv("SAM31_MAX_ARTIFACT_BYTES")
        max_bytes = DEFAULT_MAX_ARTIFACT_BYTES if raw is None else int(raw)
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max artifact return size must be a positive integer")
    return max_bytes


def _artifact_selected(relative: str, allow_paths: list[str]) -> bool:
    return any(relative == entry or (entry.endswith("/") and relative.startswith(entry)) for entry in allow_paths)


def collect_artifacts(
    output_dir: Path,
    *,
    max_bytes: int | None = None,
    allow_paths: list[str] | None = None,
    max_file_bytes: int | None = None,
    skipped: list[dict[str, Any]] | None = None,
) -> dict[str, bytes]:
    root = output_dir.resolve()
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact path is a symlink: {path.relative_to(output_dir)}")
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"artifact path resolves outside output root: {path.relative_to(output_dir)}")
        files.append(path)
    if allow_paths is not None:
        # Selection happens before the size checks so unselected bulk (masks) is never counted.
        selected = []
        for path in files:
            relative = path.relative_to(output_dir).as_posix()
            if not _artifact_selected(relative, allow_paths):
                continue
            size = path.stat().st_size
            if max_file_bytes is not None and size > max_file_bytes:
                if skipped is not None:
                    skipped.append({"path": relative, "bytes": size})
                continue
            selected.append(path)
        files = selected
    total_bytes = sum(path.stat().st_size for path in files)
    limit = _artifact_size_limit(max_bytes)
    if total_bytes > limit:
        raise ValueError(
            f"artifact return size {total_bytes} exceeds configured artifact return limit {limit}"
        )
    return {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in files
    }


def _artifact_destination(relative_path: str, output_dir: Path) -> Path:
    if not isinstance(relative_path, str) or not relative_path or "\x00" in relative_path:
        raise ValueError(f"invalid artifact path: {relative_path!r}")
    if relative_path != relative_path.strip() or "\\" in relative_path:
        raise ValueError(f"invalid artifact path: {relative_path!r}")
    path = PurePosixPath(relative_path)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid artifact path: {relative_path!r}")
    root = output_dir.resolve()
    destination = (root / Path(*path.parts)).resolve()
    if destination == root or root not in destination.parents:
        raise ValueError(f"invalid artifact path: {relative_path!r}")
    return destination


def write_artifacts(artifacts: dict[str, bytes], output_dir: Path) -> None:
    destinations = [
        (_artifact_destination(relative_path, output_dir), data)
        for relative_path, data in artifacts.items()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for destination, data in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def _add_mask_references(rows: list[dict[str, Any]], masks: dict[str, Any]) -> None:
    for row in rows:
        key = str(row["observation_key"])
        if key in masks:
            row["mask_ref"] = f"masks/{_mask_filename(key)}"


def _add_clean_mask_references(rows: list[dict[str, Any]], masks: dict[str, Any]) -> None:
    for row in rows:
        key = str(row["observation_key"])
        if key in masks:
            row["clean_mask_ref"] = f"masks_clean/{_mask_filename(key)}"


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _video_metadata(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        return {
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def _load_video_frames(path: Path, frame_start: int, frame_end: int) -> dict[int, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        frames: dict[int, Any] = {}
        for frame_index in range(frame_start, frame_end + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not read video frame: {frame_index}")
            frames[frame_index] = frame.copy()
        return frames
    finally:
        capture.release()


def _run_v2_stats(video_path: Path, output_dir: Path, observations, *, masks=None, clean_masks=None,
                  config=None, number_predictor=None, number_detector=None, on_stage=None,
                  court_run_info=None):
    """Run stats from SAM3.1 rows; never construct a legacy player tracker."""
    config = config or PipelineConfig.from_env()
    kwargs = {
        "masks_by_observation": masks,
        "clean_masks_by_observation": clean_masks,
        "number_predictor": number_predictor,
        "number_detector": number_detector,
    }
    if on_stage is not None:
        kwargs["on_stage"] = on_stage
    if court_run_info is not None:
        kwargs["court_run_info"] = court_run_info
    return run_stats_from_v2_observations(video_path, config, output_dir, observations, **kwargs)


def _validate_possession_engine(value: str) -> str:
    if value not in {"v3", "v4", "v5"}:
        raise ValueError("--possession-engine must be 'v3', 'v4', or 'v5'")
    return value


def _validate_possession_only_engine(value: str) -> str:
    if value not in {"legacy", "v3", "v4", "v5"}:
        raise ValueError("--possession-engine must be 'legacy', 'v3', 'v4', or 'v5'")
    return value


def _session_objects(output: dict[str, Any]) -> list[dict[str, Any]]:
    ids = output.get("out_obj_ids")
    boxes = output.get("out_boxes_xywh")
    probabilities = output.get("out_probs")
    masks = output.get("out_binary_masks")
    if ids is None and boxes is None:
        values = (probabilities, masks)
        if all(value is None or _sequence_is_empty(value) for value in values):
            return []
    if ids is None or boxes is None:
        raise RuntimeError("SAM3.1 propagation response lacks object ids or boxes")
    try:
        object_count = len(ids)
        box_count = len(boxes)
    except TypeError as error:
        raise RuntimeError("SAM3.1 object ids and boxes must be sequences") from error
    if object_count != box_count:
        raise RuntimeError("SAM3.1 object ids and boxes have different cardinalities")
    for name, values in (("probabilities", probabilities), ("masks", masks)):
        if values is not None:
            try:
                value_count = len(values)
            except TypeError as error:
                raise RuntimeError(f"SAM3.1 {name} must be a sequence") from error
            if value_count != object_count:
                raise RuntimeError(f"SAM3.1 {name} have different cardinality")
    return [
        {
            "object_id": int(object_id),
            "bbox": list(box),
            "bbox_format": "xywh",
            "probability": None if probabilities is None else float(probabilities[index]),
            "mask": None if masks is None else masks[index],
        }
        for index, (object_id, box) in enumerate(zip(ids, boxes))
    ]


def _sequence_is_empty(value: Any) -> bool:
    try:
        return len(value) == 0
    except TypeError:
        return False


def _start_session(predictor: Any, video_path: Path) -> dict[str, Any]:
    model = predictor.model
    original_init_state = model.init_state
    init_state = inspect.signature(original_init_state)
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in init_state.parameters.values())
    request = {
        "type": "start_session",
        "resource_path": str(video_path),
        "offload_state_to_cpu": True,
        "offload_video_to_cpu": True,
        "async_loading_frames": False,
    }
    supported = set(init_state.parameters)
    request_kwargs = set(request) - {"type"}
    if accepts_kwargs or request_kwargs <= supported:
        return predictor.handle_request(request)

    def compatible_init_state(**kwargs: Any):
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
        return original_init_state(**filtered_kwargs)

    model.init_state = compatible_init_state
    try:
        return predictor.handle_request(request)
    finally:
        model.init_state = original_init_state


def _track_video(
    predictor: Any,
    video_path: Path,
    prompt: str,
    frame_start: int,
    frame_end: int,
    *,
    local_propagation: bool = False,
    on_progress=None,
):
    start = _start_session(predictor, video_path)
    session_id = start["session_id"]
    predictor.handle_request(
        {"type": "add_prompt", "session_id": session_id, "frame_index": frame_start, "text": prompt}
    )
    rows: list[dict[str, Any]] = []
    masks: dict[str, Any] = {}
    try:
        request = {"type": "propagate_in_video", "session_id": session_id}
        if local_propagation:
            request.update({"start_frame_index": frame_start, "max_frame_num_to_track": frame_end - frame_start + 1})
        stream = predictor.handle_stream_request(request)
        for output in stream:
            frame_index = int(output.get("frame_index", output.get("frame_idx", -1)))
            if not frame_start <= frame_index <= frame_end:
                continue
            objects = _session_objects(output.get("outputs", output))
            frame_rows = normalize_multiplex_output(
                frame_index, 0.0, objects, bbox_coordinate_space="normalized"
            )
            rows.extend(frame_rows)
            for row, obj in zip(frame_rows, objects):
                if obj["mask"] is not None:
                    masks[row["observation_key"]] = obj["mask"]
            if on_progress is not None:
                on_progress((frame_index - frame_start + 1) / (frame_end - frame_start + 1))
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    return rows, masks


def _track_segmented_video(
    predictor: Any,
    video_path: Path,
    prompt: str,
    frame_start: int,
    frame_end: int,
    work_dir: Path,
    *,
    window: int,
    overlap: int = 1,
    seam_min_score: float = DEFAULT_SEAM_MIN_SCORE,
    seam_ambiguity_margin: float = DEFAULT_SEAM_AMBIGUITY_MARGIN,
    seam_diagnostics: list[dict[str, Any]] | None = None,
    overflow_diagnostics: list[dict[str, Any]] | None = None,
    on_progress=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run independent bounded SAM sessions, then stitch their raw output."""

    all_rows: list[dict[str, Any]] = []
    all_masks: dict[str, Any] = {}
    segments = list(plan_segment_windows(frame_start, frame_end, window=window, overlap=overlap))
    for position, segment in enumerate(segments):
        segment_path = work_dir / f"segment-{segment.ordinal:04d}.mp4"
        extract_segment_video(video_path, segment_path, segment)
        # The temporary clip always starts at local frame zero.  This is also
        # what prevents a session from retaining the full source video.
        rows, masks = _track_video(
            predictor, segment_path, prompt, 0, segment.frame_count - 1,
            local_propagation=True,
        )
        rows = prune_segment_object_overflow(
            rows,
            diagnostics=overflow_diagnostics,
            max_objects=SAM31_MAX_NUM_OBJECTS,
            grace_max_objects=SAM31_SEGMENT_GRACE_MAX_OBJECTS,
            segment_index=segment.ordinal,
        )
        retained_keys = {str(row["observation_key"]) for row in rows}
        masks = {key: value for key, value in masks.items() if key in retained_keys}
        assert_segment_object_limit(rows, max_objects=SAM31_MAX_NUM_OBJECTS)
        remapped_rows, remapped_masks = remap_segment_rows(rows, masks, segment)
        all_rows.extend(remapped_rows)
        all_masks.update(remapped_masks)
        if on_progress is not None:
            on_progress((position + 1) / len(segments))
    stitched_rows, stitched_masks = stitch_segment_seams(
        all_rows,
        all_masks,
        min_score=seam_min_score,
        ambiguity_margin=seam_ambiguity_margin,
        diagnostics=seam_diagnostics,
    )
    return deduplicate_segment_observations(
        stitched_rows,
        stitched_masks,
        diagnostics=seam_diagnostics,
    )


def _build_run_info(
    *, court_counters: Mapping[str, Any], seam_diagnostics: list[dict[str, Any]] | None,
    overflow_diagnostics: list[dict[str, Any]] | None, segment_count: int | None,
    segment_overlap: int | None, acceptance_reasons, input_sha256: str | None,
) -> dict[str, Any]:
    """Counts-only run summary for the service; None means "not measured", never a guess."""
    tracking: dict[str, Any] = {
        "segments": segment_count, "overflow_dropped_ids": None, "seams": None, "seams_linked": None,
    }
    if segment_count is not None:
        overflow = overflow_diagnostics or []
        seams = [item for item in (seam_diagnostics or []) if item.get("event") in {"matched", "ambiguous"}]
        tracking.update({
            "overflow_dropped_ids": sum(
                len(item.get("dropped_raw_ids") or []) for item in overflow
                if item.get("event") == "segment_object_overflow"
            ),
            "seams": len(seams),
            "seams_linked": sum(1 for item in seams if item.get("event") == "matched"),
        })
    errors = court_counters.get("errors_by_type")
    run_info = {
        "court": {
            "frames_total": court_counters.get("frames_total"),
            "frames_projected": court_counters.get("frames_projected"),
            "frames_failed": court_counters.get("frames_failed"),
            "errors_by_type": dict(errors) if errors is not None else None,
        },
        "tracking": tracking,
        "acceptance_reasons": [str(reason) for reason in (acceptance_reasons or [])],
        "input_sha256": input_sha256,
        "segment_overlap": segment_overlap,
        "pipeline_git_sha": os.environ.get("PIPELINE_GIT_SHA") or None,
        "sam31_commit": SAM31_COMMIT,
    }
    return run_info


def _segmented_enabled(value: bool | None) -> bool:
    if value is not None:
        return bool(value)
    return os.getenv("SAM31_V2_SEGMENTED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _segment_setting(value: int | None, env_name: str, default: int) -> int:
    return default if value is None and os.getenv(env_name) is None else int(
        value if value is not None else os.environ[env_name]
    )


def _segment_float_setting(value: float | None, env_name: str, default: float) -> float:
    return default if value is None and os.getenv(env_name) is None else float(
        value if value is not None else os.environ[env_name]
    )


def _config_fingerprint(config: PipelineConfig) -> dict[str, Any]:
    values = asdict(config)
    values.pop("roboflow_api_key", None)
    return values


def _tracking_stage_fingerprint(
    prompt: str, frame_start: int, frame_end: int, segmented: bool | None,
    segment_window: int | None, segment_overlap: int | None,
    segment_seam_min_score: float | None, segment_seam_ambiguity_margin: float | None,
    config: PipelineConfig,
) -> str:
    return stage_fingerprint(
        "tracking", prompt, frame_start, frame_end,
        {
            "segmented": _segmented_enabled(segmented),
            "segment_window": _segment_setting(segment_window, "SAM31_V2_SEGMENT_WINDOW", 32),
            "segment_overlap": _segment_setting(segment_overlap, "SAM31_V2_SEGMENT_OVERLAP", 1),
            "seam_min_score": _segment_float_setting(segment_seam_min_score, "SAM31_V2_SEAM_MIN_SCORE", DEFAULT_SEAM_MIN_SCORE),
            "seam_ambiguity_margin": _segment_float_setting(segment_seam_ambiguity_margin, "SAM31_V2_SEAM_AMBIGUITY_MARGIN", DEFAULT_SEAM_AMBIGUITY_MARGIN),
            "max_num_objects": SAM31_MAX_NUM_OBJECTS,
            "segment_grace_max_objects": SAM31_SEGMENT_GRACE_MAX_OBJECTS,
            "reid_backend": os.getenv("REID_BACKEND", "osnet"),
            "team_embedder_backend": os.getenv("TEAM_EMBEDDER_BACKEND", "siglip"),
            "team_embedder_model_id": os.getenv("TEAM_EMBEDDER_MODEL_ID", "Marqo/marqo-fashionSigLIP"),
        },
        _config_fingerprint(config),
    )


# Bump to invalidate cached stats generations produced before numbers were wired in.
STATS_NUMBERS_WIRING_VERSION = "numbers-wired-v1"


def _numbers_enabled(config: PipelineConfig) -> bool:
    return bool(getattr(config, "number_evidence_enabled", True))


def _build_number_evidence_components(config: PipelineConfig, has_stable_players: bool = True):
    """Return (predictor, detector); client construction makes no network call."""
    if not (_numbers_enabled(config) and has_stable_players):
        return None, None
    return build_number_predictor(config), build_detector(config)


def _stats_stage_fingerprint(possession_engine: str | None, generation: str, config: PipelineConfig) -> str:
    values = ["stats", possession_engine or config.possession_engine, generation, _config_fingerprint(config)]
    if _numbers_enabled(config):
        # Appended only when enabled so disabled fingerprints stay identical to before.
        values.append(STATS_NUMBERS_WIRING_VERSION)
    return stage_fingerprint(*values)


def _possession_only_fingerprint(possession_engine: str, generation: str, config: PipelineConfig) -> str:
    return stage_fingerprint("possession-only", possession_engine, generation, _config_fingerprint(config))


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_possession_tracking_inputs(job: Path, generation: dict[str, Any]) -> tuple[list[dict], list[dict], dict, dict, float, int, int]:
    """Load only the immutable tracking package and an existing ball artifact."""
    root = job / "tracking" / "generations" / generation["generation"]
    package = json.loads((root / "tracking-manifest.json").read_text(encoding="utf-8"))
    declared = package.get("artifacts", {})
    tracks_ref = declared.get("tracks", {})
    observations = _load_jsonl_file(root / tracks_ref.get("key", "tracks.jsonl"))

    def masks(name: str) -> dict[str, Any]:
        group = declared.get(name, {})
        result = {}
        for key, reference in (group.get("references", {}) if isinstance(group, Mapping) else {}).items():
            path = root / reference["key"]
            import cv2
            value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if value is None or value.ndim != 2:
                raise ValueError(f"invalid {name} artifact")
            result[str(key)] = value
        return result

    ball_path = None
    ball_ref = declared.get("ball_tracks")
    if isinstance(ball_ref, Mapping) and isinstance(ball_ref.get("key"), str):
        ball_path = root / ball_ref["key"]
    if ball_path is None or not ball_path.is_file():
        for candidate in sorted((job / "stats" / "generations").glob("*/manifest.json")):
            try:
                stats = load_stage_generation(job / "stats", candidate.parent.name)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if stats.get("tracking_generation") != generation["generation"]:
                continue
            ref = stats.get("artifacts", {}).get("ball_tracks")
            if isinstance(ref, Mapping) and isinstance(ref.get("path"), str):
                candidate_path = candidate.parent / ref["path"]
                if candidate_path.is_file():
                    ball_path = candidate_path
                    break
    if ball_path is None or not ball_path.is_file():
        raise ValueError("validated tracking job has no persisted ball_tracks.jsonl")
    video = package.get("video", {})
    fps = float(video.get("fps") or 30.0)
    mask_values = masks("masks")
    clean_values = masks("clean_masks")
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if (not width or not height) and mask_values:
        height, width = next(iter(mask_values.values())).shape[:2]
    return observations, _load_jsonl_file(ball_path), mask_values, clean_values, fps, width, height


def _new_stage_metrics(stage: str) -> StageMetrics:
    return StageMetrics(stage, function_timeout_seconds=PIPELINE_TIMEOUT_SECONDS, container=PIPELINE_CONTAINER)


def _attempt_number(manifest: Mapping[str, Any], stage: str) -> int | None:
    value = (manifest.get("attempts") or {}).get(stage)
    return value if type(value) is int else None


def _distinct_frames(rows) -> int | None:
    try:
        return len({int(row["frame_index"]) for row in rows})
    except (KeyError, TypeError, ValueError):
        return None


PROGRESS_DICT_NAME = "scoutai-v2-progress"


def _progress_dict_sink(key: str):
    """Write each payload under `key`; a Dict failure must never fail the job."""
    progress_dict = None

    def sink(payload: dict[str, Any]) -> None:
        nonlocal progress_dict
        try:
            if progress_dict is None:
                progress_dict = modal.Dict.from_name(PROGRESS_DICT_NAME, create_if_missing=True)
            progress_dict[key] = payload
        except Exception as error:
            print(f"[v2] progress write failed: {type(error).__name__}", flush=True)

    return sink


def _video_summary(video_metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    summary: dict[str, Any] = {}
    fps = video_metadata.get("fps")
    frame_count = video_metadata.get("frame_count")
    if isinstance(fps, (int, float)) and fps > 0:
        summary["fps"] = float(fps)
    if isinstance(frame_count, int) and frame_count > 0:
        summary["frame_count"] = frame_count
    if "fps" in summary and "frame_count" in summary:
        summary["duration_seconds"] = summary["frame_count"] / summary["fps"]
    return summary or None


def _finish_stage_metrics(metrics: StageMetrics) -> bytes:
    """Serialize the record and log it; publishing time is excluded because the file is hashed into the generation."""
    record = metrics.build(gpu=detect_gpu())
    print(format_stage_line(record), flush=True)
    return metrics_artifact(record)


@app.cls(
    image=image,
    gpu="A100-80GB",
    # volumes={CACHE_DIR: model_cache, TRACKNET_VOLUME_DIR: tracknet_data}
    volumes=V2_VOLUMES,
    secrets=[hf_secret, roboflow_secret],
    timeout=PIPELINE_TIMEOUT_SECONDS,
)
class PipelineV2:
    @modal.enter()
    def load_models(self) -> None:
        """Keep container startup cheap; stages initialize models on demand."""
        self._models_loaded = True

    def _ensure_predictor(self) -> Any:
        if hasattr(self, "predictor"):
            return self.predictor
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        self.predictor = build_sam3_multiplex_video_predictor(
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
        )
        model_cache.commit()
        return self.predictor

    def _ensure_seams(self) -> Any:
        if hasattr(self, "seams"):
            return self.seams
        from pipeline.reid import build_embedder
        from pipeline.runner_v2 import (
            ConservativeStableIdentityResolver,
            RunnerV2Seams,
            RuntimeIdentityAuthority,
        )
        from pipeline.team_embedder import build_team_embedder

        self.seams = RunnerV2Seams(
            identity=ConservativeStableIdentityResolver(),
            reid=build_embedder(
                os.getenv("REID_BACKEND", "osnet"),
                weights=f"{CACHE_DIR}/reid/osnet_x1_0_msmt17.pt",
            ),
            team_embedder=build_team_embedder(
                os.getenv("TEAM_EMBEDDER_BACKEND", "siglip"),
                model_id=os.getenv("TEAM_EMBEDDER_MODEL_ID", "Marqo/marqo-fashionSigLIP"),
            ),
            identity_authority=RuntimeIdentityAuthority(),
        )
        model_cache.commit()
        return self.seams

    def _job_dir(self, job_id: str) -> Path:
        return _job_path(Path(V2_JOB_VOLUME_DIR), job_id)

    def _track_stage(
        self, *, job_id: str, video_bytes: bytes, prompt: str = "basketball player on the court",
        frame_start: int = 60, frame_end: int = 130, segmented: bool | None = None,
        segment_window: int | None = None, segment_overlap: int | None = None,
        segment_seam_min_score: float | None = None,
        segment_seam_ambiguity_margin: float | None = None,
        worker_id: str | None = None,
        return_payload: bool = False,
    ) -> dict[str, Any]:
        """Persist tracking evidence once; retries return the matching generation."""
        validate_entrypoint_args(video_bytes=video_bytes, prompt=prompt,
                                 frame_start=frame_start, frame_end=frame_end)
        metrics = _new_stage_metrics("tracking")
        job = self._job_dir(job_id)
        job.mkdir(parents=True, exist_ok=True)
        input_fingerprint = f"sha256:{hashlib.sha256(video_bytes).hexdigest()}"
        config = PipelineConfig.from_env()
        request_fingerprint = _tracking_stage_fingerprint(
            prompt, frame_start, frame_end, segmented, segment_window, segment_overlap,
            segment_seam_min_score, segment_seam_ambiguity_margin, config,
        )
        # job.json holds the latest stage's fingerprint, so match on the published tracking generations.
        existing = _find_matching_generation(job / "tracking", input_fingerprint=input_fingerprint,
                                             request_fingerprint=request_fingerprint)
        if existing is not None:
            return {"status": "tracked", "generation": existing["generation"], "fingerprint": input_fingerprint}
        owner = _stage_owner(worker_id)
        acquired = False
        try:
            leased = acquire_job_lease(Path(V2_JOB_VOLUME_DIR), job_id, owner=owner, stage="tracking",
                                       input_fingerprint=input_fingerprint, request_fingerprint=request_fingerprint)
            acquired = True
            metrics.attempt = _attempt_number(leased, "tracking")
            job = self._job_dir(job_id)
            _persist_job_input(job, input_fingerprint, video_bytes)
            with tempfile.TemporaryDirectory(prefix="v2-track-") as work:
                root = Path(work)
                source = root / "input.mp4"
                source.write_bytes(video_bytes)
                segmented_run = _segmented_enabled(segmented)
                seams: list[dict[str, Any]] = []
                overflow_diagnostics: list[dict[str, Any]] = []
                with metrics.phase("model_load_seconds"):
                    predictor = self._ensure_predictor()
                if segmented_run:
                    with metrics.phase("tracking_seconds"):
                        rows, masks = _track_segmented_video(
                            predictor, source, prompt, frame_start, frame_end, root,
                            window=_segment_setting(segment_window, "SAM31_V2_SEGMENT_WINDOW", 32),
                            overlap=_segment_setting(segment_overlap, "SAM31_V2_SEGMENT_OVERLAP", 1),
                            seam_min_score=_segment_float_setting(segment_seam_min_score, "SAM31_V2_SEAM_MIN_SCORE", DEFAULT_SEAM_MIN_SCORE),
                            seam_ambiguity_margin=_segment_float_setting(segment_seam_ambiguity_margin, "SAM31_V2_SEAM_AMBIGUITY_MARGIN", DEFAULT_SEAM_AMBIGUITY_MARGIN),
                            seam_diagnostics=seams,
                            overflow_diagnostics=overflow_diagnostics,
                        )
                else:
                    with metrics.phase("tracking_seconds"):
                        rows, masks = _track_video(predictor, source, prompt, frame_start, frame_end)
                _add_mask_references(rows, masks)
                if getattr(self, "_models_loaded", False):
                    with metrics.phase("model_load_seconds"):
                        self._ensure_seams()
                with metrics.phase("frame_load_seconds"):
                    frames = _load_video_frames(source, frame_start, frame_end)
                runner_kwargs = {
                    "frames_by_index": frames,
                    "masks_by_observation": masks,
                    "bbox_sanitizer_config": config,
                }
                if hasattr(self, "seams"):
                    runner_kwargs["seams"] = self.seams
                with metrics.phase("identity_seconds"):
                    tracking_result = run_v2(rows, **runner_kwargs)
                if segmented_run:
                    tracking_result.runtime_diagnostics = {
                        **(getattr(tracking_result, "runtime_diagnostics", {}) or {}),
                        "seam_diagnostics": seams,
                        "segment_overflow_diagnostics": overflow_diagnostics,
                    }
                clean_masks = getattr(tracking_result, "cleaned_masks_by_observation", {})
                _add_clean_mask_references(tracking_result.observations, clean_masks)
                package = root / "package"
                artifact_config = {"prompt": prompt, "frame_start": frame_start, "frame_end": frame_end}
                if segmented_run:
                    artifact_config.update({
                        "sam31_segmented": True,
                        "sam31_segment_window": _segment_setting(segment_window, "SAM31_V2_SEGMENT_WINDOW", 32),
                        "sam31_segment_overlap": _segment_setting(segment_overlap, "SAM31_V2_SEGMENT_OVERLAP", 1),
                        "sam31_max_num_objects": SAM31_MAX_NUM_OBJECTS,
                        "sam31_segment_grace_max_objects": SAM31_SEGMENT_GRACE_MAX_OBJECTS,
                        "sam31_seam_min_score": _segment_float_setting(segment_seam_min_score, "SAM31_V2_SEAM_MIN_SCORE", DEFAULT_SEAM_MIN_SCORE),
                        "sam31_seam_ambiguity_margin": _segment_float_setting(segment_seam_ambiguity_margin, "SAM31_V2_SEAM_AMBIGUITY_MARGIN", DEFAULT_SEAM_AMBIGUITY_MARGIN),
                        "sam31_seam_diagnostics": seams,
                        "sam31_segment_overflow_diagnostics": overflow_diagnostics,
                    })
                sanitizer_metadata = getattr(tracking_result, "runtime_diagnostics", {}).get("bbox_sanitizer", {})
                if sanitizer_metadata.get("corrected_count", 0) or sanitizer_metadata.get("unresolved_count", 0):
                    artifact_config["bbox_sanitizer"] = sanitizer_metadata
                with metrics.phase("artifact_write_seconds"):
                    package_artifacts(package, tracking_result.observations, masks, clean_masks=clean_masks,
                                      input_fingerprint=input_fingerprint, frame_bounds=[frame_start, frame_end],
                                      video_metadata=_video_metadata(source),
                                      prompts=[prompt], config=artifact_config,
                                      acceptance=getattr(tracking_result, "acceptance", {}),
                                      acceptance_reasons=getattr(tracking_result, "acceptance_reasons", []))
                artifacts = {
                    ("tracking-manifest.json" if path.name == "manifest.json" else path.relative_to(package).as_posix()): path
                    for path in package.rglob("*") if path.is_file()
                }
                metrics.set_frames(requested=frame_end - frame_start + 1,
                                   processed=_distinct_frames(tracking_result.observations))
                # Tracking runs SAM, ReID and the team embedder locally; no hosted model is called.
                metrics.set_api_calls(roboflow_shot_detection_frames=0, roboflow_court_detection_frames=0,
                                      roboflow_number_detector_frames=0, roboflow_ocr_crops=0)
                artifacts[METRICS_FILENAME] = _finish_stage_metrics(metrics)
                generation = _finalize_stage(
                    Path(V2_JOB_VOLUME_DIR), job_id, owner=owner,
                    publish=lambda: publish_stage_generation(
                        job / "tracking", stage="tracking",
                        input_fingerprint=input_fingerprint,
                        request_fingerprint=request_fingerprint, artifacts=artifacts,
                        frame_bounds=[frame_start, frame_end],
                    ),
                    manifest=lambda published: {
                        "state": "tracked", "stage": "tracking",
                        "tracking_generation": published["generation"], "lease": None,
                    },
                )
            result = {"status": "tracked", "generation": generation["generation"], "fingerprint": input_fingerprint}
            if return_payload:
                result["tracking_payload"] = {
                    "observations": tracking_result.observations,
                    "masks_by_observation": masks,
                    "clean_masks_by_observation": clean_masks,
                }
            return result
        except Exception as error:
            if acquired:
                _record_job_error_if_owner(Path(V2_JOB_VOLUME_DIR), job_id, "tracking", error, owner)
            raise

    @modal.method()
    def track_stage(
        self, *, job_id: str, video_bytes: bytes, prompt: str = "basketball player on the court",
        frame_start: int = 60, frame_end: int = 130, segmented: bool | None = None,
        segment_window: int | None = None, segment_overlap: int | None = None,
        segment_seam_min_score: float | None = None,
        segment_seam_ambiguity_margin: float | None = None,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        return self._track_stage(
            job_id=job_id, video_bytes=video_bytes, prompt=prompt,
            frame_start=frame_start, frame_end=frame_end, segmented=segmented,
            segment_window=segment_window, segment_overlap=segment_overlap,
            segment_seam_min_score=segment_seam_min_score,
            segment_seam_ambiguity_margin=segment_seam_ambiguity_margin,
            worker_id=worker_id,
        )

    def _stats_stage(self, *, job_id: str, possession_engine: str | None = None,
                     worker_id: str | None = None, tracking_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Consume one validated tracking generation and publish bounded stats metadata."""
        metrics = _new_stage_metrics("stats")
        job = self._job_dir(job_id)
        started = time.perf_counter()
        _stats_progress("job manifest read start")
        manifest = read_job_manifest(job)
        _stats_progress(f"job manifest read complete: elapsed={time.perf_counter() - started:.2f}s")
        started = time.perf_counter()
        _stats_progress("stage generation validation/hash checks start")
        generation = load_stage_generation(job / "tracking", manifest.get("tracking_generation"))
        metrics.add_phase("generation_validation_seconds", time.perf_counter() - started)
        _stats_progress(
            "stage generation validation/hash checks complete: "
            f"elapsed={time.perf_counter() - started:.2f}s"
        )
        generation_dir = job / "tracking" / "generations" / generation["generation"]
        config = PipelineConfig.from_env(**({"possession_engine": possession_engine} if possession_engine else {}))
        stats_request = _stats_stage_fingerprint(possession_engine, generation["generation"], config)
        existing = _find_matching_generation(job / "stats", input_fingerprint=manifest["input_fingerprint"],
                                             request_fingerprint=stats_request,
                                             tracking_generation=generation["generation"])
        if existing is not None:
            return {"status": "complete", "generation": existing["generation"],
                    "tracking_generation": generation["generation"]}
        try:
            owner = _stage_owner(worker_id)
            acquired = False
            leased = acquire_job_lease(Path(V2_JOB_VOLUME_DIR), job_id, owner=owner, stage="stats",
                                       input_fingerprint=manifest["input_fingerprint"], request_fingerprint=stats_request)
            acquired = True
            metrics.attempt = _attempt_number(leased, "stats")
            with tempfile.TemporaryDirectory(prefix="v2-stats-") as work:
                output = Path(work) / "stats"
                _stats_progress("stats start")
                # Stable-id presence isn't cheap to read here; the runner skips when no rows qualify.
                number_predictor, number_detector = _build_number_evidence_components(config)
                result = run_stats_from_v2_generation(
                    _committed_job_input(job, generation["input_fingerprint"]), config,
                    job / "tracking", output,
                    generation, validated=True, tracking_payload=tracking_payload,
                    number_predictor=number_predictor, number_detector=number_detector,
                    metrics=metrics)
                bounds = generation.get("frame_bounds") or []
                metrics.set_frames(
                    requested=bounds[1] - bounds[0] + 1 if len(bounds) == 2 else None,
                    processed=getattr(getattr(result, "summary", None), "frames_processed", None),
                )
                # Written before the rglob so the file is published and hashed with the generation.
                (output / METRICS_FILENAME).write_bytes(_finish_stage_metrics(metrics))
                artifacts = {path.relative_to(output).as_posix(): path for path in output.rglob("*") if path.is_file()}
                published = _finalize_stage(
                    Path(V2_JOB_VOLUME_DIR), job_id, owner=owner,
                    publish=lambda: publish_stage_generation(
                        job / "stats", stage="stats",
                        input_fingerprint=manifest["input_fingerprint"],
                        request_fingerprint=stats_request,
                        tracking_generation=generation["generation"], artifacts=artifacts,
                        frame_bounds=generation["frame_bounds"],
                    ),
                    manifest=lambda published: {
                        "state": "complete", "stats_generation": published["generation"], "lease": None,
                    },
                )
            return {"status": "complete", "generation": published["generation"], "tracking_generation": generation["generation"]}
        except Exception as error:
            if acquired:
                _record_job_error_if_owner(Path(V2_JOB_VOLUME_DIR), job_id, "stats", error, owner)
            raise

    def _possession_only_stage(self, *, job_id: str, possession_engine: str,
                               worker_id: str | None = None) -> dict[str, Any]:
        engine = _validate_possession_only_engine(possession_engine)
        metrics = _new_stage_metrics("possession")
        job = self._job_dir(job_id)
        manifest = read_job_manifest(job)
        with metrics.phase("generation_validation_seconds"):
            tracking = load_stage_generation(job / "tracking", manifest.get("tracking_generation"))
        config = PipelineConfig.from_env(possession_engine=engine)
        request = _possession_only_fingerprint(engine, tracking["generation"], config)
        existing = _find_matching_generation(
            job / "possession", input_fingerprint=manifest["input_fingerprint"],
            request_fingerprint=request, tracking_generation=tracking["generation"],
        )
        if existing is not None:
            return {"status": "complete", "generation": existing["generation"],
                    "tracking_generation": tracking["generation"], "engine": engine,
                    "fingerprint": request}
        owner = _stage_owner(worker_id)
        acquired = False
        try:
            leased = acquire_job_lease(Path(V2_JOB_VOLUME_DIR), job_id, owner=owner, stage="possession",
                                       input_fingerprint=manifest["input_fingerprint"], request_fingerprint=request)
            acquired = True
            metrics.attempt = _attempt_number(leased, "possession")
            observations, ball_rows, masks, clean_masks, fps, width, height = _load_possession_tracking_inputs(job, tracking)
            with tempfile.TemporaryDirectory(prefix="v2-possession-") as work:
                output = Path(work) / "possession"
                with metrics.phase("possession_seconds"):
                    run_possession_only_from_v2_artifacts(
                        output, observations, ball_rows, fps=fps, config=config,
                        masks_by_observation=masks, clean_masks_by_observation=clean_masks,
                        frame_width=width, frame_height=height,
                    )
                bounds = tracking.get("frame_bounds") or []
                metrics.set_frames(requested=bounds[1] - bounds[0] + 1 if len(bounds) == 2 else None,
                                   processed=_distinct_frames(observations))
                # No hosted model is called here; the possession engines work on persisted evidence.
                metrics.set_api_calls(roboflow_shot_detection_frames=0, roboflow_court_detection_frames=0,
                                      roboflow_number_detector_frames=0, roboflow_ocr_crops=0)
                (output / METRICS_FILENAME).write_bytes(_finish_stage_metrics(metrics))
                artifacts = {path.relative_to(output).as_posix(): path for path in output.rglob("*") if path.is_file()}
                published = _finalize_stage(
                    Path(V2_JOB_VOLUME_DIR), job_id, owner=owner,
                    publish=lambda: publish_stage_generation(
                        job / "possession", stage="possession",
                        input_fingerprint=manifest["input_fingerprint"], request_fingerprint=request,
                        tracking_generation=tracking["generation"], artifacts=artifacts,
                        frame_bounds=tracking["frame_bounds"], metadata={"engine": engine},
                    ),
                    manifest=lambda value: {
                        "state": "complete", "lease": None,
                        "possession_generations": {
                            **(read_job_manifest(job).get("possession_generations") or {}),
                            engine: value["generation"],
                        },
                    },
                )
            return {"status": "complete", "generation": published["generation"],
                    "tracking_generation": tracking["generation"], "engine": engine,
                    "fingerprint": request}
        except Exception as error:
            if acquired:
                _record_job_error_if_owner(Path(V2_JOB_VOLUME_DIR), job_id, "possession", error, owner)
            raise

    @modal.method()
    def possession_only_stage(self, *, job_id: str, possession_engine: str = "v5",
                               worker_id: str | None = None) -> dict[str, Any]:
        return self._possession_only_stage(job_id=job_id, possession_engine=possession_engine, worker_id=worker_id)

    @modal.method()
    def stats_stage(self, *, job_id: str, possession_engine: str | None = None,
                    worker_id: str | None = None) -> dict[str, Any]:
        return self._stats_stage(job_id=job_id, possession_engine=possession_engine, worker_id=worker_id)

    @modal.method()
    def process(
        self,
        *,
        video_bytes: bytes,
        prompt: str = "basketball player on the court",
        frame_start: int = 60,
        frame_end: int = 130,
        return_artifacts: bool = False,
        return_diagnostics: bool = False,
        max_artifact_bytes: int | None = None,
        segmented: bool | None = None,
        segment_window: int | None = None,
        segment_overlap: int | None = None,
        segment_seam_min_score: float | None = None,
        segment_seam_ambiguity_margin: float | None = None,
        possession_engine: str | None = None,
        staged: bool = False,
        job_id: str | None = None,
        possession_only: bool = False,
        progress_key: str | None = None,
        return_artifact_paths: list[str] | None = None,
        return_artifact_max_file_bytes: int | None = None,
    ) -> dict[str, Any]:
        if possession_only:
            if not job_id:
                raise ValueError("job_id is required for possession-only execution")
            if staged:
                raise ValueError("--possession-only is mutually exclusive with --staged")
            return self._possession_only_stage(
                job_id=job_id, possession_engine=_validate_possession_only_engine(possession_engine or "v5")
            )
        if staged:
            if not job_id:
                raise ValueError("job_id is required for staged execution")
            tracking = self._track_stage(video_bytes=video_bytes, prompt=prompt, frame_start=frame_start,
                                         frame_end=frame_end, job_id=job_id, segmented=segmented,
                                         segment_window=segment_window, segment_overlap=segment_overlap,
                                         segment_seam_min_score=segment_seam_min_score,
                                         segment_seam_ambiguity_margin=segment_seam_ambiguity_margin,
                                         return_payload=True)
            return self._stats_stage(job_id=job_id, possession_engine=possession_engine,
                                     tracking_payload=tracking.get("tracking_payload"))
        if possession_engine is not None:
            _validate_possession_engine(possession_engine)
        if return_artifact_max_file_bytes is not None and (
            type(return_artifact_max_file_bytes) is not int or return_artifact_max_file_bytes <= 0
        ):
            raise ValueError("return_artifact_max_file_bytes must be a positive integer")
        validate_entrypoint_args(
            video_bytes=video_bytes,
            prompt=prompt,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        with tempfile.TemporaryDirectory(prefix="sam31-v2-") as work_dir:
            root = Path(work_dir)
            video_path = root / "input.mp4"
            output_dir = root / "artifacts"
            video_path.write_bytes(video_bytes)
            started = time.perf_counter()
            court_counters: dict[str, Any] = {}
            direct_metrics = _new_stage_metrics("direct")
            reporter = ProgressReporter(
                _progress_dict_sink(progress_key) if progress_key is not None else None,
                run_id=uuid.uuid4().hex,
            )
            print("[v2] progress=0")
            try:
                segmented_run = _segmented_enabled(segmented)
                reporter.stage("model_load", 0.0)
                if getattr(self, "_models_loaded", False):
                    self._ensure_seams()
                predictor = self._ensure_predictor()
                reporter.stage("model_load", 1.0)
                reporter.stage("tracking", 0.0)
                tracking_progress = {}
                if progress_key is not None:
                    tracking_progress["on_progress"] = lambda fraction: reporter.stage("tracking", fraction)
                seam_diagnostics: list[dict[str, Any]] = []
                overflow_diagnostics: list[dict[str, Any]] = []
                if segmented_run:
                    rows, masks = _track_segmented_video(
                        predictor, video_path, prompt, frame_start, frame_end, root,
                        window=_segment_setting(segment_window, "SAM31_V2_SEGMENT_WINDOW", 32),
                        overlap=_segment_setting(segment_overlap, "SAM31_V2_SEGMENT_OVERLAP", 1),
                        seam_min_score=_segment_float_setting(segment_seam_min_score, "SAM31_V2_SEAM_MIN_SCORE", DEFAULT_SEAM_MIN_SCORE),
                        seam_ambiguity_margin=_segment_float_setting(segment_seam_ambiguity_margin, "SAM31_V2_SEAM_AMBIGUITY_MARGIN", DEFAULT_SEAM_AMBIGUITY_MARGIN),
                        seam_diagnostics=seam_diagnostics,
                        overflow_diagnostics=overflow_diagnostics,
                        **tracking_progress,
                    )
                else:
                    rows, masks = _track_video(
                        predictor, video_path, prompt, frame_start, frame_end,
                        **tracking_progress,
                    )
                _add_mask_references(rows, masks)
                frames = _load_video_frames(video_path, frame_start, frame_end)
                reporter.stage("tracking", 1.0)
                reporter.stage("identity", 0.0)
                config_overrides = {}
                if possession_engine is not None:
                    config_overrides["possession_engine"] = _validate_possession_engine(possession_engine)
                config = PipelineConfig.from_env(**config_overrides)
                runner_kwargs = {"frames_by_index": frames, "masks_by_observation": masks}
                runner_kwargs["bbox_sanitizer_config"] = config
                if hasattr(self, "seams"):
                    runner_kwargs["seams"] = self.seams
                result = run_v2(rows, **runner_kwargs)
                reporter.stage("identity", 1.0)
                reporter.stage("packaging", 0.0)
                if segmented_run:
                    result.runtime_diagnostics = {
                        **(getattr(result, "runtime_diagnostics", {}) or {}),
                        "seam_diagnostics": seam_diagnostics,
                        "segment_overflow_diagnostics": overflow_diagnostics,
                    }
                clean_masks = getattr(result, "cleaned_masks_by_observation", {})
                _add_clean_mask_references(result.observations, clean_masks)
                video_metadata = _video_metadata(video_path)
                input_fingerprint = _file_fingerprint(video_path)
                artifact_config = {"prompt": prompt, "frame_start": frame_start, "frame_end": frame_end}
                if config.possession_engine in {"v3", "v4", "v5"}:
                    artifact_config["possession_engine"] = config.possession_engine
                if segmented_run:
                    artifact_config.update({
                        "sam31_segmented": True,
                        "sam31_segment_window": _segment_setting(segment_window, "SAM31_V2_SEGMENT_WINDOW", 32),
                        "sam31_segment_overlap": _segment_setting(segment_overlap, "SAM31_V2_SEGMENT_OVERLAP", 1),
                        "sam31_max_num_objects": SAM31_MAX_NUM_OBJECTS,
                        "sam31_segment_grace_max_objects": SAM31_SEGMENT_GRACE_MAX_OBJECTS,
                        "sam31_seam_min_score": _segment_float_setting(segment_seam_min_score, "SAM31_V2_SEAM_MIN_SCORE", DEFAULT_SEAM_MIN_SCORE),
                        "sam31_seam_ambiguity_margin": _segment_float_setting(segment_seam_ambiguity_margin, "SAM31_V2_SEAM_AMBIGUITY_MARGIN", DEFAULT_SEAM_AMBIGUITY_MARGIN),
                        "sam31_seam_diagnostics": seam_diagnostics,
                        "sam31_segment_overflow_diagnostics": overflow_diagnostics,
                    })
                sanitizer_metadata = getattr(result, "runtime_diagnostics", {}).get("bbox_sanitizer", {})
                if sanitizer_metadata.get("corrected_count", 0) or sanitizer_metadata.get("unresolved_count", 0):
                    artifact_config["bbox_sanitizer"] = sanitizer_metadata
                artifact_manifest = package_artifacts(
                    output_dir,
                    result.observations,
                    masks,
                    clean_masks=clean_masks,
                    input_fingerprint=input_fingerprint,
                    video_metadata=video_metadata,
                    frame_bounds=[frame_start, frame_end],
                    prompts=[prompt],
                    config=artifact_config,
                    benchmark={"wall_seconds": time.perf_counter() - started},
                    acceptance=result.acceptance,
                    acceptance_reasons=result.acceptance_reasons,
                )
                has_stable_players = any(row.get("stable_player_id") for row in result.observations)
                number_predictor, number_detector = _build_number_evidence_components(
                    config, has_stable_players
                )
                stats_kwargs = {
                    "masks": masks, "clean_masks": clean_masks, "config": config,
                    "number_predictor": number_predictor,
                    "number_detector": number_detector,
                    "court_run_info": court_counters,
                }
                if progress_key is not None:
                    stats_kwargs["on_stage"] = reporter.stage
                stats_result = _run_v2_stats(
                    video_path, output_dir / "stats", result.observations, **stats_kwargs
                )
                stats_payload = stats_result.to_dict()
                artifact_manifest["artifacts"]["legacy_stats"] = {
                    name: _artifact_ref(output_dir / "stats" / filename, f"stats/{filename}")
                    for name, filename in {
                        "results": "results.json",
                        "tracks": stats_payload["tracks_file"],
                         "teams": stats_payload["teams_file"],
                         "numbers": stats_payload["numbers_file"],
                         "number_predictions": "number_predictions.json" if stats_payload["numbers_file"] else None,
                         "number_prediction_cases": "number_prediction_cases.jsonl" if stats_payload["numbers_file"] else None,
                         "number_manifest": "number_manifest.json" if stats_payload["numbers_file"] else None,
                        "shot_events": stats_payload["shot_events_file"],
                        "shot_evidence": stats_payload.get("shot_evidence_file"),
                        "possession_events": stats_payload.get("possession_events_file"),
                        "possession_transitions": stats_payload.get("possession_transitions_file"),
                        "ball_tracks": stats_payload.get("ball_tracks_file"),
                        "projected_tracks": stats_payload.get("projected_tracks_file"),
                         "projected_shots": stats_payload.get("projected_shots_file"),
                         "projected_shots_image": stats_payload.get("projected_shots_image_file"),
                         "annotated_video": stats_payload.get("annotated_video_file"),
                         "possession_v4_diagnostics": stats_payload.get("possession_v4_diagnostics_file"),
                     }.items()
                    if filename is not None
                }
                artifact_manifest["legacy_stats_results"] = "stats/results.json"
                artifact_manifest["artifacts"]["manifest"]["sha256"] = _manifest_hash(artifact_manifest)
                (output_dir / "manifest.json").write_text(
                    json.dumps(artifact_manifest, indent=2, sort_keys=True), encoding="utf-8"
                )
            except Exception as error:
                print(f"[v2] error={type(error).__name__}: {error}", flush=True)
                raise
            print("[v2] progress=100")
            print("[v2] overall V2 complete", flush=True)
            _finish_stage_metrics(direct_metrics)
            reporter.stage("packaging", 1.0)
            payload = {
                "acceptance": result.acceptance,
                "timings": reporter.timings,
                "gpu": detect_gpu(),
                "frames": {"requested": frame_end - frame_start + 1, "processed": _distinct_frames(rows)},
            }
            video_summary = _video_summary(video_metadata)
            if video_summary is not None:
                payload["video"] = video_summary
            used_overlap = _segment_setting(segment_overlap, "SAM31_V2_SEGMENT_OVERLAP", 1) if segmented_run else None
            payload["run_info"] = _build_run_info(
                court_counters=court_counters,
                seam_diagnostics=seam_diagnostics, overflow_diagnostics=overflow_diagnostics,
                segment_count=len(list(plan_segment_windows(
                    frame_start, frame_end,
                    window=_segment_setting(segment_window, "SAM31_V2_SEGMENT_WINDOW", 32), overlap=used_overlap,
                ))) if segmented_run else None,
                segment_overlap=used_overlap, acceptance_reasons=result.acceptance_reasons,
                input_sha256=input_fingerprint.removeprefix("sha256:"),
            )
            if return_diagnostics:
                payload["debug_diagnostics"] = result.runtime_diagnostics
            if return_artifacts:
                skipped: list[dict[str, Any]] = []
                payload["artifacts"] = collect_artifacts(
                    output_dir,
                    max_bytes=max_artifact_bytes,
                    allow_paths=return_artifact_paths,
                    max_file_bytes=return_artifact_max_file_bytes,
                    skipped=skipped,
                )
                if skipped:
                    payload["artifacts_skipped"] = skipped
            return payload


def _validate_local_entrypoint_mode(
    *, video_path: str, job_id: str, staged: bool, stats_only: bool, tracking_only: bool,
    possession_only: bool = False,
) -> None:
    selected_modes = sum((staged, stats_only, tracking_only, possession_only))
    if selected_modes > 1:
        raise ValueError("--staged, --stats-only, --tracking-only, and --possession-only are mutually exclusive")
    if selected_modes and not job_id:
        raise ValueError("--job-id is required with --staged, --stats-only, --tracking-only, or --possession-only")
    if job_id and not selected_modes:
        raise ValueError(
            "--job-id requires --staged, --stats-only, --tracking-only, or --possession-only"
        )
    if (stats_only or possession_only) and video_path:
        raise ValueError("--video-path cannot be used with --stats-only or --possession-only")
    if not stats_only and not possession_only and not video_path:
        raise ValueError("--video-path is required")


@app.local_entrypoint()
def main(
    video_path: str = "",
    output_dir: str = "",
    prompt: str = "basketball player on the court",
    frame_start: int = 60,
    frame_end: int = 130,
    max_artifact_bytes: int = 0,
    return_diagnostics: bool = False,
    segmented: bool | None = None,
    segment_window: int | None = None,
    segment_overlap: int | None = None,
    segment_seam_min_score: float | None = None,
    segment_seam_ambiguity_margin: float | None = None,
    possession_engine: str = "v3",
    staged: bool = False,
    stats_only: bool = False,
    tracking_only: bool = False,
    possession_only: bool = False,
    job_id: str = "",
) -> None:
    _validate_local_entrypoint_mode(
        video_path=video_path, job_id=job_id, staged=staged, stats_only=stats_only,
        tracking_only=tracking_only, possession_only=possession_only,
    )
    (_validate_possession_only_engine if possession_only else _validate_possession_engine)(possession_engine)

    pipeline = PipelineV2()
    # Client-side wall time: unlike the container-side stage numbers it includes cold start and model load.
    wall_started = time.perf_counter()
    if possession_only:
        payload = pipeline.possession_only_stage.remote(
            job_id=job_id, possession_engine=possession_engine
        )
    elif stats_only:
        payload = pipeline.stats_stage.remote(
            job_id=job_id, possession_engine=possession_engine
        )
    else:
        source = Path(video_path)
        if not source.is_file():
            raise SystemExit(f"Video not found: {source}")
        video_bytes = source.read_bytes()
        validate_entrypoint_args(
            video_bytes=video_bytes,
            prompt=prompt,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        if tracking_only:
            payload = pipeline.track_stage.remote(
                job_id=job_id,
                video_bytes=video_bytes,
                prompt=prompt,
                frame_start=frame_start,
                frame_end=frame_end,
                segmented=segmented,
                segment_window=segment_window,
                segment_overlap=segment_overlap,
                segment_seam_min_score=segment_seam_min_score,
                segment_seam_ambiguity_margin=segment_seam_ambiguity_margin,
            )
        else:
            payload = pipeline.process.remote(
                video_bytes=video_bytes,
                prompt=prompt,
                frame_start=frame_start,
                frame_end=frame_end,
                return_artifacts=bool(output_dir),
                return_diagnostics=return_diagnostics,
                segmented=segmented,
                segment_window=segment_window,
                segment_overlap=segment_overlap,
                segment_seam_min_score=segment_seam_min_score,
                segment_seam_ambiguity_margin=segment_seam_ambiguity_margin,
                possession_engine=possession_engine,
                max_artifact_bytes=max_artifact_bytes or None,
                staged=staged,
                job_id=job_id or None,
                possession_only=possession_only,
            )
    artifacts = payload.pop("artifacts", None)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if artifacts is not None:
        write_artifacts(artifacts, Path(output_dir))
    print(format_job_line(job_id or None, time.perf_counter() - wall_started), flush=True)
