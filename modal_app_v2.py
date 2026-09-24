"""Isolated SAM 3.1 Modal boundary with direct local artifact return."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import tempfile
import time
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import modal

from pipeline.results import (
    _artifact_ref,
    _manifest_hash,
    _mask_filename,
    package_artifacts,
)
from pipeline.config import PipelineConfig
from pipeline.runner import run_stats_from_v2_observations
from pipeline.runner_v2 import run_v2
from pipeline.sam31_tracking import normalize_multiplex_output

SAM31_REPOSITORY = "git+https://github.com/facebookresearch/sam3.git"
SAM31_COMMIT = "2345a4a"
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


def collect_artifacts(output_dir: Path, *, max_bytes: int | None = None) -> dict[str, bytes]:
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


def _run_v2_stats(video_path: Path, output_dir: Path, observations):
    """Run stats from SAM3.1 rows; never construct a legacy player tracker."""
    return run_stats_from_v2_observations(
        video_path, PipelineConfig.from_env(), output_dir, observations
    )


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


def _track_video(predictor: Any, video_path: Path, prompt: str, frame_start: int, frame_end: int):
    start = _start_session(predictor, video_path)
    session_id = start["session_id"]
    predictor.handle_request(
        {"type": "add_prompt", "session_id": session_id, "frame_index": frame_start, "text": prompt}
    )
    rows: list[dict[str, Any]] = []
    masks: dict[str, Any] = {}
    try:
        stream = predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        )
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
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    return rows, masks


@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={CACHE_DIR: model_cache, TRACKNET_VOLUME_DIR: tracknet_data},
    secrets=[hf_secret, roboflow_secret],
    timeout=1800,
)
class PipelineV2:
    @modal.enter()
    def load_models(self) -> None:
        """Warm SAM, ReID, and team models once per Modal container."""
        from pipeline.reid import build_embedder
        from pipeline.runner_v2 import (
            ConservativeStableIdentityResolver,
            RunnerV2Seams,
            RuntimeIdentityAuthority,
        )
        from pipeline.team_embedder import build_team_embedder

        self._ensure_predictor()
        reid_backend = os.getenv("REID_BACKEND", "osnet")
        team_backend = os.getenv("TEAM_EMBEDDER_BACKEND", "siglip")
        team_model_id = os.getenv("TEAM_EMBEDDER_MODEL_ID", "Marqo/marqo-fashionSigLIP")
        self.seams = RunnerV2Seams(
            identity=ConservativeStableIdentityResolver(),
            reid=build_embedder(reid_backend, weights=f"{CACHE_DIR}/reid/osnet_x1_0_msmt17.pt"),
            team_embedder=build_team_embedder(team_backend, model_id=team_model_id),
            identity_authority=RuntimeIdentityAuthority(),
        )
        model_cache.commit()

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
    ) -> dict[str, Any]:
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
            print("[v2] progress=0")
            try:
                rows, masks = _track_video(
                    self._ensure_predictor(), video_path, prompt, frame_start, frame_end
                )
                _add_mask_references(rows, masks)
                frames = _load_video_frames(video_path, frame_start, frame_end)
                config = PipelineConfig.from_env()
                runner_kwargs = {"frames_by_index": frames, "masks_by_observation": masks}
                runner_kwargs["bbox_sanitizer_config"] = config
                if hasattr(self, "seams"):
                    runner_kwargs["seams"] = self.seams
                result = run_v2(rows, **runner_kwargs)
                clean_masks = getattr(result, "cleaned_masks_by_observation", {})
                _add_clean_mask_references(result.observations, clean_masks)
                video_metadata = _video_metadata(video_path)
                artifact_config = {"prompt": prompt, "frame_start": frame_start, "frame_end": frame_end}
                sanitizer_metadata = getattr(result, "runtime_diagnostics", {}).get("bbox_sanitizer", {})
                if sanitizer_metadata.get("corrected_count", 0) or sanitizer_metadata.get("unresolved_count", 0):
                    artifact_config["bbox_sanitizer"] = sanitizer_metadata
                artifact_manifest = package_artifacts(
                    output_dir,
                    result.observations,
                    masks,
                    clean_masks=clean_masks,
                    input_fingerprint=_file_fingerprint(video_path),
                    video_metadata=video_metadata,
                    frame_bounds=[frame_start, frame_end],
                    prompts=[prompt],
                    config=artifact_config,
                    benchmark={"wall_seconds": time.perf_counter() - started},
                    acceptance=result.acceptance,
                    acceptance_reasons=result.acceptance_reasons,
                )
                stats_result = _run_v2_stats(video_path, output_dir / "stats", result.observations)
                stats_payload = stats_result.to_dict()
                artifact_manifest["artifacts"]["legacy_stats"] = {
                    name: _artifact_ref(output_dir / "stats" / filename, f"stats/{filename}")
                    for name, filename in {
                        "results": "results.json",
                        "tracks": stats_payload["tracks_file"],
                        "teams": stats_payload["teams_file"],
                        "numbers": stats_payload["numbers_file"],
                        "shot_events": stats_payload["shot_events_file"],
                        "shot_evidence": stats_payload.get("shot_evidence_file"),
                        "possession_events": stats_payload.get("possession_events_file"),
                        "possession_transitions": stats_payload.get("possession_transitions_file"),
                        "ball_tracks": stats_payload.get("ball_tracks_file"),
                        "projected_tracks": stats_payload.get("projected_tracks_file"),
                        "projected_shots": stats_payload.get("projected_shots_file"),
                        "projected_shots_image": stats_payload.get("projected_shots_image_file"),
                        "annotated_video": stats_payload.get("annotated_video_file"),
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
            payload = {"acceptance": result.acceptance}
            if return_diagnostics:
                payload["debug_diagnostics"] = result.runtime_diagnostics
            if return_artifacts:
                payload["artifacts"] = collect_artifacts(output_dir, max_bytes=max_artifact_bytes)
            return payload


@app.local_entrypoint()
def main(
    video_path: str = "",
    output_dir: str = "",
    prompt: str = "basketball player on the court",
    frame_start: int = 60,
    frame_end: int = 130,
    max_artifact_bytes: int = 0,
    return_diagnostics: bool = False,
) -> None:
    if not video_path:
        raise ValueError("--video-path is required")
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
    payload = PipelineV2().process.remote(
        video_bytes=video_bytes,
        prompt=prompt,
        frame_start=frame_start,
        frame_end=frame_end,
        return_artifacts=bool(output_dir),
        return_diagnostics=return_diagnostics,
        max_artifact_bytes=max_artifact_bytes or None,
    )
    artifacts = payload.pop("artifacts", None)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if artifacts is not None:
        write_artifacts(artifacts, Path(output_dir))
