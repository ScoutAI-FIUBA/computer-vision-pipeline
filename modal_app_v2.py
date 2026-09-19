"""Isolated SAM 3.1 Modal boundary with direct local artifact return."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

from pipeline.results import _mask_filename, package_artifacts
from pipeline.runner_v2 import run_v2
from pipeline.sam31_tracking import normalize_multiplex_output

SAM31_REPOSITORY = "git+https://github.com/facebookresearch/sam3.git"
CACHE_DIR = "/cache"

app = modal.App("scoutai-pipeline-v2")
model_cache = modal.Volume.from_name("scoutai-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.0.0",
        "huggingface_hub==1.7.1",
        "opencv-python-headless==4.10.0.84",
        "numpy",
        "scipy==1.18.0",
        "umap-learn==0.5.9.post2",
        "boxmot==22.0.0",
        "timm==1.0.22",
        "python-dotenv==1.2.2",
    )
    .pip_install(SAM31_REPOSITORY)
    .env({"HF_HOME": f"{CACHE_DIR}/huggingface", "HF_HUB_CACHE": f"{CACHE_DIR}/huggingface/hub"})
    .add_local_python_source("pipeline")
)

hf_secret = modal.Secret.from_name("scoutai-huggingface", required_keys=["HF_TOKEN"])


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


def collect_artifacts(output_dir: Path) -> dict[str, bytes]:
    return {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }


def write_artifacts(artifacts: dict[str, bytes], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, data in artifacts.items():
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def _add_mask_references(rows: list[dict[str, Any]], masks: dict[str, Any]) -> None:
    for row in rows:
        key = str(row["observation_key"])
        if key in masks:
            row["mask_ref"] = f"masks/{_mask_filename(key)}"


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


def _session_objects(output: dict[str, Any]) -> list[dict[str, Any]]:
    ids = output.get("out_obj_ids")
    boxes = output.get("out_boxes_xywh")
    probabilities = output.get("out_probs")
    masks = output.get("out_binary_masks")
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


def _track_video(predictor: Any, video_path: Path, prompt: str, frame_start: int, frame_end: int):
    start = predictor.handle_request({"type": "start_session", "resource_path": str(video_path)})
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
            objects = _session_objects(output)
            frame_rows = normalize_multiplex_output(frame_index, 0.0, objects)
            rows.extend(frame_rows)
            for row, obj in zip(frame_rows, objects):
                if obj["mask"] is not None:
                    masks[row["observation_key"]] = obj["mask"]
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    return rows, masks


@app.cls(image=image, gpu="L4", volumes={CACHE_DIR: model_cache}, secrets=[hf_secret], timeout=1800)
class PipelineV2:
    def _ensure_predictor(self) -> Any:
        if hasattr(self, "predictor"):
            return self.predictor
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        self.predictor = build_sam3_multiplex_video_predictor()
        model_cache.commit()
        return self.predictor

    @modal.method()
    def process(
        self,
        *,
        video_bytes: bytes,
        prompt: str = "basketball player",
        frame_start: int = 60,
        frame_end: int = 130,
        return_artifacts: bool = False,
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
            rows, masks = _track_video(
                self._ensure_predictor(), video_path, prompt, frame_start, frame_end
            )
            _add_mask_references(rows, masks)
            result = run_v2(rows)
            package_artifacts(
                output_dir,
                result.observations,
                masks,
                input_fingerprint=_file_fingerprint(video_path),
                video_metadata=_video_metadata(video_path),
                frame_bounds=[frame_start, frame_end],
                prompts=[prompt],
                config={"prompt": prompt, "frame_start": frame_start, "frame_end": frame_end},
                benchmark={"wall_seconds": time.perf_counter() - started},
                acceptance=result.acceptance,
            )
            payload = {"acceptance": result.acceptance}
            if return_artifacts:
                payload["artifacts"] = collect_artifacts(output_dir)
            return payload


@app.local_entrypoint()
def main(
    video_path: str = "",
    output_dir: str = "",
    prompt: str = "basketball player",
    frame_start: int = 60,
    frame_end: int = 130,
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
    )
    artifacts = payload.pop("artifacts", None)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if artifacts is not None:
        write_artifacts(artifacts, Path(output_dir))
