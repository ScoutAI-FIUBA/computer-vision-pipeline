"""Isolated SAM 3.1 Modal boundary.

This wrapper only produces raw tracking evidence. Visualization remains a local
consumer of the returned artifact references; the remote job never encodes it.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import modal

from pipeline.results import _manifest_hash, package_artifacts
from pipeline.runner_v2 import run_v2
from pipeline.sam31_tracking import normalize_multiplex_output
from pipeline.storage import R2Storage

SAM31_REPOSITORY = "git+https://github.com/facebookresearch/sam3.git"
CACHE_DIR = "/cache"

app = modal.App("scoutai-pipeline-v2")
model_cache = modal.Volume.from_name("scoutai-model-cache", create_if_missing=True)

# SAM 3.1 needs the upstream SAM 3 repository and PyTorch 2.7+; these pins are
# deliberately separate from the legacy SAM2 image. Existing V2 runner seams
# only require the dependencies below.
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
        "boto3==1.35.36",
        "python-dotenv==1.2.2",
    )
    .pip_install(SAM31_REPOSITORY)
    .env({"HF_HOME": f"{CACHE_DIR}/huggingface", "HF_HUB_CACHE": f"{CACHE_DIR}/huggingface/hub"})
    .add_local_python_source("pipeline")
)

hf_secret = modal.Secret.from_name("scoutai-huggingface", required_keys=["HF_TOKEN"])
r2_secret = modal.Secret.from_name("scoutai-r2")


def validate_entrypoint_args(
    *, video_bytes: bytes | None, r2_key: str | None, output_prefix: str
) -> None:
    if (video_bytes is None) == (r2_key is None):
        raise ValueError("Pass exactly one of r2_key or video_bytes")
    if not output_prefix.strip():
        raise ValueError("output_prefix is required for V2 artifacts")


def artifact_retrieval_response(manifest: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {
        "manifest": {
            "complete": bool(manifest.get("complete", False)),
            "key": manifest["artifacts"]["manifest"]["key"],
        },
        "artifact_keys": list(keys),
    }


def upload_artifacts(directory: Path, output_prefix: str) -> tuple[list[str], list[str]]:
    try:
        return R2Storage().upload_directory(directory, output_prefix), []
    except Exception as error:  # noqa: BLE001 - preserve transfer failure in response
        return [], [f"{type(error).__name__}: {error}"]


def _session_objects(output: dict[str, Any]) -> list[dict[str, Any]]:
    ids = output.get("out_obj_ids")
    boxes = output.get("out_boxes_xywh")
    probabilities = output.get("out_probs")
    masks = output.get("out_binary_masks")
    if ids is None or boxes is None:
        raise RuntimeError("SAM3.1 propagation response lacks object ids or boxes")
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
    start = predictor.handle_request(
        {"type": "start_session", "resource_path": str(video_path)}
    )
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


@app.cls(image=image, gpu="L4", volumes={CACHE_DIR: model_cache}, secrets=[hf_secret, r2_secret], timeout=1800)
class PipelineV2:
    @modal.enter()
    def load_model(self) -> None:
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        self.predictor = build_sam3_multiplex_video_predictor()
        model_cache.commit()

    @modal.method()
    def process(
        self,
        *,
        video_bytes: bytes | None = None,
        r2_key: str | None = None,
        output_prefix: str,
        prompt: str = "basketball player",
        frame_start: int = 60,
        frame_end: int = 130,
    ) -> dict[str, Any]:
        validate_entrypoint_args(video_bytes=video_bytes, r2_key=r2_key, output_prefix=output_prefix)
        with tempfile.TemporaryDirectory(prefix="sam31-v2-") as work_dir:
            root = Path(work_dir)
            video_path = root / "input.mp4"
            if r2_key is not None:
                R2Storage().download(r2_key, video_path)
            else:
                video_path.write_bytes(video_bytes or b"")
            rows, masks = _track_video(self.predictor, video_path, prompt, frame_start, frame_end)
            result = run_v2(rows)
            output_dir = root / "artifacts"
            manifest = package_artifacts(
                output_dir,
                result.observations,
                masks,
                frame_bounds=[frame_start, frame_end],
                prompts=[prompt],
                acceptance=result.acceptance,
                output_prefix=output_prefix,
            )
            keys, transfer_errors = upload_artifacts(output_dir, output_prefix)
            if transfer_errors:
                manifest["transfer_errors"] = transfer_errors
                manifest["truncated"] = True
                manifest["complete"] = False
                manifest["artifacts"]["manifest"]["sha256"] = _manifest_hash(manifest)
                (output_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
                )
            response = artifact_retrieval_response(manifest, keys)
            response["transfer_errors"] = transfer_errors
            response["acceptance"] = result.acceptance
            return response


@app.local_entrypoint()
def main(
    video_path: str = "",
    r2_key: str = "",
    output_prefix: str = "",
    prompt: str = "basketball player",
    frame_start: int = 60,
    frame_end: int = 130,
) -> None:
    if bool(video_path) == bool(r2_key):
        raise ValueError("Pass exactly one of --video-path or --r2-key")
    prefix = output_prefix or os.getenv("SAM31_OUTPUT_PREFIX", "")
    validate_entrypoint_args(video_bytes=b"local" if video_path else None, r2_key=r2_key or None, output_prefix=prefix)
    payload = {"r2_key": r2_key or None, "output_prefix": prefix, "prompt": prompt, "frame_start": frame_start, "frame_end": frame_end}
    if video_path:
        payload["video_bytes"] = Path(video_path).read_bytes()
    response = PipelineV2().process.remote(**payload)
    print(json.dumps(response, indent=2, sort_keys=True))
