"""Modal deployment wrapper.

This file is the ONLY place Modal appears. It imports `pipeline/` and calls it;
it contains no tracking or detection logic of its own. Keeping the split strict
means the pipeline stays testable off Modal and we are not locked to one vendor.

Quick start (see SETUP.md for the full sequence):

    modal run modal_app.py --video-path clips/short.mp4 --max-frames 90
    modal deploy modal_app.py
"""

from __future__ import annotations

import hmac
import json
import os
import tempfile
from pathlib import Path

import modal
from fastapi import Header, HTTPException

MAIN_PIPELINE = "stats"
PLAYER_POSITIONS_PIPELINE = "player-positions"
PIPELINE_CHOICES = {MAIN_PIPELINE, PLAYER_POSITIONS_PIPELINE}

# Pinned to the official Meta repository at a specific commit so image rebuilds
# are reproducible. Note: the `sam2` package on PyPI is NOT published by Meta —
# do not switch to it.
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"

# Where the Volume holding downloaded model weights is mounted.
CACHE_DIR = "/cache"
TRACKNET_VOLUME_DIR = "/tracknet-data"
TRACKNET_CHECKPOINT = (
    f"{TRACKNET_VOLUME_DIR}/tracknet-finetune/runs/"
    "basketball-v1-20ep-r3/checkpoints/TrackNet_best.pt"
)

app = modal.App("scoutai-pipeline")

# Weights live on a Volume rather than being baked into the image or downloaded
# per job. SAM2 checkpoints are a few hundred MB; re-downloading them on every
# cold start would add time to each job and, at per-second billing, real money.
model_cache = modal.Volume.from_name("scoutai-model-cache", create_if_missing=True)
tracknet_data = modal.Volume.from_name(
    "tracknet-data", environment_name="main", create_if_missing=False
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # ffmpeg/libgl are OpenCV's runtime dependencies. libgl1 is needed even for
    # headless OpenCV builds on debian-slim.
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        # Versions are pinned so a rebuild six months from now produces the same
        # container. Bump deliberately, never implicitly.
        "torch==2.5.1",
        "torchvision==0.20.1",
        "ftfy==6.3.1",
        "open_clip_torch==2.32.0",
        "timm==1.0.22",
        "transformers==5.0.0",
        "huggingface_hub==1.7.1",
        "opencv-python-headless==4.10.0.84",
        "supervision==0.29.1",
        # Used for optimal (Hungarian) detection-to-track assignment. Pinned
        # explicitly rather than relying on it arriving as someone else's
        # transitive dependency.
        "scipy==1.18.0",
        "umap-learn==0.5.9.post2",
        "inference-sdk==1.3.3",
        "boto3==1.35.36",
        "fastapi==0.115.6",
        # Supplies the osnet person re-ID model used to keep player identities
        # apart. Verified compatible with the torch pin above.
        "boxmot==22.0.0",
    )
    # Plain URL form, NOT `sam2 @ git+...`: the repo declares its project name as
    # "sam-2" with a hyphen, so the PEP 508 named form fails the name check and
    # pip then falls back to the unrelated third-party `sam2` package on PyPI.
    .pip_install(f"git+https://github.com/facebookresearch/sam2.git@{SAM2_COMMIT}")
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "TRACKNET_ENABLED": "1",
            "TRACKNET_CHECKPOINT": TRACKNET_CHECKPOINT,
             "BALL_ALL_FRAMES": "1",
        }
    )
    # Modal requires add_local_* calls to be the final image build steps.
    # Ship our own package and TrackNet adapter into the container.
    .add_local_python_source("pipeline")
    .add_local_file("scripts/ball/tracknet.py", remote_path="/root/scripts/ball/tracknet.py")
    .add_local_file("scripts/ball/candidates.py", remote_path="/root/scripts/ball/candidates.py")
)

# Secrets are created once with `modal secret create` (see SETUP.md) and never
# committed. Names must match exactly.
roboflow_secret = modal.Secret.from_name("scoutai-roboflow")
r2_secret = modal.Secret.from_name("scoutai-r2")
auth_secret = modal.Secret.from_name("scoutai-auth")
# Where to report job status. Optional: created only once the backend is
# deployed, and `modal run` works fine without it.
callback_secret = modal.Secret.from_name("scoutai-callback", required_keys=[])


def _run_selected_pipeline(
    pipeline: str,
    *,
    video_path: Path,
    config,
    output_dir: Path,
    tracker,
    embedder,
    tracknet_detector=None,
    on_progress=None,
):
    if pipeline == MAIN_PIPELINE:
        from pipeline.runner import run_pipeline

        return run_pipeline(
            video_path,
            config,
            output_dir,
            tracker=tracker,
            embedder=embedder,
            tracknet_detector=tracknet_detector,
            on_progress=on_progress,
        )

    if pipeline == PLAYER_POSITIONS_PIPELINE:
        from pipeline.player_positions import run_player_position_pipeline

        # No progress hook on this one yet, so it reports RUNNING and then DONE.
        return run_player_position_pipeline(video_path, config, output_dir, tracker=tracker)

    raise ValueError(f"Unknown pipeline: {pipeline!r}")


def _build_process_config_overrides(
    *,
    pipeline: str,
    frame_stride: int | None,
    max_frames: int | None,
    confidence: float | None,
    target_fps: float | None,
    ball_target_fps: float | None = None,
    ball_all_frames: bool | None = None,
    sam2_model_id: str | None,
    max_missed_detections: int | None,
    appearance_weight: float | None,
    team_embedder_backend: str | None,
    team_embedder_model_id: str | None,
    team_count: int | None,
    min_team_samples_per_track: int | None,
    team_sample_stride: int | None,
    min_team_sample_quality: float | None,
    max_team_samples_per_track: int | None,
    team_debug_enabled: bool | None,
    number_predictor_backend: str | None,
    tracknet_enabled: bool | None = None,
    write_annotated_video: bool,
) -> dict:
    config_overrides = {
        "frame_stride": frame_stride or None,
        "max_frames": max_frames or None,
        "confidence": confidence,
        "target_fps": target_fps,
        "ball_target_fps": ball_target_fps,
        "ball_all_frames": ball_all_frames,
        "sam2_model_id": sam2_model_id,
        "max_missed_detections": max_missed_detections,
        "appearance_weight": appearance_weight,
        "team_embedder_backend": team_embedder_backend,
        "team_embedder_model_id": team_embedder_model_id,
        "team_count": team_count,
        "min_team_samples_per_track": min_team_samples_per_track,
        "team_sample_stride": team_sample_stride,
        "min_team_sample_quality": min_team_sample_quality,
        "max_team_samples_per_track": max_team_samples_per_track,
        "team_debug_enabled": team_debug_enabled,
        "number_predictor_backend": number_predictor_backend,
        "tracknet_enabled": tracknet_enabled,
        "write_annotated_video": True if write_annotated_video else None,
    }
    if pipeline == PLAYER_POSITIONS_PIPELINE:
        from pipeline.player_positions import (
            DEFAULT_PLAYER_POSITION_CONFIDENCE,
            DEFAULT_PLAYER_POSITION_FPS,
            DEFAULT_PLAYER_POSITION_FRAME_STRIDE,
        )

        config_overrides.update(
            confidence=DEFAULT_PLAYER_POSITION_CONFIDENCE if confidence is None else confidence,
            target_fps=DEFAULT_PLAYER_POSITION_FPS if target_fps is None else target_fps,
            frame_stride=DEFAULT_PLAYER_POSITION_FRAME_STRIDE if frame_stride is None else frame_stride,
            allowed_classes=("player",),
            write_annotated_video=False,
        )
    return config_overrides


def _progress_reporter(job_id: str | None, *, ceiling: int = 90, every_percent: int = 10):
    """Build an `on_progress` callback that reports tracking progress.

    Throttled to every `every_percent` so a long run makes ~10 status calls, not
    one per segment.

    Reports only up to `ceiling` (default 90), not 100: tracking is the bulk of a
    run, but the stats pipeline still has work left afterwards (team ID, jersey
    OCR, shot detection, court projection) that reports no progress. Holding at
    90 until the DONE callback fills the last stretch reads as "finalizing"
    rather than "done but still churning".
    """

    if not job_id:
        return None

    last_reported = {"bucket": -1}

    def report(frames_done: int, frames_total: int) -> None:
        if frames_total <= 0:
            return
        percent = min(ceiling, int(ceiling * frames_done / frames_total))
        bucket = percent // every_percent
        if bucket > last_reported["bucket"]:
            last_reported["bucket"] = bucket
            _report_status(job_id, "RUNNING", progress_percent=percent)

    return report


def _report_status(
    job_id: str | None,
    status: str,
    *,
    artifacts: list[str] | None = None,
    summary: dict | None = None,
    error_message: str | None = None,
    progress_percent: int | None = None,
) -> None:
    """Tell the backend how a job is going.

    No-ops when there is no `job_id` (a `modal run` test) or no configured
    callback, so local runs work without a backend to report to.

    Never raises. A failed status report must not fail an otherwise good job —
    the GPU work is the expensive part and is already done by the time most of
    these fire.
    """

    import os

    if not job_id:
        return

    url = (os.getenv("PROCESSING_CALLBACK_URL") or "").strip()
    token = (os.getenv("PROCESSING_CALLBACK_TOKEN") or "").strip()
    if not url or not token:
        print(f"[job {job_id}] status={status} (no callback configured, not reported)")
        return

    body: dict = {"job_id": job_id, "status": status}
    if artifacts is not None:
        body["artifacts"] = artifacts
    if summary is not None:
        body["summary"] = summary
    if error_message is not None:
        body["error_message"] = error_message[:1000]
    if progress_percent is not None:
        body["progress_percent"] = progress_percent

    try:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            print(f"[job {job_id}] reported {status} ({response.status})")
    except Exception as error:  # noqa: BLE001 - reporting must never break a job
        print(f"[job {job_id}] could not report {status}: {error!r}")


def _load_tracknet_detector(*, enabled: bool | None = None):
    if enabled is None:
        enabled = os.getenv("TRACKNET_ENABLED", "1").lower() in {"1", "true", "yes"}
    if not enabled:
        print("TrackNet disabled; using hosted ball detection compatibility path")
        return None

    from pipeline.ball import build_tracknet_detector

    checkpoint = Path(os.getenv("TRACKNET_CHECKPOINT", TRACKNET_CHECKPOINT))
    if not checkpoint.exists():
        raise RuntimeError(f"TrackNet is enabled but checkpoint is missing: {checkpoint}")
    try:
        detector = build_tracknet_detector(checkpoint)
    except Exception as error:
        raise RuntimeError(f"Could not load TrackNet checkpoint {checkpoint}: {error}") from error
    print(f"TrackNet loaded: {checkpoint}")
    return detector


@app.cls(
    image=image,
    # COST: L4 is ~24GB VRAM at roughly a quarter the price of an A100. The
    # pipeline processes one ~16-frame segment at a time, so peak VRAM is a few
    # hundred MB and a bigger GPU would buy nothing but a larger bill.
    gpu="L4",
    volumes={CACHE_DIR: model_cache, TRACKNET_VOLUME_DIR: tracknet_data},
    secrets=[roboflow_secret, r2_secret, callback_secret],
    # COST: how long an idle container stays alive after finishing. Long enough
    # that back-to-back jobs reuse a warm container and skip the ~60s model load,
    # short enough that a forgotten container costs cents rather than dollars.
    # Idle time IS billed, so do not raise this casually.
    scaledown_window=60,
    # COST: hard ceiling on parallel GPUs. With a $30/month budget, an
    # accidental burst (retry loop, someone testing) must not be able to spin up
    # ten GPUs. Raise only when there is a real queue to justify it.
    max_containers=2,
    # COST: hard per-job wall-clock ceiling. A hung job cannot bill forever.
    # 30 min is roughly a 10-15 minute clip on L4 with hiera-small; raise it when
    # you start processing full games, and watch the first few runs.
    timeout=1800,
)
class Pipeline:
    """One warm container: SAM2 loaded once, many jobs served."""

    @modal.enter()
    def load_model(self) -> None:
        """Load SAM2 once per container, not once per job.

        Runs on cold start only. The weights come from the mounted Volume after
        the first download.
        """

        from pipeline.config import PipelineConfig
        from pipeline.reid import build_embedder
        from pipeline.tracking import Sam2SegmentTracker

        model_id = os.getenv("SAM2_MODEL_ID", PipelineConfig.__dataclass_fields__["sam2_model_id"].default)
        self.tracker = Sam2SegmentTracker(model_id)
        print(f"SAM2 loaded: {model_id} on {self.tracker.device}")

        # Re-ID weights also live on the Volume, so they download once ever
        # rather than on every cold start.
        backend = os.getenv("REID_BACKEND", PipelineConfig.__dataclass_fields__["reid_backend"].default)
        self.embedder = build_embedder(backend, weights=f"{CACHE_DIR}/reid/osnet_x1_0_msmt17.pt")
        print(f"ReID loaded: {self.embedder.name} on {self.embedder.device}")

        # TrackNet is loaded lazily: player-position jobs do not need its
        # checkpoint or volume, and should retain their existing cold start.
        self.tracknet_detector = None

        model_cache.commit()

    def _ensure_tracknet_detector(self):
        if self.tracknet_detector is None:
            self.tracknet_detector = _load_tracknet_detector(enabled=True)
        return self.tracknet_detector

    @modal.method()
    def process(
        self,
        *,
        pipeline: str = MAIN_PIPELINE,
        # Identifies the backend job this run belongs to. Echoed back on every
        # status callback; without it the backend cannot tell which job finished.
        job_id: str | None = None,
        r2_key: str | None = None,
        video_bytes: bytes | None = None,
        output_prefix: str | None = None,
        frame_stride: int | None = None,
        max_frames: int | None = None,
        confidence: float | None = None,
        target_fps: float | None = None,
        ball_target_fps: float | None = None,
        ball_all_frames: bool | None = None,
        sam2_model_id: str | None = None,
        max_missed_detections: int | None = None,
        appearance_weight: float | None = None,
        team_embedder_backend: str | None = None,
        team_embedder_model_id: str | None = None,
        team_count: int | None = None,
        min_team_samples_per_track: int | None = None,
        team_sample_stride: int | None = None,
        min_team_sample_quality: float | None = None,
        max_team_samples_per_track: int | None = None,
        team_debug_enabled: bool | None = None,
        number_predictor_backend: str | None = None,
        tracknet_enabled: bool | None = None,
        write_annotated_video: bool = False,
        return_artifacts: bool = False,
    ) -> dict:
        """Run the pipeline on one video.

        Source is either an R2 key or raw bytes (handy for `modal run` testing
        before any R2 wiring exists). When `output_prefix` is set the outputs are
        uploaded back to R2 under that prefix.
        """

        import logging

        from pipeline.config import PipelineConfig

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

        if pipeline not in PIPELINE_CHOICES:
            raise ValueError(f"Unknown pipeline: {pipeline!r}")

        if (r2_key is None) == (video_bytes is None):
            raise ValueError("Pass exactly one of r2_key or video_bytes")

        config_overrides = _build_process_config_overrides(
            pipeline=pipeline,
            frame_stride=frame_stride,
            max_frames=max_frames,
            confidence=confidence,
            target_fps=target_fps,
            ball_target_fps=ball_target_fps,
            ball_all_frames=ball_all_frames,
            sam2_model_id=sam2_model_id,
            max_missed_detections=max_missed_detections,
            appearance_weight=appearance_weight,
            team_embedder_backend=team_embedder_backend,
            team_embedder_model_id=team_embedder_model_id,
            team_count=team_count,
            min_team_samples_per_track=min_team_samples_per_track,
            team_sample_stride=team_sample_stride,
            min_team_sample_quality=min_team_sample_quality,
            max_team_samples_per_track=max_team_samples_per_track,
            team_debug_enabled=team_debug_enabled,
            number_predictor_backend=number_predictor_backend,
            tracknet_enabled=tracknet_enabled,
            write_annotated_video=write_annotated_video,
        )

        config = PipelineConfig.from_env(
            **config_overrides,
        )

        # A/B testing checkpoints means loading a different model than the one
        # this container warmed up with. Costs a model load, so only do it when
        # explicitly asked.
        tracker = self.tracker
        if sam2_model_id and sam2_model_id != self.tracker.model_id:
            from pipeline.tracking import Sam2SegmentTracker

            print(f"Loading requested SAM2 checkpoint for this job: {sam2_model_id}")
            tracker = Sam2SegmentTracker(sam2_model_id)

        tracknet_detector = (
            self._ensure_tracknet_detector()
            if pipeline == MAIN_PIPELINE and config.tracknet_enabled
            else None
        )

        try:
            return self._run_job(
                pipeline=pipeline,
                job_id=job_id,
                r2_key=r2_key,
                video_bytes=video_bytes,
                output_prefix=output_prefix,
                config=config,
                tracker=tracker,
                tracknet_detector=tracknet_detector,
                return_artifacts=return_artifacts,
            )
        except Exception as error:
            # Report before re-raising. A job left QUEUED blocks its video
            # forever, because the backend allows only one active job per video.
            _report_status(job_id, "FAILED", error_message=f"{type(error).__name__}: {error}")
            raise

    def _run_job(
        self,
        *,
        pipeline: str,
        job_id: str | None,
        r2_key: str | None,
        video_bytes: bytes | None,
        output_prefix: str | None,
        config,
        tracker,
        tracknet_detector,
        return_artifacts: bool,
    ) -> dict:
        with tempfile.TemporaryDirectory(prefix="job_") as work_dir:
            work = Path(work_dir)
            video_path = work / "input.mp4"
            output_dir = work / "out"

            if r2_key is not None:
                from pipeline.storage import R2Storage

                storage = R2Storage()
                storage.download(r2_key, video_path)
            else:
                video_path.write_bytes(video_bytes)

            # Reported once the video is local and the models are warm, so
            # RUNNING means "actually processing", not "queued behind a download".
            _report_status(job_id, "RUNNING", progress_percent=0)

            results = _run_selected_pipeline(
                pipeline,
                video_path=video_path,
                config=config,
                output_dir=output_dir,
                tracker=tracker,
                embedder=self.embedder,
                tracknet_detector=tracknet_detector,
                on_progress=_progress_reporter(job_id),
            )
            payload = results.to_dict() if hasattr(results, "to_dict") else dict(results)

            uploaded_filenames: list[str] = []
            if output_prefix:
                from pipeline.storage import R2Storage

                storage = R2Storage()
                keys = storage.upload_directory(output_dir, output_prefix)
                payload["output_keys"] = keys
                # The backend stores these so it can presign download URLs
                # without ever listing or reading the bucket.
                prefix = output_prefix.rstrip("/") + "/"
                uploaded_filenames = [
                    key[len(prefix):] if key.startswith(prefix) else key for key in keys
                ]

            _report_status(
                job_id,
                "DONE",
                artifacts=uploaded_filenames,
                summary=payload.get("summary"),
                progress_percent=100,
            )

            if return_artifacts:
                # Ship the output files back to the caller so `modal run` can save
                # them locally — this is how you actually watch the annotated
                # video without wiring up R2. Only sensible for short clips: a
                # long run's artifacts belong in R2, not a return value.
                payload["artifacts"] = {
                    str(path.relative_to(output_dir)): path.read_bytes()
                    for path in sorted(output_dir.rglob("*"))
                    if path.is_file()
                }
            elif not output_prefix:
                # Nothing was persisted anywhere, so at least show some rows.
                preview_name = "tracks.jsonl" if pipeline == MAIN_PIPELINE else "player_positions.jsonl"
                preview_key = "tracks_preview" if pipeline == MAIN_PIPELINE else "player_positions_preview"
                payload[preview_key] = (output_dir / preview_name).read_text(
                    encoding="utf-8"
                ).splitlines()[:20]

            return payload


@app.function(image=image, secrets=[auth_secret], timeout=30)
@modal.fastapi_endpoint(method="POST")
def submit(payload: dict, authorization: str = Header(default="")) -> dict:
    """Authenticated job submission. Spawns and returns immediately.

    Never waits on the GPU: the caller gets a Modal call id straight away and
    polls elsewhere for the result. An unauthenticated version of this endpoint
    would let anyone on the internet spend our GPU budget, so the token check is
    not optional.

    Not wired to the Render backend yet — that is a later phase. It is deployed
    now so the URL and token flow can be tested end to end.
    """

    expected = os.environ["MODAL_WEBHOOK_TOKEN"]
    supplied = authorization.removeprefix("Bearer ").strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

    r2_key = payload.get("r2_key")
    if not r2_key:
        raise HTTPException(status_code=400, detail="r2_key is required")

    call = Pipeline().process.spawn(
        pipeline=payload.get("pipeline", MAIN_PIPELINE),
        # Carried through so status callbacks can name the job they refer to.
        job_id=payload.get("job_id"),
        r2_key=r2_key,
        output_prefix=payload.get("output_prefix"),
        frame_stride=payload.get("frame_stride"),
        max_frames=payload.get("max_frames"),
        confidence=payload.get("confidence"),
        target_fps=payload.get("target_fps"),
        ball_target_fps=payload.get("ball_target_fps"),
        ball_all_frames=payload.get("ball_all_frames"),
        team_embedder_backend=payload.get("team_embedder_backend"),
        team_embedder_model_id=payload.get("team_embedder_model_id"),
        team_count=payload.get("team_count"),
        min_team_samples_per_track=payload.get("min_team_samples_per_track"),
        team_sample_stride=payload.get("team_sample_stride"),
        min_team_sample_quality=payload.get("min_team_sample_quality"),
        max_team_samples_per_track=payload.get("max_team_samples_per_track"),
        team_debug_enabled=payload.get("team_debug_enabled"),
        number_predictor_backend=payload.get("number_predictor_backend"),
        tracknet_enabled=payload.get("tracknet_enabled", True),
        write_annotated_video=bool(payload.get("write_annotated_video", False)),
    )
    return {"call_id": call.object_id, "status": "queued"}


@app.local_entrypoint()
def main(
    pipeline: str = MAIN_PIPELINE,
    video_path: str = "",
    r2_key: str = "",
    output_prefix: str = "",
    output_dir: str = "",
    frame_stride: int = 0,
    max_frames: int = 0,
    confidence: float = -1.0,
    target_fps: float = 0.0,
    ball_target_fps: float = 0.0,
    ball_all_frames: bool | None = None,
    native_fps: bool = False,
    sam2_model_id: str = "",
    max_missed_detections: int = -1,
    appearance_weight: float = -1.0,
    team_embedder_backend: str = "",
    team_embedder_model_id: str = "",
    team_count: int = 0,
    min_team_samples_per_track: int = 0,
    team_sample_stride: int = 0,
    min_team_sample_quality: float = -1.0,
    max_team_samples_per_track: int = 0,
    team_debug_enabled: bool | None = None,
    number_predictor_backend: str = "",
    tracknet_enabled: bool = True,
    annotated: bool = False,
) -> None:
    """Run one job without deploying. Start here.

        modal run modal_app.py --video-path clips/short.mp4 --max-frames 90

        modal run modal_app.py --pipeline player-positions --video-path clips/short.mp4 --output-dir outputs/positions

    To see what it is actually tracking, add `--annotated` and `--output-dir`:

        modal run modal_app.py --video-path clips/short.mp4 --annotated --output-dir outputs/check

    That writes annotated.mp4 locally so you can watch the boxes follow players.

    Keep `--max-frames` small on the first run: it bounds what you spend while
    you are still finding out whether the image builds and the model loads.
    """

    import json

    if pipeline not in PIPELINE_CHOICES:
        raise SystemExit(f"Unknown --pipeline value: {pipeline!r}")

    if bool(video_path) == bool(r2_key):
        raise SystemExit("Pass exactly one of --video-path or --r2-key")

    destination = Path(output_dir) if output_dir else None

    kwargs: dict = {
        "pipeline": pipeline,
        "output_prefix": output_prefix or None,
        "frame_stride": frame_stride or None,
        "max_frames": max_frames or None,
        "confidence": None if confidence < 0 else confidence,
        # `--native-fps` processes every frame, i.e. no downsampling. It is a
        # separate flag because 0.0 cannot be told apart from "not supplied"
        # once it reaches the config's override filter.
        "target_fps": 0.0 if native_fps else (target_fps or None),
        "ball_target_fps": ball_target_fps or None,
        "ball_all_frames": ball_all_frames,
        "sam2_model_id": sam2_model_id or None,
        "max_missed_detections": None if max_missed_detections < 0 else max_missed_detections,
        "appearance_weight": None if appearance_weight < 0 else appearance_weight,
        "team_embedder_backend": team_embedder_backend or None,
        "team_embedder_model_id": team_embedder_model_id or None,
        "team_count": team_count or None,
        "min_team_samples_per_track": min_team_samples_per_track or None,
        "team_sample_stride": team_sample_stride or None,
        "min_team_sample_quality": None if min_team_sample_quality < 0 else min_team_sample_quality,
        "max_team_samples_per_track": max_team_samples_per_track or None,
        "team_debug_enabled": team_debug_enabled,
        "number_predictor_backend": number_predictor_backend or None,
        "tracknet_enabled": tracknet_enabled,
        "write_annotated_video": annotated,
        "return_artifacts": destination is not None,
    }

    if video_path:
        source = Path(video_path)
        if not source.exists():
            raise SystemExit(f"Video not found: {source}")
        size_mb = source.stat().st_size / 1024 / 1024
        print(f"Uploading {source.name} ({size_mb:.1f} MB) to the container...")
        kwargs["video_bytes"] = source.read_bytes()
    else:
        kwargs["r2_key"] = r2_key

    results = Pipeline().process.remote(**kwargs)

    artifacts = results.pop("artifacts", None)
    print(json.dumps(results, indent=2))

    if artifacts and destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        print(f"\nSaved to {destination}:")
        for name, data in artifacts.items():
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            (destination / name).write_bytes(data)
            print(f"  {name}  ({len(data) / 1024:.0f} KB)")
