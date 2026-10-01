"""Backend-facing service layer for the SAM 3.1 (v2) pipeline.

`modal_app_v2.py` is deliberately an *isolated* Modal boundary: no R2, no backend
wiring, direct local artifact return. A test enforces that isolation. This file
is the thin integration layer that lets the Render backend drive v2 **without
touching that boundary** — it calls `PipelineV2` remotely by name, and does the
R2 and callback plumbing around it:

    backend  --POST /submit-->  this app  --.remote()-->  scoutai-pipeline-v2
             <--callbacks (RUNNING/DONE/FAILED)--  this app  --upload stats--> R2

Deploy order (both are your call; this does nothing on its own):

    modal deploy modal_app_v2.py          # the isolated GPU pipeline
    modal deploy modal_app_v2_service.py  # this backend-facing wrapper

Then point the backend at this app's `submit` URL via Render env vars
`PROCESSING_SERVICE_URL` and `PROCESSING_SERVICE_TOKEN`.
"""

from __future__ import annotations

import hmac
import json
import os
import subprocess
import tempfile
from pathlib import Path

import modal
from fastapi import Header, HTTPException

APP_NAME = "scoutai-pipeline-v2-service"
# The GPU app and class this wrapper drives, referenced by name so we never
# import v2's module (keeping its boundary intact) and so it can be redeployed
# independently. We point straight at the teammate's deployed scoutai-pipeline-v2
# so there is no forked copy to keep in sync.
V2_APP = "scoutai-pipeline-v2"
V2_CLASS = "PipelineV2"

# v2 exposes no progress signal, so while the GPU job runs we emit a time-based
# estimate that creeps toward a ceiling — enough to keep the UI bar moving. The
# DONE callback fills the rest once results are uploaded. Replace this with real
# progress once v2 emits it upstream.
PROGRESS_HEARTBEAT_SECONDS = 12
PROGRESS_STEP = 8
PROGRESS_CEILING = 85

DEFAULT_PROMPT = "basketball player on the court"
# By default we process the WHOLE clip: v2's bounded window (60-130) only tracks
# players there, so any shot outside it gets a stale, wrong court position. We
# probe the video length and pass 0..last. A payload may still override with
# explicit frame_start/frame_end. These are the fallback if probing fails.
FALLBACK_FRAME_START = 60
FALLBACK_FRAME_END = 130

# SAM 3.1 is capped at 16 active objects per session, so a single full-video
# session overflows and identity/team resolution collapses. The intended way to
# cover a whole clip is SEGMENTED mode: bounded windows (each under the limit)
# with overlap, whose identities are stitched across the seams.
#
# The right window is clip-dependent: it must hold <=16 distinct tracks. We
# default to the pipeline's documented default (32/1), which is conservative and
# the most likely to stay under the cap on dense footage. A payload can raise it
# for sparser clips (e.g. 150/20). NOTE: a clip with spurious/fragmented tracks
# can still exceed 16 even at a small window -- that is a pipeline data-quality
# issue, not something the window size alone can fix.
DEFAULT_SEGMENTED = True
DEFAULT_SEGMENT_WINDOW = 32
DEFAULT_SEGMENT_OVERLAP = 1

# v2 nests its box-score outputs under this subdirectory; the backend's stats
# endpoints read them at the results-key root, so we flatten this prefix away on
# upload.
STATS_PREFIX = "stats/"

# v2 returns artifacts inline (it cannot upload to R2 itself, by design), so the
# whole output set crosses the wire into this wrapper. Only the small stats files
# are uploaded, but the GPU side still packs everything, so raise the cap above
# v2's 64 MB default to leave headroom for a window with many masks.
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024

app = modal.App(APP_NAME)

# A light CPU image: no GPU, no models. It needs an S3 client (R2 is
# S3-compatible), the pipeline package for its R2Storage helper, and ffprobe
# (from ffmpeg) to read the clip's frame count so we can track the whole video.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("boto3==1.35.36", "fastapi==0.115.6")
    .add_local_python_source("pipeline")
)

r2_secret = modal.Secret.from_name("scoutai-r2")
auth_secret = modal.Secret.from_name("scoutai-auth")
# Optional: present only once the backend is deployed. Without it, status
# callbacks are skipped and a `modal run`-style invocation still works.
callback_secret = modal.Secret.from_name("scoutai-callback", required_keys=[])


def _report_status(
    job_id: str | None,
    status: str,
    *,
    artifacts: list[str] | None = None,
    summary: dict | None = None,
    error_message: str | None = None,
    progress_percent: int | None = None,
) -> None:
    """Tell the backend how a job is going. Never raises.

    No-ops without a `job_id` or a configured callback, so this wrapper is still
    runnable in isolation. A failed status report must not fail an otherwise good
    job — the GPU work is the expensive part and is already done.
    """

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


def _frame_count(video_path: Path) -> int | None:
    """Total video frames via ffprobe, or None if it can't be read."""

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=nokey=1:noprint_wrappers=1", str(video_path),
            ],
            capture_output=True, text=True, timeout=180,
        )
        return int(result.stdout.strip())
    except Exception:  # noqa: BLE001 - probing is best-effort; fall back on failure
        return None


def _frame_bounds(payload: dict, video_path: Path) -> tuple[int, int]:
    """The frame range to track: explicit payload bounds, else the whole clip.

    v2's default window (60-130) only tracks players there, so shots elsewhere
    get a stale court position. Unless the caller pins a range, we track the
    entire video so every shot has real positions to project from.
    """

    if payload.get("frame_start") is not None and payload.get("frame_end") is not None:
        return int(payload["frame_start"]), int(payload["frame_end"])
    count = _frame_count(video_path)
    if count and count > 1:
        return 0, count - 1
    return FALLBACK_FRAME_START, FALLBACK_FRAME_END


def _await_with_progress(call, job_id: str | None):
    """Block on the GPU job, emitting a time-based progress estimate meanwhile.

    v2 reports no real progress, and `.get()` blocks, so we poll with a timeout
    and nudge the bar up a step on each tick until the job returns. If the job
    raised, `.get()` re-raises it here (not a TimeoutError), and the caller turns
    it into a FAILED callback.
    """

    percent = 0
    while True:
        try:
            return call.get(timeout=PROGRESS_HEARTBEAT_SECONDS)
        except TimeoutError:
            percent = min(PROGRESS_CEILING, percent + PROGRESS_STEP)
            _report_status(job_id, "RUNNING", progress_percent=percent)


@app.function(image=image, secrets=[r2_secret, callback_secret], timeout=1800)
def run_backed_job(payload: dict) -> dict:
    """Fetch the video from R2, run v2 on the GPU, upload its stats, report back.

    This is the orchestrator the submit endpoint spawns. It runs on CPU; the GPU
    work happens inside the separate `scoutai-pipeline-v2` app, called remotely.
    """

    from pipeline.storage import R2Storage

    job_id = payload.get("job_id")
    r2_key = payload.get("r2_key")
    output_prefix = payload.get("output_prefix")

    try:
        storage = R2Storage()
        with tempfile.TemporaryDirectory(prefix="v2svc_") as work_dir:
            work = Path(work_dir)
            video_path = work / "input.mp4"
            storage.download(r2_key, video_path)

            # RUNNING only once the video is local, so it means "processing", not
            # "queued behind a download".
            _report_status(job_id, "RUNNING", progress_percent=0)

            frame_start, frame_end = _frame_bounds(payload, video_path)
            pipeline_v2 = modal.Cls.from_name(V2_APP, V2_CLASS)
            call = pipeline_v2().process.spawn(
                video_bytes=video_path.read_bytes(),
                prompt=payload.get("prompt") or DEFAULT_PROMPT,
                frame_start=frame_start,
                frame_end=frame_end,
                return_artifacts=True,
                possession_engine=payload.get("possession_engine"),
                # Segmented by default so a whole-clip run stays under SAM 3.1's
                # 16-object-per-session limit and keeps team/identity resolution.
                segmented=payload.get("segmented", DEFAULT_SEGMENTED),
                segment_window=payload.get("segment_window", DEFAULT_SEGMENT_WINDOW),
                segment_overlap=payload.get("segment_overlap", DEFAULT_SEGMENT_OVERLAP),
                max_artifact_bytes=MAX_ARTIFACT_BYTES,
            )
            result = _await_with_progress(call, job_id)

            artifacts = result.get("artifacts") or {}
            uploaded: list[str] = []
            if output_prefix:
                prefix = output_prefix.rstrip("/") + "/"
                for relative_path, data in artifacts.items():
                    # Only the box-score outputs matter to the backend; raw masks
                    # and the manifest are left on the GPU side. The stats files
                    # are flattened to the results-key root, which is where the
                    # stats endpoints read `projected_shots.json`, `teams.json`,
                    # `numbers.json`, `projected_tracks.jsonl`, etc.
                    if not relative_path.startswith(STATS_PREFIX):
                        continue
                    name = relative_path[len(STATS_PREFIX):]
                    staged = work / "upload.bin"
                    staged.write_bytes(data)
                    storage.upload(staged, prefix + name)
                    uploaded.append(name)

            _report_status(
                job_id,
                "DONE",
                artifacts=uploaded,
                summary={"engine": "v2", "acceptance": result.get("acceptance")},
                progress_percent=100,
            )
            return {"status": "DONE", "artifacts": uploaded}
    except Exception as error:
        # Report before re-raising: a job left RUNNING blocks its video forever,
        # because the backend allows only one active job per video.
        _report_status(job_id, "FAILED", error_message=f"{type(error).__name__}: {error}")
        raise


@app.function(image=image, secrets=[auth_secret], timeout=30)
@modal.fastapi_endpoint(method="POST")
def submit(payload: dict, authorization: str = Header(default="")) -> dict:
    """Authenticated job submission. Spawns and returns immediately.

    The token check is not optional: an open endpoint would let anyone on the
    internet spend our GPU budget. Mirrors the v1 `submit` contract so the
    backend's existing processing-service client works unchanged.
    """

    expected = os.environ["MODAL_WEBHOOK_TOKEN"]
    supplied = authorization.removeprefix("Bearer ").strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not payload.get("r2_key"):
        raise HTTPException(status_code=400, detail="r2_key is required")

    call = run_backed_job.spawn(payload)
    return {"call_id": call.object_id, "status": "queued"}
