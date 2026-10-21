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
import time
import urllib.error
import urllib.request
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

# Real progress comes from the Modal Dict that PipelineV2 writes (keyed by job id).
# If the Dict is unreachable we fall back to a time-based creep toward a ceiling so
# the UI bar still moves; the DONE callback fills the rest.
PROGRESS_DICT_NAME = "scoutai-v2-progress"
PROGRESS_HEARTBEAT_SECONDS = 12
PROGRESS_STEP = 8
PROGRESS_CEILING = 85
PROGRESS_MIN = 2
PROGRESS_MAX_RUNNING = 99
UPLOAD_BAND = (96, 99)

# Hard limits on the GPU call, across every ladder attempt. The Modal function
# timeout (3600) stays above the deadline so we can report FAILED ourselves.
JOB_DEADLINE_SECONDS = 3300
STALL_SECONDS = 15 * 60

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
# The window that holds <=16 tracks is clip-dependent (a dense clip has more
# distinct tracks in the same span). Rather than guess one value, we try a ladder
# largest-first -- a bigger window means fewer segments, so it's faster and makes
# fewer client round-trips when it fits. 17-22 simultaneous objects are pruned by
# the pipeline and the run continues; only more than 22 trips the cap error, and
# then we step down to a smaller window and retry. A payload may pin
# segment_window to skip the ladder.
DEFAULT_SEGMENTED = True
SEGMENT_WINDOW_LADDER = (96, 48, 24)

# v2 nests its box-score outputs under this subdirectory; the backend's stats
# endpoints read them at the results-key root, so we flatten this prefix away on
# upload.
STATS_PREFIX = "stats/"

# v2 returns artifacts inline (it cannot upload to R2 itself, by design), so we ask
# it for only the files the backend reads and cap each one; the GPU side then never
# packs masks or videos.
RETURN_ARTIFACT_PATHS = [
    "stats/projected_shots.json",
    "stats/teams.json",
    "stats/numbers.json",
    "stats/results.json",
    "stats/shot_events.jsonl",
    "stats/player_summary.json",
    "stats/player_paths.jsonl",
    "stats/possession_events.jsonl",
    "stats/possession_transitions.jsonl",
    "stats/projected_tracks.jsonl",
    # Evidence behind the numbers (never tracks.jsonl or masks: size and privacy).
    "stats/possession_v3_manifest.json",
    "stats/possession_v3/generations/",
    "stats/possession_v3_failure.json",
    "stats/ball_tracks.jsonl",
    "stats/shot_evidence.jsonl",
    "stats/shot_signals.jsonl",
    "stats/shot_raw_predictions.jsonl",
    "stats/number_predictions.json",
]
# Files the v3 possession engine writes under a generation directory; they are
# uploaded flattened, and only from the generation the manifest names.
V3_GENERATIONS_PREFIX = "stats/possession_v3/generations/"
V3_MANIFEST_PATH = "stats/possession_v3_manifest.json"
V3_FAILURE_PATH = "stats/possession_v3_failure.json"
V3_EVENTS_NAME = "possession_events_v3.jsonl"
V3_BASENAMES = (
    "possession_v3.jsonl",
    "possession_primitives_v3.jsonl",
    "ball_transitions_v3.jsonl",
    V3_EVENTS_NAME,
    "possession_v4_diagnostics.jsonl",
)
RETURN_ARTIFACT_MAX_FILE_BYTES = 30 * 1024 * 1024
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
DEFAULT_POSSESSION_ENGINE = "v5"

# Status callbacks: DONE/FAILED get 3 attempts with these pauses between them.
REPORT_BACKOFF_SECONDS = (2, 5, 15)

# Test seams for the deadline and stall clocks and the callback backoff.
_clock = time.monotonic
_sleep = time.sleep


def _git_sha() -> str | None:
    """Short git sha of this checkout at deploy time, else the baked-in env value."""

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=Path(__file__).resolve().parent,
        )
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            return sha
    except Exception:  # noqa: BLE001 - provenance is best-effort, never blocks a deploy
        pass
    return (os.getenv("SERVICE_GIT_SHA") or "").strip() or None


SERVICE_GIT_SHA = _git_sha()

app = modal.App(APP_NAME)

# A light CPU image: no GPU, no models. It needs only an S3 client (R2 is
# S3-compatible) and ffprobe (from ffmpeg) to read the clip's frame count. We do
# NOT ship the `pipeline` package: importing it pulls in cv2/torch etc. (via
# pipeline/__init__ -> results), which this slim image lacks — so R2 access is
# done with boto3 directly.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("boto3==1.35.36", "fastapi==0.115.6")
)
if SERVICE_GIT_SHA:
    # Evaluated where the module is loaded for deploy; the container has no git.
    image = image.env({"SERVICE_GIT_SHA": SERVICE_GIT_SHA})

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
    progress_phase: str | None = None,
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
    if progress_phase is not None:
        body["progress_phase"] = progress_phase[:40]

    # DONE/FAILED decide the job's fate in the backend, so they are retried on a
    # transient failure. RUNNING ticks every few seconds and the next one
    # supersedes it, so one attempt is enough (and never stalls the poll loop).
    attempts = len(REPORT_BACKOFF_SECONDS) if status in ("DONE", "FAILED") else 1
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                print(f"[job {job_id}] reported {status} ({response.status})")
            return
        except Exception as error:  # noqa: BLE001 - reporting must never break a job
            print(f"[job {job_id}] could not report {status} (attempt {attempt + 1}/{attempts}): {error!r}")
            code = getattr(error, "code", None)
            if isinstance(code, int) and code < 500:
                return  # the backend rejected it; retrying the same body cannot help
            if attempt + 1 < attempts:
                _sleep(REPORT_BACKOFF_SECONDS[attempt])


def _r2_client():
    """A boto3 S3 client for R2, plus the bucket name, from the scoutai-r2 secret.

    Done inline rather than importing pipeline.storage: that would pull in the
    whole pipeline package (cv2/torch), which this slim CPU image does not carry.
    """

    import boto3

    required = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME", "R2_ENDPOINT_URL")
    missing = [name for name in required if not (os.getenv(name) or "").strip()]
    if missing:
        raise RuntimeError(f"Missing R2 environment variables: {', '.join(missing)}")
    client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"].strip(),
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"].strip(),
        endpoint_url=os.environ["R2_ENDPOINT_URL"].strip(),
    )
    return client, os.environ["R2_BUCKET_NAME"].strip()


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


def _segment_windows(payload: dict) -> tuple[int, ...]:
    """Segment windows to try, largest first. A pinned payload value wins."""

    pinned = payload.get("segment_window")
    if pinned is not None:
        return (int(pinned),)
    return SEGMENT_WINDOW_LADDER


def _segment_overlap(payload: dict, window: int) -> int:
    """Seam overlap for a window: a payload value, else a fraction of the window."""

    pinned = payload.get("segment_overlap")
    if pinned is not None:
        return int(pinned)
    return max(1, window // 8)


def _is_object_cap_error(exc: Exception) -> bool:
    """Whether a run failed because a segment exceeded SAM 3.1's object cap."""

    message = str(exc).lower()
    return "active objects" in message and "maximum" in message


class JobAbortedError(RuntimeError):
    """The GPU call was cancelled by the deadline or the stall watchdog."""


class _JobProgress:
    """Progress state for a whole job, shared by every ladder attempt.

    The percent never goes down: a retried window or a restarted call (new
    run_id) holds the last value and reports phase 'restarted' until the new run
    gets past it.
    """

    def __init__(self) -> None:
        self.started = _clock()
        self.percent = 0
        self.phase: str | None = None
        self.run_id: str | None = None
        self.creep = 0
        self.last_phase = None

    def advance(self, percent: int, phase: str | None) -> tuple[int, str | None]:
        if percent > self.percent:
            self.percent = percent
            self.phase = phase
        return self.percent, self.phase

    def apply_payload(self, payload: dict) -> tuple[int, str | None]:
        run_id = payload.get("run_id")
        phase = payload.get("phase")
        try:
            percent = int(float(payload.get("percent") or 0))
        except (TypeError, ValueError):
            percent = 0
        percent = max(PROGRESS_MIN, min(PROGRESS_MAX_RUNNING, percent))
        if run_id != self.run_id:
            restarted = self.run_id is not None
            self.run_id = run_id
            if restarted:
                self.phase = "restarted"
        if percent > self.percent:
            self.percent, self.phase = percent, phase
        elif self.phase == "restarted":
            pass
        elif percent == self.percent and phase:
            self.phase = phase
        if phase:
            self.last_phase = phase
        return self.percent, self.phase


def _open_progress_dict():
    return modal.Dict.from_name(PROGRESS_DICT_NAME, create_if_missing=True)


def _await_with_progress(call, job_id: str | None, state: _JobProgress | None = None):
    """Block on the GPU job, reporting the pipeline's real progress meanwhile.

    `.get()` blocks, so we poll with a timeout and read the pipeline's progress
    from a Modal Dict on each tick. If the job raised, `.get()` re-raises it here
    (not a TimeoutError) and the caller turns it into a FAILED callback. The call
    is cancelled when the job deadline passes or the Dict stops changing.
    """

    state = state or _JobProgress()
    progress_dict = None
    dict_ok = True
    try:
        progress_dict = _open_progress_dict()
    except Exception as error:  # noqa: BLE001 - progress is best-effort
        print(f"[job {job_id}] progress dict unavailable, using estimate: {error!r}")
        dict_ok = False

    last_signature = None
    last_change = _clock()
    try:
        while True:
            try:
                return call.get(timeout=PROGRESS_HEARTBEAT_SECONDS)
            except TimeoutError:
                pass

            now = _clock()
            if now - state.started > JOB_DEADLINE_SECONDS:
                call.cancel()
                raise JobAbortedError(
                    f"deadline exceeded after {JOB_DEADLINE_SECONDS}s "
                    f"(last phase: {state.last_phase or 'unknown'})"
                )

            payload = None
            if dict_ok:
                try:
                    payload = progress_dict[job_id] if job_id else None
                except KeyError:
                    payload = None
                except Exception as error:  # noqa: BLE001
                    print(f"[job {job_id}] progress dict read failed, using estimate: {error!r}")
                    dict_ok = False

            if dict_ok:
                signature = (
                    (payload.get("run_id"), payload.get("percent"), payload.get("phase"),
                     payload.get("updated_at"))
                    if isinstance(payload, dict) else None
                )
                if signature != last_signature:
                    last_signature, last_change = signature, now
                elif now - last_change > STALL_SECONDS:
                    call.cancel()
                    raise JobAbortedError(
                        f"stalled: no progress for {STALL_SECONDS // 60} minutes "
                        f"(last phase: {state.last_phase or 'unknown'})"
                    )
                if isinstance(payload, dict):
                    percent, phase = state.apply_payload(payload)
                else:
                    percent, phase = state.advance(PROGRESS_MIN, state.phase)
            else:
                state.creep = min(PROGRESS_CEILING, state.creep + PROGRESS_STEP)
                percent, phase = state.advance(max(PROGRESS_MIN, state.creep), state.phase)
            _report_status(
                job_id, "RUNNING", progress_percent=percent, progress_phase=phase
            )
    finally:
        if dict_ok and progress_dict is not None and job_id:
            try:
                progress_dict.pop(job_id)
            except Exception:  # noqa: BLE001 - key may not exist; cleanup is best-effort
                pass


def _is_returnable(relative_path: str) -> bool:
    """Whether a returned artifact is on the allowlist (exact or directory prefix)."""

    return any(
        relative_path == entry or (entry.endswith("/") and relative_path.startswith(entry))
        for entry in RETURN_ARTIFACT_PATHS
    )


def _v3_generation(artifacts: dict[str, bytes]) -> str | None:
    """The generation the returned v3 manifest names, or None if unreadable."""

    try:
        generation = json.loads(artifacts[V3_MANIFEST_PATH]).get("generation")
    except Exception:  # noqa: BLE001 - absent or malformed manifest: no generation
        return None
    return generation if isinstance(generation, str) and generation else None


def _upload_name(relative_path: str, v3_generation: str | None) -> str | None:
    """The flat upload name for a returned artifact, or None if it must not go up.

    `stats/<name>` -> `<name>`. A v3 generation file -> its basename, only when its
    generation is the one the manifest names (stale generations from earlier runs
    must not shadow the current files).
    """

    if not relative_path.startswith(STATS_PREFIX):
        return None
    name = relative_path[len(STATS_PREFIX):]
    if relative_path.startswith(V3_GENERATIONS_PREFIX):
        generation, _, base = relative_path[len(V3_GENERATIONS_PREFIX):].partition("/")
        if not v3_generation or generation != v3_generation or not base or "/" in base:
            return None
        return base
    return name if name and "/" not in name else None


def _plan_uploads(artifacts: dict[str, bytes]) -> tuple[list[tuple[str, str]], list[dict]]:
    """(relative_path, upload_name) pairs to upload, plus the files skipped here."""

    generation = _v3_generation(artifacts)
    plan: list[tuple[str, str]] = []
    skipped: list[dict] = []
    taken: set[str] = set()
    for relative_path, data in artifacts.items():
        if not _is_returnable(relative_path) or not relative_path.startswith(STATS_PREFIX):
            continue
        name = _upload_name(relative_path, generation)
        reason = None
        if name is None:
            reason = "v3_generation_mismatch"
        elif name in taken:
            reason = "name_collision"
        if reason:
            skipped.append({"path": relative_path, "bytes": len(data), "reason": reason})
            continue
        taken.add(name)
        plan.append((relative_path, name))
    return plan, skipped


# Names a healthy run is expected to upload (flat), for `missing_expected`.
EXPECTED_ARTIFACT_NAMES = tuple(
    entry[len(STATS_PREFIX):] for entry in RETURN_ARTIFACT_PATHS
    if not entry.endswith("/") and entry not in (V3_FAILURE_PATH,)
) + (V3_EVENTS_NAME,)
COURT_FAILED_RATIO_DEGRADED = 0.05


def _json_rows(data: bytes | None) -> list[dict]:
    rows = []
    for line in (data or b"").decode("utf-8", "replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _court_stage(config: dict, uploaded: list[str], run_info: dict | None) -> dict:
    court = (run_info or {}).get("court") if isinstance((run_info or {}).get("court"), dict) else {}
    stage = {
        "status": "unknown",
        "frames_total": court.get("frames_total"),
        "frames_projected": court.get("frames_projected"),
        "frames_failed": court.get("frames_failed"),
        "errors_by_type": court.get("errors_by_type"),
    }
    if not config:
        return stage
    if not config.get("court_model_id"):
        stage["status"] = "disabled"
    elif "projected_tracks.jsonl" not in uploaded:
        stage["status"] = "empty"
    elif stage["frames_projected"] == 0 and stage["frames_total"]:
        stage["status"] = "empty"
    elif stage["frames_failed"]:
        stage["status"] = "partial"
    else:
        stage["status"] = "ok"
    return stage


def build_quality(
    results: dict,
    uploaded: list[str],
    skipped: list[dict],
    run_info: dict | None,
    *,
    acceptance: dict | None = None,
    team_assignments: int | None = None,
    v3_events: bytes | None = None,
) -> dict:
    """What the user can trust in this run: a level, the reasons, per-stage facts.

    `results` is the run's results.json. Anything that only the GPU side knows
    comes from `run_info` and stays None when it is absent (never invented).
    `acceptance`, `team_assignments` and `v3_events` are facts the service reads
    from the GPU result and the returned files themselves.
    """

    results = results or {}
    run_info = run_info if isinstance(run_info, dict) else {}
    acceptance = acceptance or {}
    config = results.get("config") or {}
    reasons: list[dict] = []

    def reason(code: str, severity: str, detail: str) -> None:
        reasons.append({"code": code, "severity": severity, "detail": detail})

    court = _court_stage(config, uploaded, run_info)
    if config.get("court_model_id") and "projected_tracks.jsonl" not in uploaded:
        reason("court_projection_empty", "critical",
               "court model configured but no projected_tracks.jsonl was produced")
    if team_assignments == 0:
        reason("teams_empty", "critical", "teams.json has no track assignments")

    observability = results.get("possession_v3_observability") or {}
    failed = observability.get("status") == "failed" or "possession_v3_failure.json" in uploaded
    events_uploaded = V3_EVENTS_NAME in uploaded
    counts = observability.get("counts") or {}
    v3 = {"status": "ok", "transitions": counts.get("transitions"),
          "events": counts.get("events"), "passes": None}
    if failed:
        v3["status"] = "failed"
        reason("possession_v3_failed", "degraded", str(observability.get("error") or "v3 possession failed"))
    elif not events_uploaded:
        v3["status"] = "missing"
    if not failed and not events_uploaded:
        reason("possession_v3_events_missing", "degraded", f"{V3_EVENTS_NAME} was not returned")
    if events_uploaded and v3_events is not None:
        v3["passes"] = sum(1 for row in _json_rows(v3_events) if row.get("event_type") == "PASS")

    for item in skipped:
        reason("artifact_skipped", "degraded", f"{item.get('path')}: {item.get('reason') or 'too_large'}")

    ratio_total, ratio_failed = court["frames_total"], court["frames_failed"]
    if ratio_total and ratio_failed and ratio_failed / ratio_total > COURT_FAILED_RATIO_DEGRADED:
        reason("court_projection_partial", "degraded", f"{ratio_failed}/{ratio_total} frames failed to project")

    tracking_info = run_info.get("tracking") if isinstance(run_info.get("tracking"), dict) else {}
    tracking = {key: tracking_info.get(key)
                for key in ("segments", "overflow_dropped_ids", "seams", "seams_linked")}
    if tracking["overflow_dropped_ids"]:
        reason("tracking_overflow_dropped_ids", "degraded",
               f"{tracking['overflow_dropped_ids']} track ids dropped by the object cap")

    numbers_ok = "numbers.json" in uploaded
    if not numbers_ok:
        reason("jersey_numbers_missing", "degraded", "numbers.json was not returned")

    tracking_identity = acceptance.get("tracking_identity")
    if tracking_identity is not None and tracking_identity != "accepted":
        reason("tracking_identity_not_accepted", "degraded", f"tracking_identity={tracking_identity}")

    if any(r["severity"] == "critical" for r in reasons):
        level = "critical"
    elif reasons:
        level = "degraded"
    else:
        level = "ok"

    return {
        "schema_version": 1,
        "level": level,
        "reasons": reasons,
        "stages": {
            "court_projection": court,
            "possession_v3": v3,
            "jersey_numbers": {"status": "ok" if numbers_ok else "missing"},
            "tracking": tracking,
        },
        "identity": {
            "tracking_identity": tracking_identity,
            "excluded_observations": acceptance.get("identity_excluded_observations"),
            "reasons": run_info.get("acceptance_reasons"),
        },
        "artifacts": {
            "uploaded": list(uploaded),
            "missing_expected": [name for name in EXPECTED_ARTIFACT_NAMES if name not in uploaded],
            "skipped": list(skipped),
        },
    }


def build_provenance(
    results: dict,
    *,
    engine: str,
    window: int,
    prompt: str,
    run_info: dict | None,
    v3_generation: str | None,
    service_git_sha: str | None = None,
) -> dict:
    """Which code and inputs produced this run. GPU-side facts come from `run_info`."""

    config = (results or {}).get("config") or {}
    run_info = run_info if isinstance(run_info, dict) else {}
    return {
        "service_git_sha": service_git_sha,
        "pipeline_git_sha": run_info.get("pipeline_git_sha"),
        "possession_engine": engine,
        "segment_window": window,
        "segment_overlap": run_info.get("segment_overlap"),
        "prompt": prompt,
        "court_model_id": config.get("court_model_id"),
        "number_predictor_backend": config.get("number_predictor_backend"),
        "input_sha256": run_info.get("input_sha256"),
        "possession_v3_generation": v3_generation,
    }


def _build_summary(
    result: dict,
    *,
    engine: str,
    window: int,
    attempts: int,
    artifacts: dict[str, bytes] | None = None,
    uploaded: list[str] | None = None,
    skipped: list[dict] | None = None,
    prompt: str = DEFAULT_PROMPT,
) -> dict:
    """The DONE summary: the pipeline's own facts, each optional."""

    video = result.get("video") or {}
    frames = result.get("frames")
    artifacts = artifacts or {}
    run_info = result.get("run_info")
    generation = _v3_generation(artifacts)
    results_json = {}
    teams_assignments = None
    try:
        results_json = json.loads(artifacts["stats/results.json"])
    except Exception:  # noqa: BLE001 - quality degrades to "unknown", never fails the job
        pass
    try:
        teams_assignments = len(json.loads(artifacts["stats/teams.json"])["assignments_by_track_id"])
    except Exception:  # noqa: BLE001
        pass
    quality = build_quality(
        results_json, uploaded or [], skipped or [], run_info,
        acceptance=result.get("acceptance"), team_assignments=teams_assignments,
        v3_events=artifacts.get(f"{V3_GENERATIONS_PREFIX}{generation}/{V3_EVENTS_NAME}"),
    )
    return {
        "engine": "v2",
        "contract_version": 2,
        "acceptance": result.get("acceptance"),
        "possession_engine": engine,
        "segment_window": window,
        "attempts": attempts,
        "frames": frames,
        "fps": video.get("fps"),
        "duration_seconds": video.get("duration_seconds"),
        "gpu": result.get("gpu"),
        "timings": result.get("timings"),
        "quality": quality,
        "provenance": build_provenance(
            results_json, engine=engine, window=window, prompt=prompt, run_info=run_info,
            v3_generation=generation, service_git_sha=SERVICE_GIT_SHA,
        ),
    }


@app.function(image=image, secrets=[r2_secret, callback_secret], timeout=3600)
def run_backed_job(payload: dict) -> dict:
    """Fetch the video from R2, run v2 on the GPU, upload its stats, report back.

    This is the orchestrator the submit endpoint spawns. It runs on CPU; the GPU
    work happens inside the separate `scoutai-pipeline-v2` app, called remotely.
    """

    job_id = payload.get("job_id")
    state = _JobProgress()
    r2_key = payload.get("r2_key")
    output_prefix = payload.get("output_prefix")

    try:
        s3, bucket = _r2_client()
        with tempfile.TemporaryDirectory(prefix="v2svc_") as work_dir:
            work = Path(work_dir)
            video_path = work / "input.mp4"
            s3.download_file(bucket, r2_key, str(video_path))

            # RUNNING only once the video is local, so it means "processing", not
            # "queued behind a download".
            percent, phase = state.advance(PROGRESS_MIN, "download")
            _report_status(job_id, "RUNNING", progress_percent=percent, progress_phase=phase)

            frame_start, frame_end = _frame_bounds(payload, video_path)
            pipeline_v2 = modal.Cls.from_name(V2_APP, V2_CLASS)
            video_bytes = video_path.read_bytes()
            windows = _segment_windows(payload)

            result = None
            engine = payload.get("possession_engine") or DEFAULT_POSSESSION_ENGINE
            window = None
            attempts = 0
            for attempt, window in enumerate(windows):
                attempts = attempt + 1
                overlap = _segment_overlap(payload, window)
                call = pipeline_v2().process.spawn(
                    video_bytes=video_bytes,
                    prompt=payload.get("prompt") or DEFAULT_PROMPT,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    return_artifacts=True,
                    possession_engine=engine,
                    # Segmented so a whole-clip run stays under SAM 3.1's
                    # 16-object-per-session limit and keeps team/identity resolution.
                    segmented=payload.get("segmented", DEFAULT_SEGMENTED),
                    segment_window=window,
                    segment_overlap=overlap,
                    max_artifact_bytes=MAX_ARTIFACT_BYTES,
                    return_artifact_paths=RETURN_ARTIFACT_PATHS,
                    return_artifact_max_file_bytes=RETURN_ARTIFACT_MAX_FILE_BYTES,
                    progress_key=job_id,
                )
                try:
                    result = _await_with_progress(call, job_id, state)
                    break
                except Exception as exc:  # noqa: BLE001 - only the cap error is retriable
                    if _is_object_cap_error(exc) and attempt < len(windows) - 1:
                        print(
                            f"[job {job_id}] segment window {window} tripped the 16-object "
                            f"cap; retrying with a smaller window"
                        )
                        # Hold the percent: the bar must not go backwards on a retry.
                        percent, phase = state.advance(0, "restarted")
                        _report_status(
                            job_id, "RUNNING", progress_percent=percent,
                            progress_phase="restarted",
                        )
                        continue
                    raise

            result = result or {}
            artifacts = result.get("artifacts") or {}
            for skipped in result.get("artifacts_skipped") or []:
                print(f"[job {job_id}] artifact skipped on the GPU side: {skipped}")
            percent = max(state.percent, UPLOAD_BAND[0])
            _report_status(job_id, "RUNNING", progress_percent=percent, progress_phase="upload")
            uploaded: list[str] = []
            skipped = [dict(item, reason=item.get("reason") or "too_large")
                       for item in result.get("artifacts_skipped") or []]
            if output_prefix:
                prefix = output_prefix.rstrip("/") + "/"
                # Only allowlisted outputs are uploaded, flattened to the
                # results-key root where the stats endpoints read them.
                plan, plan_skipped = _plan_uploads(artifacts)
                skipped.extend(plan_skipped)
                for relative_path, name in plan:
                    staged = work / "upload.bin"
                    staged.write_bytes(artifacts[relative_path])
                    s3.upload_file(str(staged), bucket, prefix + name)
                    uploaded.append(name)

            _report_status(
                job_id,
                "DONE",
                artifacts=uploaded,
                summary=_build_summary(
                    result, engine=engine, window=window, attempts=attempts,
                    artifacts=artifacts, uploaded=uploaded, skipped=skipped,
                    prompt=payload.get("prompt") or DEFAULT_PROMPT,
                ),
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
