#!/usr/bin/env python3
"""Build a redacted V2 acceptance report from a `modal run` log and a job folder.

Usage:
    python scripts/ops/acceptance_report.py --log run.log --job outputs/<id>-job/<id> \
        [--commit SHA] [--replay-output DIR] [--json] [--output FILE]

Every figure comes from the log or the downloaded job folder. Anything the
inputs cannot prove is reported as NOT_MEASURED (or NOT_RUN for steps that were
never exercised), together with the source that would provide it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PASS, FAIL, NOT_RUN, NOT_MEASURED = "PASS", "FAIL", "NOT_RUN", "NOT_MEASURED"

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_GPU = re.compile(r"CUDA:(\d+)\s+\(([^,)]+?)(?:,\s*(\d+)\s*MiB)?\)")
_ELAPSED = re.compile(r"^\[v2-stats\]\s+(.+?) complete: elapsed=([0-9.]+)s")
_PAYLOAD_ROWS = re.compile(r"^\[v2-stats\] in-memory tracking payload: (\d+) rows")
_PROJECTED_ROWS = re.compile(r"^\[v2-stats\] court projection complete: (\d+) projected rows")
_PHASE_DONE = re.compile(
    r"^\[v2-stats\] (TrackNet|shot detection) progress: phase complete "
    r"\((\d+)%; processed (\d+)/(\d+) frames\)"
)
_SAM3_TS = re.compile(
    r"INFO (\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3}) .*?"
    r"(started new session|removed session)"
)
_WARNING = re.compile(
    r"\b(WARNING|ERROR|Traceback)\b|UserWarning|FutureWarning|DeprecationWarning|"
    r"Error\b|^Missing keys"
)
_SMOKE_OK = re.compile(r"smoke", re.I)

# (name, pattern, replacement). Order matters: URLs and tokens first.
_REDACTIONS: list[tuple[str, re.Pattern[str], str]] = [
    ("modal_url", re.compile(r"https?://(?:[\w-]+\.)*modal\.(?:com|run)\S*"), "<modal-url>"),
    ("token_github", re.compile(r"\b(?:gho|ghp|ghu|ghs|ghr)_[A-Za-z0-9]{6,}"), "<token>"),
    ("token_github", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{6,}"), "<token>"),
    ("token_sk", re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}"), "<token>"),
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=\-]+", re.I), "Bearer <token>"),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "<email>"),
    ("home_path", re.compile(r"/(?:home|Users)/[^/\s\"']+"), "<home>"),
]


# ---------------------------------------------------------------- redaction
def redact_text(text: str, counts: dict[str, int] | None = None) -> str:
    for name, pattern, repl in _REDACTIONS:
        text, n = pattern.subn(repl, text)
        if counts is not None and n:
            counts[name] = counts.get(name, 0) + n
    return text


def redact_obj(obj: Any, counts: dict[str, int] | None = None) -> Any:
    if isinstance(obj, str):
        return redact_text(obj, counts)
    if isinstance(obj, list):
        return [redact_obj(v, counts) for v in obj]
    if isinstance(obj, dict):
        return {k: redact_obj(v, counts) for k, v in obj.items()}
    return obj


# -------------------------------------------------------------- log parsing
def _clean_lines(text: str) -> list[str]:
    text = _ANSI.sub("", text).replace("\r", "\n")
    return [line.rstrip() for line in text.split("\n")]


def _find_final_json(text: str) -> dict[str, Any] | None:
    """Last JSON object in the log that carries generation/status keys."""
    decoder = json.JSONDecoder()
    found = None
    for match in re.finditer(r"^\{", text, re.M):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and ("generation" in obj or "status" in obj):
            found = obj
    return found


def _to_seconds(date: str, h: str, m: str, s: str, ms: str) -> float:
    from datetime import datetime

    base = datetime.strptime(date, "%Y-%m-%d").toordinal() * 86400
    return base + int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_log(text: str) -> dict[str, Any]:
    lines = _clean_lines(text)
    clean_text = "\n".join(lines)
    gpus: list[dict[str, Any]] = []
    stage_elapsed: dict[str, float] = {}
    phases: dict[str, dict[str, int]] = {}
    out: dict[str, Any] = {
        "tracking_rows": None,
        "projected_rows": None,
        "stats_complete": False,
        "app_completed": False,
        "app_url_count": 0,
        "stats_markers": [],
        "smoke_test_result_lines": [],
        "created_functions": [],
    }
    warnings: list[str] = []
    seen_warn: set[str] = set()
    sessions = 0
    removed = 0
    first_ts = last_ts = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        gm = _GPU.search(stripped)
        if gm:
            gpu = {"device": f"CUDA:{gm.group(1)}", "name": gm.group(2).strip()}
            if gm.group(3):
                gpu["memory_mib"] = int(gm.group(3))
            if gpu not in gpus:
                gpus.append(gpu)
        if m := _ELAPSED.match(stripped):
            stage_elapsed[m.group(1)] = float(m.group(2))
        if m := _PAYLOAD_ROWS.match(stripped):
            out["tracking_rows"] = int(m.group(1))
        if m := _PROJECTED_ROWS.match(stripped):
            out["projected_rows"] = int(m.group(1))
        if m := _PHASE_DONE.match(stripped):
            phases[m.group(1)] = {
                "percent": int(m.group(2)),
                "processed_frames": int(m.group(3)),
                "total_frames": int(m.group(4)),
            }
        if stripped.startswith("[v2-stats]"):
            out["stats_markers"].append(stripped[len("[v2-stats] "):])
        if stripped == "[v2-stats] stats complete":
            out["stats_complete"] = True
        if "App completed" in stripped:
            out["app_completed"] = True
        if "modal.com" in stripped or "modal.run" in stripped:
            out["app_url_count"] += 1
        if m := re.search(r"Created function (\S+?)\.?$", stripped):
            out["created_functions"].append(m.group(1))
        elif _SMOKE_OK.search(stripped) and "Created" not in stripped:
            out["smoke_test_result_lines"].append(stripped[:200])
        tm = _SAM3_TS.search(stripped)
        if tm:
            ts = _to_seconds(*tm.group(1, 2, 3, 4, 5))
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)
            if tm.group(6) == "started new session":
                sessions += 1
            else:
                removed += 1
        if (
            _WARNING.search(stripped)
            and "propagate_in_video" not in stripped
            and "frame loading" not in stripped
            and not stripped.startswith("warnings.warn")
        ):
            key = stripped[:300]
            if key not in seen_warn:
                seen_warn.add(key)
                warnings.append(key)
    out["gpus"] = gpus
    out["stage_elapsed_seconds"] = stage_elapsed
    out["phase_complete"] = phases
    out["warnings"] = warnings
    out["sam3_sessions"] = sessions
    out["sam3_sessions_removed"] = removed
    out["sam3_log_span_seconds"] = (
        round(last_ts - first_ts, 3) if first_ts is not None and last_ts > first_ts else None
    )
    out["final_json"] = _find_final_json(clean_text)
    return out


# --------------------------------------------------------------- job folder
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def verify_generation(gen_dir: Path) -> dict[str, Any]:
    """Recompute SHA-256 of files listed in <gen_dir>/manifest.json."""
    manifest = _read_json(gen_dir / "manifest.json")
    result: dict[str, Any] = {
        "generation": gen_dir.name,
        "manifest_present": isinstance(manifest, dict),
        "listed_count": 0,
        "listed_bytes": 0,
        "verified_ok": 0,
        "mismatched": [],
        "missing": [],
        "size_mismatched": [],
        "artifacts": [],
        "disk_bytes": _dir_bytes(gen_dir) if gen_dir.is_dir() else 0,
        "complete_flag": None,
        "frame_bounds": None,
        "stage": None,
    }
    if not isinstance(manifest, dict):
        return result
    result["complete_flag"] = manifest.get("complete")
    result["frame_bounds"] = manifest.get("frame_bounds")
    result["stage"] = manifest.get("stage")
    result["manifest_tracking_generation"] = manifest.get("tracking_generation")
    artifacts = manifest.get("artifacts") or {}
    for key in sorted(artifacts):
        entry = artifacts[key]
        rel = entry.get("path", key)
        expected = entry.get("sha256")
        size = entry.get("size_bytes")
        result["listed_count"] += 1
        result["listed_bytes"] += size or 0
        file_path = gen_dir / rel
        row = {"key": key, "path": rel, "size_bytes": size, "sha256": expected}
        if not file_path.is_file():
            row["status"] = "missing"
            result["missing"].append(rel)
        else:
            if size is not None and file_path.stat().st_size != size:
                result["size_mismatched"].append(rel)
            actual = sha256_file(file_path)
            if actual != expected:
                row["status"] = "mismatch"
                row["actual_sha256"] = actual
                result["mismatched"].append(rel)
            else:
                row["status"] = "ok"
                result["verified_ok"] += 1
        result["artifacts"].append(row)
    return result


def _ball_rows(gen_dir: Path | None) -> dict[str, Any] | None:
    if gen_dir is None:
        return None
    path = gen_dir / "ball_tracks.jsonl"
    if not path.is_file():
        return None
    total = non_lost = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        total += 1
        if row.get("track_state") not in (None, "lost"):
            non_lost += 1
    return {"rows": total, "non_lost_rows": non_lost}


def inspect_job(job_dir: Path) -> dict[str, Any]:
    job = _read_json(job_dir / "job.json") or {}
    info: dict[str, Any] = {"job": job, "job_json_present": bool(job)}
    fingerprint = job.get("input_fingerprint") or ""
    expected_sha = fingerprint.split(":", 1)[1] if fingerprint.startswith("sha256:") else None
    inputs = sorted((job_dir / "inputs").glob("*")) if (job_dir / "inputs").is_dir() else []
    inputs = [p for p in inputs if p.is_file()]
    input_info: dict[str, Any] = {
        "fingerprint": fingerprint or None,
        "file": None,
        "size_bytes": None,
        "sha256_recomputed": None,
        "sha256_matches_fingerprint": None,
    }
    match = [p for p in inputs if expected_sha and expected_sha in p.name] or inputs[:1]
    if match:
        path = match[0]
        input_info["file"] = path.name
        input_info["size_bytes"] = path.stat().st_size
        input_info["sha256_recomputed"] = sha256_file(path)
        if expected_sha:
            input_info["sha256_matches_fingerprint"] = (
                input_info["sha256_recomputed"] == expected_sha
            )
    info["input"] = input_info

    generations: dict[str, dict[str, Any]] = {}
    for stage in ("tracking", "stats"):
        stage_dir = job_dir / stage
        current = _read_json(stage_dir / "current.json") or {}
        gen_id = job.get(f"{stage}_generation") or current.get("generation")
        gens_root = stage_dir / "generations"
        all_dirs = sorted(p for p in gens_root.iterdir() if p.is_dir()) if gens_root.is_dir() else []
        entry: dict[str, Any] = {
            "id": gen_id,
            "current_pointer": current.get("generation"),
            "dir_present": bool(gen_id and (gens_root / gen_id).is_dir()),
            "storage_bytes_by_generation": {p.name: _dir_bytes(p) for p in all_dirs},
            "verification": None,
        }
        if entry["dir_present"]:
            entry["verification"] = verify_generation(gens_root / gen_id)
        generations[stage] = entry
    info["generations"] = generations

    tracking_manifest = None
    tid = generations["tracking"]["id"]
    if tid:
        tracking_manifest = _read_json(
            job_dir / "tracking" / "generations" / tid / "tracking-manifest.json"
        )
    info["tracking_manifest"] = tracking_manifest if isinstance(tracking_manifest, dict) else None
    results = None
    sid = generations["stats"]["id"]
    stats_dir = job_dir / "stats" / "generations" / sid if sid else None
    if stats_dir and stats_dir.is_dir():
        results = _read_json(stats_dir / "results.json")
    info["stats_results"] = results if isinstance(results, dict) else None
    info["ball"] = _ball_rows(stats_dir if stats_dir and stats_dir.is_dir() else None)
    info["download_bytes"] = _dir_bytes(job_dir) if job_dir.is_dir() else None
    return info


# ----------------------------------------------------------------- checklist
def build_checklist(log: dict[str, Any], jobinfo: dict[str, Any], replay: dict[str, Any]) -> list[dict[str, str]]:
    job = jobinfo["job"]
    gens = jobinfo["generations"]
    tm = jobinfo["tracking_manifest"] or {}
    res = jobinfo["stats_results"] or {}
    items: list[dict[str, str]] = []

    def add(item: str, status: str, evidence: str) -> None:
        items.append({"item": item, "status": status, "evidence": evidence})

    smoke = log["smoke_test_result_lines"]
    if smoke:
        add("SAM3.1 dependency smoke test passes in the deployed image", PASS,
            "log line: " + smoke[0])
    else:
        add("SAM3.1 dependency smoke test passes in the deployed image", NOT_RUN,
            "no smoke-test result in the log (function listed as created: "
            f"{'sam31_dependency_smoke_test' in log['created_functions']}); "
            "run `modal run` of the smoke entrypoint and attach its log")

    gpu_names = ", ".join(g["name"] for g in log["gpus"]) or None
    final = log["final_json"] or {}
    done = job.get("state") == "complete" and final.get("status") == "complete"
    if done and gpu_names:
        add("A short live V2 run completes on the intended GPU", PASS,
            f"job state=complete, log final status=complete, GPU seen: {gpu_names} "
            "(the intended GPU is not recorded in the inputs, compare manually)")
    elif job or final:
        add("A short live V2 run completes on the intended GPU", FAIL,
            f"job state={job.get('state')!r}, log final status={final.get('status')!r}, "
            f"GPU seen: {gpu_names!r}")
    else:
        add("A short live V2 run completes on the intended GPU", NOT_MEASURED,
            "neither job.json nor a final log JSON was provided")

    obs = tm.get("observation_count")
    trk_ver = gens["tracking"]["verification"]
    if obs and trk_ver and trk_ver["complete_flag"] and trk_ver["listed_count"] > 0:
        add("The run publishes a non-empty root tracking package", PASS,
            f"tracking generation {gens['tracking']['id']}: complete, observation_count={obs}, "
            f"{trk_ver['listed_count']} listed artifacts")
    elif gens["tracking"]["id"]:
        add("The run publishes a non-empty root tracking package", FAIL,
            f"tracking generation {gens['tracking']['id']}: observation_count={obs!r}, "
            f"manifest present={bool(trk_ver)}")
    else:
        add("The run publishes a non-empty root tracking package", NOT_MEASURED,
            "no tracking generation in the job folder")

    ball = jobinfo["ball"]
    if ball is None:
        add("TrackNet produces a non-lost ball row on a suitable clip", NOT_MEASURED,
            "ball_tracks.jsonl not present in the stats generation")
    elif ball["non_lost_rows"] > 0:
        add("TrackNet produces a non-lost ball row on a suitable clip", PASS,
            f"ball_tracks.jsonl: {ball['non_lost_rows']} non-lost of {ball['rows']} rows")
    else:
        add("TrackNet produces a non-lost ball row on a suitable clip", FAIL,
            f"ball_tracks.jsonl: 0 non-lost of {ball['rows']} rows (clip may be unsuitable)")

    cfg = res.get("config") or {}
    keys = {k: cfg[k] for k in ("tracknet_enabled", "court_model_id", "number_predictor_backend") if k in cfg}
    add("Configured external model credentials and optional integrations are verified or explicitly marked unavailable",
        NOT_MEASURED,
        f"configuration seen in results.json: {json.dumps(keys, sort_keys=True)}; "
        "the log does not show credential checks (Roboflow/OCR/ReID); "
        "a successful stats stage is indirect evidence only")

    both = bool(gens["tracking"]["id"] and gens["stats"]["id"])
    flags = [
        (gens[s]["verification"] or {}).get("complete_flag") is True for s in ("tracking", "stats")
    ]
    if both and all(flags) and job.get("state") == "complete":
        add("One-command staged execution publishes both generations", PASS,
            f"tracking={gens['tracking']['id']} stats={gens['stats']['id']}, both manifests complete=true "
            f"(attempts: {json.dumps(job.get('attempts'), sort_keys=True)}). "
            "That the run was one command is taken from the log being a single `modal run`")
    elif both:
        add("One-command staged execution publishes both generations", FAIL,
            f"both ids present but manifest complete flags={flags}, job state={job.get('state')!r}")
    else:
        add("One-command staged execution publishes both generations", NOT_MEASURED,
            "job folder lacks one or both generation ids")

    add("Repeating the same stage reuses the matching generation without recomputation", NOT_RUN,
        "a single run was provided; repeat the command and compare generation ids and timings")
    add("Stats-only retry succeeds after a tracking failure or expired lease", NOT_RUN,
        f"job attempts={json.dumps(job.get('attempts'), sort_keys=True)}, errors={job.get('errors')!r}; "
        "no simulated failure was exercised")

    if replay["requested"]:
        if replay["exists"]:
            add("Downloaded artifacts render with debug_v2.py using an explicit generation", PASS,
                f"replay output present ({replay['files']} files, {replay['bytes']} bytes); "
                "existence only, the render command itself is not re-run by this script")
        else:
            add("Downloaded artifacts render with debug_v2.py using an explicit generation", FAIL,
                "--replay-output was given but the path does not exist or is empty")
    else:
        add("Downloaded artifacts render with debug_v2.py using an explicit generation", NOT_RUN,
            "no --replay-output provided")

    add("GPU time, timeout margin, Volume growth, and download size are recorded", NOT_MEASURED,
        f"download size measured: {jobinfo['download_bytes']} bytes. NOT in the inputs: GPU runtime "
        "(Modal billing / dashboard usage export), timeout margin (function timeout vs. Modal "
        "function-call duration), Volume growth (`modal volume ls` before/after)")

    verifs = [gens[s]["verification"] for s in ("tracking", "stats") if gens[s]["verification"]]
    bad = sum(len(v["mismatched"]) + len(v["missing"]) for v in verifs)
    if not verifs:
        add("Artifact hashes match the manifests", NOT_MEASURED, "no generation folder available")
    elif bad:
        add("Artifact hashes match the manifests", FAIL, f"{bad} mismatched or missing files")
    else:
        add("Artifact hashes match the manifests", PASS,
            f"{sum(v['verified_ok'] for v in verifs)} of {sum(v['listed_count'] for v in verifs)} listed files recomputed OK")

    add("The acceptance report is redacted and reproducible", PASS,
        "redaction applied to all report text (see the Redaction line at the top); output is a deterministic "
        "function of the log, the job folder and --commit")
    add("Cleanup/retention implications are recorded for follow-up", NOT_MEASURED,
        "per-generation storage bytes are listed above; retention policy and Volume cleanup "
        "are not derivable from the inputs and need a follow-up decision")
    return items


# -------------------------------------------------------------------- build
def resolve_commit(explicit: str | None) -> dict[str, str]:
    if explicit:
        return {"sha": explicit, "source": "--commit"}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        if sha:
            return {"sha": sha, "source": "git rev-parse HEAD (best effort, may differ from deployed code)"}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"sha": NOT_MEASURED, "source": "git unavailable and --commit not given"}


def _replay_info(path: str | None) -> dict[str, Any]:
    info = {"requested": path is not None, "exists": False, "files": 0, "bytes": 0}
    if path is not None:
        p = Path(path)
        if p.is_file():
            info.update(exists=True, files=1, bytes=p.stat().st_size)
        elif p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            info.update(exists=bool(files), files=len(files), bytes=sum(f.stat().st_size for f in files))
    return info


def build_report(
    log_text: str,
    job_dir: Path,
    commit: str | None = None,
    replay_output: str | None = None,
) -> dict[str, Any]:
    log = parse_log(log_text)
    jobinfo = inspect_job(job_dir)
    job = jobinfo["job"]
    tm = jobinfo["tracking_manifest"] or {}
    video = tm.get("video") or (jobinfo["stats_results"] or {}).get("video") or {}
    gens = jobinfo["generations"]
    final = log["final_json"] or {}

    consistency = {
        "log_stats_generation_matches_job": (
            final.get("generation") == gens["stats"]["id"] if final.get("generation") and gens["stats"]["id"] else None
        ),
        "log_tracking_generation_matches_job": (
            final.get("tracking_generation") == gens["tracking"]["id"]
            if final.get("tracking_generation") and gens["tracking"]["id"] else None
        ),
    }

    durations = {
        "log_reported_seconds": log["stage_elapsed_seconds"],
        "sam3_tracking_log_span_seconds": log["sam3_log_span_seconds"],
        "sam3_tracking_log_span_note": (
            "difference between first and last timestamped SAM3 session log lines; "
            "covers the SAM3 segments only, not model load, TrackNet or artifact writing"
        ),
        "total_job_seconds": NOT_MEASURED,
        "total_job_source": "Modal function-call duration (dashboard / `modal app` logs); the CLI log has no wall-clock stamps outside SAM3",
        "tracknet_and_shot_phases": log["phase_complete"],
        "benchmark_in_tracking_manifest": tm.get("benchmark") or None,
    }

    storage = {}
    for stage in ("tracking", "stats"):
        v = gens[stage]["verification"]
        storage[stage] = {
            "generation": gens[stage]["id"],
            "manifest_listed_bytes": v["listed_bytes"] if v else None,
            "bytes_on_disk_by_generation": gens[stage]["storage_bytes_by_generation"],
        }

    verification = {}
    for stage in ("tracking", "stats"):
        v = gens[stage]["verification"]
        if v:
            verification[stage] = {k: v[k] for k in (
                "generation", "listed_count", "listed_bytes", "verified_ok", "mismatched", "missing",
                "size_mismatched", "artifacts")}

    replay = _replay_info(replay_output)
    report: dict[str, Any] = {
        "report_version": 1,
        "job_id": job.get("job_id") or NOT_MEASURED,
        "commit": resolve_commit(commit),
        "modal_app_image_version": {
            "status": NOT_MEASURED,
            "source": "app URL is redacted; image id/tag is not printed by `modal run` (use `modal app list` / deploy output)",
            "functions_created_in_log": log["created_functions"],
        },
        "input_clip": {
            **jobinfo["input"],
            "fps": video.get("fps"),
            "frame_count": video.get("frame_count"),
            "width": video.get("width"),
            "height": video.get("height"),
            "duration_seconds": (jobinfo["stats_results"] or {}).get("video", {}).get("duration_seconds"),
        },
        "gpu_and_configuration": {
            "gpus_seen_in_log": log["gpus"],
            "tracking_config": {k: (tm.get("config") or {}).get(k) for k in (
                "prompt", "frame_start", "frame_end", "sam31_max_num_objects") if k in (tm.get("config") or {})},
            "stats_config": (jobinfo["stats_results"] or {}).get("config"),
            "tracking_rows_in_log": log["tracking_rows"],
            "projected_rows_in_log": log["projected_rows"],
        },
        "job": {
            "state": job.get("state"),
            "stage": job.get("stage"),
            "attempts": job.get("attempts"),
            "errors": job.get("errors"),
            "tracking_generation": gens["tracking"]["id"],
            "stats_generation": gens["stats"]["id"],
            "log_final_json": final or None,
            "consistency": consistency,
        },
        "durations": durations,
        "artifact_summary": {
            stage: {
                "generation": gens[stage]["id"],
                "listed_files": (gens[stage]["verification"] or {}).get("listed_count"),
                "listed_bytes": (gens[stage]["verification"] or {}).get("listed_bytes"),
            } for stage in ("tracking", "stats")
        },
        "hash_verification": verification,
        "hash_verification_ok": (
            bool(verification) and all(not v["mismatched"] and not v["missing"] for v in verification.values())
        ),
        "storage": {**storage, "download_bytes": jobinfo["download_bytes"]},
        "ball_tracks": jobinfo["ball"],
        "warnings": log["warnings"],
        "not_measured": {
            "gpu_runtime_billing": "Modal usage/billing export for the app run (not printed in `modal run` logs)",
            "cost": "GPU seconds x Modal GPU rate; needs billed GPU runtime",
            "cost_per_video_minute": "cost / clip duration; needs cost",
            "timeout_margin": "configured function timeout minus Modal function-call duration",
            "volume_growth": "`modal volume ls`/size before and after the run",
            "model_api_calls": "Roboflow dashboard / API usage counters",
            "total_job_duration": durations["total_job_source"],
        },
        "replay_output": replay,
        "log_stats": {
            "stats_complete": log["stats_complete"],
            "app_completed": log["app_completed"],
            "sam3_sessions": log["sam3_sessions"],
            "sam3_sessions_removed": log["sam3_sessions_removed"],
        },
        "rollback_cleanup_notes": (
            "Not derivable from inputs. Generations live under v2-job-data/<job>/{tracking,stats}/generations/<id>; "
            "sizes above bound what a cleanup would reclaim."
        ),
    }
    report["checklist"] = build_checklist(log, jobinfo, replay)

    counts: dict[str, int] = {}
    redact_text(log_text, counts)  # count what the raw log would have leaked
    report = redact_obj(report, counts)
    report["redaction"] = {
        "applied": True,
        "note": "replacement counts include matches found in the raw log, even if the report never quotes them",
        "replacements": sum(counts.values()),
        "by_rule": dict(sorted(counts.items())),
        "rules": ["modal app URLs", "tokens (gho_/ghp_/sk-/Bearer)", "email addresses", "absolute home directory paths"],
    }
    return report


# ------------------------------------------------------------------ markdown
def _fmt(value: Any) -> str:
    return NOT_MEASURED if value is None else str(value)


def render_markdown(report: dict[str, Any], max_files: int = 40) -> str:
    L: list[str] = []
    a = L.append
    clip = report["input_clip"]
    a(f"# V2 acceptance report: {report['job_id']}")
    a("")
    a(f"- Commit: `{report['commit']['sha']}` (source: {report['commit']['source']})")
    a(f"- Modal app/image version: {NOT_MEASURED} ({report['modal_app_image_version']['source']})")
    r = report["redaction"]
    a(f"- Redaction: applied ({r['replacements']} replacements; rules: {', '.join(r['rules'])})")
    a("")
    a("## Input clip")
    a(f"- File: `{_fmt(clip['file'])}`, size: {_fmt(clip['size_bytes'])} bytes")
    a(f"- job.json fingerprint: `{_fmt(clip['fingerprint'])}`")
    a(f"- Recomputed sha256: `{_fmt(clip['sha256_recomputed'])}` (matches fingerprint: {_fmt(clip['sha256_matches_fingerprint'])})")
    a(f"- Frames: {_fmt(clip['frame_count'])}, fps: {_fmt(clip['fps'])}, resolution: {_fmt(clip['width'])}x{_fmt(clip['height'])}, duration: {_fmt(clip['duration_seconds'])} s")
    a("")
    cfg = report["gpu_and_configuration"]
    a("## GPU and configuration")
    gpus = "; ".join(f"{g['device']} {g['name']}" + (f", {g['memory_mib']} MiB" if 'memory_mib' in g else "") for g in cfg["gpus_seen_in_log"])
    a(f"- GPU (from log): {gpus or NOT_MEASURED}")
    a(f"- Tracking config: `{json.dumps(cfg['tracking_config'], sort_keys=True)}`")
    a(f"- Stats config: `{json.dumps(cfg['stats_config'], sort_keys=True)}`")
    a(f"- Tracking rows in log: {_fmt(cfg['tracking_rows_in_log'])}, projected rows: {_fmt(cfg['projected_rows_in_log'])}")
    a("")
    job = report["job"]
    a("## Job and generations")
    a(f"- State: {_fmt(job['state'])} (stage: {_fmt(job['stage'])}), attempts: `{json.dumps(job['attempts'], sort_keys=True)}`, errors: `{job['errors']}`")
    a(f"- Tracking generation: `{_fmt(job['tracking_generation'])}`")
    a(f"- Stats generation: `{_fmt(job['stats_generation'])}`")
    a(f"- Log final JSON: `{json.dumps(job['log_final_json'], sort_keys=True)}`; consistency with job.json: `{json.dumps(job['consistency'], sort_keys=True)}`")
    a("")
    d = report["durations"]
    a("## Durations")
    for name, secs in d["log_reported_seconds"].items():
        a(f"- {name} (log): {secs} s")
    a(f"- SAM3 tracking log span: {_fmt(d['sam3_tracking_log_span_seconds'])} s ({d['sam3_tracking_log_span_note']})")
    for name, ph in d["tracknet_and_shot_phases"].items():
        a(f"- {name} phase: {ph['processed_frames']}/{ph['total_frames']} frames ({ph['percent']}%) (no duration in log)")
    a(f"- Total job duration: {NOT_MEASURED} (source: {d['total_job_source']})")
    a("")
    a("## Artifacts and hash verification")
    ok = report["hash_verification_ok"]
    a(f"- Overall hash verification: {'PASS' if ok else 'FAIL or NOT_MEASURED'}")
    for stage, v in report["hash_verification"].items():
        a(f"- {stage} `{v['generation']}`: listed {v['listed_count']} files / {v['listed_bytes']} bytes, verified OK {v['verified_ok']}, mismatched {len(v['mismatched'])}, missing {len(v['missing'])}, size-mismatched {len(v['size_mismatched'])}")
        for p in v["mismatched"][:20]:
            a(f"  - MISMATCH: `{p}`")
        for p in v["missing"][:20]:
            a(f"  - MISSING: `{p}`")
        rows = sorted(v["artifacts"], key=lambda x: -(x["size_bytes"] or 0))
        shown = rows[:max_files]
        a("")
        a(f"  | file | bytes | sha256 (manifest) | status |")
        a(f"  |---|---|---|---|")
        for row in shown:
            a(f"  | `{row['path']}` | {_fmt(row['size_bytes'])} | `{row['sha256']}` | {row['status']} |")
        if len(rows) > len(shown):
            a(f"  | ... {len(rows) - len(shown)} more files omitted here (largest {len(shown)} shown); all are in the `--json` output | | | |")
        a("")
    a("## Storage")
    for stage in ("tracking", "stats"):
        s = report["storage"][stage]
        a(f"- {stage}: manifest-listed {_fmt(s['manifest_listed_bytes'])} bytes; on disk per generation: `{json.dumps(s['bytes_on_disk_by_generation'], sort_keys=True)}`")
    a(f"- Downloaded job folder total: {_fmt(report['storage']['download_bytes'])} bytes")
    a("")
    a("## Known warnings (deduplicated from log)")
    if report["warnings"]:
        for w in report["warnings"]:
            a(f"- `{w}`")
    else:
        a("- none found")
    a("")
    a("## Acceptance checklist")
    a("")
    a("| Criterion | Status | Evidence |")
    a("|---|---|---|")
    for c in report["checklist"]:
        a(f"| {c['item']} | **{c['status']}** | {c['evidence'].replace('|', '/')} |")
    a("")
    a("## Not measured (and where it would come from)")
    for k, v in report["not_measured"].items():
        a(f"- {k}: {NOT_MEASURED}; source: {v}")
    a("")
    a("## Rollback and cleanup")
    a(report["rollback_cleanup_notes"])
    a("")
    return "\n".join(L)


# ----------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, help="`modal run` log file")
    parser.add_argument("--job", required=True, help="downloaded job folder (contains job.json)")
    parser.add_argument("--commit", help="commit SHA of the deployed code (default: git rev-parse HEAD)")
    parser.add_argument("--replay-output", help="debug_v2.py render output, if the replay was run")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    parser.add_argument("--output", help="write to this file instead of stdout")
    parser.add_argument("--max-files", type=int, default=40, help="max artifact rows per generation in Markdown")
    args = parser.parse_args(argv)

    log_path, job_dir = Path(args.log), Path(args.job)
    if not log_path.is_file():
        parser.error(f"log file not found: {log_path}")
    if not job_dir.is_dir():
        parser.error(f"job folder not found: {job_dir}")
    report = build_report(
        log_path.read_text(encoding="utf-8", errors="replace"),
        job_dir, commit=args.commit, replay_output=args.replay_output,
    )
    text = json.dumps(report, indent=2, sort_keys=True) if args.json else render_markdown(report, args.max_files)
    text = redact_text(text) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if report["hash_verification_ok"] or not report["hash_verification"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
