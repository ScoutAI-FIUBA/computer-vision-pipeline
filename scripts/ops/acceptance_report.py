#!/usr/bin/env python3
"""Build a redacted V2 acceptance report from a `modal run` log and a job folder.

Usage:
    python scripts/ops/acceptance_report.py --log run.log --job outputs/<id>-job/<id> \
        [--commit SHA] [--replay-output DIR|FILE.mp4] [--notes FILE.md] \
        [--also-log RUN2.log ...] [--gpu-rate-usd-per-hour RATE] [--video-seconds S] \
        [--volume-listing-before FILE --volume-listing-after FILE] \
        [--repeat-log FILE] [--retry-log FILE] [--retention-policy PATH] [--json] [--output FILE]
    python scripts/ops/acceptance_report.py --diff A.json B.json

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
_GPU_STRING = re.compile(r"CUDA:\d+\s+\(([^)]*)\)")
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
    gpu_strings: list[str] = []
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
        sm = _GPU_STRING.search(stripped)
        if sm and sm.group(1).strip() not in gpu_strings:
            gpu_strings.append(sm.group(1).strip())
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
    out["gpu_strings"] = gpu_strings
    out["stage_elapsed_seconds"] = stage_elapsed
    out["phase_complete"] = phases
    out["warnings"] = warnings
    out["sam3_sessions"] = sessions
    out["sam3_sessions_removed"] = removed
    out["sam3_log_span_seconds"] = (
        round(last_ts - first_ts, 3) if first_ts is not None and last_ts > first_ts else None
    )
    out["final_json"] = _find_final_json(clean_text)
    timing = parse_timing_lines(lines)
    out["timing_stages"] = timing["stages"]
    out["timing_jobs"] = timing["jobs"]
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


def _jsonl_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            invalid += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            invalid += 1
    return rows, invalid


def _manifest_field(tm: dict[str, Any], key: str) -> Any:
    """Look in the manifest root and its acceptance block; None means absent."""
    for holder in (tm, tm.get("acceptance") if isinstance(tm.get("acceptance"), dict) else {}):
        if key in holder:
            return holder[key]
    return None


def _identity_info(tm: dict[str, Any], tracks_dirs: list[Path | None]) -> dict[str, Any]:
    acceptance = tm.get("acceptance") if isinstance(tm.get("acceptance"), dict) else {}
    info: dict[str, Any] = {
        "tracking_identity": acceptance.get("tracking_identity"),
        "acceptance_reasons": tm.get("acceptance_reasons"),
        # None means the manifest predates the field; do not report 0 for it
        "identity_excluded_observations": _manifest_field(tm, "identity_excluded_observations"),
        "identity_excluded_observation_keys": _manifest_field(tm, "identity_excluded_observation_keys"),
        "tracks_source": None,
        "rows_total": None,
        "rows_with_stable_player_id": None,
        "rows_without_stable_player_id": None,
        "rows_unparseable": None,
    }
    for gen_dir in tracks_dirs:
        if gen_dir is not None and (gen_dir / "tracks.jsonl").is_file():
            rows, invalid = _jsonl_rows(gen_dir / "tracks.jsonl")
            with_id = sum(1 for r in rows if r.get("stable_player_id"))
            info.update(
                tracks_source=f"{gen_dir.parent.parent.name}/tracks.jsonl",
                rows_total=len(rows), rows_with_stable_player_id=with_id,
                rows_without_stable_player_id=len(rows) - with_id, rows_unparseable=invalid,
            )
            break
    return info


def _numbers_info(res: dict[str, Any] | None, stats_dir: Path | None, stats_manifest: Any) -> dict[str, Any]:
    cfg = (res or {}).get("config") or {}
    numbers_file = (res or {}).get("numbers_file")
    listed = sorted(
        k for k in ((stats_manifest or {}).get("artifacts") or {}) if "number" in k.lower()
    ) if isinstance(stats_manifest, dict) else []
    return {
        "results_present": res is not None,
        "numbers_file": numbers_file,
        "file_exists": bool(numbers_file and stats_dir and (stats_dir / numbers_file).is_file()),
        "manifest_number_artifacts": listed,
        "number_predictor_backend": cfg.get("number_predictor_backend"),
    }


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int] | None:
    """Counts of a field; None when no item carries the key (field not emitted by that run)."""
    if not any(key in i for i in items):
        return None
    out: dict[str, int] = {}
    for i in items:
        label = "absent" if key not in i else str(i[key])
        out[label] = out.get(label, 0) + 1
    return dict(sorted(out.items()))


def _outcome_info(stats_dir: Path | None, ball: dict[str, Any] | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "events": None, "by_outcome": None, "by_outcome_source": None,
        "projected_shots": None, "by_shooter_attribution_method": None,
        "shots_with_null_shooter": None, "ball_tracks": ball,
    }
    if stats_dir is None:
        return info
    if (stats_dir / "shot_events.jsonl").is_file():
        events, _ = _jsonl_rows(stats_dir / "shot_events.jsonl")
        info.update(events=len(events), by_outcome=_count_by(events, "outcome") or {},
                    by_outcome_source=_count_by(events, "outcome_source"))
    shots_doc = _read_json(stats_dir / "projected_shots.json")
    if isinstance(shots_doc, dict) and isinstance(shots_doc.get("shots"), list):
        shots = [s for s in shots_doc["shots"] if isinstance(s, dict)]
        info.update(
            projected_shots=len(shots),
            by_shooter_attribution_method=_count_by(shots, "shooter_attribution_method"),
            shots_with_null_shooter=sum(
                1 for s in shots if s.get("shooter_track_id") is None and s.get("shooter_stable_player_id") is None
            ),
        )
        if info["by_outcome_source"] is None:
            info["by_outcome_source"] = _count_by(shots, "outcome_source")
    return info


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
    trk_dir = job_dir / "tracking" / "generations" / tid if tid else None
    stats_ok = stats_dir if stats_dir and stats_dir.is_dir() else None
    info["identity"] = _identity_info(
        info["tracking_manifest"] or {}, [stats_ok, trk_dir if trk_dir and trk_dir.is_dir() else None]
    )
    info["numbers"] = _numbers_info(
        info["stats_results"], stats_ok, _read_json(stats_ok / "manifest.json") if stats_ok else None
    )
    info["outcomes"] = _outcome_info(stats_ok, info["ball"])
    info["download_bytes"] = _dir_bytes(job_dir) if job_dir.is_dir() else None
    return info


# ------------------------------------------------------- performance and cost
STAGES = ("tracking", "stats", "possession")
METRICS_FILE = "stage_metrics.json"
API_KEYS = (
    "roboflow_shot_detection_frames", "roboflow_court_detection_frames",
    "roboflow_number_detector_frames", "roboflow_ocr_crops",
)
_TIMING_STAGE = re.compile(
    r"\[v2-timing\]\s+stage=(\S+)\s+started=(\S+)\s+finished=(\S+)\s+elapsed=(-?[0-9.]+)s?\s+"
    r"timeout_margin=(-?[0-9.]+)s?\s+gpu=(.*)$"
)
_TIMING_JOB = re.compile(r"\[v2-timing\]\s+job\s+id=(\S+)\s+total_wall_seconds=(-?[0-9.]+)")
_SIZE_UNITS = {
    "b": 1, "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4,
    "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4,
}
_HUMAN_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)\s*[│|]?\s*$", re.I)
_BOX = str.maketrans({c: " " for c in "│┃┆┇┊┋║┌┐└┘├┤┬┴┼─━═╭╮╯╰┏┓┗┛"})
REASON_NO_METRICS = "run predates stage metrics"
REASON_NO_RATE = "pass the rate from the Modal plan"
DISCLAIMER = (
    "Estimates, not the invoice; the Modal dashboard is the source of truth. The lower figure is GPU "
    "time only (sum of container-side stage elapsed): container CPU/memory and cold start are excluded. "
    "The upper figure uses client-side wall time, which adds cold start, model load and transfer."
)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _nm(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": NOT_MEASURED, "reason": reason, **extra}


def parse_timing_lines(lines: list[str]) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for line in lines:
        if m := _TIMING_STAGE.search(line):
            gpu = m.group(6).strip()
            stages.append({
                "stage": m.group(1), "started": m.group(2), "finished": m.group(3),
                "elapsed_seconds": float(m.group(4)), "timeout_margin_seconds": float(m.group(5)),
                "gpu": None if gpu.lower() in ("none", "") else gpu,
            })
        elif m := _TIMING_JOB.search(line):
            jobs.append({"job_id": None if m.group(1) == "none" else m.group(1),
                         "total_wall_seconds": float(m.group(2))})
    return {"stages": stages, "jobs": jobs}


def read_stage_metrics(job_dir: Path, job: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per stage: current generation id, its metrics (or None) and how many generations have metrics."""
    out: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        stage_dir = job_dir / stage
        gens_root = stage_dir / "generations"
        published = sorted(p for p in gens_root.iterdir() if p.is_dir()) if gens_root.is_dir() else []
        current = _read_json(stage_dir / "current.json")
        current_id = (current.get("generation") if isinstance(current, dict) else None) or job.get(f"{stage}_generation")
        found: dict[str, dict[str, Any]] = {}
        invalid = 0
        for gen in published:
            path = gen / METRICS_FILE
            if not path.is_file():
                continue
            doc = _read_json(path)
            if isinstance(doc, dict) and _num(doc.get("elapsed_seconds")) is not None:
                found[gen.name] = doc
            else:
                invalid += 1
        out[stage] = {
            "generation": current_id,
            "published": [p.name for p in published],
            "bytes_by_generation": {p.name: _dir_bytes(p) for p in published},
            "metrics": found.get(current_id) if current_id else None,
            "generations_with_metrics": len(found),
            "invalid_metrics_files": invalid,
        }
    return out


def _gpu_from_metrics(doc: dict[str, Any]) -> dict[str, Any] | None:
    gpu = doc.get("gpu")
    if not isinstance(gpu, dict):
        return None
    return {"name": gpu.get("name"), "memory_mib": gpu.get("memory_mib")}


def parse_volume_listing(text: str) -> dict[str, Any]:
    """Total file bytes from a JSON {path: bytes} or a `modal volume ls` text copy; fail soft."""
    result: dict[str, Any] = {"total_bytes": None, "files": 0, "approximate": False,
                              "unparsed_lines": 0, "format": None}
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            doc = json.loads(stripped)
        except ValueError:
            doc = None
        if isinstance(doc, dict):
            sizes = [int(v) for v in doc.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            result.update(format="json", files=len(sizes), total_bytes=sum(sizes))
            return result
    total = files = unparsed = 0
    approx = False
    for raw in _clean_lines(text):
        line = raw.translate(_BOX).strip()
        if not line or set(line) <= set("-=+ "):
            continue
        parts = line.split()
        if re.search(r"\bdir(ectory)?\b", line, re.I) and not _HUMAN_SIZE.search(line):
            continue
        if len(parts) < 2:
            unparsed += 1
            continue
        if m := _HUMAN_SIZE.search(line):
            unit = m.group(2).lower()
            total += int(float(m.group(1)) * _SIZE_UNITS[unit])
            approx = approx or unit != "b"
            files += 1
        elif re.fullmatch(r"\d+", parts[-1]):
            total += int(parts[-1])
            files += 1
        else:
            unparsed += 1
    result.update(format="text", files=files, approximate=approx, unparsed_lines=unparsed,
                  total_bytes=total if files else None)
    return result


def _read_listing(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        return parse_volume_listing(Path(path).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return {"total_bytes": None, "files": 0, "approximate": False, "unparsed_lines": 0,
                "format": None, "error": "listing file could not be read"}


def _video_seconds(tm: dict[str, Any], bounds: Any, override: float | None) -> dict[str, Any]:
    if override is not None:
        return {"seconds": override, "source": "--video-seconds"}
    fps = _num((tm.get("video") or {}).get("fps"))
    if not fps or fps <= 0:
        return {"seconds": None, "source": "tracking manifest has no fps; pass --video-seconds"}
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and all(isinstance(b, int) for b in bounds):
        return {"seconds": round((bounds[1] - bounds[0] + 1) / fps, 3),
                "source": f"tracking frame_bounds {list(bounds)} (inclusive) / video fps {fps:g}"}
    frames = _num((tm.get("video") or {}).get("frame_count"))
    if frames:
        return {"seconds": round(frames / fps, 3), "source": f"tracking manifest frame_count {frames:g} / video fps {fps:g}"}
    return {"seconds": None, "source": "tracking manifest has no frame bounds; pass --video-seconds"}


def build_performance(
    log: dict[str, Any], job_dir: Path, jobinfo: dict[str, Any], *,
    gpu_rate: float | None = None, video_seconds: float | None = None,
    volume_included_gib: float | None = None, volume_rate: float | None = None,
    listing_before: str | None = None, listing_after: str | None = None,
) -> dict[str, Any]:
    job = jobinfo["job"]
    tm = jobinfo["tracking_manifest"] or {}
    raw = read_stage_metrics(job_dir, job)
    log_stage = {t["stage"]: t for t in log["timing_stages"]}  # last line per stage wins
    stages: dict[str, dict[str, Any]] = {}
    without: list[str] = []
    for stage in STAGES:
        r = raw[stage]
        doc, lt = r["metrics"], log_stage.get(stage)
        if doc is None and lt is None:
            if r["published"]:
                without.append(stage)
            continue
        entry: dict[str, Any] = {"generation": r["generation"]}
        if doc is not None:
            timeout = _num(doc.get("function_timeout_seconds"))
            entry.update(
                source=METRICS_FILE, started_at=doc.get("started_at"), finished_at=doc.get("finished_at"),
                elapsed_seconds=_num(doc["elapsed_seconds"]),
                phases={k: v for k, v in (doc.get("phases") or {}).items() if _num(v) is not None},
                gpu=_gpu_from_metrics(doc), function_timeout_seconds=timeout,
                timeout_margin_seconds=_num(doc.get("timeout_margin_seconds")),
                container=doc.get("container") if isinstance(doc.get("container"), dict) else None,
                frames=doc.get("frames") if isinstance(doc.get("frames"), dict) else None,
                api_calls={k: (doc.get("api_calls") or {}).get(k) for k in API_KEYS},
                attempt=doc.get("attempt"),
            )
            if lt is not None:
                diff = abs(lt["elapsed_seconds"] - entry["elapsed_seconds"])
                entry["log_crosscheck"] = {
                    "log_elapsed_seconds": lt["elapsed_seconds"], "difference_seconds": round(diff, 3),
                    "agrees": diff <= 1.0,
                }
        else:
            entry.update(
                source="log [v2-timing]", started_at=lt["started"], finished_at=lt["finished"],
                elapsed_seconds=lt["elapsed_seconds"], phases={},
                gpu={"name": lt["gpu"], "memory_mib": None} if lt["gpu"] else None,
                function_timeout_seconds=None, timeout_margin_seconds=lt["timeout_margin_seconds"],
                container=None, frames=None, api_calls={k: None for k in API_KEYS}, attempt=None,
            )
        margin, timeout = entry["timeout_margin_seconds"], entry["function_timeout_seconds"]
        if timeout is None and margin is not None:
            timeout = entry["elapsed_seconds"] + margin  # log lines carry no timeout; derived
            entry["function_timeout_derived"] = True
        entry["function_timeout_seconds"] = timeout
        entry["timeout_used_percent"] = (
            round(100 * (timeout - margin) / timeout, 2) if timeout and margin is not None else None
        )
        stages[stage] = entry

    measured = [s for s in STAGES if s in stages]
    if stages:
        sources = {e["source"] for e in stages.values()}
        sm: dict[str, Any] = {"status": "MEASURED", "sources": sorted(sources)}
        if without:
            sm["stages_without_metrics"] = without
            sm["reason_for_missing"] = (
                "published generation has no stage_metrics.json (stage reused an earlier generation, "
                "or " + REASON_NO_METRICS + ")"
            )
    else:
        sm = _nm(REASON_NO_METRICS if not any(r["generations_with_metrics"] for r in raw.values())
                 else "no metrics for the current generations",
                 source=f"{METRICS_FILE} per generation or [v2-timing] log lines")
    invalid = sum(r["invalid_metrics_files"] for r in raw.values())
    if invalid:
        sm["invalid_metrics_files"] = invalid

    published = [s for s in STAGES if raw[s]["published"]]
    elapsed_sum = round(sum(stages[s]["elapsed_seconds"] for s in measured), 3) if measured else None
    complete = bool(measured) and all(s in stages for s in published)
    jobs = log["timing_jobs"]
    pick = [j for j in jobs if j["job_id"] == job.get("job_id")] or jobs
    wall = pick[-1]["total_wall_seconds"] if pick else None
    overhead = round(wall - elapsed_sum, 3) if wall is not None and elapsed_sum is not None and complete else None

    margins = [(s, stages[s]["timeout_margin_seconds"]) for s in measured if stages[s]["timeout_margin_seconds"] is not None]
    timeout_info = None
    if margins:
        tight, tight_margin = min(margins, key=lambda x: x[1])
        pcts = [(stages[s]["timeout_used_percent"], s) for s in measured if stages[s]["timeout_used_percent"] is not None]
        timeout_info = {
            "smallest_margin_seconds": tight_margin, "smallest_margin_stage": tight,
            "percent_of_timeout_used_at_smallest_margin": stages[tight]["timeout_used_percent"],
            "max_percent_of_timeout_used": max(pcts)[0] if pcts else None,
        }

    gpu_names = sorted({e["gpu"]["name"] for e in stages.values() if e["gpu"] and e["gpu"].get("name")})
    gpu_mem = sorted({e["gpu"]["memory_mib"] for e in stages.values() if e["gpu"] and e["gpu"].get("memory_mib")})
    gpu = {"names": gpu_names, "memory_mib": gpu_mem,
           "from": "stage metrics / timing lines" if gpu_names else None}
    if not gpu_names and log["gpus"]:
        gpu = {"names": sorted({g["name"] for g in log["gpus"]}),
               "memory_mib": sorted({g["memory_mib"] for g in log["gpus"] if "memory_mib" in g}),
               "from": "CUDA line in the log"}

    totals: dict[str, Any] = {}
    for key in API_KEYS:
        vals = {s: stages[s]["api_calls"].get(key) for s in measured}
        known = [v for v in vals.values() if isinstance(v, int) and not isinstance(v, bool)]
        unknown = [s for s, v in vals.items() if v is None]
        totals[key] = {"total": sum(known) if known else None, "unknown_in_stages": unknown,
                       "partial": bool(known) and bool(unknown)}

    attempts = job.get("attempts") if isinstance(job.get("attempts"), dict) else {}
    errors = job.get("errors") if isinstance(job.get("errors"), list) else []
    retries = {
        "attempts_by_stage": attempts or None,
        "retried_stages": sorted(s for s, n in attempts.items() if isinstance(n, int) and n > 1),
        "recorded_error_count": len(errors),
        "recorded_errors": [str(e)[:300] for e in errors[:10]],
        "note": "a stage that reused an existing generation records no new metrics",
    }

    gen_info = {s: {"count": len(raw[s]["published"]),
                    "bytes": sum(raw[s]["bytes_by_generation"].values()),
                    "bytes_by_generation": raw[s]["bytes_by_generation"]} for s in STAGES if raw[s]["published"]}
    published_bytes = sum(g["bytes"] for g in gen_info.values())
    generations = {"by_stage": gen_info, "total_count": sum(g["count"] for g in gen_info.values()),
                   "total_bytes": published_bytes, "download_bytes": jobinfo["download_bytes"]}

    # volume growth
    before, after = _read_listing(listing_before), _read_listing(listing_after)
    if before is None or after is None:
        growth: dict[str, Any] = _nm(
            "give --volume-listing-before and --volume-listing-after (copies of `modal volume ls v2-job-data <job>`)",
            published_bytes_as_downloaded=published_bytes)
    elif before["total_bytes"] is None or after["total_bytes"] is None:
        growth = _nm("a listing had no parseable sizes (accepted: JSON {path: bytes} or `modal volume ls` text with sizes)",
                     published_bytes_as_downloaded=published_bytes, before=before, after=after)
    else:
        growth = {"status": "MEASURED", "growth_bytes": after["total_bytes"] - before["total_bytes"],
                  "before_bytes": before["total_bytes"], "after_bytes": after["total_bytes"],
                  "approximate": before["approximate"] or after["approximate"],
                  "published_bytes_as_downloaded": published_bytes}

    # costs
    vid = _video_seconds(tm, tm.get("frame_bounds") or (jobinfo["generations"]["tracking"]["verification"] or {}).get("frame_bounds"), video_seconds)
    if gpu_rate is None:
        cost: dict[str, Any] = _nm(REASON_NO_RATE)
    elif elapsed_sum is None:
        cost = _nm(f"no stage timings: {REASON_NO_METRICS}", rate_usd_per_hour=gpu_rate)
    else:
        lower = elapsed_sum / 3600 * gpu_rate
        upper = wall / 3600 * gpu_rate if wall is not None else None
        cost = {"status": "MEASURED", "rate_usd_per_hour": gpu_rate,
                "estimated_usd_gpu_stage_time": round(lower, 6),
                "estimated_usd_upper_bound_wall_time": round(upper, 6) if upper is not None else None,
                "upper_bound_note": None if upper is not None else "no client-side wall time in the log",
                "stages_complete": complete, "disclaimer": DISCLAIMER}
        if not complete:
            cost["partial_note"] = "some published stages have no timings; the figures cover only: " + ", ".join(measured)
    per_min: dict[str, Any]
    if cost["status"] != "MEASURED":
        per_min = _nm("needs the cost estimate: " + cost["reason"])
    elif not vid["seconds"]:
        per_min = _nm("video duration unknown: " + vid["source"])
    else:
        minutes = vid["seconds"] / 60
        per_min = {"status": "MEASURED",
                   "usd_per_video_minute_gpu_stage_time": round(cost["estimated_usd_gpu_stage_time"] / minutes, 6),
                   "usd_per_video_minute_upper_bound": (
                       round(cost["estimated_usd_upper_bound_wall_time"] / minutes, 6)
                       if cost["estimated_usd_upper_bound_wall_time"] is not None else None)}
    cost["video_seconds"] = vid["seconds"]
    cost["video_seconds_source"] = vid["source"]
    cost["per_video_minute"] = per_min

    if volume_rate is None:
        vcost: dict[str, Any] = _nm("pass --volume-rate-usd-per-gib-month from the Modal plan"
                                    + ("" if volume_included_gib is not None else " (and --volume-included-gib if the plan includes storage)"))
    else:
        basis = growth["growth_bytes"] if growth["status"] == "MEASURED" else published_bytes
        gib = max(basis, 0) / 1024**3
        included = volume_included_gib or 0.0
        billable = max(gib - included, 0.0)
        vcost = {"status": "MEASURED", "gib": round(gib, 6), "included_gib": included,
                 "billable_gib": round(billable, 6), "rate_usd_per_gib_month": volume_rate,
                 "usd_per_month": round(billable * volume_rate, 6),
                 "basis": "volume growth" if growth["status"] == "MEASURED" else "published bytes as downloaded (growth not measured)",
                 "note": "included amount is not shared with other jobs here; compare with the whole Volume on the dashboard"}

    return {
        "stage_metrics": sm, "stages": stages, "elapsed_sum_seconds": elapsed_sum,
        "elapsed_sum_complete": complete,
        "total_wall_seconds": wall,
        "overhead_seconds": overhead,
        "overhead_note": "wall minus sum of stage elapsed: cold start, model load and transfer overhead",
        "timeout": timeout_info, "gpu": gpu,
        "api_calls": {"totals": totals, "note": "null counts stay unknown, they are never reported as 0"},
        "retries": retries, "generations": generations,
        "volume_growth": growth, "volume_cost": vcost, "cost": cost,
    }


# ------------------------------------------------- repeat / retry / integrations
TRACKING_MARKERS = ("propagate_in_video", "started new session", "Running add_prompt", "frame loading (OpenCV)")
_STAGE_FAILURE = re.compile(
    r"lease.{0,40}expired|expired.{0,40}lease|stage\s+failed|previous\s+.{0,30}failed", re.I)


def analyze_stage_log(text: str) -> dict[str, Any]:
    """What a repeat/retry log proves: final JSON, tracking activity, timing lines, preemption."""
    clean = "\n".join(_clean_lines(text))
    parsed = parse_log(text)
    return {
        "final": parsed["final_json"] or {},
        "tracking_markers": [m for m in TRACKING_MARKERS if m in clean],
        "stage_timing_lines": len(parsed["timing_stages"]),
        "stats_elapsed": next((t["elapsed_seconds"] for t in reversed(parsed["timing_stages"]) if t["stage"] == "stats"), None),
        "wall_seconds": parsed["timing_jobs"][-1]["total_wall_seconds"] if parsed["timing_jobs"] else None,
        "preempted": "terminated due to preemption" in clean,
        "failure_evidence": bool(_STAGE_FAILURE.search(clean)),
    }


def _repeat_row(job: dict[str, Any], gens: dict[str, Any], text: str) -> tuple[str, str]:
    a = analyze_stage_log(text)
    final = a["final"]
    stats_id, track_id = gens["stats"]["id"], gens["tracking"]["id"]
    failed = []
    if not stats_id or final.get("generation") != stats_id:
        failed.append("(a) log generation differs from the job stats generation")
    if not track_id or final.get("tracking_generation") != track_id:
        failed.append("(a) log tracking_generation differs from the job tracking generation")
    if a["tracking_markers"]:
        failed.append("(b) tracking activity in the log: " + ", ".join(a["tracking_markers"]))
    if a["stage_timing_lines"]:
        failed.append(f"(c) {a['stage_timing_lines']} [v2-timing] stage= line(s) present, a pure reuse writes none")
    evidence = (f"log generation={final.get('generation')}, tracking_generation={final.get('tracking_generation')}; "
                f"job stats={stats_id}, tracking={track_id}; attempts={json.dumps(job.get('attempts'), sort_keys=True)}")
    if a["wall_seconds"] is not None:
        evidence += f"; client-side total_wall_seconds={a['wall_seconds']}"
    if a["preempted"]:
        evidence += "; preempted and restarted"
    return (FAIL, "failed conditions: " + "; ".join(failed) + ". " + evidence) if failed else (PASS, evidence)


def _first_error_text(job: dict[str, Any]) -> str | None:
    errors = job.get("errors")
    if not isinstance(errors, list) or not errors:
        return None
    first = errors[0]
    text = first.get("error") if isinstance(first, dict) and "error" in first else first
    return redact_text(str(text))[:200]


def _retry_row(job: dict[str, Any], gens: dict[str, Any], text: str) -> tuple[str, str]:
    a = analyze_stage_log(text)
    final = a["final"]
    error = _first_error_text(job)
    attempts = (job.get("attempts") or {}).get("stats") if isinstance(job.get("attempts"), dict) else None
    failed = []
    if error is None and not a["failure_evidence"]:
        failed.append("no recorded stage failure (job.json errors empty) and no expired/failed state in the retry log")
    if job.get("state") != "complete" or job.get("lease") is not None:
        failed.append(f"job state={job.get('state')!r}, lease={job.get('lease')!r} (need complete and null)")
    if not isinstance(attempts, int) or attempts < 2:
        failed.append(f"stats attempts={attempts!r} (need >= 2)")
    if final.get("status") != "complete":
        failed.append(f"retry log final status={final.get('status')!r}")
    if not gens["tracking"]["id"] or final.get("tracking_generation") != gens["tracking"]["id"]:
        failed.append("retry log tracking_generation differs from the job tracking generation")
    if a["tracking_markers"]:
        failed.append("tracking was recomputed, markers: " + ", ".join(a["tracking_markers"]))
    evidence = f"stats attempts={attempts}"
    if error is not None:
        evidence += f"; first recorded error: {error}"
    if a["stats_elapsed"] is not None:
        evidence += f"; retry stats stage elapsed {a['stats_elapsed']} s"
    return (FAIL, "failed conditions: " + "; ".join(failed) + ". " + evidence) if failed else (PASS, evidence)


def _jsonl_count(path: Path) -> int | None:
    return len(_jsonl_rows(path)[0]) if path.is_file() else None


def integration_status(jobinfo: dict[str, Any], stats_dir: Path | None) -> list[dict[str, Any]]:
    """Per integration: verified / failed (configured, no output) / unavailable, from job artifacts only."""
    res = jobinfo["stats_results"] or {}
    cfg = res.get("config") or {}
    nums = jobinfo["numbers"]
    out: list[dict[str, Any]] = []

    def add(name: str, status: str, evidence: str) -> None:
        out.append({"integration": name, "status": status, "evidence": evidence})

    d = stats_dir
    shot = _jsonl_count(d / "shot_raw_predictions.jsonl") if d else None
    if shot is None:
        add("hosted shot detector", "unavailable", "shot_raw_predictions.jsonl not in the stats generation")
    elif shot:
        add("hosted shot detector", "verified", f"shot_raw_predictions.jsonl: {shot} rows (frames)")
    else:
        add("hosted shot detector", "failed", "shot_raw_predictions.jsonl has no rows")

    proj = _jsonl_count(d / "projected_tracks.jsonl") if d else None
    model = cfg.get("court_model_id")
    if proj:
        add("court model", "verified", f"projected_tracks.jsonl: {proj} rows (court_model_id={model})")
    elif proj == 0 or model:
        add("court model", "failed", f"court model configured ({model}) but projected_tracks.jsonl has no rows or is missing")
    else:
        add("court model", "unavailable", "no court model configured or projected_tracks.jsonl")

    ball = jobinfo["ball"]
    if ball and ball["non_lost_rows"] > 0:
        add("TrackNet", "verified", f"ball_tracks: {ball['non_lost_rows']} non-lost of {ball['rows']} rows")
    elif cfg.get("tracknet_enabled") is True or ball:
        add("TrackNet", "failed", f"enabled but ball_tracks has no non-lost rows ({ball['rows'] if ball else 'file missing'} rows)")
    else:
        add("TrackNet", "unavailable", "not enabled in this job")

    teams = _read_json(d / "teams.json") if d else None
    assigned = (teams or {}).get("assignments_by_track_id") if isinstance(teams, dict) else None
    if assigned:
        add("team embedder / team ID", "verified", f"teams.json: {len(assigned)} track assignments")
    elif teams is not None:
        add("team embedder / team ID", "failed", "teams.json has no assignments")
    else:
        add("team embedder / team ID", "unavailable", "teams.json not in the stats generation")

    if nums["numbers_file"] and nums["file_exists"] and nums["manifest_number_artifacts"]:
        add("hosted jersey-number OCR", "verified",
            f"numbers_file={nums['numbers_file']}; artifacts listed: {', '.join(nums['manifest_number_artifacts'])}")
    elif nums["number_predictor_backend"] == "roboflow-hosted":
        add("hosted jersey-number OCR", "failed",
            f"configured (number_predictor_backend=roboflow-hosted) but numbers_file={nums['numbers_file']!r}, "
            f"file exists={nums['file_exists']}, number artifacts: {nums['manifest_number_artifacts'] or 'none'}")
    else:
        add("hosted jersey-number OCR", "unavailable: not run in this job",
            f"number_predictor_backend={nums['number_predictor_backend']!r}, no numbers artifact")

    tm = jobinfo["tracking_manifest"] or {}
    trk = (jobinfo["generations"]["tracking"]["verification"] or {})
    if tm.get("observation_count") and trk.get("listed_count"):
        add("SAM 3.1 tracking", "verified",
            f"tracking package: observation_count={tm['observation_count']}, {trk['listed_count']} listed artifacts")
    elif jobinfo["generations"]["tracking"]["id"]:
        add("SAM 3.1 tracking", "failed", "tracking generation exists but is empty")
    else:
        add("SAM 3.1 tracking", "unavailable", "no tracking generation in the job folder")
    return out


# ----------------------------------------------------------------- checklist
def build_checklist(
    log: dict[str, Any], jobinfo: dict[str, Any], replay: dict[str, Any], perf: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    inputs = inputs or {}
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

    ident = jobinfo["identity"]
    status_txt = ident["tracking_identity"] or "absent"
    if ident["rows_total"] is None:
        add("Stable player identity present in tracks", NOT_MEASURED,
            f"no parseable tracks.jsonl in the job folder; tracking_identity={status_txt}")
    elif ident["rows_total"] == 0:
        add("Stable player identity present in tracks", NOT_MEASURED,
            f"{ident['tracks_source']} has no parseable rows; tracking_identity={status_txt}")
    elif ident["rows_with_stable_player_id"] == 0 or ident["tracking_identity"] == "rejected":
        reasons = tm.get("acceptance_reasons") or []
        add("Stable player identity present in tracks", FAIL,
            f"tracking_identity={status_txt}; {ident['rows_with_stable_player_id']} of {ident['rows_total']} "
            f"rows in {ident['tracks_source']} carry stable_player_id; reasons: "
            f"{'; '.join(map(str, reasons[:5])) or 'none recorded'}"
            + (f" (+{len(reasons) - 5} more)" if len(reasons) > 5 else ""))
    else:
        add("Stable player identity present in tracks", PASS,
            f"tracking_identity={status_txt}; {ident['rows_with_stable_player_id']} of {ident['rows_total']} "
            f"rows in {ident['tracks_source']} carry stable_player_id")

    nums = jobinfo["numbers"]
    if not nums["results_present"]:
        add("Jersey number OCR ran", NOT_MEASURED, "stats results.json not available")
    elif nums["numbers_file"] and nums["file_exists"]:
        add("Jersey number OCR ran", PASS,
            f"numbers_file={nums['numbers_file']} exists in the stats generation "
            f"(manifest number artifacts: {nums['manifest_number_artifacts'] or 'none listed'}); "
            "this proves an artifact was produced, not that its numbers are correct")
    else:
        why = (
            "configured (NUMBER_PREDICTOR_BACKEND=roboflow-hosted in the image) but no numbers artifact was produced"
            if nums["number_predictor_backend"] == "roboflow-hosted"
            else f"no numbers artifact was produced (number_predictor_backend={nums['number_predictor_backend']!r})"
        )
        add("Jersey number OCR ran", FAIL,
            f"{why}; numbers_file={nums['numbers_file']!r}, file exists={nums['file_exists']}, "
            f"manifest number artifacts: {nums['manifest_number_artifacts'] or 'none'}")

    cfg = res.get("config") or {}
    keys = {k: cfg[k] for k in ("tracknet_enabled", "court_model_id", "number_predictor_backend") if k in cfg}
    integ_item = "Configured external model credentials and optional integrations are verified or explicitly marked unavailable"
    integrations = jobinfo.get("integrations") or []
    hosted_evidence = any(i["integration"] in ("hosted shot detector", "court model") and i["status"] == "verified"
                          for i in integrations)
    if hosted_evidence:
        bad = [i for i in integrations if i["status"] == "failed"]
        listing = "; ".join(f"{i['integration']}: {i['status']} ({i['evidence']})" for i in integrations)
        add(integ_item, FAIL if bad else PASS,
            (f"configured but no output: {', '.join(i['integration'] for i in bad)}. " if bad else "")
            + listing + ". Names only, no credential values are read or printed")
    else:
        add(integ_item, NOT_MEASURED,
            f"configuration seen in results.json: {json.dumps(keys, sort_keys=True)}; "
            "no hosted-model artifacts (shot_raw_predictions.jsonl / projected_tracks.jsonl) to derive per-integration "
            "status from; a successful stats stage is indirect evidence only. Whether OCR actually ran is judged "
            "separately by the 'Jersey number OCR ran' row, which depends on an artifact, not on configuration")

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

    repeat_item = "Repeating the same stage reuses the matching generation without recomputation"
    if inputs.get("repeat_log") is not None:
        add(repeat_item, *_repeat_row(job, gens, inputs["repeat_log"]))
    else:
        add(repeat_item, NOT_RUN,
            "a single run was provided; repeat the command and pass its log with --repeat-log")
    retry_item = "Stats-only retry succeeds after a tracking failure or expired lease"
    if inputs.get("retry_log") is not None:
        add(retry_item, *_retry_row(job, gens, inputs["retry_log"]))
    else:
        add(retry_item, NOT_RUN,
            f"job attempts={json.dumps(job.get('attempts'), sort_keys=True)}, errors={job.get('errors')!r}; "
            "no --retry-log provided")

    video = replay.get("video")
    replay_item = "Downloaded artifacts render with debug_v2.py using an explicit generation"
    if video:
        if not video["opens"]:
            add(replay_item, FAIL, f"{video['file']} could not be opened with cv2")
        elif video["expected_frames"] is None:
            add(replay_item, NOT_MEASURED,
                f"{video['file']} opens with cv2 ({video['frame_count']} frames) but the tracking "
                "generation has no readable frame_bounds to compare")
        elif video["frame_count"] == video["expected_frames"]:
            add(replay_item, PASS,
                f"{video['file']} opens with cv2; {video['frame_count']} frames == tracking frame bounds "
                f"{video['frame_bounds']} ({video['expected_frames']} frames)")
        else:
            add(replay_item, FAIL,
                f"{video['file']} has {video['frame_count']} frames, tracking frame bounds "
                f"{video['frame_bounds']} imply {video['expected_frames']}")
    elif replay["requested"]:
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

    perf = perf or {}
    ops_item = "GPU time, timeout margin, Volume growth, and download size are recorded"
    stages_perf = perf.get("stages") or {}
    missing = []
    if perf.get("elapsed_sum_seconds") is None:
        missing.append(f"GPU/stage time ({REASON_NO_METRICS}; no stage_metrics.json or [v2-timing] lines)")
    elif not perf.get("elapsed_sum_complete"):
        missing.append("GPU/stage time for stages: " + ", ".join(perf["stage_metrics"].get("stages_without_metrics", [])))
    if not perf.get("timeout"):
        missing.append("timeout margin")
    if jobinfo["download_bytes"] is None:
        missing.append("download size")
    if (perf.get("volume_growth") or {}).get("status") != "MEASURED":
        missing.append("Volume growth (no usable --volume-listing-before/--volume-listing-after)")
    if missing:
        add(ops_item, NOT_MEASURED,
            "missing: " + "; ".join(missing) + f". Present: download size {jobinfo['download_bytes']} bytes")
    else:
        add(ops_item, PASS,
            f"stage elapsed sum {perf['elapsed_sum_seconds']} s, smallest timeout margin "
            f"{perf['timeout']['smallest_margin_seconds']} s, Volume growth "
            f"{perf['volume_growth']['growth_bytes']} bytes, download {jobinfo['download_bytes']} bytes")

    sm = perf.get("stage_metrics") or {}
    if sm.get("status") == "MEASURED" and not sm.get("stages_without_metrics"):
        add("Operational metrics recorded by the pipeline", PASS,
            f"{len(stages_perf)} stage(s) with metrics from {', '.join(sm['sources'])}: {', '.join(stages_perf)}")
    elif sm.get("status") == "MEASURED":
        add("Operational metrics recorded by the pipeline", NOT_MEASURED,
            f"metrics for {', '.join(stages_perf)}; missing for {', '.join(sm['stages_without_metrics'])} "
            f"({sm['reason_for_missing']})")
    else:
        add("Operational metrics recorded by the pipeline", NOT_MEASURED,
            f"{sm.get('reason', REASON_NO_METRICS)}; expected {sm.get('source', METRICS_FILE)}")

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
    retention_item = "Cleanup/retention implications are recorded for follow-up"
    policy = inputs.get("retention_policy")
    if policy and policy["size_bytes"] > 0:
        published = ((perf.get("generations") or {}).get("total_bytes"))
        add(retention_item, PASS,
            f"retention policy file {policy['name']} ({policy['size_bytes']} bytes); "
            f"per-job published bytes {published}")
    else:
        add(retention_item, NOT_MEASURED,
            "per-generation storage bytes are listed above; retention policy and Volume cleanup "
            "are not derivable from the inputs and need a follow-up decision (--retention-policy PATH)")
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


def _probe_video(path: Path, bounds: Any) -> dict[str, Any]:
    expected = None
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and all(isinstance(b, int) for b in bounds):
        expected = bounds[1] - bounds[0] + 1
    out = {"file": path.name, "opens": False, "frame_count": None,
           "frame_bounds": bounds if expected is not None else None, "expected_frames": expected}
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            if cap.isOpened():
                out["opens"] = True
                # header count only: decoding every frame is not worth it for a sanity check
                out["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
    except Exception:  # cv2 missing or broken: report as not opening rather than crash
        pass
    return out


def _replay_info(path: str | None, bounds: Any = None) -> dict[str, Any]:
    info: dict[str, Any] = {"requested": path is not None, "exists": False, "files": 0, "bytes": 0, "video": None}
    if path is not None:
        p = Path(path)
        mp4 = None
        if p.is_file():
            info.update(exists=True, files=1, bytes=p.stat().st_size)
            mp4 = p if p.suffix.lower() == ".mp4" else None
        elif p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            info.update(exists=bool(files), files=len(files), bytes=sum(f.stat().st_size for f in files))
            mp4 = next((f for f in sorted(files) if f.suffix.lower() == ".mp4"), None)
        if mp4 is not None:
            info["video"] = _probe_video(mp4, bounds)
    return info


def _not_measured(perf: dict[str, Any], durations: dict[str, Any]) -> dict[str, str]:
    out = {"gpu_runtime_billing": "Modal usage/billing export for the app run (the dashboard is the source of truth)"}
    if perf["cost"]["status"] != "MEASURED":
        out["cost"] = perf["cost"]["reason"]
    if perf["cost"]["per_video_minute"]["status"] != "MEASURED":
        out["cost_per_video_minute"] = perf["cost"]["per_video_minute"]["reason"]
    if not perf["timeout"]:
        out["timeout_margin"] = perf["stage_metrics"].get("reason", "no stage with a recorded margin")
    if perf["volume_growth"]["status"] != "MEASURED":
        out["volume_growth"] = perf["volume_growth"]["reason"]
    if perf["total_wall_seconds"] is None:
        out["total_job_duration"] = durations["total_job_source"]
    totals = perf["api_calls"]["totals"]
    if not any(t["total"] is not None for t in totals.values()):
        out["model_api_calls"] = "api_calls in stage_metrics.json (unknown, not zero); otherwise Roboflow dashboard counters"
    return out


def build_report(
    log_text: str,
    job_dir: Path,
    commit: str | None = None,
    replay_output: str | None = None,
    notes: str | None = None,
    also_logs: dict[str, str] | None = None,
    gpu_rate_usd_per_hour: float | None = None,
    video_seconds: float | None = None,
    volume_included_gib: float | None = None,
    volume_rate_usd_per_gib_month: float | None = None,
    volume_listing_before: str | None = None,
    volume_listing_after: str | None = None,
    repeat_log: str | None = None,
    retry_log: str | None = None,
    retention_policy: str | None = None,
) -> dict[str, Any]:
    log = parse_log(log_text)
    jobinfo = inspect_job(job_dir)
    stats_id = jobinfo["generations"]["stats"]["id"]
    stats_gen_dir = job_dir / "stats" / "generations" / stats_id if stats_id else None
    jobinfo["integrations"] = integration_status(
        jobinfo, stats_gen_dir if stats_gen_dir and stats_gen_dir.is_dir() else None)
    perf = build_performance(
        log, job_dir, jobinfo, gpu_rate=gpu_rate_usd_per_hour, video_seconds=video_seconds,
        volume_included_gib=volume_included_gib, volume_rate=volume_rate_usd_per_gib_month,
        listing_before=volume_listing_before, listing_after=volume_listing_after,
    )
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
        "total_job_seconds": perf["total_wall_seconds"] if perf["total_wall_seconds"] is not None else NOT_MEASURED,
        "total_job_source": (
            "client-side [v2-timing] job line (includes cold start and model load)"
            if perf["total_wall_seconds"] is not None else
            "no [v2-timing] job line in the log; Modal function-call duration (dashboard / `modal app` logs) has it"
        ),
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

    bounds = tm.get("frame_bounds") or (gens["tracking"]["verification"] or {}).get("frame_bounds")
    replay = _replay_info(replay_output, bounds)
    gpu_logs = {"primary": log["gpu_strings"]}
    for name, text in (also_logs or {}).items():
        gpu_logs[name] = parse_log(text)["gpu_strings"]
    distinct = {tuple(v) for v in gpu_logs.values() if v}
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
            "gpu_strings_by_log": gpu_logs,
            "gpu_variants_differ_across_logs": len(distinct) > 1,
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
        "tracking_identity": jobinfo["identity"],
        "numbers": jobinfo["numbers"],
        "outcome_summary": jobinfo["outcomes"],
        "operational_notes": notes,
        "warnings": log["warnings"],
        "performance": perf,
        "not_measured": _not_measured(perf, durations),
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
    policy = None
    if retention_policy:
        try:
            policy = {"name": Path(retention_policy).name, "size_bytes": Path(retention_policy).stat().st_size}
        except OSError:
            policy = None
    report["integrations"] = jobinfo["integrations"]
    report["evidence_inputs"] = {
        "repeat_log": repeat_log is not None,
        "retry_log": retry_log is not None,
        "retention_policy": policy["name"] if policy else None,
        "retention_policy_size_bytes": policy["size_bytes"] if policy else None,
    }
    report["checklist"] = build_checklist(
        log, jobinfo, replay, perf,
        {"repeat_log": repeat_log, "retry_log": retry_log, "retention_policy": policy})

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
def _u(value: Any) -> str:
    return "unknown" if value is None else str(value)


def _usd(value: Any) -> str:
    return NOT_MEASURED if value is None else f"{value:.4f} USD"


def render_performance(p: dict[str, Any]) -> list[str]:
    L: list[str] = []
    a = L.append
    a("## Performance and cost")
    sm = p["stage_metrics"]
    if sm["status"] == "MEASURED":
        a(f"- Stage metrics: measured from {', '.join(sm['sources'])}")
        if sm.get("stages_without_metrics"):
            a(f"- Stages with a published generation but no metrics: {', '.join(sm['stages_without_metrics'])} ({sm['reason_for_missing']})")
    else:
        a(f"- Stage metrics: {NOT_MEASURED} ({sm['reason']}; expected {sm['source']})")
    if sm.get("invalid_metrics_files"):
        a(f"- Unreadable stage_metrics.json files ignored: {sm['invalid_metrics_files']}")
    for stage, e in p["stages"].items():
        a(f"- {stage} `{_u(e['generation'])}` ({e['source']}): elapsed {e['elapsed_seconds']} s, "
          f"timeout margin {_u(e['timeout_margin_seconds'])} s of {_u(e['function_timeout_seconds'])} s"
          f"{' (derived)' if e.get('function_timeout_derived') else ''}, "
          f"{_u(e['timeout_used_percent'])}% of timeout used, attempt {_u(e['attempt'])}")
        if e["phases"]:
            a("  - phases: " + ", ".join(f"{k} {v}" for k, v in e["phases"].items()))
        if e["frames"]:
            a(f"  - frames requested {_u(e['frames'].get('requested'))}, processed {_u(e['frames'].get('processed'))}")
        if e["container"]:
            a(f"  - container cpu {_u(e['container'].get('cpu'))}, memory {_u(e['container'].get('memory_mib'))} MiB")
        if e.get("log_crosscheck"):
            c = e["log_crosscheck"]
            a(f"  - log cross-check: [v2-timing] elapsed {c['log_elapsed_seconds']} s, "
              f"{'agrees' if c['agrees'] else 'DIFFERS'} (delta {c['difference_seconds']} s)")
    if p["elapsed_sum_seconds"] is None:
        a(f"- Sum of stage elapsed (container-side): {NOT_MEASURED}")
    else:
        a(f"- Sum of stage elapsed (container-side): {p['elapsed_sum_seconds']} s"
          f"{'' if p['elapsed_sum_complete'] else ' (partial: some published stages have no timings)'}")
    a(f"- Client-side total wall time: {_fmt(p['total_wall_seconds'])}{' s' if p['total_wall_seconds'] is not None else ''}")
    if p["overhead_seconds"] is not None:
        a(f"- Difference wall - stage sum: {p['overhead_seconds']} s ({p['overhead_note']})")
    else:
        a(f"- Difference wall - stage sum: {NOT_MEASURED} (needs the wall time and complete stage timings)")
    t = p["timeout"]
    if t:
        a(f"- Smallest timeout margin: {t['smallest_margin_seconds']} s in stage {t['smallest_margin_stage']} "
          f"({_u(t['percent_of_timeout_used_at_smallest_margin'])}% of its timeout used; highest use across stages "
          f"{_u(t['max_percent_of_timeout_used'])}%)")
    else:
        a(f"- Smallest timeout margin: {NOT_MEASURED} ({sm.get('reason', 'no margin recorded')})")
    g = p["gpu"]
    a(f"- GPU: {', '.join(g['names']) or NOT_MEASURED}"
      f"{', memory ' + ', '.join(map(str, g['memory_mib'])) + ' MiB' if g['memory_mib'] else ''}"
      f"{' (from ' + g['from'] + ')' if g['from'] else ''}")
    totals = p["api_calls"]["totals"]
    if p["stages"]:
        for key, tot in totals.items():
            extra = f" (unknown in: {', '.join(tot['unknown_in_stages'])}{'; total is a lower bound' if tot['partial'] else ''})" if tot["unknown_in_stages"] else ""
            a(f"- API/model calls, {key}: total {_u(tot['total'])}{extra}")
        for stage, e in p["stages"].items():
            a(f"  - {stage}: " + ", ".join(f"{k.replace('roboflow_', '')}={_u(v)}" for k, v in e["api_calls"].items()))
    else:
        a(f"- API/model calls: {NOT_MEASURED} (no stage metrics)")
    r = p["retries"]
    a(f"- Attempts: `{json.dumps(r['attempts_by_stage'], sort_keys=True)}`; retried stages: {', '.join(r['retried_stages']) or 'none'}; "
      f"recorded errors: {r['recorded_error_count']}" + (f" `{r['recorded_errors']}`" if r["recorded_errors"] else "")
      + f" ({r['note']})")
    gi = p["generations"]
    for stage, info in gi["by_stage"].items():
        a(f"- Published generations, {stage}: {info['count']} totalling {info['bytes']} bytes")
    a(f"- Published generations total: {gi['total_count']} / {gi['total_bytes']} bytes; downloaded job folder {_fmt(gi['download_bytes'])} bytes")
    vg = p["volume_growth"]
    if vg["status"] == "MEASURED":
        a(f"- Volume growth: {vg['growth_bytes']} bytes (before {vg['before_bytes']}, after {vg['after_bytes']}"
          f"{'; sizes rounded by human units, approximate' if vg['approximate'] else ''})")
    else:
        a(f"- Volume growth: {NOT_MEASURED} ({vg['reason']}); published bytes (as downloaded): {vg['published_bytes_as_downloaded']}")
    c = p["cost"]
    a(f"- Estimated GPU cost: " + (NOT_MEASURED + f" ({c['reason']})" if c["status"] != "MEASURED" else
      f"{_usd(c['estimated_usd_gpu_stage_time'])} at {c['rate_usd_per_hour']} USD/hour (stage time); "
      f"upper bound with wall time {_usd(c['estimated_usd_upper_bound_wall_time'])}"
      + (f" ({c['upper_bound_note']})" if c.get("upper_bound_note") else "")))
    if c.get("partial_note"):
        a(f"  - {c['partial_note']}")
    a(f"- Video duration: {_fmt(c['video_seconds'])}{' s' if c['video_seconds'] is not None else ''} (source: {c['video_seconds_source']})")
    pm = c["per_video_minute"]
    if pm["status"] == "MEASURED":
        a(f"- Cost per video minute: {_usd(pm['usd_per_video_minute_gpu_stage_time'])} (stage time); upper bound {_usd(pm['usd_per_video_minute_upper_bound'])}")
    else:
        a(f"- Cost per video minute: {NOT_MEASURED} ({pm['reason']})")
    vc = p["volume_cost"]
    if vc["status"] == "MEASURED":
        a(f"- Volume storage cost: {_usd(vc['usd_per_month'])}/month for {vc['gib']} GiB, {vc['included_gib']} GiB included "
          f"(basis: {vc['basis']})")
    else:
        a(f"- Volume storage cost: {NOT_MEASURED} ({vc['reason']})")
    if c["status"] == "MEASURED":
        a(f"- Note: {c['disclaimer']}")
    else:
        a(f"- Note: {DISCLAIMER}")
    a("")
    return L


def _fmt(value: Any) -> str:
    return NOT_MEASURED if value is None else str(value)


def _fmt_absent(value: Any) -> str:
    return "absent" if value is None else str(value)


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
    for name, strings in cfg["gpu_strings_by_log"].items():
        a(f"- GPU string, {name} log: {'; '.join(f'`{x}`' for x in strings) or NOT_MEASURED}")
    if cfg["gpu_variants_differ_across_logs"]:
        a("- NOTE: the logs report different GPU variants for runs of this job")
    a("")
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
    total = d["total_job_seconds"]
    a(f"- Total job duration: {total}{' s' if total != NOT_MEASURED else ''} (source: {d['total_job_source']})")
    a("")
    L.extend(render_performance(report["performance"]))
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
    ident = report["tracking_identity"]
    a("## Tracking identity")
    a(f"- tracking_identity: {_fmt_absent(ident['tracking_identity'])}")
    a(f"- acceptance_reasons: `{json.dumps(ident['acceptance_reasons'])}`" if ident["acceptance_reasons"] is not None else "- acceptance_reasons: absent")
    a(f"- identity_excluded_observations: {_fmt_absent(ident['identity_excluded_observations'])}")
    a(f"- identity_excluded_observation_keys: {_fmt_absent(ident['identity_excluded_observation_keys'])}")
    if ident["rows_total"] is None:
        a(f"- tracks rows: {NOT_MEASURED} (no tracks.jsonl found)")
    else:
        a(f"- tracks rows ({ident['tracks_source']}): {ident['rows_total']} total, "
          f"{ident['rows_with_stable_player_id']} with stable_player_id, "
          f"{ident['rows_without_stable_player_id']} without, {ident['rows_unparseable']} unparseable lines")
    a("")
    nums = report["numbers"]
    a("## External integration evidence")
    a(f"- number_predictor_backend (config): {_fmt_absent(nums['number_predictor_backend'])}")
    a(f"- numbers_file: {_fmt_absent(nums['numbers_file'])}, file exists: {nums['file_exists']}")
    a(f"- number artifacts listed in the stats manifest: {nums['manifest_number_artifacts'] or 'none'}")
    a("")
    out = report["outcome_summary"]
    a("## Outcome summary (facts only, no accuracy claim)")
    a(f"- shot events: {_fmt(out['events'])}; projected shots: {_fmt(out['projected_shots'])}")
    a(f"- by outcome: `{json.dumps(out['by_outcome'], sort_keys=True)}`")
    a(f"- by outcome_source: `{json.dumps(out['by_outcome_source'], sort_keys=True) if out['by_outcome_source'] is not None else 'absent (field not emitted)'}`")
    a(f"- by shooter_attribution_method: `{json.dumps(out['by_shooter_attribution_method'], sort_keys=True) if out['by_shooter_attribution_method'] is not None else 'absent (field not emitted)'}`")
    a(f"- shots with null shooter: {_fmt(out['shots_with_null_shooter'])}")
    ball = out["ball_tracks"]
    a(f"- ball track rows: {ball['rows'] - ball['non_lost_rows']} lost, {ball['non_lost_rows']} non-lost of {ball['rows']}" if ball else f"- ball track rows: {NOT_MEASURED}")
    a("")
    a("## Operational findings")
    a(report["operational_notes"] if report["operational_notes"] else "No operational notes were provided (--notes).")
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


# ---------------------------------------------------------------------- diff
def _perf_numbers(p: Any) -> dict[str, Any]:
    p = p if isinstance(p, dict) else {}
    cost = p.get("cost") or {}
    pm = cost.get("per_video_minute") or {}
    out = {
        "perf.elapsed_sum_seconds": p.get("elapsed_sum_seconds"),
        "perf.total_wall_seconds": p.get("total_wall_seconds"),
        "perf.smallest_timeout_margin_seconds": (p.get("timeout") or {}).get("smallest_margin_seconds"),
        "perf.cost_usd_stage_time": cost.get("estimated_usd_gpu_stage_time"),
        "perf.cost_usd_upper_bound": cost.get("estimated_usd_upper_bound_wall_time"),
        "perf.cost_usd_per_video_minute": pm.get("usd_per_video_minute_gpu_stage_time"),
        "perf.volume_growth_bytes": (p.get("volume_growth") or {}).get("growth_bytes"),
        "perf.published_bytes": (p.get("generations") or {}).get("total_bytes"),
    }
    for stage, e in (p.get("stages") or {}).items():
        out[f"perf.elapsed_seconds.{stage}"] = e.get("elapsed_seconds")
    return out


def _key_numbers(r: dict[str, Any]) -> dict[str, Any]:
    ident, nums, out = r.get("tracking_identity") or {}, r.get("numbers") or {}, r.get("outcome_summary") or {}
    return {
        "job_id": r.get("job_id"),
        "tracking_generation": (r.get("job") or {}).get("tracking_generation"),
        "stats_generation": (r.get("job") or {}).get("stats_generation"),
        "hash_verification_ok": r.get("hash_verification_ok"),
        "tracking_identity": ident.get("tracking_identity"),
        "rows_with_stable_player_id": ident.get("rows_with_stable_player_id"),
        "rows_without_stable_player_id": ident.get("rows_without_stable_player_id"),
        "identity_excluded_observations": ident.get("identity_excluded_observations"),
        "numbers_file": nums.get("numbers_file"),
        "shot_events": out.get("events"),
        "by_outcome": out.get("by_outcome"),
        "by_outcome_source": out.get("by_outcome_source"),
        "by_shooter_attribution_method": out.get("by_shooter_attribution_method"),
        "shots_with_null_shooter": out.get("shots_with_null_shooter"),
        **_perf_numbers(r.get("performance")),
        "gpu_strings": ((r.get("gpu_and_configuration") or {}).get("gpu_strings_by_log") or {}).get("primary"),
    }


def diff_reports(a: dict[str, Any], b: dict[str, Any]) -> str:
    def cell(v: Any) -> str:
        return ("absent" if v is None else json.dumps(v, sort_keys=True)).replace("|", "/")

    status = lambda r: {c["item"]: c["status"] for c in r.get("checklist", [])}
    sa, sb = status(a), status(b)
    rows = [(f"checklist: {k}", sa.get(k), sb.get(k)) for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)]
    ka, kb = _key_numbers(a), _key_numbers(b)
    rows += [(k, ka.get(k), kb.get(k)) for k in dict.fromkeys([*ka, *kb]) if ka.get(k) != kb.get(k)]
    lines = ["| field | A | B |", "|---|---|---|"]
    lines += [f"| {k} | {cell(x)} | {cell(y)} |" for k, x, y in rows]
    if not rows:
        lines.append("| (no differences in checklist statuses or key numbers) | | |")
    return "\n".join(lines)


# ----------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", help="`modal run` log file")
    parser.add_argument("--job", help="downloaded job folder (contains job.json)")
    parser.add_argument("--diff", nargs=2, metavar=("A.json", "B.json"), help="compare two JSON reports and exit 0")
    parser.add_argument("--notes", help="markdown file with operational findings written by a human, embedded after redaction")
    parser.add_argument("--also-log", action="append", default=[], help="extra run log of the same job, only its GPU strings are listed (repeatable)")
    parser.add_argument("--commit", help="commit SHA of the deployed code (default: git rev-parse HEAD)")
    parser.add_argument("--replay-output", help="debug_v2.py render output (dir or .mp4), if the replay was run")
    parser.add_argument("--gpu-rate-usd-per-hour", type=float, help="GPU rate from the Modal plan; no default, cost is NOT_MEASURED without it")
    parser.add_argument("--video-seconds", type=float, help="video duration in seconds (default: derived from the tracking manifest frame bounds and fps)")
    parser.add_argument("--volume-included-gib", type=float, help="GiB of Volume storage included in the plan (default: none)")
    parser.add_argument("--volume-rate-usd-per-gib-month", type=float, help="Volume storage rate; needed for the storage cost")
    parser.add_argument("--volume-listing-before", help="listing of the job on the Volume before the run (JSON {path: bytes} or `modal volume ls` text)")
    parser.add_argument("--volume-listing-after", help="same listing after the run")
    parser.add_argument("--repeat-log", help="`modal run` log of repeating the identical staged command on the same job")
    parser.add_argument("--retry-log", help="log of a `--stats-only` retry after a failed stats stage or expired lease")
    parser.add_argument("--retention-policy", help="retention/cleanup policy file; only its name and size are reported")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    parser.add_argument("--output", help="write to this file instead of stdout")
    parser.add_argument("--max-files", type=int, default=40, help="max artifact rows per generation in Markdown")
    args = parser.parse_args(argv)

    if args.diff:
        loaded = []
        for name in args.diff:
            loaded.append(_read_json(Path(name)))
        if not all(isinstance(r, dict) for r in loaded):
            text = "diff skipped: could not read both reports as JSON objects\n"
        else:
            text = redact_text(diff_reports(*loaded)) + "\n"
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0
    if not args.log or not args.job:
        parser.error("--log and --job are required unless --diff is used")
    log_path, job_dir = Path(args.log), Path(args.job)
    if not log_path.is_file():
        parser.error(f"log file not found: {log_path}")
    if not job_dir.is_dir():
        parser.error(f"job folder not found: {job_dir}")
    also_logs = {}
    for extra in args.also_log:
        if not Path(extra).is_file():
            parser.error(f"--also-log file not found: {extra}")
        also_logs[Path(extra).name] = Path(extra).read_text(encoding="utf-8", errors="replace")
    notes = None
    if args.notes:
        if not Path(args.notes).is_file():
            parser.error(f"notes file not found: {args.notes}")
        notes = Path(args.notes).read_text(encoding="utf-8", errors="replace")
    extra: dict[str, str | None] = {}
    for opt in ("repeat_log", "retry_log"):
        path = getattr(args, opt)
        if path and not Path(path).is_file():
            parser.error(f"--{opt.replace('_', '-')} file not found: {path}")
        extra[opt] = Path(path).read_text(encoding="utf-8", errors="replace") if path else None
    report = build_report(
        log_path.read_text(encoding="utf-8", errors="replace"),
        job_dir, commit=args.commit, replay_output=args.replay_output,
        notes=notes, also_logs=also_logs,
        gpu_rate_usd_per_hour=args.gpu_rate_usd_per_hour, video_seconds=args.video_seconds,
        volume_included_gib=args.volume_included_gib,
        volume_rate_usd_per_gib_month=args.volume_rate_usd_per_gib_month,
        volume_listing_before=args.volume_listing_before, volume_listing_after=args.volume_listing_after,
        retention_policy=args.retention_policy, **extra,
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
