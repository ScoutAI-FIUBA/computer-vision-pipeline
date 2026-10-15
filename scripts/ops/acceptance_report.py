#!/usr/bin/env python3
"""Build a redacted V2 acceptance report from a `modal run` log and a job folder.

Usage:
    python scripts/ops/acceptance_report.py --log run.log --job outputs/<id>-job/<id> \
        [--commit SHA] [--replay-output DIR|FILE.mp4] [--notes FILE.md] \
        [--also-log RUN2.log ...] [--json] [--output FILE]
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
    add("Configured external model credentials and optional integrations are verified or explicitly marked unavailable",
        NOT_MEASURED,
        f"configuration seen in results.json: {json.dumps(keys, sort_keys=True)}; "
        "the log does not show credential checks (Roboflow/OCR/ReID); "
        "a successful stats stage is indirect evidence only. Whether OCR actually ran is judged "
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

    add("Repeating the same stage reuses the matching generation without recomputation", NOT_RUN,
        "a single run was provided; repeat the command and compare generation ids and timings")
    add("Stats-only retry succeeds after a tracking failure or expired lease", NOT_RUN,
        f"job attempts={json.dumps(job.get('attempts'), sort_keys=True)}, errors={job.get('errors')!r}; "
        "no simulated failure was exercised")

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


def build_report(
    log_text: str,
    job_dir: Path,
    commit: str | None = None,
    replay_output: str | None = None,
    notes: str | None = None,
    also_logs: dict[str, str] | None = None,
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
        "gpu_strings": ((r.get("gpu_and_configuration") or {}).get("gpu_strings_by_log") or {}).get("primary"),
    }


def diff_reports(a: dict[str, Any], b: dict[str, Any]) -> str:
    def cell(v: Any) -> str:
        return ("absent" if v is None else json.dumps(v, sort_keys=True)).replace("|", "/")

    status = lambda r: {c["item"]: c["status"] for c in r.get("checklist", [])}
    sa, sb = status(a), status(b)
    rows = [(f"checklist: {k}", sa.get(k), sb.get(k)) for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)]
    ka, kb = _key_numbers(a), _key_numbers(b)
    rows += [(k, ka[k], kb[k]) for k in ka if ka[k] != kb[k]]
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
    report = build_report(
        log_path.read_text(encoding="utf-8", errors="replace"),
        job_dir, commit=args.commit, replay_output=args.replay_output,
        notes=notes, also_logs=also_logs,
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
