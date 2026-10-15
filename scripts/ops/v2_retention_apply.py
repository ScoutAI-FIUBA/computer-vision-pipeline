"""Apply a ``v2_retention_report.py`` plan to the ``v2-job-data`` Modal Volume (the ONLY deleting code).

Safety model, all enforced in code:

* volume name is hard-locked to ``v2-job-data``;
* ``--only JOB[,JOB...]`` is mandatory, so nobody deletes someone else's job by accident;
* dry-run by default; deleting needs ``--apply --yes`` AND ``--plan-hash`` matching the plan;
* only ELIGIBLE items of kind generation/orphan whose path matches an exact allowlist regex
  (``<job>/<stage>/generations/<id>`` or a ``.stage-*`` / ``.*.tmp`` orphan) can ever be deleted;
  job.json, current.json, inputs, ``.job.lock`` and protected-name paths never match;
* every item is re-validated against the live volume right before it is removed;
* no wildcard, no recursive listing-based deletion: one explicit path per removal.

The volume is reached only through ``VolumeAdapter`` so tests use an in-memory fake.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Protocol

from scripts.ops.v2_retention_report import (ELIGIBLE, IN_PROGRESS_STATES, REPORT_SCHEMA, STAGES,
                                             compute_plan_hash, is_protected_name)

VOLUME_NAME = "v2-job-data"
_JOB = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_STAGE = "(?:" + "|".join(STAGES) + ")"
GENERATION_RE = re.compile(rf"^(?P<job>{_JOB})/(?P<stage>{_STAGE})/generations/(?P<id>[A-Za-z0-9][A-Za-z0-9._-]*)$")
ORPHAN_RE = re.compile(rf"^(?P<job>{_JOB})/(?:(?P<stage>{_STAGE})/)?"
                       r"(?P<id>\.stage-[A-Za-z0-9._-]+|\.current\.json\.tmp|\.[A-Za-z0-9._-]+\.tmp)$")


class VolumeAdapter(Protocol):
    def read_json(self, path: str) -> dict[str, Any] | None: ...
    def list_dir(self, path: str) -> list[str]: ...
    def exists(self, path: str) -> bool: ...
    def remove_tree(self, path: str) -> None: ...  # raises FileNotFoundError if missing


class ModalVolume:
    """Adapter over ``modal.Volume.from_name('v2-job-data')``. Never used by the tests."""

    def __init__(self) -> None:
        import modal  # imported lazily so tests and the planner never need Modal
        self._modal = modal
        self._vol = modal.Volume.from_name(VOLUME_NAME)

    def _missing(self) -> tuple[type, ...]:
        return (self._modal.exception.NotFoundError, FileNotFoundError)

    def read_json(self, path: str) -> dict[str, Any] | None:
        try:
            value = json.loads(b"".join(self._vol.read_file(path)).decode("utf-8"))
        except (*self._missing(), ValueError):
            return None
        return value if isinstance(value, dict) else None

    def list_dir(self, path: str) -> list[str]:
        try:
            return [e.path.rsplit("/", 1)[-1] for e in self._vol.listdir(path)]
        except self._missing():
            return []

    def exists(self, path: str) -> bool:
        parent, _, name = path.rpartition("/")
        return name in self.list_dir(parent)

    def remove_tree(self, path: str) -> None:
        try:
            self._vol.remove_file(path, recursive=True)
        except self._missing() as exc:
            raise FileNotFoundError(path) from exc


def allowlist_match(path: str) -> re.Match[str] | None:
    """Exact-path allowlist; None means the path can never be deleted."""
    if any(c in path for c in "*?[]{}\\") or path.startswith("/") or ".." in path.split("/"):
        return None
    if any(is_protected_name(part) for part in path.split("/")):
        return None
    return GENERATION_RE.match(path) or ORPHAN_RE.match(path)


def _lease_active(job: dict[str, Any], now: float) -> bool:
    lease = job.get("lease") or {}
    try:
        return bool(lease.get("owner")) and float(lease.get("expires_at") or 0) > now
    except (TypeError, ValueError):
        return True  # unreadable lease: assume live


def _pointer_targets(vol: VolumeAdapter, job_id: str, job: dict[str, Any], stage: str) -> set[str]:
    targets = set()
    current = vol.read_json(f"{job_id}/{stage}/current.json")
    if current and current.get("generation"):
        targets.add(str(current["generation"]))
    if job.get(f"{stage}_generation"):
        targets.add(str(job[f"{stage}_generation"]))
    engines = job.get("possession_generations")
    if stage == "possession" and isinstance(engines, dict):
        targets.update(str(v) for v in engines.values() if v)
    return targets


def _still_stale(vol: VolumeAdapter, job_id: str, job: dict[str, Any]) -> bool:
    if job.get("state") == "failed":
        return True
    if job.get("state") == "complete":
        return False
    return not any((vol.read_json(f"{job_id}/{s}/current.json") or {}).get("generation") for s in STAGES)


def revalidate(vol: VolumeAdapter, item: dict[str, Any], match: re.Match[str], stale_jobs: set[str],
               batch: set[str], now: float) -> str | None:
    """Return a skip reason if the live volume no longer justifies deleting ``item``, else None."""
    job_id = match["job"]
    job = vol.read_json(f"{job_id}/job.json")
    if job is None:
        return "job.json missing or unreadable on the live volume"
    if _lease_active(job, now):
        return "job has an active lease"
    stale = job_id in stale_jobs
    if stale and not _still_stale(vol, job_id, job):
        return "job is no longer failed/never-completed"
    if item["kind"] == "orphan":
        if job.get("state") in IN_PROGRESS_STATES and not stale:
            return "job is in progress"
        return None
    stage, gen = match["stage"], match["id"]
    if gen in _pointer_targets(vol, job_id, job, stage):
        return "now a current pointer / job.json target"
    if stage == "tracking":  # referenced by a retained downstream generation?
        for other in STAGES[1:]:
            for name in vol.list_dir(f"{job_id}/{other}/generations"):
                if f"{job_id}/{other}/generations/{name}" in batch:
                    continue  # that downstream generation is itself being removed
                manifest = vol.read_json(f"{job_id}/{other}/generations/{name}/manifest.json") or {}
                if str(manifest.get("tracking_generation")) == gen:
                    return f"referenced by retained {other}/{name}"
    return None


def apply_plan(plan: dict[str, Any], vol: VolumeAdapter, *, only: set[str], orphans_only: bool = False,
               apply: bool = False, now: float | None = None, log: Any = None) -> dict[str, Any]:
    """Process the selected plan items; returns ``{"results": [...], "totals": {...}}``."""
    now = time.time() if now is None else now
    stale_jobs = {j["job_id"] for j in plan["jobs"] if j.get("stale_candidate")}
    selected = [i for j in plan["jobs"] for i in j["items"] if i["job"] in only]
    selected = [i for i in selected if i["status"] == ELIGIBLE]
    batch = {i["path"] for i in selected}
    results = []

    def record(item: dict[str, Any], action: str, reason: str = "", *, counted: bool = False) -> None:
        row = {"ts": now, "action": action, "path": item["path"], "kind": item.get("kind"),
               "bytes": item.get("bytes", 0) if counted else 0, "files": item.get("files", 0) if counted else 0,
               "reason": reason}
        results.append(row)
        if log is not None and apply:
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()

    for item in selected:
        path = str(item["path"])
        match = allowlist_match(path)
        if match is None or item.get("kind") not in ("generation", "orphan") or match["job"] != item["job"]:
            record(item, "refused", "path not in the deletion allowlist or has a protected name")
            continue
        if orphans_only and item["kind"] != "orphan":
            record(item, "skipped", "--orphans-only")
            continue
        if not vol.exists(path):
            record(item, "already_deleted")
            continue
        reason = revalidate(vol, item, match, stale_jobs, batch, now)
        if reason:
            record(item, "skipped", reason)
            continue
        if not apply:
            record(item, "would_delete", counted=True)
            continue
        try:
            vol.remove_tree(path)
        except FileNotFoundError:
            record(item, "already_deleted")
        except Exception as exc:  # keep going; every failure is logged
            record(item, "failed", f"{type(exc).__name__}: {exc}")
        else:
            record(item, "deleted", counted=True)
    totals: dict[str, dict[str, int]] = {}
    for row in results:
        t = totals.setdefault(row["action"], {"items": 0, "files": 0, "bytes": 0})
        t["items"] += 1
        t["files"] += row["files"]
        t["bytes"] += row["bytes"]
    if log is not None and apply:
        log.write(json.dumps({"summary": totals, "plan_hash": plan["plan_hash"], "ts": now}, sort_keys=True) + "\n")
        log.flush()
    return {"results": results, "totals": totals}


def main(argv: list[str] | None = None, volume: VolumeAdapter | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a v2 retention plan to the v2-job-data volume "
                                                 "(dry-run unless --apply --yes --plan-hash).")
    parser.add_argument("--plan", required=True, type=Path, help="JSON from v2_retention_report.py --json")
    parser.add_argument("--only", required=True, help="comma-separated job ids to act on (mandatory)")
    parser.add_argument("--orphans-only", action="store_true", help="act on orphan temp files only")
    parser.add_argument("--volume", default=VOLUME_NAME, help=f"hard-locked to {VOLUME_NAME}")
    parser.add_argument("--apply", action="store_true", help="really delete (needs --yes and --plan-hash)")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--plan-hash", default=None, dest="plan_hash")
    parser.add_argument("--log", type=Path, default=None, help="JSONL log (default retention-apply-<epoch>.jsonl)")
    parser.add_argument("--now", type=float, default=None, help="epoch seconds (default: current time)")
    args = parser.parse_args(argv)

    def fail(msg: str) -> int:
        print(f"error: {msg}", file=sys.stderr)
        return 2

    if args.volume != VOLUME_NAME:
        return fail(f"refusing volume {args.volume!r}: only {VOLUME_NAME!r} is in scope")
    only = {j.strip() for j in args.only.split(",") if j.strip()}
    if not only:
        return fail("--only needs at least one job id")
    if any(not re.fullmatch(_JOB, j) or is_protected_name(j) for j in only):
        return fail("--only contains an invalid or protected job id (no wildcards)")
    if args.apply and not args.yes:
        return fail("--apply also needs --yes")
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        if plan.get("schema") != REPORT_SCHEMA or not isinstance(plan.get("jobs"), list):
            raise ValueError("not a v2 retention plan")
        actual = compute_plan_hash(plan)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return fail(f"cannot read plan: {exc}")
    if actual != plan.get("plan_hash"):
        return fail("plan_hash inside the plan does not match its content (plan was edited)")
    if args.apply and args.plan_hash != actual:
        return fail("--plan-hash is missing or does not match the plan")
    if args.plan_hash is not None and args.plan_hash != actual:
        return fail("--plan-hash does not match the plan")
    missing = only - {j["job_id"] for j in plan["jobs"]}
    if missing:
        return fail(f"jobs not in the plan: {', '.join(sorted(missing))}")

    vol = volume if volume is not None else ModalVolume()
    log = None
    if args.apply:
        log_path = args.log or Path(f"retention-apply-{int(time.time())}.jsonl")
        log = open(log_path, "a", encoding="utf-8")
    try:
        out = apply_plan(plan, vol, only=only, orphans_only=args.orphans_only, apply=args.apply,
                         now=args.now, log=log)
    finally:
        if log is not None:
            log.close()
    print(f"v2 retention apply ({'APPLY' if args.apply else 'DRY-RUN, nothing deleted'}) "
          f"volume={VOLUME_NAME} plan_hash={actual}")
    for row in out["results"]:
        print(f"  {row['action']:<14}{row['bytes']:>12} bytes {row['files']:>6} files  {row['path']}"
              + (f"  ({row['reason']})" if row["reason"] else ""))
    for action, t in sorted(out["totals"].items()):
        print(f"totals {action}: {t['items']} items, {t['files']} files, {t['bytes']} bytes")
    return 1 if out["totals"].get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
