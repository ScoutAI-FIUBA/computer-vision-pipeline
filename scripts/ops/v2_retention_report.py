"""Read-only retention planner for the ``v2-job-data`` Modal Volume layout.

This module only inspects a local tree shaped like the volume (``<root>/<job>/...``)
and reports what a future cleanup COULD remove. It contains NO delete code path:
it never unlinks, moves, writes or chmods anything. Deletion is a separate,
not-yet-implemented step (see ``docs/v2-retention-policy.md``).

Layout it understands (verified against ``modal_app_v2.py`` and ``pipeline/results.py``)::

    <job>/job.json  <job>/.job.lock  <job>/inputs/<sha256>.mp4
    <job>/{tracking,stats,possession}/current.json
    <job>/{tracking,stats,possession}/generations/<id>/manifest.json (+ artifacts)
    <job>/<stage>/.stage-*  <job>/<stage>/.current.json.tmp  <job>/**/.*.tmp

Every file under a job belongs to exactly one reported item, so totals add up.
Default thresholds are proposals for the owner, not decided policy. The tool is
deliberately conservative: keep what is useful, only objectively useless data is
ELIGIBLE (for ``v2_retention_apply.py``, which is the only deleting code and is separate).
REVIEW is reserved for mask-only compaction of old complete jobs: an open owner
decision that is NOT implemented, so nothing is REVIEW today.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "v2-retention-report-2"
STAGES = ("tracking", "stats", "possession")
IN_PROGRESS_STATES = {"created", *STAGES}
PROTECTED, REVIEW, ELIGIBLE = "PROTECTED", "REVIEW", "ELIGIBLE"
PROTECTED_REASON = "protected_data_class"
OWNERSHIP_NOTICE = ("job.json has no owner field: deletion decisions belong to whoever created the job; "
                    "this planner never deletes anything (the separate apply tool needs --only JOB, "
                    "--apply --yes and the plan hash).")

# Anything whose name matches (any path component, case-insensitive) is fine-tuning,
# model or dataset data and can never be ELIGIBLE or REVIEW.
PROTECTED_NAME_SUBSTRINGS = ("tracknet", "finetune", "fine-tune", "fine_tune", "checkpoint", "dataset",
                             "weights", "model-cache", "model_cache")
PROTECTED_NAME_PATTERNS = ("*.pt", "*.pth", "*.ckpt", "*.onnx", "runs")

# Proposed defaults, owner to confirm.
DEFAULTS = {
    "keep_generations": 2,
    "orphan_min_age_days": 7.0,
    "generation_min_age_days": 30.0,
    "failed_job_min_age_days": 30.0,
}


class GuardedRootError(ValueError):
    """Raised when --root is (or is named like) protected fine-tuning/model data."""


def is_protected_name(name: str) -> bool:
    lowered = name.lower()
    return (any(sub in lowered for sub in PROTECTED_NAME_SUBSTRINGS)
            or any(fnmatch.fnmatchcase(lowered, pat) for pat in PROTECTED_NAME_PATTERNS))


def _tree_has_protected_name(path: Path) -> bool:
    """True if any entry inside ``path`` (not following symlinks) has a protected name."""
    if not path.is_dir() or path.is_symlink():
        return False
    for _dirpath, dirs, files in os.walk(path):
        if any(is_protected_name(n) for n in (*dirs, *files)):
            return True
    return False


def check_root(root: Path) -> None:
    names = {Path(root).name, Path(os.path.abspath(root)).name, Path(os.path.realpath(root)).name}
    for name in names:
        if is_protected_name(name):
            raise GuardedRootError(f"refusing --root {root}: name {name!r} matches the protected-data guard "
                                   "(fine-tuning/model/dataset data is never in scope; only v2-job-data is)")


def _walk_stats(path: Path) -> tuple[int, int, float]:
    """Return (bytes, file_count, newest_mtime) without following symlinks."""
    if path.is_symlink() or path.is_file():
        st = path.lstat()
        return st.st_size, 1, st.st_mtime
    total = count = 0
    newest = 0.0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            st = os.lstat(os.path.join(dirpath, name))
            total += st.st_size
            count += 1
            newest = max(newest, st.st_mtime)
    if newest == 0.0:
        newest = path.lstat().st_mtime
    return total, count, newest


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _is_orphan_name(name: str) -> bool:
    return name.startswith(".stage-") or (name.startswith(".") and name.endswith(".tmp"))


class _Job:
    def __init__(self, path: Path, rel: str, now: float, policy: dict[str, float]) -> None:
        self.path, self.job_id, self.now, self.policy = path, rel, now, policy
        self.items: list[dict[str, Any]] = []
        self.manifest = _read_json(path / "job.json") or {}
        self.state = self.manifest.get("state")
        lease = self.manifest.get("lease") or {}
        self.lease_active = bool(lease.get("owner")) and _num(lease.get("expires_at")) > now
        job_json = path / "job.json"
        mtime = job_json.stat().st_mtime if job_json.is_file() else path.stat().st_mtime
        self.age_days = max(0.0, (now - mtime) / 86400)
        self.has_pointer = any((_read_json(path / st / "current.json") or {}).get("generation") for st in STAGES)
        # Failed or never-completed (no current pointer, not complete), no live lease, old enough.
        self.stale = not self.lease_active and self.age_days > policy["failed_job_min_age_days"] and (
            self.state == "failed" or (self.state != "complete" and not self.has_pointer))
        self.in_progress = self.state in IN_PROGRESS_STATES and not self.stale

    def add(self, kind: str, stage: str | None, ident: str, path: Path, status: str, reason: str) -> dict[str, Any]:
        size, count, newest = _walk_stats(path)
        rel = Path(self.job_id) / path.relative_to(self.path)
        guarded = any(is_protected_name(part) for part in rel.parts) or _tree_has_protected_name(path)
        item = {"kind": kind, "job": self.job_id, "stage": stage, "id": ident,
                "path": str(rel), "bytes": size, "files": count, "status": status, "reason": reason,
                "_mtime": newest, "_guard": guarded, "_pinned": False}
        self.items.append(item)
        return item

    def pointer_targets(self, stage: str) -> set[str]:
        targets = set()
        current = _read_json(self.path / stage / "current.json")
        if current and current.get("generation"):
            targets.add(str(current["generation"]))
        field = self.manifest.get(f"{stage}_generation")
        if field:
            targets.add(str(field))
        if stage == "possession":
            engines = self.manifest.get("possession_generations")
            if isinstance(engines, dict):
                targets.update(str(v) for v in engines.values() if v)
        return targets

    def classify(self) -> None:
        job = self.path
        retained_tracking_refs: dict[str, str] = {}
        gens: dict[str, list[dict[str, Any]]] = {}
        keep = int(self.policy["keep_generations"])
        min_age = self.policy["generation_min_age_days"]

        # Pass 1: generations, per stage. Only old, non-current, beyond-keep ones are REVIEW.
        for stage in STAGES:
            root = job / stage / "generations"
            if not root.is_dir():
                continue
            pointers = self.pointer_targets(stage)
            valid_entries, invalid_entries = [], []
            for gen_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()):
                manifest = _read_json(gen_dir / "manifest.json")
                if manifest and manifest.get("complete") is True:
                    mtime = (gen_dir / "manifest.json").stat().st_mtime
                    valid_entries.append((gen_dir, manifest, mtime))
                else:
                    invalid_entries.append((gen_dir, None, gen_dir.stat().st_mtime))
            valid_entries.sort(key=lambda e: (-e[2], e[0].name))
            for rank, (gen_dir, manifest, mtime) in enumerate(valid_entries):
                name = gen_dir.name
                age_days = (self.now - mtime) / 86400
                pinned = name in pointers
                if pinned:
                    status, reason = PROTECTED, "current pointer / job.json target"
                elif rank < keep:
                    status, reason = PROTECTED, f"within newest {keep} generations"
                elif age_days <= min_age:
                    status, reason = PROTECTED, f"non-current but younger than {min_age:g} days"
                else:
                    status, reason = ELIGIBLE, (f"superseded: non-current, beyond keep-generations "
                                                f"(rank {rank + 1}), older than {min_age:g} days")
                item = self.add("generation", stage, name, gen_dir, status, reason)
                item["_manifest"] = manifest
                item["_pinned"] = pinned
                gens.setdefault(stage, []).append(item)
                if status == PROTECTED and manifest.get("tracking_generation") and stage != "tracking":
                    retained_tracking_refs[str(manifest["tracking_generation"])] = f"{stage}/{name}"
                self._unlisted_files(gen_dir, manifest, stage, name, status)
            for gen_dir, _manifest, _mtime in invalid_entries:
                item = self.add("generation", stage, gen_dir.name, gen_dir, PROTECTED,
                                "generation without a complete manifest (maybe in flight); not a candidate")
                item["_manifest"] = None
                gens.setdefault(stage, []).append(item)

        # Pass 2: generations referenced by retained downstream manifests stay protected.
        for item in gens.get("tracking", []):
            ref = retained_tracking_refs.get(item["id"])
            if ref and item["status"] == ELIGIBLE:
                item["status"], item["reason"] = PROTECTED, f"referenced by retained {ref} (tracking_generation)"
                item["_pinned"] = True

        # Inputs are never candidates by themselves.
        inputs = job / "inputs"
        if inputs.is_dir():
            for path in sorted(inputs.iterdir()):
                if _is_orphan_name(path.name):
                    self._orphan(path, "input")
                else:
                    self.add("input", None, path.name, path, PROTECTED, "inputs are never candidates by themselves")

        # Orphans and metadata in job and stage dirs; everything else is unclassified.
        for path in sorted(job.iterdir()):
            if path.name in ("inputs",) or path.name in STAGES and path.is_dir():
                continue
            if _is_orphan_name(path.name):
                self._orphan(path, None)
            elif path.name in ("job.json", ".job.lock"):
                self.add("meta", None, path.name, path, PROTECTED, "job metadata")
            else:
                self.add("other", None, path.name, path, PROTECTED, "unclassified, never a candidate")
        for stage in STAGES:
            sdir = job / stage
            if not sdir.is_dir():
                continue
            for path in sorted(sdir.iterdir()):
                if path.name == "generations":
                    for extra in sorted(p for p in path.iterdir() if not p.is_dir()):
                        self.add("other", stage, extra.name, extra, PROTECTED, "unclassified, never a candidate")
                elif _is_orphan_name(path.name):
                    self._orphan(path, stage)
                elif path.name == "current.json":
                    self.add("meta", stage, path.name, path, PROTECTED, "stage pointer")
                else:
                    self.add("other", stage, path.name, path, PROTECTED, "unclassified, never a candidate")

        self._apply_job_level()

    def _unlisted_files(self, gen_dir: Path, manifest: dict[str, Any], stage: str, gen: str, status: str) -> None:
        listed = {"manifest.json", *(str(k) for k in (manifest.get("artifacts") or {}))}
        extra = []
        for dirpath, _dirs, files in os.walk(gen_dir):
            for name in files:
                rel = os.path.relpath(os.path.join(dirpath, name), gen_dir)
                if rel not in listed:
                    extra.append(os.path.join(dirpath, name))
        if not extra:
            return
        sizes = [os.lstat(p).st_size for p in extra]
        # Reported as a note on the generation item (it stays owned by the generation).
        for item in self.items:
            if item["kind"] == "generation" and item["stage"] == stage and item["id"] == gen:
                item["unlisted_files"] = {"files": len(extra), "bytes": sum(sizes)}

    def _orphan(self, path: Path, stage: str | None) -> None:
        age_days = (self.now - _walk_stats(path)[2]) / 86400
        min_age = self.policy["orphan_min_age_days"]
        if age_days <= min_age:
            self.add("orphan", stage, path.name, path, PROTECTED,
                     f"temporary/staging entry not older than {min_age:g} days (maybe in flight)")
        else:
            self.add("orphan", stage, path.name, path, ELIGIBLE,
                     f"orphan temporary/staging entry older than {min_age:g} days")

    def _apply_job_level(self) -> None:
        if self.lease_active:
            for item in self.items:
                item["status"], item["reason"] = PROTECTED, "job has an active lease"
        elif self.in_progress:
            for item in self.items:
                item["status"], item["reason"] = PROTECTED, f"in-progress stage (state={self.state!r}, lease expired)"
        elif self.stale:
            for item in self.items:
                # Whole failed/never-completed job: generations and orphans only. Inputs, job.json,
                # pointers and anything a current pointer still needs stay PROTECTED.
                if item["kind"] in ("generation", "orphan") and item["status"] == PROTECTED and not item["_pinned"]:
                    item["status"] = ELIGIBLE
                    item["reason"] = (f"job {self.state!r}, no live lease, no current pointer needing it, older than "
                                      f"{self.policy['failed_job_min_age_days']:g} days (was: {item['reason']})")
        # Hard guard last: protected data classes can never be ELIGIBLE or REVIEW.
        for item in self.items:
            if item["_guard"]:
                item["status"], item["reason"] = PROTECTED, PROTECTED_REASON


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sum(items: list[dict[str, Any]]) -> dict[str, int]:
    return {"items": len(items), "files": sum(i["files"] for i in items), "bytes": sum(i["bytes"] for i in items)}


def build_report(root: Path, *, now: float | None = None, **overrides: float) -> dict[str, Any]:
    """Build the retention plan. Pure inspection; never modifies ``root``."""
    root = Path(root)
    check_root(root)
    policy = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    now = time.time() if now is None else now
    jobs, skipped, all_items, protected_paths = [], [], [], []
    for path in sorted(root.iterdir()):
        if is_protected_name(path.name):
            size, count, _ = _walk_stats(path)
            protected_paths.append({"path": path.name, "bytes": size, "files": count, "reason": PROTECTED_REASON})
            skipped.append(path.name)
        elif path.is_dir() and not path.is_symlink() and (path / "job.json").is_file():
            job = _Job(path, path.name, now, policy)
            job.classify()
            items = sorted(job.items, key=lambda i: (i["stage"] or "", i["kind"], i["id"]))
            for item in items:
                item.pop("_mtime", None)
                item.pop("_manifest", None)
                item.pop("_pinned", None)
                if item.pop("_guard"):
                    protected_paths.append({"path": item["path"], "bytes": item["bytes"], "files": item["files"],
                                            "reason": PROTECTED_REASON})
            all_items += items
            jobs.append({"job_id": job.job_id, "state": job.state, "lease_active": job.lease_active,
                         "age_days": round(job.age_days, 3), "stale_candidate": job.stale, "bytes": sum(i["bytes"] for i in items),
                         "files": sum(i["files"] for i in items),
                         **{c.lower(): _sum([i for i in items if i["status"] == c])
                            for c in (PROTECTED, REVIEW, ELIGIBLE)},
                         "items": items})
        else:
            skipped.append(path.name)
    report = {"schema": REPORT_SCHEMA, "root": str(root), "now": now, "dry_run": True, "policy": policy,
              "defaults": dict(DEFAULTS), "ownership_notice": OWNERSHIP_NOTICE,
              "jobs": jobs, "skipped": skipped, "protected_paths": protected_paths,
              "totals": {"all": _sum(all_items),
                         **{c.lower(): _sum([i for i in all_items if i["status"] == c])
                            for c in (PROTECTED, REVIEW, ELIGIBLE)}}}
    report["plan_hash"] = compute_plan_hash(report)
    return report


def compute_plan_hash(report: dict[str, Any]) -> str:
    """sha256 over the ELIGIBLE selection (what an apply tool may act on), the policy and the schema."""
    eligible = sorted([i["path"], i["kind"], i["bytes"], i["files"]]
                      for j in report["jobs"] for i in j["items"] if i["status"] == ELIGIBLE)
    stale = sorted(j["job_id"] for j in report["jobs"] if j.get("stale_candidate"))
    blob = json.dumps({"schema": report["schema"], "policy": report["policy"], "eligible": eligible,
                       "stale_jobs": stale}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _fmt_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return str(n)


def format_table(report: dict[str, Any]) -> str:
    lines = [f"v2 retention report (READ-ONLY, nothing is deleted)  root={report['root']}",
             "active policy (defaults are proposed, owner to confirm): "
             + ", ".join(f"{k}={v:g}" for k, v in report["policy"].items()),
             "plan hash: " + report["plan_hash"],
             "defaults: " + ", ".join(f"{k}={v:g}" for k, v in report["defaults"].items()),
             "NOTE: " + report["ownership_notice"]]
    for job in report["jobs"]:
        lines += ["", f"job {job['job_id']}  state={job['state']}  lease_active={job['lease_active']}  "
                      f"age={job['age_days']:.1f}d  {_fmt_bytes(job['bytes'])}  {job['files']} files"]
        lines.append(f"  {'STATUS':<10}{'KIND':<11}{'STAGE':<11}{'BYTES':>12}{'FILES':>8}  ID / REASON")
        for i in job["items"]:
            ident = i["id"] if len(i["id"]) <= 20 else i["id"][:16] + "..."
            lines.append(f"  {i['status']:<10}{i['kind']:<11}{(i['stage'] or '-'):<11}{i['bytes']:>12}{i['files']:>8}  "
                         f"{ident}: {i['reason']}")
    t = report["totals"]
    lines += ["", "totals:"]
    for key in ("all", "protected", "review", "eligible"):
        lines.append(f"  {key:<10}{t[key]['items']:>6} items{t[key]['files']:>9} files{t[key]['bytes']:>14} bytes "
                     f"({_fmt_bytes(t[key]['bytes'])})")
    if report["protected_paths"]:
        lines.append("protected data-class paths (never candidates):")
        lines += [f"  {p['path']}  ({p['files']} files, {p['bytes']} bytes)" for p in report["protected_paths"]]
    if report["skipped"]:
        lines.append("skipped (no job.json): " + ", ".join(report["skipped"]))
    return "\n".join(lines)


def _parse_now(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value).timestamp()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only retention report for v2-job-data (never deletes).")
    parser.add_argument("--root", required=True, type=Path, help="tree shaped like the volume (<root>/<job>/...)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--now", type=_parse_now, default=None, help="epoch seconds or ISO time (default: current time)")
    parser.add_argument("--keep-generations", type=int, default=None, dest="keep_generations",
                        help=f"newest generations kept per job/stage, current included (default {DEFAULTS['keep_generations']})")
    parser.add_argument("--orphan-min-age-days", type=float, default=None, dest="orphan_min_age_days",
                        help=f"age before tmp/staging entries are ELIGIBLE (default {DEFAULTS['orphan_min_age_days']:g})")
    parser.add_argument("--generation-min-age-days", type=float, default=None, dest="generation_min_age_days",
                        help=f"age before surplus generations are REVIEW (default {DEFAULTS['generation_min_age_days']:g})")
    parser.add_argument("--failed-job-min-age-days", type=float, default=None, dest="failed_job_min_age_days",
                        help=f"age before failed/stale jobs are REVIEW (default {DEFAULTS['failed_job_min_age_days']:g})")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f"--root is not a directory: {args.root}")
    try:
        check_root(args.root)
    except GuardedRootError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = build_report(args.root, now=args.now, keep_generations=args.keep_generations,
                          orphan_min_age_days=args.orphan_min_age_days,
                          generation_min_age_days=args.generation_min_age_days,
                          failed_job_min_age_days=args.failed_job_min_age_days)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
