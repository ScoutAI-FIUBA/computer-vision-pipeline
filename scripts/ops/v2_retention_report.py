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
Default thresholds are proposals for the owner, not decided policy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "v2-retention-report-1"
STAGES = ("tracking", "stats", "possession")
IN_PROGRESS_STATES = {"created", *STAGES}
PROTECTED, ELIGIBLE = "PROTECTED", "ELIGIBLE"

# Proposed defaults, owner to confirm.
DEFAULTS = {
    "keep_generations": 2,
    "max_age_days": 7.0,
    "input_max_age_days": 30.0,
    "orphan_min_age_hours": 24.0,
}


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
        self.old = self.age_days > policy["max_age_days"]
        self.in_progress = self.state in IN_PROGRESS_STATES and not self.old
        self.expired = not self.lease_active and self.old and (
            self.state == "failed" or self.state in IN_PROGRESS_STATES)

    def add(self, kind: str, stage: str | None, ident: str, path: Path, status: str, reason: str) -> dict[str, Any]:
        size, count, newest = _walk_stats(path)
        item = {"kind": kind, "job": self.job_id, "stage": stage, "id": ident,
                "path": str(Path(self.job_id) / path.relative_to(self.path)),
                "bytes": size, "files": count, "status": status, "reason": reason,
                "_mtime": newest}
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
        referenced_inputs: set[str] = set()
        if self.manifest.get("input_fingerprint"):
            referenced_inputs.add(str(self.manifest["input_fingerprint"]))
        retained_tracking_refs: dict[str, str] = {}
        gens: dict[str, list[dict[str, Any]]] = {}

        # Pass 1: generations, per stage, current/keep-N.
        for stage in STAGES:
            root = job / stage / "generations"
            if not root.is_dir():
                continue
            pointers = self.pointer_targets(stage)
            entries = []
            for gen_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()):
                manifest = _read_json(gen_dir / "manifest.json")
                size, count, _ = _walk_stats(gen_dir)
                mtime = (gen_dir / "manifest.json").stat().st_mtime if manifest else gen_dir.stat().st_mtime
                entries.append((gen_dir, manifest, mtime))
            entries.sort(key=lambda e: (-e[2], e[0].name))
            for rank, (gen_dir, manifest, _mtime) in enumerate(entries):
                name = gen_dir.name
                valid = bool(manifest and manifest.get("complete") is True)
                if name in pointers:
                    status, reason = PROTECTED, "current pointer / job.json target"
                elif not valid:
                    status, reason = ELIGIBLE, "orphan: generation without a complete manifest"
                elif rank < self.policy["keep_generations"]:
                    status, reason = PROTECTED, f"within newest {int(self.policy['keep_generations'])} generations"
                else:
                    status, reason = ELIGIBLE, f"non-current, beyond keep-generations (rank {rank + 1})"
                item = self.add("generation", stage, name, gen_dir, status, reason)
                item["_manifest"] = manifest if valid else None
                gens.setdefault(stage, []).append(item)
                if status == PROTECTED and valid:
                    if manifest.get("input_fingerprint"):
                        referenced_inputs.add(str(manifest["input_fingerprint"]))
                    if manifest.get("tracking_generation") and stage != "tracking":
                        retained_tracking_refs[str(manifest["tracking_generation"])] = f"{stage}/{name}"
                if valid:
                    self._unlisted_files(gen_dir, manifest, stage, name, status)

        # Pass 2: tracking generations referenced by retained downstream manifests.
        for item in gens.get("tracking", []):
            ref = retained_tracking_refs.get(item["id"])
            if ref and item["status"] == ELIGIBLE:
                item["status"], item["reason"] = PROTECTED, f"referenced by retained {ref} (tracking_generation)"
                if item["_manifest"].get("input_fingerprint"):
                    referenced_inputs.add(str(item["_manifest"]["input_fingerprint"]))

        # Inputs.
        inputs = job / "inputs"
        if inputs.is_dir():
            for path in sorted(inputs.iterdir()):
                if _is_orphan_name(path.name):
                    self._orphan(path, "input")
                    continue
                fingerprint = f"sha256:{path.stem}"
                age_days = (self.now - path.lstat().st_mtime) / 86400
                if fingerprint in referenced_inputs:
                    self.add("input", None, path.name, path, PROTECTED, "referenced by job.json or a retained generation")
                elif age_days > self.policy["input_max_age_days"]:
                    self.add("input", None, path.name, path, ELIGIBLE,
                             f"unreferenced input older than {self.policy['input_max_age_days']:g} days")
                else:
                    self.add("input", None, path.name, path, PROTECTED, "unreferenced but younger than raw-input retention")

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
        age_h = (self.now - _walk_stats(path)[2]) / 3600
        if age_h < self.policy["orphan_min_age_hours"]:
            self.add("orphan", stage, path.name, path, PROTECTED,
                     f"temporary/staging entry younger than {self.policy['orphan_min_age_hours']:g}h (maybe in flight)")
        else:
            self.add("orphan", stage, path.name, path, ELIGIBLE, "orphan temporary/staging entry")

    def _apply_job_level(self) -> None:
        if self.lease_active:
            for item in self.items:
                item["status"], item["reason"] = PROTECTED, "job has an active lease"
        elif self.in_progress:
            for item in self.items:
                item["status"], item["reason"] = PROTECTED, f"in-progress stage (state={self.state!r}, lease expired)"
        elif self.expired:
            for item in self.items:
                if item["kind"] != "other" and item["status"] == PROTECTED:
                    item["status"] = ELIGIBLE
                    item["reason"] = (f"job {self.state!r} and older than "
                                      f"{self.policy['max_age_days']:g} days (was: {item['reason']})")


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
    policy = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    now = time.time() if now is None else now
    jobs, skipped, all_items = [], [], []
    for path in sorted(root.iterdir()):
        if path.is_dir() and not path.is_symlink() and (path / "job.json").is_file():
            job = _Job(path, path.name, now, policy)
            job.classify()
            items = sorted(job.items, key=lambda i: (i["stage"] or "", i["kind"], i["id"]))
            for item in items:
                item.pop("_mtime", None)
                item.pop("_manifest", None)
            all_items += items
            jobs.append({"job_id": job.job_id, "state": job.state, "lease_active": job.lease_active,
                         "age_days": round(job.age_days, 3), "bytes": sum(i["bytes"] for i in items),
                         "files": sum(i["files"] for i in items),
                         "protected": _sum([i for i in items if i["status"] == PROTECTED]),
                         "eligible": _sum([i for i in items if i["status"] == ELIGIBLE]),
                         "items": items})
        else:
            skipped.append(path.name)
    return {"schema": REPORT_SCHEMA, "root": str(root), "now": now, "dry_run": True, "policy": policy,
            "jobs": jobs, "skipped": skipped,
            "totals": {"all": _sum(all_items),
                       "protected": _sum([i for i in all_items if i["status"] == PROTECTED]),
                       "eligible": _sum([i for i in all_items if i["status"] == ELIGIBLE])}}


def _fmt_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return str(n)


def format_table(report: dict[str, Any]) -> str:
    lines = [f"v2 retention report (READ-ONLY, nothing is deleted)  root={report['root']}",
             "policy (proposed, owner to confirm): " + ", ".join(f"{k}={v:g}" for k, v in report["policy"].items())]
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
    for key in ("all", "protected", "eligible"):
        lines.append(f"  {key:<10}{t[key]['items']:>6} items{t[key]['files']:>9} files{t[key]['bytes']:>14} bytes "
                     f"({_fmt_bytes(t[key]['bytes'])})")
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
                        help=f"generations kept per job/stage (proposed {DEFAULTS['keep_generations']})")
    parser.add_argument("--max-age-days", type=float, default=None, dest="max_age_days",
                        help=f"failed/expired job age (proposed {DEFAULTS['max_age_days']:g})")
    parser.add_argument("--input-max-age-days", type=float, default=None, dest="input_max_age_days",
                        help=f"unreferenced raw input age (proposed {DEFAULTS['input_max_age_days']:g})")
    parser.add_argument("--orphan-min-age-hours", type=float, default=None, dest="orphan_min_age_hours",
                        help=f"minimum age before tmp/staging entries are eligible (proposed {DEFAULTS['orphan_min_age_hours']:g})")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f"--root is not a directory: {args.root}")
    report = build_report(args.root, now=args.now, keep_generations=args.keep_generations,
                          max_age_days=args.max_age_days, input_max_age_days=args.input_max_age_days,
                          orphan_min_age_hours=args.orphan_min_age_hours)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
