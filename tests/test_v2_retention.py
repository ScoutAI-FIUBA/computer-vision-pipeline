import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.ops.v2_retention_report import REPORT_SCHEMA, build_report

NOW = 2_000_000_000.0
DAY = 86400


def _w(path: Path, data: str | bytes = "x", mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _digest(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def make_generation(job, stage, gen, *, input_fp, tracking=None, mtime=NOW - DAY, complete=True):
    d = job / stage / "generations" / gen
    _w(d / "artifact.jsonl", "a" * 10, mtime)
    manifest = {"schema": "v2-stage-1", "stage": stage, "generation": gen, "input_fingerprint": input_fp,
                "tracking_generation": tracking, "complete": complete, "artifacts": {"artifact.jsonl": {}}}
    _w(d / "manifest.json", json.dumps(manifest), mtime)


def make_job(root, name="job", *, state="complete", lease=None, age_days=1.0, current=None,
             inputs=("in1",), extra_job=None):
    job = root / name
    fps = {i: "sha256:" + _digest(i) for i in inputs}
    manifest = {"schema": "v2-job-1", "job_id": name, "state": state, "lease": lease, "attempts": {},
                "errors": [], "input_fingerprint": fps[inputs[0]]}
    manifest.update(extra_job or {})
    _w(job / "job.json", json.dumps(manifest), NOW - age_days * DAY)
    _w(job / ".job.lock", "")
    for i, fp in fps.items():
        _w(job / "inputs" / (fp.removeprefix("sha256:") + ".mp4"), b"v" * 100, NOW - 40 * DAY)
    for stage, gen in (current or {}).items():
        _w(job / stage / "current.json", json.dumps({"generation": gen, "stage": stage}))
    return job, fps


def items(report, **match):
    return [i for j in report["jobs"] for i in j["items"] if all(i[k] == v for k, v in match.items())]


def one(report, **match):
    found = items(report, **match)
    assert len(found) == 1, found
    return found[0]


def hash_tree(root: Path):
    out = {}
    for p in sorted(root.rglob("*")):
        out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir"
        if p.is_file():
            out[str(p.relative_to(root)) + "#mtime"] = p.stat().st_mtime_ns
    return out


def standard_tree(root):
    job, fps = make_job(root, current={"tracking": "t3", "stats": "s3"},
                        extra_job={"tracking_generation": "t3", "stats_generation": "s3"})
    fp = fps["in1"]
    for n, age in (("t1", 5), ("t2", 4), ("t3", 3)):
        make_generation(job, "tracking", n, input_fp=fp, mtime=NOW - age * DAY)
    for n, t, age in (("s1", "t1", 5), ("s2", "t2", 4), ("s3", "t3", 3)):
        make_generation(job, "stats", n, input_fp=fp, tracking=t, mtime=NOW - age * DAY)
    return job


def test_current_pointer_protected_even_when_old(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t1"})
    for n, age in (("t1", 9), ("t2", 3), ("t3", 2)):
        make_generation(job, "tracking", n, input_fp=fps["in1"], mtime=NOW - age * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=1)
    assert one(r, id="t1")["status"] == "PROTECTED" and "current" in one(r, id="t1")["reason"]
    assert one(r, id="t3")["status"] == "PROTECTED"
    assert one(r, id="t2")["status"] == "ELIGIBLE"


def test_job_json_generation_fields_protected(tmp_path):
    job, fps = make_job(tmp_path, extra_job={"stats_generation": "s1", "possession_generations": {"e": "p1"}})
    make_generation(job, "stats", "s1", input_fp=fps["in1"], mtime=NOW - 9 * DAY)
    make_generation(job, "possession", "p1", input_fp=fps["in1"], mtime=NOW - 9 * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=0)
    assert one(r, id="s1")["status"] == "PROTECTED"
    assert one(r, id="p1")["status"] == "PROTECTED"


def test_keep_n(tmp_path):
    standard_tree(tmp_path)
    r = build_report(tmp_path, now=NOW, keep_generations=2)
    assert one(r, stage="stats", id="s1")["status"] == "ELIGIBLE"
    assert one(r, stage="stats", id="s2")["status"] == "PROTECTED"
    r = build_report(tmp_path, now=NOW, keep_generations=3)
    assert not [i for i in items(r, kind="generation") if i["id"].startswith("s") and i["status"] == "ELIGIBLE"]


def test_active_lease_protects_whole_job(tmp_path):
    job, fps = make_job(tmp_path, state="stats", lease={"owner": "w", "expires_at": NOW + 60}, age_days=90)
    for n, age in (("t1", 9), ("t2", 8), ("t3", 7)):
        make_generation(job, "tracking", n, input_fp=fps["in1"], mtime=NOW - age * DAY)
    _w(job / "tracking" / ".stage-abc" / "f", "z", NOW - 30 * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=0)
    assert r["jobs"][0]["lease_active"] is True
    assert all(i["status"] == "PROTECTED" for i in items(r))
    # Same tree after the lease expired and the job is old -> abandoned, eligible.
    r = build_report(tmp_path, now=NOW + 3600, keep_generations=0)
    assert r["jobs"][0]["lease_active"] is False
    assert one(r, id="t1")["status"] == "ELIGIBLE"


def test_in_progress_young_job_protected(tmp_path):
    job, fps = make_job(tmp_path, state="tracking", lease={"owner": "w", "expires_at": NOW - 10}, age_days=1)
    make_generation(job, "tracking", "t1", input_fp=fps["in1"], mtime=NOW - 9 * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=0)
    assert all(i["status"] == "PROTECTED" for i in items(r))
    assert "in-progress" in one(r, id="t1")["reason"]


def test_failed_job_expiry(tmp_path):
    job, fps = make_job(tmp_path, "old", state="failed", age_days=10, inputs=("a",))
    make_job(tmp_path, "new", state="failed", age_days=2, inputs=("b",))
    r = build_report(tmp_path, now=NOW, max_age_days=7)
    assert all(i["status"] == "ELIGIBLE" for i in items(r, job="old"))
    assert all(i["status"] == "PROTECTED" for i in items(r, job="new"))


def test_referenced_tracking_generation_protected(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t3", "stats": "s3"})
    fp = fps["in1"]
    for n, age in (("t1", 9), ("t2", 8), ("t3", 3)):
        make_generation(job, "tracking", n, input_fp=fp, mtime=NOW - age * DAY)
    make_generation(job, "stats", "s3", input_fp=fp, tracking="t1", mtime=NOW - 3 * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=1)
    t1 = one(r, id="t1")
    assert t1["status"] == "PROTECTED" and "stats/s3" in t1["reason"]
    assert one(r, id="t2")["status"] == "ELIGIBLE"


def test_unretained_stats_does_not_protect_tracking(tmp_path):
    standard_tree(tmp_path)
    r = build_report(tmp_path, now=NOW, keep_generations=1)
    assert one(r, stage="stats", id="s1")["status"] == "ELIGIBLE"
    assert one(r, stage="tracking", id="t1")["status"] == "ELIGIBLE"


def test_referenced_input_protected_and_unreferenced_eligible(tmp_path):
    job, fps = make_job(tmp_path, inputs=("in1", "in2", "in3"), current={"tracking": "t1"})
    make_generation(job, "tracking", "t1", input_fp=fps["in2"], mtime=NOW - DAY)
    r = build_report(tmp_path, now=NOW)
    name = lambda k: _digest(k) + ".mp4"
    assert one(r, kind="input", id=name("in1"))["status"] == "PROTECTED"  # job.json
    assert one(r, kind="input", id=name("in2"))["status"] == "PROTECTED"  # retained generation
    assert one(r, kind="input", id=name("in3"))["status"] == "ELIGIBLE"  # 40d old, unreferenced
    r = build_report(tmp_path, now=NOW, input_max_age_days=60)
    assert one(r, kind="input", id=name("in3"))["status"] == "PROTECTED"


def test_input_of_eligible_generation_not_protected(tmp_path):
    job, fps = make_job(tmp_path, inputs=("in1", "in2"), current={"tracking": "t3"})
    make_generation(job, "tracking", "t1", input_fp=fps["in2"], mtime=NOW - 9 * DAY)
    make_generation(job, "tracking", "t3", input_fp=fps["in1"], mtime=NOW - DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=1)
    assert one(r, id="t1")["status"] == "ELIGIBLE"
    assert one(r, kind="input", id=_digest("in2") + ".mp4")["status"] == "ELIGIBLE"


def test_orphans(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t1"})
    make_generation(job, "tracking", "t1", input_fp=fps["in1"])
    old = NOW - 3 * DAY
    _w(job / "tracking" / ".stage-xyz" / "masks" / "m.png", "m", old)
    _w(job / "tracking" / ".current.json.tmp", "{}", old)
    _w(job / ".job.json.abc.tmp", "{}", old)
    _w(job / "inputs" / ".v.mp4.abc.tmp", "v", old)
    _w(job / "tracking" / ".stage-fresh" / "f", "f", NOW - 60)
    make_generation(job, "stats", "bad", input_fp=fps["in1"], complete=False, mtime=old)
    r = build_report(tmp_path, now=NOW)
    elig = {i["id"] for i in items(r, status="ELIGIBLE", kind="orphan")}
    assert elig == {".stage-xyz", ".current.json.tmp", ".job.json.abc.tmp", ".v.mp4.abc.tmp"}
    assert one(r, id=".stage-fresh")["status"] == "PROTECTED"
    assert one(r, id="bad")["status"] == "ELIGIBLE"
    assert one(r, id=".stage-xyz")["files"] == 1


def test_unlisted_file_noted(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t1"})
    make_generation(job, "tracking", "t1", input_fp=fps["in1"])
    _w(job / "tracking" / "generations" / "t1" / "stray.png", "s" * 7)
    r = build_report(tmp_path, now=NOW)
    assert one(r, id="t1")["unlisted_files"] == {"files": 1, "bytes": 7}


def test_idempotent(tmp_path):
    standard_tree(tmp_path)
    assert build_report(tmp_path, now=NOW) == build_report(tmp_path, now=NOW)


def test_totals_add_up_and_cover_every_file(tmp_path):
    standard_tree(tmp_path)
    r = build_report(tmp_path, now=NOW, keep_generations=1)
    t = r["totals"]
    for k in ("items", "files", "bytes"):
        assert t["all"][k] == t["protected"][k] + t["eligible"][k]
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert t["all"]["files"] == len(files)
    assert t["all"]["bytes"] == sum(p.stat().st_size for p in files)


def test_no_mutation(tmp_path):
    standard_tree(tmp_path)
    job = tmp_path / "job"
    _w(job / "tracking" / ".stage-old" / "f", "q", NOW - 9 * DAY)
    before = hash_tree(tmp_path)
    build_report(tmp_path, now=NOW, keep_generations=0)
    out = subprocess.run([sys.executable, "-m", "scripts.ops.v2_retention_report", "--root", str(tmp_path),
                          "--json", "--now", str(NOW)], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr
    json.loads(out.stdout)
    assert hash_tree(tmp_path) == before


def test_json_schema_keys(tmp_path):
    standard_tree(tmp_path)
    r = build_report(tmp_path, now=NOW)
    assert r["schema"] == REPORT_SCHEMA and r["dry_run"] is True
    assert {"schema", "root", "now", "dry_run", "policy", "jobs", "skipped", "totals"} <= set(r)
    assert set(r["totals"]) == {"all", "protected", "eligible"}
    assert set(r["totals"]["all"]) == {"items", "files", "bytes"}
    job = r["jobs"][0]
    assert {"job_id", "state", "lease_active", "age_days", "bytes", "files", "protected", "eligible", "items"} <= set(job)
    for item in job["items"]:
        assert {"kind", "job", "stage", "id", "path", "bytes", "files", "status", "reason"} <= set(item)
        assert item["status"] in ("PROTECTED", "ELIGIBLE")
    json.dumps(r)


def test_non_job_dirs_skipped(tmp_path):
    standard_tree(tmp_path)
    (tmp_path / "scratch").mkdir()
    assert build_report(tmp_path, now=NOW)["skipped"] == ["scratch"]
