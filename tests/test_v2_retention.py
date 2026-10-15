import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ops.v2_retention_report import DEFAULTS, REPORT_SCHEMA, GuardedRootError, build_report

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


def statuses(report, **match):
    return {i["status"] for i in items(report, **match)}


def test_defaults_values():
    assert DEFAULTS == {"keep_generations": 2, "orphan_min_age_days": 7.0,
                        "generation_min_age_days": 30.0, "failed_job_min_age_days": 30.0}


def test_defaults_complete_old_job_is_protected(tmp_path):
    job, fps = make_job(tmp_path, age_days=1000, current={"tracking": "t1", "stats": "s1"},
                        extra_job={"tracking_generation": "t1", "stats_generation": "s1"})
    make_generation(job, "tracking", "t1", input_fp=fps["in1"], mtime=NOW - 1000 * DAY)
    make_generation(job, "stats", "s1", input_fp=fps["in1"], tracking="t1", mtime=NOW - 1000 * DAY)
    r = build_report(tmp_path, now=NOW)
    assert statuses(r) == {"PROTECTED"}
    assert r["totals"]["review"]["items"] == 0 and r["totals"]["eligible"]["items"] == 0


def test_current_pointer_protected_even_when_old(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t1"})
    for n, age in (("t1", 400), ("t2", 300), ("t3", 200)):
        make_generation(job, "tracking", n, input_fp=fps["in1"], mtime=NOW - age * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=1)
    assert one(r, id="t1")["status"] == "PROTECTED" and "current" in one(r, id="t1")["reason"]
    assert one(r, id="t3")["status"] == "PROTECTED"  # newest, within keep
    assert one(r, id="t2")["status"] == "ELIGIBLE"


def test_job_json_generation_fields_protected(tmp_path):
    job, fps = make_job(tmp_path, extra_job={"stats_generation": "s1", "possession_generations": {"e": "p1"}})
    make_generation(job, "stats", "s1", input_fp=fps["in1"], mtime=NOW - 900 * DAY)
    make_generation(job, "possession", "p1", input_fp=fps["in1"], mtime=NOW - 900 * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=0)
    assert one(r, id="s1")["status"] == "PROTECTED"
    assert one(r, id="p1")["status"] == "PROTECTED"


def test_superseded_generation_beyond_keep2_and_30_days(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t5"})
    for n, age in (("t1", 200), ("t2", 150), ("t3", 100), ("t4", 95), ("t5", 3)):
        make_generation(job, "tracking", n, input_fp=fps["in1"], mtime=NOW - age * DAY)
    r = build_report(tmp_path, now=NOW)  # keep 2: t5, t4 retained
    assert {n: one(r, id=n)["status"] for n in ("t1", "t2", "t3", "t4", "t5")} == {
        "t1": "ELIGIBLE", "t2": "ELIGIBLE", "t3": "ELIGIBLE", "t4": "PROTECTED", "t5": "PROTECTED"}
    r = build_report(tmp_path, now=NOW, keep_generations=1, generation_min_age_days=96)
    assert one(r, id="t4")["status"] == "PROTECTED"  # 95 days < 96
    assert one(r, id="t3")["status"] == "ELIGIBLE"


def test_young_surplus_generation_protected(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t3"})
    for n, age in (("t1", 20), ("t2", 10), ("t3", 3)):
        make_generation(job, "tracking", n, input_fp=fps["in1"], mtime=NOW - age * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=1)
    assert statuses(r) == {"PROTECTED"}


def test_keep_n_counts_only_valid_generations(tmp_path):
    standard_tree(tmp_path)
    make_generation(tmp_path / "job", "stats", "bad", input_fp="sha256:x", complete=False, mtime=NOW - 500 * DAY)
    r = build_report(tmp_path, now=NOW)
    assert one(r, id="bad")["status"] == "PROTECTED"
    assert statuses(r, kind="generation") == {"PROTECTED"}


def test_active_lease_protects_whole_job(tmp_path):
    job, fps = make_job(tmp_path, state="stats", lease={"owner": "w", "expires_at": NOW + 60}, age_days=900)
    for n, age in (("t1", 400), ("t2", 300), ("t3", 200)):
        make_generation(job, "tracking", n, input_fp=fps["in1"], mtime=NOW - age * DAY)
    _w(job / "tracking" / ".stage-abc" / "f", "z", NOW - 300 * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=0)
    assert r["jobs"][0]["lease_active"] is True
    assert statuses(r) == {"PROTECTED"}
    # Same tree after the lease expired and the job is very old and never completed -> abandoned.
    r = build_report(tmp_path, now=NOW + 3600, keep_generations=0)
    assert r["jobs"][0]["lease_active"] is False
    assert one(r, id="t1")["status"] == "ELIGIBLE"
    assert one(r, id=".stage-abc")["status"] == "ELIGIBLE"  # old orphan, job no longer in progress
    assert r["jobs"][0]["stale_candidate"] is True


def test_in_progress_young_job_protected(tmp_path):
    job, fps = make_job(tmp_path, state="tracking", lease={"owner": "w", "expires_at": NOW - 10}, age_days=1)
    make_generation(job, "tracking", "t1", input_fp=fps["in1"], mtime=NOW - 900 * DAY)
    _w(job / "tracking" / ".stage-old" / "f", "z", NOW - 300 * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=0)
    assert statuses(r) == {"PROTECTED"}
    assert "in-progress" in one(r, id="t1")["reason"]


def test_failed_job_eligible_after_30_days(tmp_path):
    for name, state, age, fp in (("old", "failed", 31, "a"), ("new", "failed", 29, "b"),
                                 ("done", "complete", 900, "c"), ("never", "stats", 40, "d")):
        job, fps = make_job(tmp_path, name, state=state, age_days=age, inputs=(fp,),
                            current={"tracking": "t2"} if name == "done" else None)
        for n, gage in (("t1", 31), ("t2", 31)):
            make_generation(job, "tracking", n, input_fp=fps[fp], mtime=NOW - gage * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=5)
    assert statuses(r, job="old", kind="generation") == {"ELIGIBLE"}  # whole failed job, generations
    assert statuses(r, job="never", kind="generation") == {"ELIGIBLE"}
    assert statuses(r, job="old", kind="input") | statuses(r, job="old", kind="meta") == {"PROTECTED"}
    assert statuses(r, job="new") == {"PROTECTED"}
    assert statuses(r, job="done") == {"PROTECTED"}
    assert {j["job_id"] for j in r["jobs"] if j["stale_candidate"]} == {"old", "never"}
    assert "REVIEW" not in statuses(r)


def test_failed_job_keeps_current_pointer_target(tmp_path):
    job, fps = make_job(tmp_path, state="failed", age_days=60, current={"tracking": "t1"})
    for n in ("t1", "t2"):
        make_generation(job, "tracking", n, input_fp=fps["in1"], mtime=NOW - 60 * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=5)
    assert one(r, id="t1")["status"] == "PROTECTED" and one(r, id="t2")["status"] == "ELIGIBLE"


def test_plan_hash_tracks_eligible_selection(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t3"})
    for n, age in (("t1", 90), ("t2", 80), ("t3", 3)):
        make_generation(job, "tracking", n, input_fp=fps["in1"], mtime=NOW - age * DAY)
    a = build_report(tmp_path, now=NOW)
    assert a["plan_hash"] == build_report(tmp_path, now=NOW + 5)["plan_hash"]
    assert a["plan_hash"] != build_report(tmp_path, now=NOW, keep_generations=3)["plan_hash"]


def test_inputs_never_listed_alone(tmp_path):
    job, fps = make_job(tmp_path, inputs=("in1", "in2", "in3"), current={"tracking": "t1"}, age_days=900)
    make_generation(job, "tracking", "t1", input_fp=fps["in2"], mtime=NOW - 900 * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=0, orphan_min_age_days=0,
                     generation_min_age_days=0, failed_job_min_age_days=0)
    assert len(items(r, kind="input")) == 3
    assert statuses(r, kind="input") == {"PROTECTED"}  # unreferenced and 40 days old, still protected


def test_orphans_eligible_only_after_7_days(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t1"})
    make_generation(job, "tracking", "t1", input_fp=fps["in1"])
    old = NOW - 8 * DAY
    _w(job / "tracking" / ".stage-xyz" / "masks" / "m.png", "m", old)
    _w(job / "tracking" / ".current.json.tmp", "{}", old)
    _w(job / ".job.json.abc.tmp", "{}", old)
    _w(job / "inputs" / ".v.mp4.abc.tmp", "v", old)
    _w(job / "tracking" / ".stage-young" / "f", "f", NOW - 6 * DAY)
    _w(job / "stats" / ".stage-fresh" / "f", "f", NOW - 60)
    make_generation(job, "stats", "bad", input_fp=fps["in1"], complete=False, mtime=old)
    r = build_report(tmp_path, now=NOW)
    elig = {i["id"] for i in items(r, status="ELIGIBLE")}
    assert elig == {".stage-xyz", ".current.json.tmp", ".job.json.abc.tmp", ".v.mp4.abc.tmp"}
    assert all(i["kind"] == "orphan" for i in items(r, status="ELIGIBLE"))
    assert one(r, id=".stage-young")["status"] == "PROTECTED"
    assert one(r, id=".stage-fresh")["status"] == "PROTECTED"
    assert one(r, id="bad")["status"] == "PROTECTED"
    assert one(r, id=".stage-xyz")["files"] == 1


def test_protected_names_inside_job(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t1"}, age_days=900)
    make_generation(job, "tracking", "t1", input_fp=fps["in1"])
    old = NOW - 800 * DAY
    _w(job / "tracking" / ".stage-tracknet-x" / "f", "f", old)  # orphan-looking, protected name
    _w(job / "tracking" / ".stage-ok" / "weights.bin", "f", old)  # protected name inside orphan
    _w(job / "tracking" / ".stage-plain" / "f", "f", old)  # plain orphan stays ELIGIBLE
    _w(job / "finetune" / "ckpt.bin", "f", old)
    _w(job / "Checkpoints" / "a", "f", old)
    _w(job / "best.PT", "f", old)
    _w(job / "runs" / "x", "f", old)
    make_generation(job, "tracking", "old1", input_fp=fps["in1"], mtime=old)
    make_generation(job, "tracking", "old2", input_fp=fps["in1"], mtime=old)
    make_generation(job, "tracking", "old3", input_fp=fps["in1"], mtime=old)
    make_generation(job, "tracking", "old4", input_fp=fps["in1"], mtime=old)
    _w(job / "tracking" / "generations" / "old4" / "model.onnx", "o", old)
    r = build_report(tmp_path, now=NOW)
    for ident in (".stage-tracknet-x", ".stage-ok", "finetune", "Checkpoints", "best.PT", "runs", "old4"):
        it = one(r, id=ident)
        assert it["status"] == "PROTECTED" and it["reason"] == "protected_data_class", it
    assert one(r, id=".stage-plain")["status"] == "ELIGIBLE"
    assert "old2" in {i["id"] for i in items(r, status="ELIGIBLE")}  # rank 4 (t1, old1, old2 kept)
    listed = {p["path"] for p in r["protected_paths"]}
    assert {"job/finetune", "job/best.PT", "job/runs", "job/tracking/generations/old4"} <= listed
    r = build_report(tmp_path, now=NOW, keep_generations=0, orphan_min_age_days=0,
                     generation_min_age_days=0, failed_job_min_age_days=0)
    guarded = [i for i in items(r) if i["path"].split("/")[-1] in (".stage-tracknet-x", "finetune", "best.PT", "runs")]
    assert guarded and all(i["status"] == "PROTECTED" for i in guarded)


@pytest.mark.parametrize("name", ["tracknet-data", "tracknet-finetune", "fine_tune_set", "ckpt.pt",
                                  "my-checkpoints", "dataset-v2", "weights", "model_cache", "runs", "x.onnx"])
def test_protected_top_level_dirs(tmp_path, name):
    standard_tree(tmp_path)
    make_job(tmp_path, name, state="failed", age_days=900)  # even with a valid job.json
    _w(tmp_path / name / "f", "x", NOW - 900 * DAY)
    r = build_report(tmp_path, now=NOW)
    assert name in r["skipped"]
    assert name in {p["path"] for p in r["protected_paths"]}
    assert name not in {j["job_id"] for j in r["jobs"]}
    assert "ELIGIBLE" not in statuses(r) and "REVIEW" not in statuses(r)


@pytest.mark.parametrize("name", ["tracknet-data", "scoutai-model-cache", "TrackNet_Finetune"])
def test_guarded_root_refused(tmp_path, name):
    root = tmp_path / name
    standard_tree(root)
    with pytest.raises(GuardedRootError):
        build_report(root, now=NOW)
    out = subprocess.run([sys.executable, "-m", "scripts.ops.v2_retention_report", "--root", str(root)],
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 2 and "refusing" in out.stderr and out.stdout == ""


def test_referenced_tracking_generation_protected(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t3", "stats": "s3"})
    fp = fps["in1"]
    for n, age in (("t1", 300), ("t2", 280), ("t3", 3)):
        make_generation(job, "tracking", n, input_fp=fp, mtime=NOW - age * DAY)
    make_generation(job, "stats", "s3", input_fp=fp, tracking="t1", mtime=NOW - 3 * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=1)
    t1 = one(r, id="t1")
    assert t1["status"] == "PROTECTED" and "stats/s3" in t1["reason"]
    assert one(r, id="t2")["status"] == "ELIGIBLE"


def test_unretained_stats_does_not_protect_tracking(tmp_path):
    job, fps = make_job(tmp_path, current={"tracking": "t3", "stats": "s3"})
    fp = fps["in1"]
    for n, t, age in (("s1", "t1", 300), ("s2", "t2", 250), ("s3", "t3", 3)):
        make_generation(job, "stats", n, input_fp=fp, tracking=t, mtime=NOW - age * DAY)
    for n, age in (("t1", 300), ("t2", 250), ("t3", 3)):
        make_generation(job, "tracking", n, input_fp=fp, mtime=NOW - age * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=1)
    assert one(r, stage="stats", id="s1")["status"] == "ELIGIBLE"
    assert one(r, stage="tracking", id="t1")["status"] == "ELIGIBLE"


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
    make_generation(tmp_path / "job", "tracking", "t0", input_fp="sha256:x", mtime=NOW - 400 * DAY)
    _w(tmp_path / "job" / "tracking" / ".stage-old" / "f", "q", NOW - 30 * DAY)
    r = build_report(tmp_path, now=NOW, keep_generations=1)
    t = r["totals"]
    assert t["review"]["items"] == 0 and t["eligible"]["items"] >= 2
    for k in ("items", "files", "bytes"):
        assert t["all"][k] == t["protected"][k] + t["review"][k] + t["eligible"][k]
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
    table = subprocess.run([sys.executable, "-m", "scripts.ops.v2_retention_report", "--root", str(tmp_path),
                            "--now", str(NOW)], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parents[1])
    assert "never deletes" in table.stdout and "keep_generations=2" in table.stdout
    assert hash_tree(tmp_path) == before


def test_json_schema_keys(tmp_path):
    standard_tree(tmp_path)
    r = build_report(tmp_path, now=NOW)
    assert r["schema"] == REPORT_SCHEMA and r["dry_run"] is True
    assert {"schema", "root", "now", "dry_run", "policy", "defaults", "ownership_notice", "jobs", "skipped",
            "protected_paths", "totals", "plan_hash"} <= set(r)
    assert r["defaults"] == DEFAULTS and set(r["policy"]) == set(DEFAULTS)
    assert "never deletes" in r["ownership_notice"] and "whoever created the job" in r["ownership_notice"]
    assert set(r["totals"]) == {"all", "protected", "review", "eligible"}
    assert set(r["totals"]["all"]) == {"items", "files", "bytes"}
    job = r["jobs"][0]
    assert {"job_id", "state", "lease_active", "age_days", "bytes", "files", "protected", "review", "eligible", "items"} <= set(job)
    for item in job["items"]:
        assert {"kind", "job", "stage", "id", "path", "bytes", "files", "status", "reason"} <= set(item)
        assert item["status"] in ("PROTECTED", "REVIEW", "ELIGIBLE")
    json.dumps(r)


def test_non_job_dirs_skipped(tmp_path):
    standard_tree(tmp_path)
    (tmp_path / "scratch").mkdir()
    assert build_report(tmp_path, now=NOW)["skipped"] == ["scratch"]
