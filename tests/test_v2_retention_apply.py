import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.ops import v2_retention_apply as apply_mod
from scripts.ops.v2_retention_apply import allowlist_match, main
from scripts.ops.v2_retention_report import build_report, compute_plan_hash

NOW = 2_000_000_000.0
DAY = 86400


class FakeVolume:
    """In-memory stand-in for the Modal Volume adapter (tests never touch Modal)."""

    def __init__(self, files=None):
        self.files = dict(files or {})
        self.removed = []

    def read_json(self, path):
        try:
            value = json.loads(self.files[path])
        except (KeyError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def list_dir(self, path):
        pre = path.rstrip("/") + "/"
        return sorted({p[len(pre):].split("/")[0] for p in self.files if p.startswith(pre)})

    def exists(self, path):
        return path in self.files or any(p.startswith(path + "/") for p in self.files)

    def remove_tree(self, path):
        hit = [p for p in self.files if p == path or p.startswith(path + "/")]
        if not hit:
            raise FileNotFoundError(path)
        for p in hit:
            del self.files[p]
        self.removed.append(path)


def _w(path, data="x", mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _gen(job, stage, gen, fp, *, tracking=None, age=100):
    d = job / stage / "generations" / gen
    mt = NOW - age * DAY
    _w(d / "a.bin", "a" * 10, mt)
    _w(d / "manifest.json", json.dumps({"stage": stage, "complete": True, "input_fingerprint": fp,
                                        "tracking_generation": tracking, "artifacts": {"a.bin": {}}}), mt)


def _job(root, name, *, state="complete", age=1, lease=None, current=None):
    fp = "sha256:" + hashlib.sha256(name.encode()).hexdigest()
    job = root / name
    _w(job / "job.json", json.dumps({"job_id": name, "state": state, "lease": lease, "input_fingerprint": fp}),
       NOW - age * DAY)
    _w(job / "inputs" / (fp[7:] + ".mp4"), "v" * 50, NOW - 400 * DAY)
    for stage, gen in (current or {}).items():
        _w(job / stage / "current.json", json.dumps({"generation": gen}))
    return job, fp


def to_volume(root: Path) -> FakeVolume:
    return FakeVolume({str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()})


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "snap"
    job, fp = _job(root, "mine", current={"tracking": "t4"})
    for n, age in (("t1", 120), ("t2", 100), ("t3", 90), ("t4", 2)):
        _gen(job, "tracking", n, fp, age=age)
    _w(job / "tracking" / ".stage-dead" / "f", "z", NOW - 20 * DAY)
    _w(job / "tracking" / ".stage-live" / "f", "z", NOW - 1 * DAY)
    other, ofp = _job(root, "theirs", current={"tracking": "o3"})
    for n, age in (("o0", 150), ("o1", 100), ("o3", 2)):
        _gen(other, "tracking", n, ofp, age=age)
    failed, ffp = _job(root, "dead", state="failed", age=60)
    _gen(failed, "tracking", "f1", ffp, age=60)
    plan = build_report(root, now=NOW)
    planfile = tmp_path / "plan.json"
    planfile.write_text(json.dumps(plan))
    return root, plan, planfile, to_volume(root)


def run(planfile, vol, *extra, only="mine", tmp=None):
    argv = ["--plan", str(planfile), "--now", str(NOW), *extra]
    if only is not None:
        argv += ["--only", only]
    return main(argv, volume=vol)


def test_plan_has_expected_eligible(world):
    _, plan, _, _ = world
    elig = {i["path"] for j in plan["jobs"] for i in j["items"] if i["status"] == "ELIGIBLE"}
    assert elig == {"mine/tracking/generations/t1", "mine/tracking/generations/t2", "mine/tracking/.stage-dead",
                    "theirs/tracking/generations/o0", "dead/tracking/generations/f1"}


def test_allowlist_regex():
    ok = ["j/tracking/generations/abc123", "j/stats/generations/x", "j/.stage-abc", "j/tracking/.stage-abc",
          "j/tracking/.current.json.tmp", "j/.job.json.abc.tmp"]
    bad = ["j/job.json", "j/tracking/current.json", "j/inputs/abc.mp4", "j/inputs/.x.tmp", "j/.job.lock", "j",
           "j/tracking/generations", "j/tracking/generations/*", "j/tracking/generations/a/b", "/j/.stage-x",
           "j/tracking/generations/../x", "j/other/generations/a", "j/tracking/generations/tracknet-1",
           "j/tracking/generations/w.pt", "tracknet-data/tracking/generations/a", "j/.stage-checkpoint"]
    assert all(allowlist_match(p) for p in ok)
    assert not any(allowlist_match(p) for p in bad), [p for p in bad if allowlist_match(p)]


def test_only_is_mandatory(world, capsys):
    _, _, planfile, vol = world
    with pytest.raises(SystemExit) as e:
        run(planfile, vol, only=None)
    assert e.value.code == 2


def test_wrong_volume_refused(world, capsys):
    _, _, planfile, vol = world
    assert run(planfile, vol, "--volume", "tracknet-data") == 2
    assert run(planfile, vol, "--volume", "scoutai-model-cache") == 2
    assert "refusing volume" in capsys.readouterr().err


def test_dry_run_deletes_nothing(world, capsys):
    _, _, planfile, vol = world
    before = dict(vol.files)
    assert run(planfile, vol) == 0
    assert vol.files == before and vol.removed == []
    assert "would_delete" in capsys.readouterr().out


def test_apply_needs_yes_and_matching_hash(world):
    _, plan, planfile, vol = world
    before = dict(vol.files)
    assert run(planfile, vol, "--apply", "--plan-hash", plan["plan_hash"]) == 2  # no --yes
    assert run(planfile, vol, "--apply", "--yes") == 2  # no hash
    assert run(planfile, vol, "--apply", "--yes", "--plan-hash", "0" * 64) == 2  # wrong hash
    assert vol.files == before


def test_edited_plan_refused_by_embedded_hash(world):
    _, plan, planfile, vol = world
    plan["jobs"][0]["items"][0]["status"] = "ELIGIBLE"
    for j in plan["jobs"]:
        for i in j["items"]:
            if i["path"].endswith("job.json"):
                i["status"] = "ELIGIBLE"
    planfile.write_text(json.dumps(plan))
    before = dict(vol.files)
    assert run(planfile, vol, "--apply", "--yes", "--plan-hash", plan["plan_hash"]) == 2
    assert vol.files == before


def test_apply_deletes_exactly_selected_eligible(world, tmp_path):
    _, plan, planfile, vol = world
    log = tmp_path / "log.jsonl"
    assert run(planfile, vol, "--apply", "--yes", "--plan-hash", plan["plan_hash"], "--log", str(log)) == 0
    assert sorted(vol.removed) == ["mine/tracking/.stage-dead", "mine/tracking/generations/t1",
                                   "mine/tracking/generations/t2"]
    left = set(vol.files)
    for keep in ("mine/job.json", "mine/tracking/current.json", "mine/tracking/generations/t3/manifest.json",
                 "mine/tracking/.stage-live/f", "theirs/tracking/generations/o0/manifest.json",
                 "dead/tracking/generations/f1/manifest.json", "dead/job.json"):
        assert keep in left, keep
    assert any(p.startswith("mine/inputs/") for p in left)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    deleted = [r for r in rows if r.get("action") == "deleted"]
    assert {r["path"] for r in deleted} == set(vol.removed)
    gen = next(r for r in deleted if r["path"].endswith("generations/t1"))
    assert gen["bytes"] > 0 and gen["files"] == 2
    assert rows[-1]["summary"]["deleted"]["items"] == 3 and rows[-1]["plan_hash"] == plan["plan_hash"]


def test_orphans_only(world):
    _, plan, planfile, vol = world
    assert run(planfile, vol, "--apply", "--yes", "--plan-hash", plan["plan_hash"], "--orphans-only",
               "--log", str(planfile.parent / "l.jsonl")) == 0
    assert vol.removed == ["mine/tracking/.stage-dead"]


def test_revalidation_skips_item_that_became_current(world, tmp_path):
    _, plan, planfile, vol = world
    vol.files["mine/tracking/current.json"] = json.dumps({"generation": "t1"}).encode()
    log = tmp_path / "log.jsonl"
    run(planfile, vol, "--apply", "--yes", "--plan-hash", plan["plan_hash"], "--log", str(log))
    assert "mine/tracking/generations/t1" not in vol.removed
    assert "mine/tracking/generations/t2" in vol.removed
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    skipped = [r for r in rows if r.get("action") == "skipped"]
    assert skipped and "current pointer" in skipped[0]["reason"]


def test_revalidation_skips_leased_job_and_referenced_tracking(world, tmp_path):
    _, plan, planfile, vol = world
    job = json.loads(vol.files["mine/job.json"])
    job["lease"] = {"owner": "w", "expires_at": NOW + 60}
    vol.files["mine/job.json"] = json.dumps(job).encode()
    run(planfile, vol, "--apply", "--yes", "--plan-hash", plan["plan_hash"], "--log", str(tmp_path / "l"))
    assert vol.removed == []
    job["lease"] = None
    vol.files["mine/job.json"] = json.dumps(job).encode()
    vol.files["mine/stats/generations/s9/manifest.json"] = json.dumps(
        {"complete": True, "tracking_generation": "t2"}).encode()
    run(planfile, vol, "--apply", "--yes", "--plan-hash", plan["plan_hash"], "--log", str(tmp_path / "l"))
    assert "mine/tracking/generations/t2" not in vol.removed
    assert "mine/tracking/generations/t1" in vol.removed


def test_failed_job_that_resumed_is_skipped(world, tmp_path):
    _, plan, planfile, vol = world
    job = json.loads(vol.files["dead/job.json"])
    job["state"] = "complete"
    vol.files["dead/job.json"] = json.dumps(job).encode()
    run(planfile, vol, "--apply", "--yes", "--plan-hash", plan["plan_hash"], "--log", str(tmp_path / "l"),
        only="dead")
    assert vol.removed == []


def hand_edited_plan(plan, planfile, path, kind):
    """Inject a forbidden item as ELIGIBLE and recompute the hash, as a malicious editor would."""
    job_id = path.split("/")[0]
    job = next(j for j in plan["jobs"] if j["job_id"] == job_id)
    job["items"].append({"kind": kind, "job": job_id, "stage": None, "id": path.rsplit("/", 1)[-1], "path": path,
                         "bytes": 1, "files": 1, "status": "ELIGIBLE", "reason": "forged"})
    plan["plan_hash"] = compute_plan_hash(plan)
    planfile.write_text(json.dumps(plan))
    return plan["plan_hash"]


@pytest.mark.parametrize("path,kind", [
    ("mine/job.json", "meta"), ("mine/tracking/current.json", "meta"),
    ("mine/inputs/abc.mp4", "input"), ("mine/tracking/generations/*", "generation"),
    ("mine/tracknet-finetune/runs", "generation"), ("mine/tracking/generations/best.pt", "generation"),
    ("mine/tracking/.stage-weights", "orphan"), ("mine/stats/generations/checkpoint-1", "generation"),
    ("mine", "orphan"), ("mine/tracking/generations/t3/../t1", "generation"),
])
def test_forged_plan_items_refused(world, tmp_path, path, kind):
    _, plan, planfile, vol = world
    h = hand_edited_plan(plan, planfile, path, kind)
    vol.files[path] = b"keep"
    log = tmp_path / "l.jsonl"
    run(planfile, vol, "--apply", "--yes", "--plan-hash", h, "--log", str(log))
    assert path not in vol.removed and path in vol.files
    assert any(json.loads(line).get("action") == "refused" for line in log.read_text().splitlines())


def test_other_jobs_never_touched(world, tmp_path):
    _, plan, planfile, vol = world
    run(planfile, vol, "--apply", "--yes", "--plan-hash", plan["plan_hash"], "--log", str(tmp_path / "l"))
    assert not any(p.startswith(("theirs", "dead")) for p in vol.removed)


def test_idempotent_second_run(world, tmp_path):
    _, plan, planfile, vol = world
    args = ("--apply", "--yes", "--plan-hash", plan["plan_hash"])
    assert run(planfile, vol, *args, "--log", str(tmp_path / "a.jsonl")) == 0
    after = dict(vol.files)
    assert run(planfile, vol, *args, "--log", str(tmp_path / "b.jsonl")) == 0
    assert vol.files == after
    rows = [json.loads(line) for line in (tmp_path / "b.jsonl").read_text().splitlines()]
    assert {r["action"] for r in rows if "action" in r} == {"already_deleted"}


def test_failure_is_logged_and_nonzero(world, tmp_path):
    _, plan, planfile, vol = world

    def boom(path):
        raise RuntimeError("volume unavailable")
    vol.remove_tree = boom
    log = tmp_path / "l.jsonl"
    assert run(planfile, vol, "--apply", "--yes", "--plan-hash", plan["plan_hash"], "--log", str(log)) == 1
    assert "volume unavailable" in log.read_text()


def test_apply_module_has_no_wildcard_or_listing_delete():
    src = Path(apply_mod.__file__).read_text()
    assert src.count("remove_tree(path)") == 1  # single explicit-path deletion site
