from __future__ import annotations

import hashlib
import json
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import modal_app_v2_service as svc


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeCall:
    """get() raises TimeoutError once per scripted step, then returns/raises."""

    def __init__(self, clock, steps, outcome=None, tick=12.0, on_tick=None):
        self.clock, self.steps, self.outcome = clock, steps, outcome
        self.tick, self.on_tick = tick, on_tick
        self.cancelled = False
        self.calls = 0

    def get(self, timeout=None):
        self.calls += 1
        self.clock.now += self.tick
        if self.calls <= self.steps:
            if self.on_tick:
                self.on_tick(self.calls)
            raise TimeoutError
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def cancel(self):
        self.cancelled = True


class FakeDict(dict):
    def __init__(self, fail_read=False):
        super().__init__()
        self.fail_read = fail_read

    def __getitem__(self, key):
        if self.fail_read:
            raise RuntimeError("dict down")
        return super().__getitem__(key)


@pytest.fixture
def env(monkeypatch):
    clock = FakeClock()
    reports = []
    monkeypatch.setattr(svc, "_clock", clock)
    monkeypatch.setattr(
        svc, "_report_status",
        lambda job_id, status, **kw: reports.append((status, kw.get("progress_percent"), kw.get("progress_phase"), kw)),
    )
    store = FakeDict()
    monkeypatch.setattr(svc, "_open_progress_dict", lambda: store)
    return types.SimpleNamespace(clock=clock, reports=reports, store=store)


def payload(run_id, percent, phase, t=0.0):
    return {"run_id": run_id, "percent": percent, "phase": phase, "updated_at": t}


def test_percent_phases_and_key_deleted(env):
    seq = {1: ("a", 10, "tracking"), 2: ("a", 30, "identity"), 3: ("a", 60, "court_projection")}

    def tick(i):
        env.store["j"] = payload(*seq[i], t=i)

    call = FakeCall(env.clock, 3, outcome={"ok": 1}, on_tick=tick)
    assert svc._await_with_progress(call, "j") == {"ok": 1}
    seqs = [(p, ph) for _, p, ph, _ in env.reports]
    print("callback sequence:", seqs)
    assert seqs == [(10, "tracking"), (30, "identity"), (60, "court_projection")]
    assert "j" not in env.store


def test_monotonic_across_retry_and_run_id_change(env):
    state = svc._JobProgress()
    env.store["j"] = payload("a", 40, "identity")
    svc._await_with_progress(FakeCall(env.clock, 1, outcome=1), "j", state)
    assert state.percent == 40
    # retry: new run_id with a lower percent holds and says restarted
    steps = {1: ("b", 6, "tracking"), 2: ("b", 30, "identity"), 3: ("b", 55, "court_projection")}
    call = FakeCall(env.clock, 3, outcome=1, on_tick=lambda i: env.store.__setitem__("j", payload(*steps[i], t=i)))
    env.reports.clear()
    svc._await_with_progress(call, "j", state)
    seqs = [(p, ph) for _, p, ph, _ in env.reports]
    assert seqs == [(40, "restarted"), (40, "restarted"), (55, "court_projection")]
    assert [p for p, _ in seqs] == sorted(p for p, _ in seqs)


def test_dict_failure_falls_back_to_creep_capped(env, monkeypatch):
    def boom():
        raise RuntimeError("no dict")

    monkeypatch.setattr(svc, "_open_progress_dict", boom)
    call = FakeCall(env.clock, 14, outcome=1)
    svc._await_with_progress(call, "j")
    percents = [p for _, p, _, _ in env.reports]
    assert percents[0] == 8 and max(percents) == 85 and percents == sorted(percents)


def test_dict_read_failure_falls_back(env):
    env.store.fail_read = True
    svc._await_with_progress(FakeCall(env.clock, 2, outcome=1), "j")
    assert [p for _, p, _, _ in env.reports] == [8, 16]


def test_key_deleted_on_error(env):
    env.store["j"] = payload("a", 10, "tracking")
    with pytest.raises(ValueError):
        svc._await_with_progress(FakeCall(env.clock, 1, outcome=ValueError("x")), "j")
    assert "j" not in env.store


def test_deadline_cancels(env):
    env.store["j"] = payload("a", 10, "tracking")
    state = svc._JobProgress()
    env.clock.now += svc.JOB_DEADLINE_SECONDS
    call = FakeCall(env.clock, 5, outcome=1)
    with pytest.raises(svc.JobAbortedError, match="deadline exceeded"):
        svc._await_with_progress(call, "j", state)
    assert call.cancelled and "j" not in env.store


def test_stall_watchdog_cancels_with_last_phase(env):
    env.store["j"] = payload("a", 10, "identity", t=1)
    call = FakeCall(env.clock, 10_000, tick=300.0)
    with pytest.raises(svc.JobAbortedError, match="stalled.*identity"):
        svc._await_with_progress(call, "j")
    assert call.cancelled


def test_summary_fields_and_none_safe():
    result = {
        "acceptance": {"ok": True}, "video": {"fps": 30, "duration_seconds": 12.5},
        "gpu": {"name": "L4"}, "frames": {"requested": 5, "processed": 5}, "timings": {"tracking": 3.0},
    }
    summary = svc._build_summary(result, engine="v5", window=48, attempts=2)
    core = {k: summary[k] for k in (
        "engine", "acceptance", "possession_engine", "segment_window", "attempts", "frames",
        "fps", "duration_seconds", "gpu", "timings")}
    assert core == {
        "engine": "v2", "acceptance": {"ok": True}, "possession_engine": "v5", "segment_window": 48,
        "attempts": 2, "frames": {"requested": 5, "processed": 5}, "fps": 30, "duration_seconds": 12.5,
        "gpu": {"name": "L4"}, "timings": {"tracking": 3.0},
    }
    assert summary["contract_version"] == 2
    assert summary["quality"]["schema_version"] == 1 and "provenance" in summary
    empty = svc._build_summary({}, engine="v5", window=96, attempts=1)
    assert empty["fps"] is None and empty["timings"] is None and empty["gpu"] is None


def test_allowlist_constants():
    assert len(svc.RETURN_ARTIFACT_PATHS) == 18
    assert "stats/tracks.jsonl" not in svc.RETURN_ARTIFACT_PATHS
    assert "stats/projected_shots.json" in svc.RETURN_ARTIFACT_PATHS
    assert svc.RETURN_ARTIFACT_MAX_FILE_BYTES == 30 * 1024 * 1024
    assert svc.MAX_ARTIFACT_BYTES == 128 * 1024 * 1024
    assert svc.JOB_DEADLINE_SECONDS == 3300
    assert svc._is_returnable("stats/teams.json")
    assert not svc._is_returnable("stats/masks.npz")
    assert svc._is_returnable("stats/possession_v3/generations/abc/possession_events_v3.jsonl")


def test_report_status_payload_phase_only_when_set(monkeypatch):
    sent = []
    monkeypatch.setenv("PROCESSING_CALLBACK_URL", "http://x")
    monkeypatch.setenv("PROCESSING_CALLBACK_TOKEN", "t")
    import urllib.request

    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(request, timeout=0):
        import json
        sent.append(json.loads(request.data))
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    svc._report_status("j", "RUNNING", progress_percent=5)
    svc._report_status("j", "RUNNING", progress_percent=5, progress_phase="tracking")
    assert "progress_phase" not in sent[0] and sent[1]["progress_phase"] == "tracking"


class _FakeS3:
    def __init__(self):
        self.uploaded = []

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(b"video")

    def upload_file(self, path, bucket, key):
        self.uploaded.append(key)


def _run_job(monkeypatch, env, result, payload_extra=None):
    s3 = _FakeS3()
    spawned = []
    monkeypatch.setattr(svc, "_r2_client", lambda: (s3, "b"))
    monkeypatch.setattr(svc, "_frame_bounds", lambda p, v: (0, 9))

    class Proc:
        def spawn(self, **kw):
            spawned.append(kw)
            return FakeCall(env.clock, 0, outcome=result)

    class Inst:
        process = Proc()

    monkeypatch.setattr(svc.modal.Cls, "from_name", staticmethod(lambda a, c: (lambda: Inst())))
    fn = svc.run_backed_job.get_raw_f()
    body = {"job_id": "j", "r2_key": "k", "output_prefix": "out", **(payload_extra or {})}
    return fn(body), s3, spawned


def test_upload_allowlist_default_engine_and_skipped_logged(monkeypatch, env, capsys):
    result = {
        "artifacts": {
            "stats/teams.json": b"1", "stats/masks.npz": b"2", "stats/projected_shots.json": b"3",
            "tracks.jsonl": b"4",
        },
        "artifacts_skipped": [{"path": "stats/projected_tracks.jsonl", "bytes": 99}],
        "video": {"fps": 30.0},
    }
    out, s3, spawned = _run_job(monkeypatch, env, result)
    assert sorted(s3.uploaded) == ["out/projected_shots.json", "out/teams.json"]
    assert spawned[0]["possession_engine"] == "v5"
    assert spawned[0]["progress_key"] == "j"
    assert spawned[0]["return_artifact_paths"] == svc.RETURN_ARTIFACT_PATHS
    assert spawned[0]["return_artifact_max_file_bytes"] == 30 * 1024 * 1024
    assert "projected_tracks.jsonl" in capsys.readouterr().out
    done = [r for r in env.reports if r[0] == "DONE"][0][3]
    assert done["summary"]["possession_engine"] == "v5" and done["summary"]["fps"] == 30.0
    assert done["progress_percent"] == 100


def test_failed_reported_on_deadline(monkeypatch, env):
    env.store["j"] = payload("a", 10, "tracking")
    s3 = _FakeS3()
    monkeypatch.setattr(svc, "_r2_client", lambda: (s3, "b"))
    monkeypatch.setattr(svc, "_frame_bounds", lambda p, v: (0, 9))
    call = FakeCall(env.clock, 10_000, tick=400.0)

    class Inst:
        class process:
            spawn = staticmethod(lambda **kw: call)

    monkeypatch.setattr(svc.modal.Cls, "from_name", staticmethod(lambda a, c: (lambda: Inst())))
    with pytest.raises(svc.JobAbortedError):
        svc.run_backed_job.get_raw_f()({"job_id": "j", "r2_key": "k"})
    failed = [r for r in env.reports if r[0] == "FAILED"]
    assert failed and "JobAbortedError" in failed[0][3]["error_message"]
    assert call.cancelled


# --- service output v2: v3 files, quality, provenance, callback retries ---------

FIXTURES = Path(__file__).parent / "fixtures" / "service_output"
GEN = "88894553b8885b54"


def _teammate_artifacts() -> dict[str, bytes]:
    """The teammate's real v3 stats (possession_v3.jsonl trimmed to 5 rows: shape only)."""

    stats = FIXTURES / "stats"
    return {
        path.relative_to(FIXTURES).as_posix(): path.read_bytes()
        for path in sorted(stats.rglob("*")) if path.is_file()
    }


def test_v3_generation_files_are_flattened_and_match_manifest():
    artifacts = _teammate_artifacts()
    plan, skipped = svc._plan_uploads(artifacts)
    names = [name for _, name in plan]
    assert skipped == []
    assert sorted(names) == sorted([
        "possession_v3_manifest.json", "possession_events_v3.jsonl", "ball_transitions_v3.jsonl",
        "possession_primitives_v3.jsonl", "possession_v3.jsonl",
    ])
    assert not any("/" in name for name in names)
    manifest = json.loads(artifacts["stats/possession_v3_manifest.json"])
    events = artifacts[f"stats/possession_v3/generations/{GEN}/possession_events_v3.jsonl"]
    assert hashlib.sha256(events).hexdigest() == manifest["artifacts"]["events"]["sha256"]


def test_second_generation_is_skipped_and_recorded():
    artifacts = _teammate_artifacts()
    stale = "stats/possession_v3/generations/deadbeef/possession_events_v3.jsonl"
    artifacts[stale] = b"stale"
    plan, skipped = svc._plan_uploads(artifacts)
    assert stale not in [path for path, _ in plan]
    assert skipped == [{"path": stale, "bytes": 5, "reason": "v3_generation_mismatch"}]
    # the current generation's events are still the ones uploaded
    assert (f"stats/possession_v3/generations/{GEN}/possession_events_v3.jsonl",
            "possession_events_v3.jsonl") in plan


def test_upload_name_rules_and_name_collision():
    assert svc._upload_name("stats/teams.json", None) == "teams.json"
    assert svc._upload_name("masks/x.npz", None) is None
    assert svc._upload_name("stats/possession_v3/generations/g1/possession_v3.jsonl", None) is None
    assert svc._upload_name("stats/possession_v3/generations/g1/possession_v3.jsonl", "g1") == "possession_v3.jsonl"
    assert svc._upload_name("stats/possession_v3/generations/g1/a/b.jsonl", "g1") is None
    artifacts = {
        "stats/possession_v3_manifest.json": b'{"generation": "g1"}',
        "stats/possession_v3/generations/g1/possession_v3.jsonl": b"1",
        "stats/possession_v3.jsonl": b"22",  # not allowlisted: ignored, not a collision
    }
    plan, skipped = svc._plan_uploads(artifacts)
    assert [name for _, name in plan] == ["possession_v3_manifest.json", "possession_v3.jsonl"]
    assert skipped == []
    # two allowlisted sources mapping to one name: first wins, second recorded
    original = svc.RETURN_ARTIFACT_PATHS
    svc.RETURN_ARTIFACT_PATHS = [*original, "stats/possession_v3.jsonl"]
    try:
        plan, skipped = svc._plan_uploads(artifacts)
    finally:
        svc.RETURN_ARTIFACT_PATHS = original
    assert [path for path, _ in plan][-1] == "stats/possession_v3/generations/g1/possession_v3.jsonl"
    assert skipped == [{"path": "stats/possession_v3.jsonl", "bytes": 2, "reason": "name_collision"}]


def test_generation_missing_manifest_skips_all_generation_files():
    artifacts = {"stats/possession_v3/generations/g1/possession_events_v3.jsonl": b"x"}
    plan, skipped = svc._plan_uploads(artifacts)
    assert plan == [] and skipped[0]["reason"] == "v3_generation_mismatch"


def _job_quality(name):
    folder = FIXTURES / "jobs" / name
    meta = json.loads((folder / "meta.json").read_text())
    results = json.loads((folder / "results.json").read_text())
    return svc.build_quality(
        results, meta["files"], [], None,
        acceptance=meta["acceptance"], team_assignments=meta["team_assignments"],
    )


def test_quality_critical_when_court_projection_silently_failed():
    quality = _job_quality("c9fad24b")
    assert quality["level"] == "critical"
    critical = [r for r in quality["reasons"] if r["severity"] == "critical"]
    assert [r["code"] for r in critical] == ["court_projection_empty"]
    assert quality["stages"]["court_projection"]["status"] == "empty"


@pytest.mark.parametrize("name", [
    "3931fd9f", "86f87c1b", "804f1784", "9d60cf76", "4b889bbd", "6dee2866", "cd5ce862", "87f5d08c",
])
def test_quality_real_good_jobs_are_not_critical(name):
    quality = _job_quality(name)
    assert quality["level"] != "critical"
    assert quality["stages"]["court_projection"]["status"] == "ok"
    assert quality["stages"]["jersey_numbers"]["status"] == "ok"
    # these downloads predate the v3 files: honest "missing", not invented
    assert quality["stages"]["possession_v3"]["status"] == "missing"
    assert quality["level"] == "degraded"


def test_quality_without_run_info_leaves_run_info_fields_none():
    quality = _job_quality("3931fd9f")
    court = quality["stages"]["court_projection"]
    assert (court["frames_total"], court["frames_projected"], court["frames_failed"],
            court["errors_by_type"]) == (None, None, None, None)
    assert quality["stages"]["tracking"] == {
        "segments": None, "overflow_dropped_ids": None, "seams": None, "seams_linked": None}
    assert quality["identity"]["reasons"] is None
    assert quality["identity"]["tracking_identity"] == "accepted"
    assert quality["identity"]["excluded_observations"] == 2
    prov = svc.build_provenance(
        {}, engine="v5", window=96, prompt="p", run_info=None, v3_generation=None)
    assert prov["pipeline_git_sha"] is None and prov["input_sha256"] is None
    assert prov["segment_overlap"] is None and prov["service_git_sha"] is None


def _good_files():
    return [name for name in svc.EXPECTED_ARTIFACT_NAMES]


def test_quality_ok_with_run_info_and_pass_count():
    results = {"config": {"court_model_id": "m"},
               "possession_v3_observability": {"status": "complete", "counts": {"transitions": 3, "events": 6}}}
    run_info = {
        "court": {"frames_total": 100, "frames_projected": 100, "frames_failed": 0, "errors_by_type": {}},
        "tracking": {"segments": 4, "overflow_dropped_ids": 0, "seams": 3, "seams_linked": 3},
        "acceptance_reasons": [], "input_sha256": "abc", "segment_overlap": 12,
        "pipeline_git_sha": "1234567", "sam31_commit": "c",
    }
    events = b'{"event_type": "PASS"}\nnot json\n{"event_type": "UNKNOWN"}\n{"event_type": "PASS"}\n'
    quality = svc.build_quality(
        results, _good_files(), [], run_info,
        acceptance={"tracking_identity": "accepted"}, team_assignments=5, v3_events=events)
    assert quality["level"] == "ok" and quality["reasons"] == []
    assert quality["stages"]["possession_v3"] == {
        "status": "ok", "transitions": 3, "events": 6, "passes": 2}
    assert quality["stages"]["tracking"]["seams_linked"] == 3
    assert quality["artifacts"]["missing_expected"] == []
    prov = svc.build_provenance(
        results, engine="v5", window=96, prompt="p", run_info=run_info, v3_generation="g")
    assert prov["pipeline_git_sha"] == "1234567" and prov["input_sha256"] == "abc"
    assert prov["segment_overlap"] == 12 and prov["court_model_id"] == "m"


def test_quality_degraded_and_critical_rules():
    base = {"config": {"court_model_id": "m"}, "possession_v3_observability": {"status": "complete", "counts": {}}}
    files = _good_files()
    ri = lambda **court: {"court": {"frames_total": 100, "frames_projected": 100 - court.get("f", 0),
                                   "frames_failed": court.get("f", 0), "errors_by_type": None}}
    assert svc.build_quality(base, files, [], ri(f=5), team_assignments=3)["level"] == "ok"
    partial = svc.build_quality(base, files, [], ri(f=6), team_assignments=3)
    assert partial["level"] == "degraded" and partial["stages"]["court_projection"]["status"] == "partial"
    assert svc.build_quality(base, files, [], None, team_assignments=0)["level"] == "critical"
    skipped = [{"path": "stats/x", "bytes": 1, "reason": "name_collision"}]
    assert svc.build_quality(base, files, skipped, None, team_assignments=3)["level"] == "degraded"
    overflow = {"tracking": {"overflow_dropped_ids": 2}}
    assert svc.build_quality(base, files, [], overflow, team_assignments=3)["level"] == "degraded"
    no_numbers = [f for f in files if f != "numbers.json"]
    q = svc.build_quality(base, no_numbers, [], None, team_assignments=3)
    assert q["level"] == "degraded" and q["stages"]["jersey_numbers"]["status"] == "missing"
    q = svc.build_quality(base, files, [], None, acceptance={"tracking_identity": "rejected"}, team_assignments=3)
    assert q["level"] == "degraded"
    failed = dict(base, possession_v3_observability={"status": "failed", "error": "boom"})
    q = svc.build_quality(failed, [f for f in files if f != svc.V3_EVENTS_NAME], [], None, team_assignments=3)
    assert q["stages"]["possession_v3"]["status"] == "failed" and q["level"] == "degraded"
    q = svc.build_quality(base, files + ["possession_v3_failure.json"], [], None, team_assignments=3)
    assert q["stages"]["possession_v3"]["status"] == "failed"
    q = svc.build_quality({"config": {}}, files, [], None, team_assignments=3)
    assert q["stages"]["court_projection"]["status"] == "unknown"
    q = svc.build_quality({"config": {"court_model_id": None}}, files, [], None, team_assignments=3)
    assert q["stages"]["court_projection"]["status"] == "disabled" and q["level"] == "ok"


def test_run_backed_job_summary_carries_quality_and_uploads_v3(monkeypatch, env):
    artifacts = _teammate_artifacts()
    artifacts["stats/teams.json"] = b'{"assignments_by_track_id": {"1": 0}}'
    artifacts["stats/numbers.json"] = b"{}"
    artifacts["stats/results.json"] = json.dumps({"config": {"court_model_id": "m"}}).encode()
    artifacts["stats/projected_tracks.jsonl"] = b"{}"
    artifacts[f"stats/possession_v3/generations/other/{svc.V3_EVENTS_NAME}"] = b"stale"
    result = {"artifacts": artifacts, "artifacts_skipped": [{"path": "stats/big", "bytes": 9}],
              "acceptance": {"tracking_identity": "accepted"}, "run_info": {"input_sha256": "ab"}}
    out, s3, _ = _run_job(monkeypatch, env, result, {"prompt": "custom"})
    assert "out/possession_events_v3.jsonl" in s3.uploaded and not any(k.count("/") > 1 for k in s3.uploaded)
    summary = [r for r in env.reports if r[0] == "DONE"][0][3]["summary"]
    assert summary["contract_version"] == 2
    assert summary["quality"]["stages"]["possession_v3"]["status"] == "ok"
    assert summary["quality"]["stages"]["possession_v3"]["passes"] == 3
    reasons = sorted(i["reason"] for i in summary["quality"]["artifacts"]["skipped"])
    assert reasons == ["too_large", "v3_generation_mismatch"]
    assert summary["quality"]["level"] == "degraded"
    assert summary["provenance"]["possession_v3_generation"] == GEN
    assert summary["provenance"]["prompt"] == "custom" and summary["provenance"]["input_sha256"] == "ab"
    json.dumps(summary)


class _Resp:
    status = 200

    def __enter__(self): return self
    def __exit__(self, *a): return False


def _callback(monkeypatch, outcomes):
    monkeypatch.setenv("PROCESSING_CALLBACK_URL", "http://x")
    monkeypatch.setenv("PROCESSING_CALLBACK_TOKEN", "t")
    calls, sleeps = [], []

    def fake_urlopen(request, timeout=0):
        calls.append(1)
        outcome = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(svc, "_sleep", sleeps.append)
    return calls, sleeps


def _http(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


def test_done_report_retries_5xx_then_succeeds(monkeypatch):
    calls, sleeps = _callback(monkeypatch, [_http(503), _http(502), "ok"])
    svc._report_status("j", "DONE", progress_percent=100)
    assert len(calls) == 3 and sleeps == [2, 5]


def test_failed_report_retries_connection_errors_and_gives_up_after_three(monkeypatch):
    calls, sleeps = _callback(monkeypatch, [urllib.error.URLError("down")])
    svc._report_status("j", "FAILED", error_message="x")  # must not raise
    assert len(calls) == 3 and sleeps == [2, 5]


def test_report_4xx_is_not_retried(monkeypatch):
    calls, sleeps = _callback(monkeypatch, [_http(400)])
    svc._report_status("j", "DONE")
    assert len(calls) == 1 and sleeps == []


def test_running_report_is_single_attempt(monkeypatch):
    calls, sleeps = _callback(monkeypatch, [_http(503)])
    svc._report_status("j", "RUNNING", progress_percent=5)
    assert len(calls) == 1 and sleeps == []


def test_git_sha_never_raises(monkeypatch):
    monkeypatch.setattr(svc.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))
    monkeypatch.setenv("SERVICE_GIT_SHA", "abc1234")
    assert svc._git_sha() == "abc1234"
    monkeypatch.delenv("SERVICE_GIT_SHA")
    assert svc._git_sha() is None
