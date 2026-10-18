from __future__ import annotations

import types
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
    assert summary == {
        "engine": "v2", "acceptance": {"ok": True}, "possession_engine": "v5", "segment_window": 48,
        "attempts": 2, "frames": {"requested": 5, "processed": 5}, "fps": 30, "duration_seconds": 12.5,
        "gpu": {"name": "L4"}, "timings": {"tracking": 3.0},
    }
    empty = svc._build_summary({}, engine="v5", window=96, attempts=1)
    assert empty["fps"] is None and empty["timings"] is None and empty["gpu"] is None


def test_allowlist_constants():
    assert len(svc.RETURN_ARTIFACT_PATHS) == 10
    assert "stats/projected_shots.json" in svc.RETURN_ARTIFACT_PATHS
    assert svc.RETURN_ARTIFACT_MAX_FILE_BYTES == 30 * 1024 * 1024
    assert svc.MAX_ARTIFACT_BYTES == 128 * 1024 * 1024
    assert svc.JOB_DEADLINE_SECONDS == 3300
    assert svc._is_returnable("stats/teams.json")
    assert not svc._is_returnable("stats/masks.npz")


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
