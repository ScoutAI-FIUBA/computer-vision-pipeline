from __future__ import annotations

import pytest

from pipeline.progress import PHASE_BANDS, ProgressReporter, percent_for


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make(**kwargs):
    clock = Clock()
    writes: list[dict] = []
    reporter = ProgressReporter(writes.append, run_id="r1", clock=clock, **kwargs)
    return reporter, clock, writes


def test_bands_are_contiguous_and_ordered():
    bands = list(PHASE_BANDS.values())
    assert bands[0][0] == 0.0 and bands[-1] == (100.0, 100.0)
    assert [b[1] for b in bands[:-2]] == [b[0] for b in bands[1:-1]]
    assert bands[-2][1] == 99.0


def test_percent_for_interpolates_and_clamps():
    assert percent_for("tracking", None) == 6.0
    assert percent_for("tracking", 0.0) == 6.0
    assert percent_for("tracking", 1.0) == 21.0
    assert percent_for("tracking", 0.5) == pytest.approx(13.5)
    assert percent_for("tracking", -3) == 6.0
    assert percent_for("tracking", 9) == 21.0
    assert percent_for("identity", float("nan")) == 21.0
    with pytest.raises(ValueError):
        percent_for("nope", 0.5)


def test_payload_shape_and_timings_accumulate():
    reporter, clock, writes = make()
    reporter.stage("model_load", 0.0)
    clock.now += 2
    reporter.stage("tracking", 0.0)
    clock.now += 3
    reporter.stage("tracking", 1.0)
    clock.now += 1
    reporter.stage("identity")

    assert set(writes[-1]) == {"run_id", "phase", "fraction", "percent", "updated_at", "timings"}
    assert writes[-1]["run_id"] == "r1" and writes[-1]["phase"] == "identity" and writes[-1]["fraction"] is None
    assert reporter.timings == {"model_load": 2.0, "tracking": 4.0, "identity": 0.0}
    clock.now += 5
    assert reporter.timings["identity"] == 5.0
    assert reporter.snapshot()["timings"]["identity"] == 5.0


def test_revisited_phase_accumulates():
    reporter, clock, _ = make()
    reporter.stage("tracking", 0.0)
    clock.now += 1
    reporter.stage("identity", 0.0)
    clock.now += 1
    reporter.stage("tracking", 0.5)
    clock.now += 1
    assert reporter.timings["tracking"] == 2.0


def test_percent_is_monotonic():
    reporter, _, writes = make(min_interval_seconds=0)
    reporter.stage("tracking", 0.8)
    reporter.stage("tracking", 0.2)
    reporter.stage("model_load", 1.0)
    assert [w["percent"] for w in writes] == [pytest.approx(18.0)] * 3


def test_throttles_midphase_but_always_writes_edges_and_phase_changes():
    reporter, clock, writes = make(min_interval_seconds=5.0)
    reporter.stage("tracking", 0.0)
    clock.now += 1
    reporter.stage("tracking", 0.3)
    clock.now += 1
    reporter.stage("tracking", 0.4)
    assert len(writes) == 1
    clock.now += 4
    reporter.stage("tracking", 0.5)
    assert len(writes) == 2
    clock.now += 0.1
    reporter.stage("tracking", 1.0)
    assert len(writes) == 3
    clock.now += 0.1
    reporter.stage("identity", 0.3)
    assert len(writes) == 4
    # Throttled updates still advance the in-memory state.
    clock.now += 0.1
    reporter.stage("identity", 0.6)
    assert len(writes) == 4 and reporter.snapshot()["fraction"] == 0.6


def test_sink_failure_is_swallowed_and_does_not_stop_reporting():
    calls = []

    def sink(payload):
        calls.append(payload)
        raise RuntimeError("down")

    reporter = ProgressReporter(sink, clock=Clock())
    reporter.stage("tracking", 0.0)
    reporter.stage("identity", 0.0)
    assert len(calls) == 2


def test_no_sink_and_unknown_phase_never_raise():
    reporter = ProgressReporter(None, clock=Clock())
    reporter.stage("tracking", 0.5)
    reporter.stage("bogus", 0.5)
    assert reporter.snapshot()["phase"] == "tracking"
    assert reporter.snapshot()["run_id"] is None
