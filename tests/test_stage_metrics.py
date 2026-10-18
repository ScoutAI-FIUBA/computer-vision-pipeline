from __future__ import annotations

import json
import re

import pytest

from pipeline import stage_metrics
from pipeline.stage_metrics import (
    JOB_LINE_RE,
    STAGE_LINE_RE,
    StageMetrics,
    format_job_line,
    format_stage_line,
    metrics_artifact,
)

ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


def test_phases_accumulate_and_require_seconds_suffix():
    metrics = StageMetrics("tracking", function_timeout_seconds=100)
    metrics.add_phase("tracking_seconds", 1.5)
    metrics.add_phase("tracking_seconds", 0.5)
    with metrics.phase("identity_seconds"):
        pass
    with pytest.raises(ValueError, match="_seconds"):
        metrics.add_phase("tracking", 1)
    record = metrics.build()
    assert record["phases"]["tracking_seconds"] == 2.0
    assert record["phases"]["identity_seconds"] >= 0


def test_phase_is_recorded_when_the_block_raises():
    metrics = StageMetrics("stats", function_timeout_seconds=100)
    with pytest.raises(RuntimeError):
        with metrics.phase("tracknet_seconds"):
            raise RuntimeError("boom")
    assert "tracknet_seconds" in metrics.phases


def test_timeout_margin_is_timeout_minus_elapsed(monkeypatch):
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(stage_metrics.time, "perf_counter", lambda: next(ticks))
    record = StageMetrics("stats", function_timeout_seconds=3600).build()
    assert record["elapsed_seconds"] == 2.5
    assert record["function_timeout_seconds"] == 3600
    assert record["timeout_margin_seconds"] == 3597.5


def test_record_has_the_agreed_shape_and_defaults_to_null():
    metrics = StageMetrics("possession", function_timeout_seconds=60)
    record = metrics.build()
    assert record["schema"] == "v2-stage-metrics-1"
    assert record["stage"] == "possession"
    assert ISO_Z.match(record["started_at"]) and ISO_Z.match(record["finished_at"])
    assert record["gpu"] is None
    assert record["container"] == {"cpu": None, "memory_mib": None}
    assert record["frames"] == {"requested": None, "processed": None}
    assert record["api_calls"] == {
        "roboflow_shot_detection_frames": None, "roboflow_court_detection_frames": None,
        "roboflow_number_detector_frames": None, "roboflow_ocr_crops": None,
    }
    assert record["attempt"] is None
    assert set(record) == {
        "schema", "stage", "started_at", "finished_at", "elapsed_seconds", "phases", "gpu",
        "function_timeout_seconds", "timeout_margin_seconds", "container", "frames", "api_calls",
        "attempt",
    }


def test_setters_and_serialization_round_trip():
    metrics = StageMetrics("stats", function_timeout_seconds=60, container={"cpu": 4})
    metrics.set_frames(requested=10, processed=9)
    metrics.set_api_calls(roboflow_shot_detection_frames=9)
    metrics.attempt = 2
    record = metrics.build(gpu={"name": "NVIDIA A100", "memory_mib": 81920})
    assert json.loads(metrics_artifact(record)) == record
    assert record["gpu"] == {"name": "NVIDIA A100", "memory_mib": 81920}
    assert record["container"] == {"cpu": 4, "memory_mib": None}
    assert record["api_calls"]["roboflow_shot_detection_frames"] == 9
    with pytest.raises(ValueError, match="unknown"):
        metrics.set_api_calls(made_up=1)


def test_detect_gpu_is_none_without_cuda(monkeypatch):
    import sys
    import types

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert stage_metrics.detect_gpu() is None
    monkeypatch.setitem(sys.modules, "torch", None)  # import fails
    assert stage_metrics.detect_gpu() is None


def test_detect_gpu_reads_name_and_memory(monkeypatch):
    import sys
    import types

    cuda = types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda index: "NVIDIA A100-SXM4-80GB",
        get_device_properties=lambda index: types.SimpleNamespace(total_memory=80 * 1024 * 1024 * 1024),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    assert stage_metrics.detect_gpu() == {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920}


def test_log_lines_match_the_documented_regexes():
    record = StageMetrics("tracking", function_timeout_seconds=3600).build(
        gpu={"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920}
    )
    match = STAGE_LINE_RE.match(format_stage_line(record))
    assert match
    assert match["stage"] == "tracking"
    assert match["started"] == record["started_at"] and match["finished"] == record["finished_at"]
    assert float(match["elapsed"]) == pytest.approx(record["elapsed_seconds"], abs=5e-4)
    assert float(match["timeout_margin"]) == pytest.approx(record["timeout_margin_seconds"], abs=5e-4)
    assert match["gpu"] == "NVIDIA A100-SXM4-80GB"

    no_gpu = STAGE_LINE_RE.match(format_stage_line({**record, "gpu": None}))
    assert no_gpu["gpu"] == "none"

    job = JOB_LINE_RE.match(format_job_line("game-1", 12.34567))
    assert job["job_id"] == "game-1" and job["total_wall_seconds"] == "12.346"
    assert JOB_LINE_RE.match(format_job_line(None, 0))["job_id"] == "none"
