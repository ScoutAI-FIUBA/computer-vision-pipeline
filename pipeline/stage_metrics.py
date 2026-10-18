"""Operational measurements for one V2 stage run (no Modal imports on purpose)."""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

SCHEMA = "v2-stage-metrics-1"
METRICS_FILENAME = "stage_metrics.json"
API_CALL_KEYS = (
    "roboflow_shot_detection_frames",
    "roboflow_court_detection_frames",
    "roboflow_number_detector_frames",
    "roboflow_ocr_crops",
)
# Single source of truth for the documented `[v2-timing]` log lines.
STAGE_LINE_RE = re.compile(
    r"\[v2-timing\] stage=(?P<stage>\S+) started=(?P<started>\S+) finished=(?P<finished>\S+) "
    r"elapsed=(?P<elapsed>\d+\.\d{3}) timeout_margin=(?P<timeout_margin>-?\d+\.\d{3}) gpu=(?P<gpu>.+)$"
)
JOB_LINE_RE = re.compile(
    r"\[v2-timing\] job id=(?P<job_id>\S+) total_wall_seconds=(?P<total_wall_seconds>\d+\.\d{3})$"
)


def _utc_iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def detect_gpu() -> dict[str, Any] | None:
    """Return the first CUDA device, or None when torch or CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "name": torch.cuda.get_device_name(0),
            "memory_mib": int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)),
        }
    except Exception:
        # Metrics must never fail a stage.
        return None


class StageMetrics:
    """Starts timing on construction; `build()` freezes the end and returns the record."""

    def __init__(self, stage: str, *, function_timeout_seconds: float,
                 container: Mapping[str, Any] | None = None) -> None:
        self.stage = stage
        self.function_timeout_seconds = function_timeout_seconds
        self.container = {"cpu": None, "memory_mib": None, **(container or {})}
        self.phases: dict[str, float] = {}
        self.api_calls: dict[str, int | None] = {key: None for key in API_CALL_KEYS}
        self.frames: dict[str, int | None] = {"requested": None, "processed": None}
        self.attempt: int | None = None
        self._started_at = datetime.now(timezone.utc)
        self._started = time.perf_counter()

    def add_phase(self, name: str, seconds: float) -> None:
        if not name.endswith("_seconds"):
            raise ValueError(f"phase name must end with _seconds: {name!r}")
        self.phases[name] = self.phases.get(name, 0.0) + float(seconds)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_phase(name, time.perf_counter() - started)

    def set_api_calls(self, **counts: int | None) -> None:
        unknown = set(counts) - set(API_CALL_KEYS)
        if unknown:
            raise ValueError(f"unknown api call counters: {sorted(unknown)}")
        self.api_calls.update(counts)

    def set_frames(self, *, requested: int | None = None, processed: int | None = None) -> None:
        self.frames = {"requested": requested, "processed": processed}

    def build(self, *, gpu: Mapping[str, Any] | None = None) -> dict[str, Any]:
        elapsed = time.perf_counter() - self._started
        finished_at = datetime.now(timezone.utc)
        return {
            "schema": SCHEMA,
            "stage": self.stage,
            "started_at": _utc_iso(self._started_at),
            "finished_at": _utc_iso(finished_at),
            "elapsed_seconds": elapsed,
            "phases": dict(self.phases),
            "gpu": dict(gpu) if gpu is not None else None,
            "function_timeout_seconds": self.function_timeout_seconds,
            "timeout_margin_seconds": self.function_timeout_seconds - elapsed,
            "container": dict(self.container),
            "frames": dict(self.frames),
            "api_calls": dict(self.api_calls),
            "attempt": self.attempt,
        }


def metrics_artifact(record: Mapping[str, Any]) -> bytes:
    return json.dumps(record, indent=2, sort_keys=True).encode("utf-8")


def format_stage_line(record: Mapping[str, Any]) -> str:
    gpu = record.get("gpu")
    return (
        f"[v2-timing] stage={record['stage']} started={record['started_at']} "
        f"finished={record['finished_at']} elapsed={record['elapsed_seconds']:.3f} "
        f"timeout_margin={record['timeout_margin_seconds']:.3f} "
        f"gpu={gpu['name'] if gpu and gpu.get('name') else 'none'}"
    )


def format_job_line(job_id: str | None, total_wall_seconds: float) -> str:
    return f"[v2-timing] job id={job_id or 'none'} total_wall_seconds={total_wall_seconds:.3f}"
