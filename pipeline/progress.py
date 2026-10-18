"""Job progress reporting shared by the V2 pipeline (no Modal imports on purpose)."""

from __future__ import annotations

import time
from typing import Any, Callable

# Percent of the whole job; download/upload are owned by the service, not the pipeline.
PHASE_BANDS: dict[str, tuple[float, float]] = {
    "download": (0.0, 2.0),
    "model_load": (2.0, 6.0),
    "tracking": (6.0, 21.0),
    "identity": (21.0, 45.0),
    "packaging": (45.0, 50.0),
    "court_projection": (50.0, 66.0),
    "ball_tracking": (66.0, 73.0),
    "shot_detection": (73.0, 83.0),
    "possession": (83.0, 86.0),
    "jersey_numbers": (86.0, 96.0),
    "upload": (96.0, 99.0),
    "done": (100.0, 100.0),
}


def percent_for(phase: str, fraction: float | None) -> float:
    if phase not in PHASE_BANDS:
        raise ValueError(f"unknown progress phase: {phase!r}")
    start, end = PHASE_BANDS[phase]
    if fraction is None or fraction != fraction:
        return start
    return start + (end - start) * min(1.0, max(0.0, float(fraction)))


class ProgressReporter:
    """`stage` is the `on_stage` hook; it never raises, so progress can never fail a job."""

    def __init__(
        self,
        sink: Callable[[dict[str, Any]], None] | None,
        *,
        run_id: str | None = None,
        min_interval_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sink = sink
        self.run_id = run_id
        self._min_interval = float(min_interval_seconds)
        self._clock = clock
        self._phase: str | None = None
        self._phase_started = 0.0
        self._fraction: float | None = None
        self._percent = 0.0
        self._closed: dict[str, float] = {}
        self._last_write: float | None = None

    @property
    def timings(self) -> dict[str, float]:
        result = dict(self._closed)
        if self._phase is not None:
            result[self._phase] = result.get(self._phase, 0.0) + (self._clock() - self._phase_started)
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self._phase,
            "fraction": self._fraction,
            "percent": self._percent,
            "updated_at": time.time(),
            "timings": self.timings,
        }

    def stage(self, name: str, fraction: float | None = None) -> None:
        try:
            self._stage(name, fraction)
        except Exception:
            pass

    def _stage(self, name: str, fraction: float | None) -> None:
        percent = percent_for(name, fraction)
        now = self._clock()
        changed = name != self._phase
        if changed:
            if self._phase is not None:
                self._closed[self._phase] = self._closed.get(self._phase, 0.0) + (now - self._phase_started)
            self._phase = name
            self._phase_started = now
        if fraction is not None:
            fraction = min(1.0, max(0.0, float(fraction)))
        self._fraction = fraction
        self._percent = max(self._percent, percent)
        always = changed or fraction in (0.0, 1.0)
        if self._sink is None:
            return
        if not always and self._last_write is not None and now - self._last_write < self._min_interval:
            return
        self._last_write = now
        try:
            self._sink(self.snapshot())
        except Exception:
            pass
