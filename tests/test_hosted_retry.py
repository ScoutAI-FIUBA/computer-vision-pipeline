from __future__ import annotations

import sys
import types

import pytest

from pipeline import detection
from pipeline.detection import HostedRoboflowDetector, PrefetchingDetector


class HTTPCallErrorError(Exception):
    """Stand-in for inference_sdk.http.errors.HTTPCallErrorError (matched by class name)."""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def infer(self, frame, model_id):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _detector(monkeypatch, outcomes):
    client = FakeClient(outcomes)
    monkeypatch.setitem(sys.modules, "inference_sdk", types.SimpleNamespace(InferenceHTTPClient=lambda **_k: client))
    sleeps: list[float] = []
    det = HostedRoboflowDetector(api_key="k", api_url="u", model_id="m", sleep=sleeps.append)
    return det, client, sleeps


def test_two_503_then_success_is_three_calls(monkeypatch):
    det, client, sleeps = _detector(monkeypatch, [HTTPCallErrorError(503), HTTPCallErrorError(503), {"predictions": []}])
    assert det.detect("f") == {"predictions": []}
    assert client.calls == 3 and det.call_count == 3
    assert sleeps == [1.0, 3.0]
    assert (det.counters.retries, det.counters.failures) == (2, 0)


def test_third_failure_raises_after_bounded_retries(monkeypatch):
    det, client, _ = _detector(monkeypatch, [HTTPCallErrorError(500)] * 5)
    with pytest.raises(HTTPCallErrorError):
        det.detect("f")
    assert client.calls == 3
    assert (det.counters.retries, det.counters.failures) == (2, 1)


def test_404_is_not_retried(monkeypatch):
    det, client, sleeps = _detector(monkeypatch, [HTTPCallErrorError(404), {"predictions": []}])
    with pytest.raises(HTTPCallErrorError):
        det.detect("f")
    assert client.calls == 1 and sleeps == []
    assert (det.counters.retries, det.counters.failures) == (0, 1)


def test_429_is_retried(monkeypatch):
    det, client, sleeps = _detector(monkeypatch, [HTTPCallErrorError(429), {"predictions": [1]}])
    assert det.detect("f") == {"predictions": [1]}
    assert client.calls == 2 and sleeps == [1.0]


def test_connection_and_timeout_errors_are_retried(monkeypatch):
    det, client, _ = _detector(monkeypatch, [ConnectionError("reset"), TimeoutError("slow"), {"ok": 1}])
    assert det.detect("f") == {"ok": 1}
    assert client.calls == 3


def test_non_transient_error_is_not_retried(monkeypatch):
    det, client, _ = _detector(monkeypatch, [ValueError("bad"), {"ok": 1}])
    with pytest.raises(ValueError):
        det.detect("f")
    assert client.calls == 1


def test_prefetching_path_retries_through_inner_and_exposes_counters(monkeypatch):
    det, client, _ = _detector(monkeypatch, [HTTPCallErrorError(502), {"ok": 1}])
    with PrefetchingDetector(det) as prefetching:
        prefetching.submit("f")
        assert prefetching.take("f") == {"ok": 1}
        assert prefetching.counters.retries == 1
    assert client.calls == 2


def test_is_retryable_helper():
    assert detection.is_retryable_detector_error(HTTPCallErrorError(503))
    assert not detection.is_retryable_detector_error(HTTPCallErrorError(400))
