from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import pipeline.runner as runner
from pipeline.config import PipelineConfig


class FakeHTTPError(Exception):
    """Stand-in for inference_sdk.http.errors.HTTPCallErrorError (not a test dependency)."""


def _config():
    return PipelineConfig(
        roboflow_api_key="test-key", court_model_id="court/1", team_embedder_backend="none",
        number_evidence_enabled=False,
    )


def _setup(monkeypatch, detector, frames=5):
    rows = [{"frame_index": f, "track_id": f + 1, "xyxy": [10, 10, 30, 50]} for f in range(frames)]
    monkeypatch.setattr(runner, "_read_video_frames", lambda _p, indices: {i: i for i in indices})
    monkeypatch.setattr(runner, "_build_detector_for_model", lambda *_a, **_k: detector)
    monkeypatch.setattr(runner, "parse_court_keypoints", lambda result, **_k: result)
    monkeypatch.setattr(
        runner, "compute_frame_homography",
        lambda _kp, **_k: SimpleNamespace(matrix="H", inliers_count=4, reprojection_mean_m=0.0),
    )
    monkeypatch.setattr(runner, "project_points", lambda _m, points: [[1.0, 2.0] for _ in points])
    return rows


def _run(tmp_path, rows, counters):
    output = tmp_path / "projected_tracks.jsonl"
    written = runner._project_v2_rows(
        tmp_path / "input.mp4", output, rows, info=SimpleNamespace(width=100, height=100),
        config=_config(), emit=lambda _m: None, counters=counters,
    )
    return written, output


def test_single_frame_http_error_keeps_other_frames_and_is_counted(tmp_path, monkeypatch):
    def detect(frame):
        if frame == 2:
            raise FakeHTTPError("503")
        return frame

    rows = _setup(monkeypatch, SimpleNamespace(detect=detect))
    counters: dict = {}
    written, output = _run(tmp_path, rows, counters)

    assert written == 4
    assert output.exists()
    assert [json.loads(line)["frame_index"] for line in output.read_text().splitlines()] == [0, 1, 3, 4]
    assert counters["frames_total"] == 5
    assert counters["frames_projected"] == 4
    assert counters["frames_failed"] == 1
    assert counters["errors_by_type"] == {"FakeHTTPError": 1}


def test_every_frame_failing_removes_file_and_counts_all(tmp_path, monkeypatch):
    def detect(_frame):
        raise KeyError("boom")

    rows = _setup(monkeypatch, SimpleNamespace(detect=detect), frames=3)
    counters: dict = {}
    written, output = _run(tmp_path, rows, counters)

    assert written == 0
    assert not output.exists()
    assert (counters["frames_total"], counters["frames_projected"], counters["frames_failed"]) == (3, 0, 3)
    assert counters["errors_by_type"] == {"KeyError": 3}


def test_detector_build_failure_returns_zero_and_counts(tmp_path, monkeypatch):
    rows = _setup(monkeypatch, None, frames=2)

    def build(*_a, **_k):
        raise FakeHTTPError("no detector")

    monkeypatch.setattr(runner, "_build_detector_for_model", build)
    counters: dict = {}
    written, output = _run(tmp_path, rows, counters)

    assert written == 0
    assert not output.exists()
    assert counters["frames_failed"] == 2
    assert counters["errors_by_type"] == {"FakeHTTPError": 2}


def test_counters_are_always_filled_without_metrics(tmp_path, monkeypatch):
    import cv2  # noqa: F401  (the stats adapter needs it importable)

    court = {}
    rows = _setup(monkeypatch, SimpleNamespace(detect=lambda frame: frame), frames=2)
    written, _ = _run(tmp_path, rows, court)
    assert written == 2
    assert court["court_detection_frames"] == 2
    assert court["errors_by_type"] == {}
