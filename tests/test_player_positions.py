"""Regression coverage for the dedicated player-position pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.video import Segment, VideoInfo  # noqa: E402


def test_build_player_position_records_projects_and_marks_homography_status():
    from pipeline.court_projection import HomographyResult
    from pipeline.player_positions import build_player_position_records

    rows = [
        {
            "track_id": 7,
            "team_id": 2,
            "track_state": "detected",
            "class_name": "player",
            "confidence": 0.9,
            "xyxy": [10.0, 20.0, 30.0, 60.0],
        }
    ]

    records = build_player_position_records(
        frame_index=15,
        fps=30.0,
        rows=rows,
        homography_result=HomographyResult(matrix="unused", inliers_count=4, reprojection_mean_m=0.25),
        project_points_fn=lambda _matrix, points: [[point[0] / 10.0, point[1] / 10.0] for point in points],
    )

    assert records == [
        {
            "frame_index": 15,
            "timestamp_seconds": 0.5,
            "player_id": 7,
            "team_id": 2,
            "track_state": "detected",
            "class_name": "player",
            "confidence": 0.9,
            "bbox_xyxy": [10.0, 20.0, 30.0, 60.0],
            "image_anchor_xy": [20.0, 60.0],
            "court_xy_m": [2.0, 6.0],
            "homography_frame_status": "ok",
            "homography_inliers_count": 4,
            "homography_reprojection_mean_m": 0.25,
        }
    ]


def test_build_player_position_records_keeps_rows_when_homography_is_missing():
    from pipeline.player_positions import build_player_position_records

    records = build_player_position_records(
        frame_index=9,
        fps=30.0,
        rows=[
            {
                "track_id": 3,
                "track_state": "propagated",
                "class_name": "player",
                "confidence": 0.8,
                "xyxy": [2.0, 4.0, 12.0, 24.0],
            }
        ],
        homography_result=None,
        homography_status="insufficient_points",
    )

    assert records == [
        {
            "frame_index": 9,
            "timestamp_seconds": 0.3,
            "player_id": 3,
            "team_id": None,
            "track_state": "propagated",
            "class_name": "player",
            "confidence": 0.8,
            "bbox_xyxy": [2.0, 4.0, 12.0, 24.0],
            "image_anchor_xy": [7.0, 24.0],
            "court_xy_m": None,
            "homography_frame_status": "insufficient_points",
            "homography_inliers_count": None,
            "homography_reprojection_mean_m": None,
        }
    ]


def test_debug_point_color_prefers_team_id_and_falls_back_to_player_id():
    from pipeline.player_positions import _debug_point_color, _track_color

    assert _debug_point_color(player_id=7, team_id=1) == _track_color(1)
    assert _debug_point_color(player_id=7, team_id=None) == _track_color(7)


def test_run_player_position_pipeline_writes_positions_and_debug_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import numpy as np
    from pipeline.player_positions import run_player_position_pipeline

    segments = [
        Segment(frame_indices=[0, 6], frames=[np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)], is_last=False),
        Segment(frame_indices=[6, 12], frames=[np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)], is_last=True),
    ]
    tracking_detector_results = [
        {"image": {"width": 32, "height": 32}, "predictions": [{"class": "player", "confidence": 0.9}]},
        {"image": {"width": 32, "height": 32}, "predictions": [{"class": "player", "confidence": 0.9}]},
    ]
    court_detector_results = [
        {"predictions": [{"class": "09", "x": 4, "y": 4, "confidence": 0.9}]},
        {"predictions": [{"class": "09", "x": 4, "y": 4, "confidence": 0.9}]},
    ]

    class FakeDetector:
        def __init__(self, results):
            self._results = results
            self.call_count = 0

        def detect(self, _frame):
            result = self._results[self.call_count]
            self.call_count += 1
            return result

    class FakeTracker:
        device = "cpu"

        def propagate_segment(self, frames, seed_rows):
            assert len(frames) == 2
            return {1: [{**dict(row), "xyxy": [8.0, 10.0, 18.0, 30.0], "track_state": "propagated"} for row in seed_rows]}

    class FakeEmbedder:
        name = "none"

    class FakeTeamEmbedder:
        name = "none"

        def embed(self, _frame, _samples):
            return []

    class FakeAssignment:
        def __init__(self, track_id, team_id):
            self.track_id = track_id
            self.team_id = team_id

    class FakeTeamResult:
        assignments = [FakeAssignment(1, 2)]
        track_diagnostics = {}

    def fake_build_detector(config, *, model_id=None):
        requested = model_id or config.player_model_id
        return FakeDetector(court_detector_results if requested == config.court_model_id else tracking_detector_results)

    monkeypatch.setattr(
        "pipeline.player_positions.probe",
        lambda _path: VideoInfo(fps=30.0, frame_count=13, width=32, height=32),
    )
    monkeypatch.setattr("pipeline.player_positions.iter_segments", lambda *_args, **_kwargs: iter(segments))
    monkeypatch.setattr("pipeline.player_positions.build_detector", fake_build_detector)
    monkeypatch.setattr(
        "pipeline.player_positions.detections_from_result",
        lambda result, **_kwargs: [
            {"class_name": prediction["class"], "confidence": prediction.get("confidence", 1.0), "xyxy": [6.0, 8.0, 16.0, 28.0]}
            for prediction in result["predictions"]
            if prediction.get("class") == "player"
        ],
    )
    monkeypatch.setattr("pipeline.player_positions.detection_rows", lambda rows: [dict(row) for row in rows])
    monkeypatch.setattr("pipeline.player_positions.build_embedder", lambda _backend: FakeEmbedder())
    monkeypatch.setattr("pipeline.player_positions.build_team_embedder", lambda *_args, **_kwargs: FakeTeamEmbedder())
    monkeypatch.setattr(
        "pipeline.player_positions.parse_court_keypoints",
        lambda *_args, **_kwargs: [
            {"label": "01", "x": 0.0, "y": 0.0, "confidence": 0.9},
            {"label": "08", "x": 0.0, "y": 50.0, "confidence": 0.9},
            {"label": "34", "x": 80.0, "y": 0.0, "confidence": 0.9},
            {"label": "41", "x": 80.0, "y": 50.0, "confidence": 0.9},
        ],
    )

    class FakeHomography:
        matrix = "unused"
        inliers_count = 4
        reprojection_mean_m = 0.0

    monkeypatch.setattr("pipeline.player_positions.compute_frame_homography", lambda *_args, **_kwargs: FakeHomography())
    monkeypatch.setattr(
        "pipeline.player_positions.project_points",
        lambda _matrix, points: [[float(point[0]) / 2.0, float(point[1]) / 4.0] for point in points],
    )
    monkeypatch.setattr(
        "pipeline.player_positions.identify_track_teams",
        lambda *_args, **_kwargs: FakeTeamResult(),
    )

    output_dir = tmp_path / "out"
    config = PipelineConfig.from_env(
        roboflow_api_key="test-key",
        court_model_id="court-model/1",
        target_fps=5.0,
        frame_stride=5,
        appearance_weight=0.0,
        reid_backend="none",
        team_embedder_backend="none",
        min_team_samples_per_track=1,
        team_sample_stride=1,
        max_team_samples_per_track=1,
        prefetch_detections=False,
        allowed_classes=("player",),
    )

    artifact = run_player_position_pipeline(tmp_path / "dummy.mp4", config, output_dir, tracker=FakeTracker())

    assert artifact["positions_file"] == "player_positions.jsonl"
    assert artifact["debug_video_file"] == "player_positions_debug.mp4"
    rows = [json.loads(line) for line in (output_dir / "player_positions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["frame_index"] for row in rows] == [0, 6, 12]
    assert {row["player_id"] for row in rows} == {1}
    assert {row["team_id"] for row in rows} == {2}
    assert rows[0]["court_xy_m"] == [5.5, 7.0]
    assert rows[-1]["court_xy_m"] == [6.5, 7.5]
    assert rows[0]["timestamp_seconds"] == 0.0
    assert rows[-1]["timestamp_seconds"] == 0.4
    assert (output_dir / "player_positions_debug.mp4").exists()


def test_run_player_position_pipeline_skips_debug_video_when_no_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import numpy as np
    from pipeline.player_positions import run_player_position_pipeline

    segments = [
        Segment(frame_indices=[0, 6], frames=[np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)], is_last=True),
    ]

    class FakeDetector:
        def __init__(self, results):
            self._results = results
            self.call_count = 0

        def detect(self, _frame):
            result = self._results[self.call_count]
            self.call_count += 1
            return result

    class FakeTracker:
        device = "cpu"

        def propagate_segment(self, frames, seed_rows):
            assert len(frames) == 2
            assert seed_rows == []
            return {1: []}

    class FakeEmbedder:
        name = "none"

    class FakeTeamEmbedder:
        name = "none"

        def embed(self, _frame, _samples):
            return []

    def fake_build_detector(config, *, model_id=None):
        requested = model_id or config.player_model_id
        if requested == config.court_model_id:
            return FakeDetector([{"predictions": []}])
        return FakeDetector([{"image": {"width": 32, "height": 32}, "predictions": []}])

    monkeypatch.setattr(
        "pipeline.player_positions.probe",
        lambda _path: VideoInfo(fps=30.0, frame_count=7, width=32, height=32),
    )
    monkeypatch.setattr("pipeline.player_positions.iter_segments", lambda *_args, **_kwargs: iter(segments))
    monkeypatch.setattr("pipeline.player_positions.build_detector", fake_build_detector)
    monkeypatch.setattr("pipeline.player_positions.detections_from_result", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("pipeline.player_positions.detection_rows", lambda rows: [dict(row) for row in rows])
    monkeypatch.setattr("pipeline.player_positions.build_embedder", lambda _backend: FakeEmbedder())
    monkeypatch.setattr("pipeline.player_positions.build_team_embedder", lambda *_args, **_kwargs: FakeTeamEmbedder())
    monkeypatch.setattr(
        "pipeline.player_positions.parse_court_keypoints",
        lambda *_args, **_kwargs: [
            {"label": "01", "x": 0.0, "y": 0.0, "confidence": 0.9},
            {"label": "08", "x": 0.0, "y": 50.0, "confidence": 0.9},
            {"label": "34", "x": 80.0, "y": 0.0, "confidence": 0.9},
            {"label": "41", "x": 80.0, "y": 50.0, "confidence": 0.9},
        ],
    )

    class FakeHomography:
        matrix = "unused"
        inliers_count = 4
        reprojection_mean_m = 0.0

    monkeypatch.setattr("pipeline.player_positions.compute_frame_homography", lambda *_args, **_kwargs: FakeHomography())
    monkeypatch.setattr(
        "pipeline.player_positions.identify_track_teams",
        lambda *_args, **_kwargs: type("FakeTeamResult", (), {"assignments": [], "track_diagnostics": {}})(),
    )
    monkeypatch.setattr(
        "pipeline.player_positions.write_player_position_debug_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("debug video should be skipped for empty outputs")),
    )

    output_dir = tmp_path / "out"
    config = PipelineConfig.from_env(
        roboflow_api_key="test-key",
        court_model_id="court-model/1",
        target_fps=5.0,
        frame_stride=5,
        appearance_weight=0.0,
        reid_backend="none",
        team_embedder_backend="none",
        min_team_samples_per_track=1,
        team_sample_stride=1,
        max_team_samples_per_track=1,
        prefetch_detections=False,
        allowed_classes=("player",),
    )

    artifact = run_player_position_pipeline(tmp_path / "dummy.mp4", config, output_dir, tracker=FakeTracker())

    assert artifact["rows_written"] == 0
    assert artifact["debug_video_file"] is None
    assert artifact["debug_frames"] == 0
    assert (output_dir / "player_positions.jsonl").read_text(encoding="utf-8") == ""
    assert not (output_dir / "player_positions_debug.mp4").exists()


def test_player_positions_main_uses_stricter_default_confidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from pipeline import player_positions

    captured: dict[str, object] = {}

    def fake_from_env(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("pipeline.player_positions.PipelineConfig.from_env", fake_from_env)
    monkeypatch.setattr("pipeline.player_positions.run_player_position_pipeline", lambda *_args, **_kwargs: {})

    exit_code = player_positions.main([str(tmp_path / "clip.mp4")])

    assert exit_code == 0
    assert captured["confidence"] == player_positions.DEFAULT_PLAYER_POSITION_CONFIDENCE


def test_player_positions_main_keeps_cli_confidence_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from pipeline import player_positions

    captured: dict[str, object] = {}

    def fake_from_env(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("pipeline.player_positions.PipelineConfig.from_env", fake_from_env)
    monkeypatch.setattr("pipeline.player_positions.run_player_position_pipeline", lambda *_args, **_kwargs: {})

    exit_code = player_positions.main([str(tmp_path / "clip.mp4"), "--confidence", "0.55"])

    assert exit_code == 0
    assert captured["confidence"] == 0.55
