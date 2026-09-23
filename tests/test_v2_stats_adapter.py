from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pipeline.config import PipelineConfig
from pipeline.runner import run_stats_from_v2_observations


def test_v2_stats_adapter_reuses_rows_without_legacy_player_tracking(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        runner,
        "probe",
        lambda _path: SimpleNamespace(
            fps=30.0, frame_count=2, width=100, height=100, duration_seconds=2 / 30
        ),
    )
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))
    monkeypatch.setattr(
        runner,
        "Sam2SegmentTracker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("SAM2 must not load")),
    )

    observations = [
        {
            "frame_index": frame,
            "raw_track_id": 17,
            "stable_player_id": "player-001",
            "bbox_xyxy": [10, 10, 30, 50],
            "team_id": 1,
            "track_state": "tracked",
            "confidence": 0.95,
        }
        for frame in (0, 1)
    ]
    result = run_stats_from_v2_observations(
        video, PipelineConfig(roboflow_api_key="test-key", team_embedder_backend="none"), tmp_path / "stats", observations
    )

    tracks = [json.loads(line) for line in (tmp_path / "stats" / "tracks.jsonl").read_text().splitlines()]
    assert [(row["stable_player_id"], row["team_id"]) for row in tracks] == [
        ("player-001", 1),
        ("player-001", 1),
    ]
    assert result.ball_tracks_file == "ball_tracks.jsonl"
    assert result.shot_evidence_file == "shot_evidence.jsonl"
    assert result.possession_events_file == "possession_events.jsonl"
    assert (tmp_path / "stats" / "ball_tracks.jsonl").exists()
    assert (tmp_path / "stats" / "shot_evidence.jsonl").exists()
    assert (tmp_path / "stats" / "possession_events.jsonl").exists()


def test_v2_stats_adapter_uses_tracknet_when_enabled(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        runner,
        "probe",
        lambda _path: SimpleNamespace(fps=4.0, frame_count=3, width=100, height=100, duration_seconds=3 / 4),
    )
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))
    builder_calls = []

    def build(checkpoint):
        builder_calls.append(checkpoint)
        return object()

    def collect(_video, *, output_path, **_kwargs):
        output_path.write_text("".join(json.dumps({
            "frame_index": frame,
            "timestamp_seconds": frame / 4,
            "track_id": 1,
            "track_state": "detected",
            "class_name": "ball",
            "confidence": 0.9,
            "xyxy": [10.0, 10.0, 20.0, 20.0],
            "center_xy": [15.0, 15.0],
        }) + "\n" for frame in range(3)))

    monkeypatch.setattr(runner, "build_tracknet_detector", build)
    monkeypatch.setattr(runner, "_collect_tracknet_rows", collect)
    run_stats_from_v2_observations(
        video,
        PipelineConfig(
            roboflow_api_key="test-key",
            team_embedder_backend="none",
            tracknet_enabled=True,
            tracknet_checkpoint="/mounted/TrackNet_best.pt",
        ),
        tmp_path / "stats",
        [
            {
                "frame_index": frame,
                "raw_track_id": 7,
                "stable_player_id": "player-7",
                "team_id": 1,
                "bbox_xyxy": [1, 1, 30, 40],
                "track_state": "tracked",
            }
            for frame in range(3)
        ],
    )

    assert builder_calls == ["/mounted/TrackNet_best.pt"]
    ball = [
        json.loads(line)
        for line in (tmp_path / "stats" / "ball_tracks.jsonl").read_text().splitlines()
    ]
    assert ball and all(row["track_state"] == "detected" for row in ball)
    possession = [
        json.loads(line)
        for line in (tmp_path / "stats" / "possession_events.jsonl").read_text().splitlines()
    ]
    assert possession[-1]["status"] == "confirmed"
    assert possession[-1]["stable_player_id"] == "player-7"
    assert possession[-1]["team_id"] == 1


def test_v2_stats_adapter_requires_checkpoint_when_tracknet_enabled(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        runner,
        "probe",
        lambda _path: SimpleNamespace(fps=30.0, frame_count=1, width=100, height=100, duration_seconds=1 / 30),
    )

    with pytest.raises(RuntimeError, match="TrackNet is enabled but TRACKNET_CHECKPOINT is not configured"):
        run_stats_from_v2_observations(
            video,
            PipelineConfig(roboflow_api_key="test-key", tracknet_enabled=True),
            tmp_path / "stats",
            [{"frame_index": 0, "raw_track_id": 7, "bbox_xyxy": [1, 1, 30, 40], "track_state": "tracked"}],
        )
