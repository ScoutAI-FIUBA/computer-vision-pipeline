from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.runner import _BoundedProgress, run_stats_from_v2_observations


def test_v2_stats_adapter_reuses_rows_without_legacy_player_tracking(tmp_path, monkeypatch, capsys):
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
    assert result.possession_transitions_file == "possession_transitions.jsonl"
    assert (tmp_path / "stats" / "ball_tracks.jsonl").exists()
    assert (tmp_path / "stats" / "shot_evidence.jsonl").exists()
    assert (tmp_path / "stats" / "possession_events.jsonl").exists()
    assert (tmp_path / "stats" / "possession_transitions.jsonl").exists()
    output = capsys.readouterr().out
    assert "[v2-stats] stats start" in output
    assert "[v2-stats] shot evidence start" in output
    assert "[v2-stats] possession start" in output
    assert "[v2-stats] artifact writing start" in output
    assert "[v2-stats] stats complete" in output


def test_v2_stats_adapter_serializes_scoped_team_conflicts_truthfully(tmp_path, monkeypatch):
    import pipeline.runner as runner
    from pipeline.results import package_artifacts

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        runner,
        "probe",
        lambda _path: SimpleNamespace(
            fps=30.0, frame_count=7, width=100, height=100, duration_seconds=7 / 30
        ),
    )
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))

    observations = [
        {"frame_index": 0, "raw_track_id": 1, "stable_player_id": "player-1", "team_id": 0,
         "bbox_xyxy": [10, 10, 30, 50], "track_state": "tracked"},
        {"frame_index": 1, "raw_track_id": 2, "stable_player_id": "player-1", "team_id": 0,
         "bbox_xyxy": [10, 10, 30, 50], "track_state": "tracked"},
        {"frame_index": 2, "raw_track_id": 3, "stable_player_id": "player-2", "team_id": None,
         "team_assignment": {"status": "contested", "reason": "conflicting_team_assignment",
                              "provenance": {"stable_player_id": "player-2"}},
         "bbox_xyxy": [10, 10, 30, 50], "track_state": "tracked"},
        {"frame_index": 3, "raw_track_id": 4, "stable_player_id": "player-2", "team_id": None,
         "team_assignment": {"status": "contested", "reason": "conflicting_team_assignment",
                              "provenance": {"stable_player_id": "player-2"}},
         "bbox_xyxy": [10, 10, 30, 50], "track_state": "tracked"},
        {"frame_index": 4, "raw_track_id": 5, "stable_player_id": "player-3", "team_id": 1,
         "bbox_xyxy": [10, 10, 30, 50], "track_state": "tracked"},
        {"frame_index": 5, "raw_track_id": 6, "stable_player_id": "player-4", "team_id": None,
         "team_assignment": {"status": "contested", "reason": "conflicting_team_assignment",
                              "provenance": {"stable_player_id": "player-4"}},
         "bbox_xyxy": [10, 10, 30, 50], "track_state": "tracked"},
        {"frame_index": 6, "raw_track_id": 7, "stable_player_id": "player-4", "team_id": None,
         "team_assignment": {"status": "contested", "reason": "conflicting_team_assignment",
                              "provenance": {"stable_player_id": "player-4"}},
         "bbox_xyxy": [10, 10, 30, 50], "track_state": "tracked"},
    ]
    acceptance_reasons = [
        "conflicting_team_assignment:stable_player_id=player-2",
        "conflicting_team_assignment:stable_player_id=player-4",
    ]
    for row in observations:
        row["observation_key"] = f"{row['frame_index']}:{row['raw_track_id']}"
    package_dir = tmp_path / "package"
    package_artifacts(
        package_dir, observations, {f"{row['frame_index']}:{row['raw_track_id']}":
                                    np.ones((2, 2), dtype=np.uint8) for row in observations},
        acceptance={"tracking_identity": "accepted", "statistics": "deferred"},
        acceptance_reasons=acceptance_reasons,
    )
    result = run_stats_from_v2_observations(
        video, PipelineConfig(roboflow_api_key="test-key", team_embedder_backend="none"),
        tmp_path / "stats", observations,
    )

    teams = json.loads((tmp_path / "stats" / result.teams_file).read_text())
    manifest = json.loads((package_dir / "manifest.json").read_text())
    tracks = [json.loads(line) for line in (package_dir / "tracks.jsonl").read_text().splitlines()]
    assert teams["assignments_by_track_id"] == {"1": 0, "2": 0, "5": 1}
    assert teams["unassigned_track_ids"] == [3, 4, 6, 7]
    assert manifest["acceptance"]["tracking_identity"] == "accepted"
    assert manifest["acceptance_reasons"] == acceptance_reasons
    assert {row["stable_player_id"] for row in tracks if row.get("team_id") is None} == {"player-2", "player-4"}


def test_v2_stats_adapter_uses_tracknet_when_enabled(tmp_path, monkeypatch, capsys):
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

    def collect(_video, *, output_path, on_progress, **_kwargs):
        on_progress(0, 3)
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
        on_progress(3, 3)

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
    output = capsys.readouterr().out
    assert "[v2-stats] TrackNet start" in output
    assert "[v2-stats] TrackNet progress: 0/3 frames (0%)" in output
    assert "[v2-stats] TrackNet progress: 3/3 frames (100%)" in output
    assert "[v2-stats] stats complete" in output


def test_v2_stats_adapter_v3_uses_actual_tracknet_and_player_shapes(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        runner, "probe",
        lambda _path: SimpleNamespace(fps=4.0, frame_count=3, width=100, height=100, duration_seconds=3 / 4),
    )
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))

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

    monkeypatch.setattr(runner, "build_tracknet_detector", lambda _checkpoint: object())
    monkeypatch.setattr(runner, "_collect_tracknet_rows", collect)
    run_stats_from_v2_observations(
        video,
        PipelineConfig(
            roboflow_api_key="test-key", team_embedder_backend="none",
            tracknet_enabled=True, tracknet_checkpoint="checkpoint.pt", possession_engine="v3",
        ),
        tmp_path / "stats",
        [{
            "frame_index": frame, "raw_track_id": 7, "stable_player_id": "player-7",
            "team_id": 1, "xyxy": [10.0, 10.0, 30.0, 40.0], "track_state": "tracked",
            "confidence": None,
        } for frame in range(3)],
    )

    observations = [
        json.loads(line)
        for line in (tmp_path / "stats" / "possession_v3" / "generations").glob("*/possession_v3.jsonl").__next__().read_text().splitlines()
    ]
    assert observations[-1]["candidates"][0]["score"] > 0.5
    assert observations[-1]["state"] == "OBSERVED"
    assert observations[-1]["stable_player_id"] == "player-7"
    legacy = [json.loads(line) for line in (tmp_path / "stats" / "possession_events.jsonl").read_text().splitlines()]
    assert legacy[-1]["status"] == "confirmed"


def test_v2_stats_adapter_writes_transition_artifact_in_same_run(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        runner, "probe", lambda _path: SimpleNamespace(fps=30.0, frame_count=2, width=100, height=100, duration_seconds=2 / 30)
    )
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))

    def write_possession(output_dir, *_args, **_kwargs):
        path = output_dir / "possession_events.jsonl"
        path.write_text("".join(json.dumps({
            "schema_version": 1, "frame_index": frame, "status": "confirmed",
            "player_track_id": player, "stable_player_id": player, "team_id": team,
            "evidence_frame_indices": [frame],
            "source_records": [f"possession_events.jsonl#frame={frame}"],
        }) + "\n" for frame, player, team in ((0, "a", 1), (1, "b", 2))))
        return path

    monkeypatch.setattr(runner, "_write_possession_events", write_possession)
    run_stats_from_v2_observations(
        video, PipelineConfig(roboflow_api_key="test-key", team_embedder_backend="none"), tmp_path / "stats",
        [{"frame_index": frame, "raw_track_id": 1, "stable_player_id": "p", "team_id": 1,
          "bbox_xyxy": [10, 10, 30, 50], "track_state": "tracked"} for frame in (0, 1)],
    )

    transitions = [json.loads(line) for line in (tmp_path / "stats" / "possession_transitions.jsonl").read_text().splitlines()]
    assert transitions[0]["event_type"] == "steal"


def test_v2_stats_adapter_converts_normalized_boxes_for_pixel_ball_geometry(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        runner,
        "probe",
        lambda _path: SimpleNamespace(fps=4.0, frame_count=3, width=100, height=200, duration_seconds=3 / 4),
    )
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))

    def collect(_video, *, output_path, **_kwargs):
        output_path.write_text("".join(json.dumps({
            "frame_index": frame,
            "timestamp_seconds": frame / 4,
            "track_id": 1,
            "track_state": "detected",
            "class_name": "ball",
            "confidence": 0.9,
            "xyxy": [19.0, 39.0, 21.0, 41.0],
            "center_xy": [20.0, 40.0],
        }) + "\n" for frame in range(3)))

    monkeypatch.setattr(runner, "build_tracknet_detector", lambda _checkpoint: object())
    monkeypatch.setattr(runner, "_collect_tracknet_rows", collect)
    observations = [
        {
            "frame_index": frame,
            "raw_track_id": 7,
            "stable_player_id": "player-7",
            "team_id": 1,
            "bbox_xyxy": [0.1, 0.1, 0.3, 0.3],
            "bbox_coordinate_space": "normalized",
            "track_state": "tracked",
        }
        for frame in range(3)
    ]

    run_stats_from_v2_observations(
        video,
        PipelineConfig(
            roboflow_api_key="test-key",
            team_embedder_backend="none",
            tracknet_enabled=True,
            tracknet_checkpoint="checkpoint.pt",
        ),
        tmp_path / "stats",
        observations,
    )

    tracks = [json.loads(line) for line in (tmp_path / "stats" / "tracks.jsonl").read_text().splitlines()]
    assert [track["xyxy"] for track in tracks] == [
        [10.0, 20.0, 30.0, 60.0],
        [10.0, 20.0, 30.0, 60.0],
        [10.0, 20.0, 30.0, 60.0],
    ]
    assert [(track["stable_player_id"], track["team_id"]) for track in tracks] == [
        ("player-7", 1),
        ("player-7", 1),
        ("player-7", 1),
    ]
    assert observations[0]["bbox_xyxy"] == [0.1, 0.1, 0.3, 0.3]
    possession = [
        json.loads(line)
        for line in (tmp_path / "stats" / "possession_events.jsonl").read_text().splitlines()
    ]
    assert possession[-1]["status"] == "confirmed"
    assert possession[-1]["stable_player_id"] == "player-7"


def test_v2_stats_adapter_keeps_pixel_boxes_and_rejects_invalid_normalized_boxes(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        runner,
        "probe",
        lambda _path: SimpleNamespace(fps=30.0, frame_count=1, width=100, height=200, duration_seconds=1 / 30),
    )
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))

    run_stats_from_v2_observations(
        video,
        PipelineConfig(roboflow_api_key="test-key", team_embedder_backend="none"),
        tmp_path / "stats",
        [
            {"frame_index": 0, "raw_track_id": 1, "bbox_xyxy": [1, 2, 11, 22], "bbox_coordinate_space": "pixel"},
            {"frame_index": 0, "raw_track_id": 2, "bbox_xyxy": [float("nan"), 0.1, 0.2, 0.3], "bbox_coordinate_space": "normalized"},
            {"frame_index": 0, "raw_track_id": 3, "bbox_xyxy": [-0.1, 0.1, 0.2, 0.3], "bbox_coordinate_space": "normalized"},
            {"frame_index": 0, "raw_track_id": 4, "bbox_xyxy": "1234", "bbox_coordinate_space": "pixel"},
            {"frame_index": 0, "raw_track_id": 5, "bbox_xyxy": [3, 4, 13, 24]},
            {"frame_index": 0, "raw_track_id": 6, "bbox_xyxy": [3, 4, 13, 24], "bbox_coordinate_space": "world"},
            {"frame_index": 0, "raw_track_id": 7, "bbox_xyxy": [3, 4, 13, 24], "bbox_coordinate_space": "normalised"},
        ],
    )

    tracks = [json.loads(line) for line in (tmp_path / "stats" / "tracks.jsonl").read_text().splitlines()]
    assert [(row["track_id"], row["xyxy"]) for row in tracks] == [
        (1, [1.0, 2.0, 11.0, 22.0]),
        (5, [3.0, 4.0, 13.0, 24.0]),
    ]


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


def test_v2_stats_adapter_reports_failing_phase(tmp_path, monkeypatch, capsys):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        runner,
        "probe",
        lambda _path: SimpleNamespace(
            fps=30.0, frame_count=1, width=100, height=100, duration_seconds=1 / 30
        ),
    )
    monkeypatch.setattr(
        runner,
        "_write_shot_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("shot model unavailable")),
    )

    with pytest.raises(RuntimeError, match="shot model unavailable"):
        run_stats_from_v2_observations(
            video,
            PipelineConfig(roboflow_api_key="test-key", team_embedder_backend="none"),
            tmp_path / "stats",
            [{"frame_index": 0, "raw_track_id": 7, "bbox_xyxy": [1, 1, 30, 40]}],
        )

    assert "[v2-stats] shot detection failed: RuntimeError: shot model unavailable" in capsys.readouterr().out


def test_bounded_progress_marks_capped_phase_complete_without_fake_source_percent(capsys):
    progress = _BoundedProgress(lambda message: print(message), "TrackNet")

    progress(0, 10)
    progress(3, 10)
    progress.complete(10)

    output = capsys.readouterr().out
    assert "TrackNet progress: phase complete (100%; processed 3 frames; source progress 3/10)" in output
    assert "3/10 frames (100%)" not in output
