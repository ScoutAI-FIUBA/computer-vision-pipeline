from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.runner import _BoundedProgress, run_stats_from_v2_observations


def _court_enabled_config():
    return PipelineConfig(
        roboflow_api_key="test-key", court_model_id="court/1", team_embedder_backend="none",
        number_evidence_enabled=False,
    )


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


def test_v2_stats_v4_emits_diagnostics_without_invoking_v3_scorer(tmp_path, monkeypatch):
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

    def fail_v3_scorer(*_args, **_kwargs):
        raise AssertionError("V3 scorer must not run for engine=v4")

    v4_calls = []
    real_v4_scorer = runner.score_candidates_v4

    def spy_v4_scorer(*args, **kwargs):
        v4_calls.append((args, kwargs))
        return real_v4_scorer(*args, **kwargs)

    monkeypatch.setattr(runner, "score_candidates", fail_v3_scorer)
    monkeypatch.setattr(runner, "score_candidates_v4", spy_v4_scorer)
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
        video,
        PipelineConfig(
            roboflow_api_key="test-key",
            team_embedder_backend="none",
            number_evidence_enabled=False,
            possession_engine="v4",
        ),
        tmp_path / "stats",
        observations,
    )

    stats_dir = tmp_path / "stats"
    manifest = json.loads((stats_dir / "possession_v3_manifest.json").read_text())
    assert manifest["v4"] == {"enabled": True, "schema_version": 1, "feature_flag": "v4",
                               "overlay_fields": [
                                   "holder_control", "holder_release", "challenger_acquisition",
                                   "mask_bbox_agreement", "gate_reason_codes",
                               ]}
    diagnostic = manifest["artifacts"]["v4_diagnostics"]
    diagnostic_path = stats_dir / diagnostic["path"]
    assert diagnostic_path.is_file()
    assert json.loads(diagnostic_path.read_text().splitlines()[0])["v4"]["feature_flag"] is True
    assert result.possession_v4_diagnostics_file == diagnostic["path"]
    assert v4_calls


def test_v2_stats_projects_authoritative_rows_and_shots(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        runner, "probe",
        lambda _path: SimpleNamespace(fps=30.0, frame_count=1, width=100, height=100, duration_seconds=1 / 30),
    )
    monkeypatch.setattr(runner, "_read_video_frames", lambda _path, indices: {0: object()})
    monkeypatch.setattr(runner, "build_detector", lambda _config, **_kwargs: SimpleNamespace(detect=lambda _frame: {}))
    monkeypatch.setattr(
        runner, "parse_court_keypoints",
        lambda *_args, **_kwargs: [{"label": "01", "x": 0, "y": 0, "confidence": 1.0}],
    )
    monkeypatch.setattr(
        runner, "compute_frame_homography",
        lambda *_args, **_kwargs: SimpleNamespace(matrix="H", inliers_count=4, reprojection_mean_m=0.0),
    )
    monkeypatch.setattr(runner, "project_points", lambda _matrix, points: [[4.0, 2.0] for _ in points])
    monkeypatch.setattr(
        runner, "_write_shot_events",
        lambda _video, path, **_kwargs: path.write_text(json.dumps({
            "event_id": 1, "start_frame_index": 0, "start_image_xy": [20, 30], "outcome": "made"
        }) + "\n"),
    )

    result = run_stats_from_v2_observations(
        video, PipelineConfig(
            roboflow_api_key="test-key", court_model_id="court/1", team_embedder_backend="none",
            number_evidence_enabled=False,
        ), tmp_path / "stats", [{
            "frame_index": 0, "raw_track_id": 17, "stable_player_id": "player-001",
            "team_id": 1, "bbox_xyxy": [10, 10, 30, 50], "track_state": "tracked",
        }],
    )

    projected = json.loads((tmp_path / "stats" / "projected_tracks.jsonl").read_text().splitlines()[0])
    assert projected["track_id"] == 17
    assert projected["stable_player_id"] == "player-001"
    assert projected["team_id"] == 1
    assert projected["court_xy_m"] == [4.0, 2.0]
    assert result.projected_tracks_file == "projected_tracks.jsonl"
    assert result.projected_shots_file == "projected_shots.json"
    assert result.projected_shots_image_file == "projected_shots.png"


def test_v2_stats_omits_projection_when_court_evidence_is_invalid(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        runner, "probe",
        lambda _path: SimpleNamespace(fps=30.0, frame_count=1, width=100, height=100, duration_seconds=1 / 30),
    )
    monkeypatch.setattr(runner, "_read_video_frames", lambda _path, indices: {0: object()})
    monkeypatch.setattr(runner, "build_detector", lambda _config, **_kwargs: SimpleNamespace(detect=lambda _frame: {}))
    monkeypatch.setattr(runner, "parse_court_keypoints", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "compute_frame_homography", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("insufficient_points")))
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))

    result = run_stats_from_v2_observations(
        video, PipelineConfig(
            roboflow_api_key="test-key", court_model_id="court/1", team_embedder_backend="none",
            number_evidence_enabled=False,
        ), tmp_path / "stats", [{
            "frame_index": 0, "raw_track_id": 17, "stable_player_id": "player-001",
            "bbox_xyxy": [10, 10, 30, 50], "track_state": "tracked",
        }],
    )

    assert result.projected_tracks_file is None
    assert result.projected_shots_file is None
    assert result.projected_shots_image_file is None
    payload = json.loads((tmp_path / "stats" / "results.json").read_text())
    assert "projected_tracks_file" not in payload


def test_v2_stats_uses_real_court_geometry_helpers(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(runner, "probe", lambda _path: SimpleNamespace(
        fps=30.0, frame_count=1, width=1000, height=600, duration_seconds=1 / 30
    ))
    monkeypatch.setattr(runner, "_read_video_frames", lambda _path, _indices: {0: object()})

    court_result = {"predictions": [
        {"class": "01", "x": 100, "y": 100, "confidence": 1.0},
        {"class": "08", "x": 100, "y": 500, "confidence": 1.0},
        {"class": "34", "x": 900, "y": 100, "confidence": 1.0},
        {"class": "41", "x": 900, "y": 500, "confidence": 1.0},
    ]}
    monkeypatch.setattr(runner, "build_detector", lambda _config, **_kwargs: SimpleNamespace(
        detect=lambda _frame: court_result
    ))
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))

    result = run_stats_from_v2_observations(
        video, _court_enabled_config(), tmp_path / "stats", [{
            "frame_index": 0, "raw_track_id": 17, "stable_player_id": "player-001",
            "team_id": 1, "bbox_xyxy": [480, 260, 520, 300], "track_state": "tracked",
        }],
    )

    projected = json.loads((tmp_path / "stats" / "projected_tracks.jsonl").read_text().splitlines()[0])
    assert projected["court_xy_m"] == pytest.approx([14.0, 7.5], abs=0.05)
    assert result.projected_tracks_file == "projected_tracks.jsonl"


@pytest.mark.parametrize("case", ["missing_frame", "detector_failure", "invalid_homography"])
def test_v2_projection_omits_all_artifacts_without_valid_court_evidence(tmp_path, monkeypatch, case):
    import pipeline.runner as runner

    rows = [{"frame_index": 0, "track_id": 1, "xyxy": [10, 10, 30, 50]}]
    output = tmp_path / "projected_tracks.jsonl"
    if case == "missing_frame":
        monkeypatch.setattr(runner, "_read_video_frames", lambda _path, _indices: {})
    else:
        monkeypatch.setattr(runner, "_read_video_frames", lambda _path, _indices: {0: object()})
    if case == "detector_failure":
        detector = SimpleNamespace(detect=lambda _frame: (_ for _ in ()).throw(RuntimeError("offline")))
    else:
        detector = SimpleNamespace(detect=lambda _frame: {})
    monkeypatch.setattr(runner, "_build_detector_for_model", lambda *_args, **_kwargs: detector)
    if case == "invalid_homography":
        monkeypatch.setattr(runner, "parse_court_keypoints", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(runner, "compute_frame_homography", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad")))

    written = runner._project_v2_rows(
        tmp_path / "input.mp4", output, rows,
        info=SimpleNamespace(width=100, height=100), config=_court_enabled_config(), emit=lambda _message: None,
    )

    assert written == 0
    assert not output.exists()


def test_v2_projection_keeps_only_valid_frames_when_homography_is_mixed(tmp_path, monkeypatch):
    import pipeline.runner as runner

    rows = [
        {"frame_index": frame, "track_id": frame + 1, "xyxy": [10, 10, 30, 50]}
        for frame in range(3)
    ]
    monkeypatch.setattr(runner, "_read_video_frames", lambda _path, _indices: {0: 0, 1: 1, 2: 2})
    detector = SimpleNamespace(detect=lambda frame: frame != 1)
    monkeypatch.setattr(runner, "_build_detector_for_model", lambda *_args, **_kwargs: detector)
    monkeypatch.setattr(runner, "parse_court_keypoints", lambda result, **_kwargs: result)
    monkeypatch.setattr(
        runner, "compute_frame_homography",
        lambda keypoints, **_kwargs: (
            (_ for _ in ()).throw(ValueError("bad"))
            if not keypoints else SimpleNamespace(matrix="H", inliers_count=4, reprojection_mean_m=0.0)
        ),
    )
    monkeypatch.setattr(runner, "project_points", lambda _matrix, points: [[1.0, 2.0] for _ in points])

    output = tmp_path / "projected_tracks.jsonl"
    written = runner._project_v2_rows(
        tmp_path / "input.mp4", output, rows,
        info=SimpleNamespace(width=100, height=100), config=_court_enabled_config(), emit=lambda _message: None,
    )

    projected = [json.loads(line) for line in output.read_text().splitlines()]
    assert written == 2
    assert [row["frame_index"] for row in projected] == [0, 2]


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


def test_v2_stats_normalization_preserves_mask_provenance():
    from pipeline.runner import _normalize_v2_stats_rows

    rows = _normalize_v2_stats_rows([{
        "frame_index": 0, "raw_track_id": 3, "stable_player_id": "p-3",
        "bbox_xyxy": [1, 2, 11, 22], "observation_key": "0:3",
        "mask_ref": "masks/0_3.png", "clean_mask_ref": "masks_clean/0_3.png",
    }], width=20, height=30)
    assert rows[0]["observation_key"] == "0:3"
    assert rows[0]["mask_ref"] == "masks/0_3.png"
    assert rows[0]["clean_mask_ref"] == "masks_clean/0_3.png"


def test_v2_stats_v3_uses_clean_mask_for_correct_segmented_id(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(runner, "probe", lambda _path: SimpleNamespace(
        fps=30.0, frame_count=1, width=40, height=40, duration_seconds=1 / 30,
    ))
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))
    monkeypatch.setattr(runner, "build_tracknet_detector", lambda _checkpoint: object())
    config = PipelineConfig(roboflow_api_key="test-key", team_embedder_backend="none", possession_engine="v3",
                            tracknet_enabled=True, tracknet_checkpoint="test.pt")
    observations = [
        {"frame_index": 0, "raw_track_id": 1, "stable_player_id": "wrong",
         "observation_key": "0:1", "bbox_xyxy": [2, 2, 30, 30], "team_id": 1},
        {"frame_index": 0, "raw_track_id": 2, "stable_player_id": "holder",
         "observation_key": "0:2", "bbox_xyxy": [2, 2, 30, 30], "team_id": 1},
    ]
    clean = {}
    for key, x in (("0:1", 5), ("0:2", 15)):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:25, x:x + 5] = True
        clean[key] = mask
    monkeypatch.setattr(runner, "_collect_tracknet_rows", lambda *args, **kwargs: (
        kwargs["output_path"].write_text(json.dumps({
            "frame_index": 0, "track_id": 1, "track_state": "detected",
            "class_name": "ball", "confidence": .9, "center_xy": [17, 17],
        }) + "\n") or 1
    ))
    result = run_stats_from_v2_observations(
        video, config, tmp_path / "stats", observations,
        clean_masks_by_observation=clean,
    )
    rows = [json.loads(line) for line in (tmp_path / "stats" / result.possession_v3_file).read_text().splitlines()]
    assert rows[0]["context"]["observed_candidate_id"] == "holder"


@pytest.mark.parametrize("clean_kind", ["object", "nan", "inf"])
def test_v2_stats_v3_falls_back_to_raw_mask_when_clean_mask_is_invalid(tmp_path, monkeypatch, clean_kind):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(runner, "probe", lambda _path: SimpleNamespace(
        fps=30.0, frame_count=1, width=40, height=40, duration_seconds=1 / 30,
    ))
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))
    monkeypatch.setattr(runner, "build_tracknet_detector", lambda _checkpoint: object())
    monkeypatch.setattr(runner, "_collect_tracknet_rows", lambda *args, **kwargs: (
        kwargs["output_path"].write_text(json.dumps({
            "frame_index": 0, "track_id": 1, "track_state": "detected",
            "class_name": "ball", "confidence": .9, "center_xy": [17, 17],
        }) + "\n") or 1
    ))
    config = PipelineConfig(roboflow_api_key="test-key", team_embedder_backend="none", possession_engine="v3",
                            tracknet_enabled=True, tracknet_checkpoint="test.pt")
    observations = [{
        "frame_index": 0, "raw_track_id": 1, "stable_player_id": "holder",
        "observation_key": "0:1", "bbox_xyxy": [2, 2, 30, 30], "team_id": 1,
    }]
    raw = np.zeros((40, 40), dtype=bool)
    raw[10:25, 15:20] = True
    clean = np.zeros((40, 40), dtype=object if clean_kind == "object" else float)
    clean[0, 0] = "mask" if clean_kind == "object" else (np.nan if clean_kind == "nan" else np.inf)
    result = run_stats_from_v2_observations(
        video, config, tmp_path / "stats", observations,
        masks_by_observation={"0:1": raw},
        clean_masks_by_observation={"0:1": clean},
    )
    rows = [json.loads(line) for line in (tmp_path / "stats" / result.possession_v3_file).read_text().splitlines()]
    assert rows[0]["context"]["observed_candidate_id"] == "holder"


def test_v2_stats_v3_falls_back_to_raw_mask_when_clean_shape_mismatches_frame(tmp_path, monkeypatch):
    import pipeline.runner as runner

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(runner, "probe", lambda _path: SimpleNamespace(
        fps=30.0, frame_count=1, width=40, height=40, duration_seconds=1 / 30,
    ))
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))
    monkeypatch.setattr(runner, "build_tracknet_detector", lambda _checkpoint: object())
    monkeypatch.setattr(runner, "_collect_tracknet_rows", lambda *args, **kwargs: (
        kwargs["output_path"].write_text(json.dumps({
            "frame_index": 0, "track_id": 1, "track_state": "detected",
            "class_name": "ball", "confidence": .9, "center_xy": [17, 17],
        }) + "\n") or 1
    ))
    config = PipelineConfig(roboflow_api_key="test-key", team_embedder_backend="none", possession_engine="v3",
                            tracknet_enabled=True, tracknet_checkpoint="test.pt")
    observations = [{
        "frame_index": 0, "raw_track_id": 1, "stable_player_id": "holder",
        "observation_key": "0:1", "bbox_xyxy": [2, 2, 30, 30], "team_id": 1,
    }]
    raw = np.zeros((40, 40), dtype=bool)
    raw[10:25, 15:20] = True
    clean = np.zeros((20, 20), dtype=bool)
    clean[0:2, 0:2] = True
    result = run_stats_from_v2_observations(
        video, config, tmp_path / "stats", observations,
        masks_by_observation={"0:1": raw},
        clean_masks_by_observation={"0:1": clean},
    )
    rows = [json.loads(line) for line in (tmp_path / "stats" / result.possession_v3_file).read_text().splitlines()]
    assert rows[0]["context"]["observed_candidate_id"] == "holder"


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


def test_v2_number_sampling_uses_stable_ids_and_preserves_conflicts(tmp_path, monkeypatch):
    import sys
    import types
    import pipeline.v2_numbers as numbers
    from pipeline.number_predictor import NumberPrediction

    class Capture:
        def __init__(self, _path): self.frame = None
        def set(self, _property, frame): self.frame = frame
        def read(self): return True, np.ones((40, 40, 3), dtype=np.uint8) * int(self.frame or 0)
        def release(self): pass

    fake_cv2 = types.SimpleNamespace(VideoCapture=Capture, CAP_PROP_POS_FRAMES=1)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    class Predictor:
        name = "fake"
        def __init__(self): self.frames = []
        def predict(self, sample):
            self.frames.append(sample.frame_index)
            return NumberPrediction("12" if sample.frame_index in {0, 2} else "13", 0.99)

    predictor = Predictor()
    monkeypatch.setattr(numbers, "build_number_predictor", lambda _config: (_ for _ in ()).throw(
        AssertionError("injected predictor must not be reconstructed")
    ))
    rows = [{"frame_index": frame, "track_id": frame + 10, "stable_player_id": "stable-a",
             "xyxy": [1, 1, 10, 10], "track_state": "tracked"} for frame in range(4)]
    config = PipelineConfig(roboflow_api_key="key", frame_stride=2, team_embedder_backend="none")
    assert numbers.run_v2_number_evidence(tmp_path / "video.mp4", rows, config, tmp_path, predictor=predictor)
    assert predictor.frames == [0, 2]
    payload = json.loads((tmp_path / "numbers.json").read_text())
    player = payload["players"][0]
    assert player["stable_player_id"] == "stable-a"
    assert player["status"] == "confirmed"
    assert player["number"] == "12"
    assert player["conflicts"] == []


def test_v2_numbers_reuses_object_detector_number_boxes_and_samples_only_matching_frames(tmp_path, monkeypatch):
    import sys
    import types
    import pipeline.v2_numbers as numbers
    from pipeline.number_predictor import NumberPrediction

    class Capture:
        def __init__(self, _path): self.frame = None
        def set(self, _property, frame): self.frame = frame
        def read(self): return True, np.full((40, 60, 3), int(self.frame or 0), dtype=np.uint8)
        def release(self): pass

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        VideoCapture=Capture, CAP_PROP_POS_FRAMES=1,
    ))

    class Detector:
        def __init__(self): self.frames = []
        def detect(self, frame):
            self.frames.append(int(frame[0, 0, 0]))
            return {"predictions": [
                {"class": "number", "x": 15, "y": 15, "width": 10, "height": 8, "confidence": 0.91},
                {"class": "player", "x": 45, "y": 15, "width": 10, "height": 8, "confidence": 0.99},
            ]}

    class Predictor:
        name = "fake"
        def __init__(self): self.samples = []
        def predict(self, sample):
            self.samples.append(sample)
            return NumberPrediction("23", 0.99)

    detector, predictor = Detector(), Predictor()
    rows = [
        {"frame_index": frame, "track_id": 10, "stable_player_id": "stable-a",
         "xyxy": [5, 5, 25, 35], "track_state": "tracked"}
        for frame in range(4)
    ]
    config = PipelineConfig(roboflow_api_key="key", frame_stride=2, team_embedder_backend="none")
    assert numbers.run_v2_number_evidence(
        tmp_path / "video.mp4", rows, config, tmp_path, predictor=predictor, detector=detector
    )

    assert detector.frames == [0, 2]
    assert [sample.frame_index for sample in predictor.samples] == [0, 2]
    assert predictor.samples[0].number_xyxy == (10.0, 11.0, 20.0, 19.0)
    evidence = json.loads((tmp_path / "number_predictions.json").read_text())["predictions"][0]["evidence"]
    assert evidence["source"] == "legacy_basketball_object_detector"
    assert evidence["association_method"] == "object_detector_number_box"
    assert evidence["overlap_ratio"] == 1.0
    assert evidence["player_overlap_ratio"] < 1.0


def test_v2_numbers_with_detector_without_number_boxes_has_no_evidence(tmp_path, monkeypatch):
    import sys
    import types
    import pipeline.v2_numbers as numbers

    class Capture:
        def __init__(self, _path): pass
        def set(self, _property, _frame): pass
        def read(self): return True, np.zeros((20, 20, 3), dtype=np.uint8)
        def release(self): pass

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        VideoCapture=Capture, CAP_PROP_POS_FRAMES=1,
    ))

    class Detector:
        def detect(self, _frame): return {"predictions": []}

    class Predictor:
        def predict(self, _sample): raise AssertionError("OCR must not run without a number box")

    config = PipelineConfig(roboflow_api_key="key", frame_stride=1, team_embedder_backend="none")
    row = {"frame_index": 0, "track_id": 1, "stable_player_id": "p-1", "xyxy": [1, 1, 10, 18]}
    assert not numbers.run_v2_number_evidence(
        tmp_path / "video.mp4", [row], config, tmp_path, predictor=Predictor(), detector=Detector()
    )
    assert all(not (tmp_path / name).exists() for name in numbers._NUMBER_ARTIFACTS)


def test_v2_number_aggregation_marks_ambiguous_and_unknown():
    from pipeline.v2_numbers import _aggregate

    records = [
        {"stable_player_id": "a", "accepted": True, "tentative_number": "7", "frame_index": 0},
        {"stable_player_id": "a", "accepted": True, "tentative_number": "8", "frame_index": 2},
        {"stable_player_id": "b", "accepted": False, "tentative_number": "unknown", "frame_index": 0},
    ]
    result = {row["stable_player_id"]: row for row in _aggregate(records)}
    assert result["a"]["status"] == "ambiguous"
    assert result["a"]["number"] == "unknown"
    assert result["b"]["status"] == "unknown"


def test_v2_number_evidence_uses_masked_torso_and_cleans_stale_outputs(tmp_path, monkeypatch):
    import sys
    import types
    import pipeline.v2_numbers as numbers
    from pipeline.number_predictor import NumberPrediction

    class Capture:
        def __init__(self, _path): pass
        def set(self, _property, _frame): pass
        def read(self): return True, np.ones((40, 40, 3), dtype=np.uint8)
        def release(self): pass

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        VideoCapture=Capture, CAP_PROP_POS_FRAMES=1,
    ))

    seen = []
    class Predictor:
        name = "fake"
        def predict(self, sample):
            seen.append(sample)
            return NumberPrediction("7", 0.99)

    config = PipelineConfig(roboflow_api_key="key", frame_stride=1, team_embedder_backend="none")
    row = {"frame_index": 0, "track_id": 3, "stable_player_id": "p-3",
           "observation_key": "0:3", "xyxy": [2, 2, 30, 38], "track_state": "tracked"}
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:25, 8:24] = 1
    assert numbers.run_v2_number_evidence(
        tmp_path / "video.mp4", [row], config, tmp_path, predictor=Predictor(),
        masks_by_observation={"0:3": mask},
    )
    assert seen[0].number_xyxy != seen[0].player_xyxy
    assert seen[0].overlap_ratio < 1.0
    evidence = json.loads((tmp_path / "number_predictions.json").read_text())["predictions"][0]["evidence"]
    assert evidence["association_method"] == "masked_torso_roi"
    assert evidence["mask_kind"] == "raw_mask"

    for name in ("numbers.json", "number_predictions.json", "number_prediction_cases.jsonl", "number_manifest.json"):
        assert (tmp_path / name).exists()
    assert not numbers.run_v2_number_evidence(
        tmp_path / "video.mp4", [{"frame_index": 1, "track_id": 3, "xyxy": [1, 1, 2, 2]}],
        config, tmp_path, predictor=Predictor(),
    )
    assert all(not (tmp_path / name).exists() for name in numbers._NUMBER_ARTIFACTS)


def test_run_stats_v2_writes_number_artifacts_and_hashed_references(tmp_path, monkeypatch):
    import sys
    import types
    import hashlib
    import pipeline.runner as runner
    from pipeline.number_predictor import NumberPrediction

    class Capture:
        def __init__(self, _path): pass
        def set(self, _property, _frame): pass
        def read(self): return True, np.ones((40, 40, 3), dtype=np.uint8)
        def release(self): pass

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        VideoCapture=Capture, CAP_PROP_POS_FRAMES=1,
    ))
    monkeypatch.setattr(runner, "probe", lambda _path: SimpleNamespace(
        fps=30.0, frame_count=2, width=40, height=40, duration_seconds=2 / 30,
    ))
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))

    class Predictor:
        name = "fake"
        def predict(self, _sample): return NumberPrediction("11", 0.99)

    output = tmp_path / "stats"
    result = run_stats_from_v2_observations(
        tmp_path / "input.mp4",
        PipelineConfig(roboflow_api_key="key", frame_stride=1, team_embedder_backend="none"),
        output,
        [{"frame_index": frame, "raw_track_id": frame + 1, "stable_player_id": "stable-1",
          "observation_key": f"{frame}:{frame + 1}", "mask_ref": f"masks/{frame}.png",
          "clean_mask_ref": f"masks_clean/{frame}.png", "bbox_xyxy": [2, 2, 30, 38],
          "track_state": "tracked"} for frame in (0, 1)],
        number_predictor=Predictor(),
        masks_by_observation={
            f"{frame}:{frame + 1}": np.pad(np.ones((15, 16), dtype=np.uint8), ((10, 15), (8, 16)))
            for frame in (0, 1)
        },
    )
    assert result.numbers_file == "numbers.json"
    references = result.artifact_references["numbers"]["artifacts"]
    for name, reference in references.items():
        data = (output / name).read_bytes()
        assert reference["size"] == len(data)
        assert reference["sha256"] == hashlib.sha256(data).hexdigest()
    prediction = json.loads((output / "number_predictions.json").read_text())["predictions"][0]
    assert prediction["evidence"]["association_method"] == "masked_torso_roi"


def test_run_stats_v2_disabled_numbers_skip_initialization_and_remove_reused_artifacts(tmp_path, monkeypatch):
    import pipeline.runner as runner

    output = tmp_path / "stats"
    output.mkdir()
    for name in ("numbers.json", "number_predictions.json", "number_prediction_cases.jsonl", "number_manifest.json"):
        (output / name).write_text("stale", encoding="utf-8")
    monkeypatch.setattr(runner, "probe", lambda _path: SimpleNamespace(
        fps=30.0, frame_count=1, width=40, height=40, duration_seconds=1 / 30,
    ))
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))

    class MustNotRun:
        def predict(self, _sample): raise AssertionError("number predictor must stay disabled")
        def detect(self, _frame): raise AssertionError("number detector must stay disabled")

    config = PipelineConfig(
        roboflow_api_key="key", team_embedder_backend="none", number_evidence_enabled=False,
    )
    result = run_stats_from_v2_observations(
        tmp_path / "input.mp4", config, output,
        [{"frame_index": 0, "raw_track_id": 1, "stable_player_id": "stable-1",
          "bbox_xyxy": [2, 2, 30, 38], "track_state": "tracked"}],
        number_predictor=MustNotRun(), number_detector=MustNotRun(),
    )
    assert result.numbers_file is None
    assert all(not (output / name).exists() for name in (
        "numbers.json", "number_predictions.json", "number_prediction_cases.jsonl", "number_manifest.json",
    ))
