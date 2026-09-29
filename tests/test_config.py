"""Pipeline config plumbing tests.

    python -m pytest tests/test_config.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.court_projection import DEFAULT_COURT_MODEL_ID  # noqa: E402


def test_from_env_reads_team_identification_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("TEAM_EMBEDDER_BACKEND", "colour")
    monkeypatch.setenv("TEAM_EMBEDDER_MODEL_ID", "custom-model")
    monkeypatch.setenv("TEAM_COUNT", "3")
    monkeypatch.setenv("MIN_TEAM_SAMPLES_PER_TRACK", "5")
    monkeypatch.setenv("TEAM_SAMPLE_STRIDE", "4")
    monkeypatch.setenv("MIN_TEAM_SAMPLE_QUALITY", "0.35")
    monkeypatch.setenv("MAX_TEAM_SAMPLES_PER_TRACK", "9")
    monkeypatch.setenv("TEAM_ASSIGNMENT_MARGIN", "0.07")
    monkeypatch.setenv("TEAM_DEBUG_ENABLED", "0")
    monkeypatch.setenv("NUMBER_PREDICTOR_BACKEND", "dummy")

    config = PipelineConfig.from_env()

    assert config.team_embedder_backend == "colour"
    assert config.team_embedder_model_id == "custom-model"
    assert config.team_count == 3
    assert config.min_team_samples_per_track == 5
    assert config.team_sample_stride == 4
    assert config.min_team_sample_quality == 0.35
    assert config.max_team_samples_per_track == 9
    assert config.team_assignment_margin == 0.07
    assert config.team_debug_enabled is False
    assert config.number_predictor_backend == "dummy"


def test_from_env_reads_v2_tracking_settings_without_changing_legacy_defaults(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SAM31_PROMPT", "basketball player")
    monkeypatch.setenv("SAM31_FRAME_START", "60")
    monkeypatch.setenv("SAM31_FRAME_END", "130")
    monkeypatch.setenv("SAM31_RETENTION_FRAMES", "4")
    monkeypatch.setenv("SAM31_OUTPUT_PREFIX", "runs/game-03")
    monkeypatch.setenv("SAM31_ACCEPTANCE_REQUIRED", "1")

    config = PipelineConfig.from_env()

    assert config.sam31_prompt == "basketball player"
    assert config.sam31_frame_start == 60
    assert config.sam31_frame_end == 130
    assert config.sam31_retention_frames == 4
    assert config.sam31_output_prefix == "runs/game-03"
    assert config.sam31_acceptance_required is True
    assert config.sam2_model_id == "facebook/sam2-hiera-small"


def test_from_env_reads_possession_transition_gap(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("POSSESSION_TRANSITION_MAX_GAP", "7")

    assert PipelineConfig.from_env().possession_transition_max_gap == 7


def test_v3_ball_gap_defaults_and_reads_env(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.delenv("POSSESSION_V3_BALL_MAX_GAP", raising=False)
    assert PipelineConfig(roboflow_api_key="test-key").possession_v3_ball_max_gap == 3
    monkeypatch.setenv("POSSESSION_V3_BALL_MAX_GAP", "6")
    assert PipelineConfig.from_env().possession_v3_ball_max_gap == 6


@pytest.mark.parametrize("value", [-1, 1.5, float("nan"), float("inf")])
def test_v3_ball_gap_rejects_invalid_values(value):
    with pytest.raises((ValueError, TypeError)):
        PipelineConfig(roboflow_api_key="test-key", possession_v3_ball_max_gap=value)


def test_v3_ball_gap_rejects_invalid_env(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    for value in ("-1", "nan", "inf"):
        monkeypatch.setenv("POSSESSION_V3_BALL_MAX_GAP", value)
        with pytest.raises((ValueError, TypeError)):
            PipelineConfig.from_env()


def test_from_env_reads_opt_in_v3_engine_and_profile(monkeypatch):
    for key, value in {"ROBOFLOW_API_KEY": "test-key", "POSSESSION_ENGINE": "v3", "POSSESSION_V3_GRACE_SECONDS": "0.4", "POSSESSION_V3_CALIBRATION_PROFILE": "pilot"}.items():
        monkeypatch.setenv(key, value)
    config = PipelineConfig.from_env()
    assert (config.possession_engine, config.possession_v3_grace_seconds, config.possession_v3_calibration_profile) == ("v3", 0.4, "pilot")


def test_v3_reverted_switch_window_defaults_and_reads_env(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("POSSESSION_V3_REVERTED_SWITCH_WINDOW_SECONDS", "0.75")
    assert PipelineConfig(roboflow_api_key="test-key").possession_v3_reverted_switch_window_seconds == .5
    assert PipelineConfig.from_env().possession_v3_reverted_switch_window_seconds == .75


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_v3_reverted_switch_window_rejects_invalid_values(value):
    with pytest.raises((ValueError, TypeError)):
        PipelineConfig(roboflow_api_key="test-key", possession_v3_reverted_switch_window_seconds=value)


def test_v3_recovery_window_defaults_and_reads_env(monkeypatch):
    assert PipelineConfig(roboflow_api_key="test-key").possession_v3_recovery_window_seconds == .8
    monkeypatch.setenv("POSSESSION_V3_RECOVERY_WINDOW_SECONDS", ".9")
    assert PipelineConfig.from_env(roboflow_api_key="test-key").possession_v3_recovery_window_seconds == .9


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "not-a-number"])
def test_v3_reverted_switch_window_rejects_invalid_environment_values(monkeypatch, value):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("POSSESSION_V3_REVERTED_SWITCH_WINDOW_SECONDS", value)
    with pytest.raises((ValueError, TypeError)):
        PipelineConfig.from_env()


def test_from_env_preserves_legacy_possession_default(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.delenv("POSSESSION_ENGINE", raising=False)

    config = PipelineConfig.from_env()

    assert config.possession_engine == "legacy"
    assert config.possession_v3_enabled is False


@pytest.mark.parametrize("field,value", [("possession_v3_hysteresis_margin", float("nan")), ("possession_v3_persistence_seconds", float("inf")), ("possession_v3_event_window_seconds", "bad"), ("possession_v3_input_contract_version", 1.5)])
def test_pipeline_config_rejects_invalid_v3_numeric_values(field, value):
    with pytest.raises((ValueError, TypeError)):
        PipelineConfig(roboflow_api_key="test-key", **{field: value})

def test_from_env_defaults_main_and_shot_models_separately(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("PLAYER_MODEL_ID", "basketball-player-detection-3-ycjdo/16")
    monkeypatch.setenv("SHOT_MODEL_ID", "basketball-player-detection-3-ycjdo/1")

    config = PipelineConfig.from_env()

    assert config.player_model_id == "basketball-player-detection-3-ycjdo/16"
    assert config.shot_model_id == "basketball-player-detection-3-ycjdo/1"
    assert config.shot_target_fps is None
    assert config.shot_frame_stride == 1


def test_from_env_uses_aligned_default_shot_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.delenv("SHOT_MODEL_ID", raising=False)

    config = PipelineConfig.from_env()

    assert config.shot_model_id == "basketball-player-detection-3-ycjdo/1"


def test_from_env_reads_shot_target_fps(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SHOT_TARGET_FPS", "7.5")

    config = PipelineConfig.from_env()

    assert config.shot_target_fps == 7.5


def test_ball_stream_defaults_to_source_frames_without_changing_player_cadence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.delenv("BALL_TARGET_FPS", raising=False)
    monkeypatch.delenv("BALL_ALL_FRAMES", raising=False)

    config = PipelineConfig.from_env()

    assert config.target_fps == 15.0
    assert config.ball_all_frames is True
    assert config.ball_target_fps is None

    monkeypatch.setenv("BALL_TARGET_FPS", "24")
    assert PipelineConfig.from_env().ball_target_fps == 24.0
    monkeypatch.setenv("BALL_ALL_FRAMES", "0")
    compatibility = PipelineConfig.from_env()
    assert compatibility.ball_all_frames is False
    assert compatibility.ball_target_fps == 24.0


def test_from_env_defaults_court_projection_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")

    config = PipelineConfig.from_env()

    assert config.court_model_id == DEFAULT_COURT_MODEL_ID
    assert config.court_keypoint_confidence == 0.7


def test_pipeline_config_rejects_invalid_team_identification_values():
    with pytest.raises(ValueError, match="team_count must be at least 1"):
        PipelineConfig(roboflow_api_key="test-key", team_count=0)

    with pytest.raises(ValueError, match="min_team_samples_per_track must be at least 1"):
        PipelineConfig(roboflow_api_key="test-key", min_team_samples_per_track=0)

    with pytest.raises(ValueError, match="team_sample_stride must be at least 1"):
        PipelineConfig(roboflow_api_key="test-key", team_sample_stride=0)

    with pytest.raises(ValueError, match="shot_frame_stride must be at least 1"):
        PipelineConfig(roboflow_api_key="test-key", shot_frame_stride=0)

    with pytest.raises(ValueError, match="min_team_sample_quality must be at least 0"):
        PipelineConfig(roboflow_api_key="test-key", min_team_sample_quality=-0.01)

    with pytest.raises(ValueError, match="max_team_samples_per_track must be at least 1"):
        PipelineConfig(roboflow_api_key="test-key", max_team_samples_per_track=0)

    with pytest.raises(ValueError, match="team_assignment_margin must be at least 0"):
        PipelineConfig(roboflow_api_key="test-key", team_assignment_margin=-0.01)

    with pytest.raises(ValueError, match="sam31_frame_end must not precede"):
        PipelineConfig(roboflow_api_key="test-key", sam31_frame_start=10, sam31_frame_end=9)

    with pytest.raises(ValueError, match="sam31_retention_frames must be at least 0"):
        PipelineConfig(roboflow_api_key="test-key", sam31_retention_frames=-1)


def test_pipeline_config_rejects_unsupported_number_predictor_backend():
    with pytest.raises(ValueError, match="Unknown number_predictor_backend"):
        PipelineConfig(roboflow_api_key="test-key", number_predictor_backend="ocr")


def test_pipeline_config_accepts_hosted_number_predictor_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("NUMBER_PREDICTOR_BACKEND", "roboflow-hosted")
    monkeypatch.setenv("NUMBER_PREDICTOR_ROBOFLOW_MODEL_ID", "basketball-jersey-numbers-ocr/6")
    monkeypatch.setenv("NUMBER_PREDICTOR_SCORE_THRESHOLD", "0.7")

    config = PipelineConfig.from_env()

    assert config.number_predictor_backend == "roboflow-hosted"
    assert config.number_predictor_roboflow_model_id == "basketball-jersey-numbers-ocr/6"
    assert config.number_predictor_score_threshold == 0.7


def test_pipeline_config_rejects_invalid_hosted_number_predictor_threshold():
    with pytest.raises(ValueError, match="number_predictor_score_threshold must be between 0 and 1"):
        PipelineConfig(
            roboflow_api_key="test-key",
            number_predictor_backend="roboflow-hosted",
            number_predictor_score_threshold=1.5,
        )


def test_pipeline_config_rejects_invalid_court_keypoint_confidence():
    with pytest.raises(ValueError, match="court_keypoint_confidence must be between 0 and 1"):
        PipelineConfig(roboflow_api_key="test-key", court_keypoint_confidence=1.5)


def test_submit_passes_number_predictor_backend(monkeypatch: pytest.MonkeyPatch):
    import modal_app

    monkeypatch.setenv("MODAL_WEBHOOK_TOKEN", "secret")
    captured: dict[str, object] = {}

    def spawn(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(object_id="call-1")

    monkeypatch.setattr(
        modal_app,
        "Pipeline",
        lambda: types.SimpleNamespace(process=types.SimpleNamespace(spawn=spawn)),
    )

    response = modal_app.submit.get_raw_f()(
        {"r2_key": "videos/game.mp4", "number_predictor_backend": "dummy"},
        authorization="Bearer secret",
    )

    assert response == {"call_id": "call-1", "status": "queued"}
    assert captured["number_predictor_backend"] == "dummy"


def test_submit_passes_tracknet_enabled(monkeypatch: pytest.MonkeyPatch):
    import modal_app

    monkeypatch.setenv("MODAL_WEBHOOK_TOKEN", "secret")
    captured: dict[str, object] = {}

    def spawn(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(object_id="call-1")

    monkeypatch.setattr(
        modal_app,
        "Pipeline",
        lambda: types.SimpleNamespace(process=types.SimpleNamespace(spawn=spawn)),
    )

    modal_app.submit.get_raw_f()(
        {"r2_key": "videos/game.mp4", "tracknet_enabled": False},
        authorization="Bearer secret",
    )

    assert captured["tracknet_enabled"] is False


def test_submit_passes_pipeline_selector(monkeypatch: pytest.MonkeyPatch):
    import modal_app

    monkeypatch.setenv("MODAL_WEBHOOK_TOKEN", "secret")
    captured: dict[str, object] = {}

    def spawn(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(object_id="call-1")

    monkeypatch.setattr(
        modal_app,
        "Pipeline",
        lambda: types.SimpleNamespace(process=types.SimpleNamespace(spawn=spawn)),
    )

    modal_app.submit.get_raw_f()(
        {"r2_key": "videos/game.mp4", "pipeline": "player-positions"},
        authorization="Bearer secret",
    )

    assert captured["pipeline"] == "player-positions"
    assert captured["tracknet_enabled"] is True


def test_main_passes_number_predictor_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import modal_app

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"clip")
    captured: dict[str, object] = {}

    def remote(**kwargs):
        captured.update(kwargs)
        return {"summary": {}}

    monkeypatch.setattr(
        modal_app,
        "Pipeline",
        lambda: types.SimpleNamespace(process=types.SimpleNamespace(remote=remote)),
    )

    modal_app.main(video_path=str(video_path), number_predictor_backend="dummy")

    assert captured["number_predictor_backend"] == "dummy"


def test_main_passes_tracknet_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import modal_app

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"clip")
    captured: dict[str, object] = {}

    def remote(**kwargs):
        captured.update(kwargs)
        return {"summary": {}}

    monkeypatch.setattr(
        modal_app,
        "Pipeline",
        lambda: types.SimpleNamespace(process=types.SimpleNamespace(remote=remote)),
    )

    modal_app.main(video_path=str(video_path), tracknet_enabled=False)

    assert captured["tracknet_enabled"] is False


def test_main_passes_pipeline_selector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import modal_app

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"clip")
    captured: dict[str, object] = {}

    def remote(**kwargs):
        captured.update(kwargs)
        return {"summary": {}}

    monkeypatch.setattr(
        modal_app,
        "Pipeline",
        lambda: types.SimpleNamespace(process=types.SimpleNamespace(remote=remote)),
    )

    modal_app.main(video_path=str(video_path), pipeline="player-positions")

    assert captured["pipeline"] == "player-positions"


def test_main_passes_player_position_confidence_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import modal_app

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"clip")
    captured: dict[str, object] = {}

    def remote(**kwargs):
        captured.update(kwargs)
        return {"summary": {}}

    monkeypatch.setattr(
        modal_app,
        "Pipeline",
        lambda: types.SimpleNamespace(process=types.SimpleNamespace(remote=remote)),
    )

    modal_app.main(video_path=str(video_path), pipeline="player-positions", confidence=0.55)

    assert captured["confidence"] == 0.55


def test_main_stats_pipeline_leaves_confidence_unset_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import modal_app

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"clip")
    captured: dict[str, object] = {}

    def remote(**kwargs):
        captured.update(kwargs)
        return {"summary": {}}

    monkeypatch.setattr(
        modal_app,
        "Pipeline",
        lambda: types.SimpleNamespace(process=types.SimpleNamespace(remote=remote)),
    )

    modal_app.main(video_path=str(video_path))

    assert captured["confidence"] is None


def test_main_stats_pipeline_preserves_explicit_zero_confidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import modal_app

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"clip")
    captured: dict[str, object] = {}

    def remote(**kwargs):
        captured.update(kwargs)
        return {"summary": {}}

    monkeypatch.setattr(
        modal_app,
        "Pipeline",
        lambda: types.SimpleNamespace(process=types.SimpleNamespace(remote=remote)),
    )

    modal_app.main(video_path=str(video_path), confidence=0.0)

    assert captured["confidence"] == 0.0


def test_process_config_overrides_preserve_explicit_zero_confidence():
    import modal_app

    overrides = modal_app._build_process_config_overrides(
        pipeline=modal_app.MAIN_PIPELINE,
        frame_stride=None,
        max_frames=None,
        confidence=0.0,
        target_fps=None,
        sam2_model_id=None,
        max_missed_detections=None,
        appearance_weight=None,
        team_embedder_backend=None,
        team_embedder_model_id=None,
        team_count=None,
        min_team_samples_per_track=None,
        team_sample_stride=None,
        min_team_sample_quality=None,
        max_team_samples_per_track=None,
        team_debug_enabled=None,
        number_predictor_backend=None,
        write_annotated_video=False,
    )

    assert overrides["confidence"] == 0.0


def test_process_config_overrides_apply_player_positions_defaults():
    import modal_app

    overrides = modal_app._build_process_config_overrides(
        pipeline=modal_app.PLAYER_POSITIONS_PIPELINE,
        frame_stride=None,
        max_frames=None,
        confidence=None,
        target_fps=None,
        sam2_model_id=None,
        max_missed_detections=None,
        appearance_weight=None,
        team_embedder_backend=None,
        team_embedder_model_id=None,
        team_count=None,
        min_team_samples_per_track=None,
        team_sample_stride=None,
        min_team_sample_quality=None,
        max_team_samples_per_track=None,
        team_debug_enabled=None,
        number_predictor_backend=None,
        write_annotated_video=False,
    )

    assert overrides["confidence"] == 0.8
    assert overrides["target_fps"] == 5.0
    assert overrides["frame_stride"] == 5
    assert overrides["allowed_classes"] == ("player",)
    assert overrides["write_annotated_video"] is False


def test_run_selected_pipeline_dispatches_player_positions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import modal_app

    expected = {"positions_file": "player_positions.jsonl"}

    monkeypatch.setitem(sys.modules, "pipeline.player_positions", types.SimpleNamespace(run_player_position_pipeline=lambda *args, **kwargs: expected))

    result = modal_app._run_selected_pipeline(
        "player-positions",
        video_path=tmp_path / "clip.mp4",
        config=object(),
        output_dir=tmp_path / "out",
        tracker=object(),
        embedder=object(),
    )

    assert result == expected
