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


def test_pipeline_config_rejects_invalid_team_identification_values():
    with pytest.raises(ValueError, match="team_count must be at least 1"):
        PipelineConfig(roboflow_api_key="test-key", team_count=0)

    with pytest.raises(ValueError, match="min_team_samples_per_track must be at least 1"):
        PipelineConfig(roboflow_api_key="test-key", min_team_samples_per_track=0)

    with pytest.raises(ValueError, match="team_sample_stride must be at least 1"):
        PipelineConfig(roboflow_api_key="test-key", team_sample_stride=0)

    with pytest.raises(ValueError, match="min_team_sample_quality must be at least 0"):
        PipelineConfig(roboflow_api_key="test-key", min_team_sample_quality=-0.01)

    with pytest.raises(ValueError, match="max_team_samples_per_track must be at least 1"):
        PipelineConfig(roboflow_api_key="test-key", max_team_samples_per_track=0)

    with pytest.raises(ValueError, match="team_assignment_margin must be at least 0"):
        PipelineConfig(roboflow_api_key="test-key", team_assignment_margin=-0.01)


def test_pipeline_config_rejects_unsupported_number_predictor_backend():
    with pytest.raises(ValueError, match="Unknown number_predictor_backend"):
        PipelineConfig(roboflow_api_key="test-key", number_predictor_backend="ocr")


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
