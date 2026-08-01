"""Pipeline config plumbing tests.

    python -m pytest tests/test_config.py
"""

from __future__ import annotations

import sys
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
