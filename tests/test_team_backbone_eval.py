"""Network-free tests for the offline team-backbone evaluator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.team.evaluate_backbones import (
    CropRecord,
    evaluate_model,
    load_crop_records,
    main,
    _metrics,
)
from pipeline.config import DEFAULT_TEAM_EMBEDDER_MODEL_ID, PipelineConfig


def _debug_dir(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    team_debug = root / "team_debug" / "track_0002"
    team_debug.mkdir(parents=True)
    cv2.imwrite(str(team_debug / "000_frame_000003_accepted.png"), np.full((8, 8, 3), 40, np.uint8))
    cv2.imwrite(str(team_debug / "001_frame_000001_accepted.png"), np.full((8, 8, 3), 80, np.uint8))
    (root / "team_debug.json").write_text(json.dumps({
        "2": {"samples": [
            {"frame_index": 3, "accepted": True, "selected": True},
            {"frame_index": 1, "accepted": True, "selected": True},
            {"frame_index": 4, "accepted": False, "selected": False},
        ]},
    }))
    return root


def test_load_crop_records_is_deterministic_and_validates_crops(tmp_path: Path):
    records = load_crop_records(_debug_dir(tmp_path))
    assert [(record.track_id, record.frame_index) for record in records] == [(2, 1), (2, 3)]

    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "team_debug.json").write_text(json.dumps({"1": {"samples": [{"frame_index": 1, "accepted": True, "selected": True}]}}))
    with pytest.raises(ValueError, match="missing crop"):
        load_crop_records(missing)


def test_evaluate_model_uses_same_order_and_reports_metrics(tmp_path: Path):
    root = _debug_dir(tmp_path)
    records = load_crop_records(root, {"2": "home"})
    seen = []

    class FakeEmbedder:
        name = "fake"

        def embed_crops(self, crops):
            seen.append([int(crop[0, 0, 0]) for crop in crops])
            return [[1.0, 0.0], [1.0, 0.0]]

    evaluation = evaluate_model(records, FakeEmbedder(), minimum_samples=1)
    assert seen == [[80, 40]]
    assert evaluation["metrics"]["track_count"] == 1
    assert evaluation["metrics"]["coverage"] == 1.0


def test_main_selects_explicit_models_and_dry_run_skips_factory(tmp_path: Path):
    output = tmp_path / "report.json"
    calls = []

    def factory(model_id):
        calls.append(model_id)
        raise AssertionError("dry-run must not load models")

    assert main([str(_debug_dir(tmp_path)), str(output), "--dry-run"], embedder_factory=factory) == 0
    report = json.loads(output.read_text())
    assert report["models"]["candidate"] == "Marqo/marqo-fashionSigLIP"
    assert calls == []


def test_main_gives_both_backbones_the_same_crop_objects(tmp_path: Path):
    output = tmp_path / "report.json"
    seen = []
    models = []

    class FakeEmbedder:
        name = "fake"

        def embed_crops(self, crops):
            seen.append(id(crops))
            return [[1.0, 0.0]] * len(crops)

    def factory(model_id):
        models.append(model_id)
        return FakeEmbedder()

    assert main([
        str(_debug_dir(tmp_path)), str(output), "--minimum-samples", "1",
        "--baseline-model-id", "baseline", "--candidate-model-id", "candidate",
    ], embedder_factory=factory) == 0
    assert models == ["baseline", "candidate"]
    assert len(seen) == 2
    assert seen[0] == seen[1]


def test_missing_metadata_is_a_clear_dry_run_error(tmp_path: Path):
    with pytest.raises(ValueError, match="missing team-debug metadata"):
        load_crop_records(tmp_path)


def test_direct_script_help_works_from_repo_root():
    result = subprocess.run(
        [sys.executable, "scripts/team/evaluate_backbones.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "offline" in result.stdout.lower()


def test_metrics_allow_more_predicted_clusters_than_truth_labels():
    records = [
        CropRecord(track_id=track_id, frame_index=1, crop_path=Path("crop.png"), ground_truth="home")
        for track_id in (1, 2, 3)
    ]
    result = SimpleNamespace(
        assignments=[SimpleNamespace(track_id=1, team_id=0), SimpleNamespace(track_id=2, team_id=1), SimpleNamespace(track_id=3, team_id=2)],
        track_diagnostics={},
    )
    metrics = _metrics(records, result)
    assert metrics["accuracy_on_assigned"] == 1.0
    assert metrics["correct_track_count"] == 3


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "non-empty object"),
        ({"1": []}, "track 1 metadata must be an object"),
        ({"1": {"samples": [{"accepted": True, "selected": True}]}}, "sample 0 is missing integer frame_index"),
        ({"1": {"samples": [None]}}, "sample 0 metadata must be an object"),
    ],
)
def test_malformed_nested_metadata_has_clear_errors(tmp_path: Path, payload, message):
    root = tmp_path / "malformed"
    root.mkdir()
    (root / "team_debug.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        load_crop_records(root)


def test_production_team_embedder_defaults_to_fashion_siglip():
    assert PipelineConfig(roboflow_api_key="test").team_embedder_model_id == DEFAULT_TEAM_EMBEDDER_MODEL_ID == "Marqo/marqo-fashionSigLIP"
