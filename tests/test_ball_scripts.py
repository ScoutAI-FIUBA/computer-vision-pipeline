from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ball.candidates import extract_candidates
from scripts.ball.test_tracknet import parse_args
from scripts.ball import tracknet
from scripts.test_ball_detection import filter_ball_predictions


def test_filter_ball_predictions_matches_ball_aliases_case_insensitively():
    result = {
        "predictions": [
            {"class": "Basketball", "confidence": 0.9},
            {"class": " player ", "confidence": 0.8},
            {"class": "BALL", "confidence": 0.7},
        ]
    }

    filtered = filter_ball_predictions(result, ("basketball", "ball"))

    assert [prediction["class"] for prediction in filtered["predictions"]] == ["Basketball", "BALL"]


def test_extract_candidates_keeps_top_spaced_peaks():
    heatmap = [
        [0.1, 0.9, 0.8, 0.1],
        [0.1, 0.2, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ]

    candidates = extract_candidates(
        heatmap,
        original_frame_size=(40, 20),
        min_score=0.5,
        min_distance=15.0,
        max_candidates=3,
    )

    assert [(candidate.heatmap_x, candidate.heatmap_y) for candidate in candidates] == [(1, 0), (3, 3)]
    assert [candidate.score for candidate in candidates] == [pytest.approx(0.9), pytest.approx(0.7)]
    assert candidates[0].x == pytest.approx(15.0)
    assert candidates[0].y == pytest.approx(2.5)


def test_default_tracknet_checkpoint_path_requires_existing_bundled_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint_path = tmp_path / "TrackNet_best.pt"
    monkeypatch.setattr(tracknet, "DEFAULT_TRACKNET_PT", checkpoint_path)

    assert tracknet.default_tracknet_checkpoint_path() is None

    checkpoint_path.write_bytes(b"stub")

    assert tracknet.default_tracknet_checkpoint_path() == checkpoint_path


def test_parse_args_allows_inspect_mode_without_video(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["test_tracknet.py", "--inspect-model", "--backend", "pytorch"])

    args = parse_args()

    assert args.inspect_model is True
    assert args.video is None
    assert args.backend == "pytorch"
