from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import test_ball_detection as ball_detection
from scripts.ball import tracknet
from scripts.ball.candidates import extract_candidates
from scripts.ball.test_tracknet import parse_args
from scripts.test_ball_detection import (
    CANDIDATE_MODEL_ID,
    DEFAULT_BALL_CLASSES,
    MODEL_ID,
    acceptance_result,
    capped_cases,
    evaluate_results,
    load_ground_truth,
    load_tracknet_ground_truth,
    parse_cli_args,
    resolved_ball_classes,
)


def _prediction(x: float, y: float, confidence: float = 0.9) -> dict:
    return {"class": "basketball", "confidence": confidence, "x": x, "y": y, "width": 10, "height": 10}


def _tracknet_game(root: Path, rows: str, *, clip: str = "clip_001", frames=(0, 1)) -> Path:
    game = root / "match01"
    (game / "csv").mkdir(parents=True)
    frame_dir = game / "frame" / clip
    frame_dir.mkdir(parents=True)
    (game / "csv" / f"{clip}_ball.csv").write_text("Frame,Visibility,X,Y\n" + rows, encoding="utf-8")
    for frame in frames:
        (frame_dir / f"{frame}.png").write_bytes(b"frame")
    return game


def test_tracknet_layout_discovery_parses_points_and_negatives(tmp_path: Path):
    game = _tracknet_game(tmp_path / "test", "0,1,12.5,20\n1,0,0,0\n")

    cases = load_tracknet_ground_truth(game.parent)

    assert [case["points"] for case in cases] == [[[12.5, 20.0]], []]
    assert [Path(case["image"]).name for case in cases] == ["0.png", "1.png"]
    assert {case["clip"] for case in cases} == {"match01/clip_001"}


def test_tracknet_capped_sampling_is_deterministic_and_keeps_negative():
    cases = [
        {"image": str(index), "points": [] if index == 7 else [[1.0, 1.0]]}
        for index in range(10)
    ]

    assert [case["image"] for case in capped_cases(cases, 3)] == ["0", "3", "7"]
    with pytest.raises(ValueError, match="at least 2"):
        capped_cases(cases, 1)


def test_tracknet_point_matching_uses_prediction_center_threshold():
    case = [{"image": "a.png", "clip": "clip-a", "points": [[10.0, 10.0]]}]

    matched = evaluate_results(case, [{"predictions": [_prediction(30, 10)]}], confidence=0.4)
    missed = evaluate_results(case, [{"predictions": [_prediction(30.1, 10)]}], confidence=0.4)

    assert matched["true_positives"] == 1
    assert matched["mean_center_error_px"] == 20.0
    assert matched["point_distance_threshold_px"] == 20.0
    assert missed["true_positives"] == 0
    assert missed["false_positives"] == 1


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("bad,1,10,20\n", "Frame and Visibility must be integers"),
        ("0,2,10,20\n", "Visibility must be 0 or 1"),
        ("0,1,bad,20\n", "X must be a finite number"),
    ],
)
def test_tracknet_csv_rejects_malformed_rows(tmp_path: Path, rows: str, message: str):
    game = _tracknet_game(tmp_path, rows, frames=(0,))

    with pytest.raises(ValueError, match=message):
        load_tracknet_ground_truth(game)


def test_tracknet_csv_rejects_missing_corresponding_frame(tmp_path: Path):
    game = _tracknet_game(tmp_path, "0,1,10,20\n1,0,0,0\n", frames=(0,))

    with pytest.raises(ValueError, match="corresponding frame not found"):
        load_tracknet_ground_truth(game)


def test_tracknet_root_rejects_missing_labels(tmp_path: Path):
    with pytest.raises(ValueError, match=r"no csv/\*_ball.csv labels found"):
        load_tracknet_ground_truth(tmp_path)


def test_tracknet_dry_run_prints_plan_without_credentials_or_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    game = _tracknet_game(tmp_path, "0,1,10,20\n1,0,0,0\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["test_ball_detection.py", "--tracknet-root", str(game), "--frame-cap", "2", "--dry-run"],
    )
    monkeypatch.setattr(ball_detection, "get_roboflow_api_key", lambda: pytest.fail("credentials were read"))
    monkeypatch.setattr(ball_detection, "roboflow_client", lambda _key: pytest.fail("API client was created"))

    ball_detection.main()

    output = capsys.readouterr().out
    assert "Planned labeled frames: 2" in output
    assert "Planned visible/no-ball frames: 1/1" in output
    assert "Planned inference calls: 4" in output
    assert "Point match threshold: 20px" in output
    assert "no API calls made" in output


def test_ball_seed_metrics_include_negative_frames_and_clip_recall():
    cases = [
        {"image": "a.jpg", "clip": "clip-a", "balls": [[0, 0, 10, 10]]},
        {"image": "b.jpg", "clip": "clip-a", "balls": []},
    ]

    metrics = evaluate_results(
        cases,
        [
            {"predictions": [_prediction(5, 5)]},
            {"predictions": [_prediction(20, 20)]},
        ],
        confidence=0.4,
    )

    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["false_positives_per_frame"] == 0.5
    assert metrics["mean_center_error_px"] == 0.0
    assert metrics["clips"]["clip-a"]["recall"] == 1.0
    assert metrics["ap50"] is None


def test_ball_seed_matching_maximizes_true_positives_before_center_error():
    metrics = evaluate_results(
        [{"image": "a.jpg", "clip": "clip-a", "balls": [[0, 0, 10, 10], [4, 0, 14, 10]]}],
        [{"predictions": [_prediction(7, 5, 0.99), _prediction(5, 5, 0.8)]}],
        confidence=0.4,
    )

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["mean_center_error_px"] == 1.0


def test_ball_seed_acceptance_handles_exact_boundaries_and_clip_regression():
    baseline = {
        "recall": 2 / 3,
        "precision": 0.8,
        "false_positives_per_frame": 0.2,
        "clips": {"clip-a": {"visible_balls": 10, "recall": 0.7}},
    }
    candidate = {
        "recall": 23 / 30,
        "precision": 0.8,
        "false_positives_per_frame": 0.2,
        "clips": {"clip-a": {"visible_balls": 10, "recall": 0.65}},
    }

    assert acceptance_result(baseline, candidate)["passed"] is True
    candidate["clips"]["clip-a"]["recall"] = 0.649
    result = acceptance_result(baseline, candidate)
    assert result["passed"] is False
    assert result["checks"]["no_clip_regresses_over_5_points"] is False


def test_ball_seed_cli_resolves_models_and_rejects_one_model():
    args = parse_cli_args(
        ["--ground-truth", "labels.jsonl", "--model-id", "baseline/1", "--model-id", "candidate/2"]
    )
    assert args.model_ids == ["baseline/1", "candidate/2"]

    defaults = parse_cli_args(["--ground-truth", "labels.jsonl"])
    assert defaults.model_ids == [MODEL_ID, CANDIDATE_MODEL_ID]

    with pytest.raises(SystemExit):
        parse_cli_args(["--ground-truth", "labels.jsonl", "--model-id", "only/1"])


def test_ball_seed_cli_rejects_one_model_before_credentials(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["test_ball_detection.py", "--ground-truth", "labels.jsonl", "--model-id", "only/1"],
    )
    monkeypatch.setattr(ball_detection, "get_roboflow_api_key", lambda: pytest.fail("credentials were read"))

    with pytest.raises(SystemExit):
        ball_detection.main()

    assert "requires at least two --model-id values" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_ball_seed_cli_rejects_invalid_point_distance(value: str, capsys):
    with pytest.raises(SystemExit):
        parse_cli_args(
            ["--tracknet-root", "processed/test", f"--point-distance-px={value}", "--dry-run"]
        )

    assert "--point-distance-px must be a finite number greater than 0" in capsys.readouterr().err


def test_ball_class_option_extends_defaults():
    args = parse_cli_args(
        ["--ground-truth", "labels.jsonl", "--ball-class", "sports-ball"]
    )

    assert resolved_ball_classes(args.ball_classes) == (*DEFAULT_BALL_CLASSES, "sports-ball")
    metrics = evaluate_results(
        [{"image": "a.jpg", "clip": "clip-a", "balls": [[0, 0, 10, 10]]}],
        [{"predictions": [{**_prediction(5, 5), "class": "sports-ball"}]}],
        confidence=0.4,
        class_names=resolved_ball_classes(args.ball_classes),
    )
    assert metrics["true_positives"] == 1


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (None, "hosted response must be an object"),
        ({"predictions": None}, "hosted response predictions must be a list"),
        ({"predictions": [None]}, "prediction[0] must be an object"),
        ({"predictions": [{**_prediction(5, 5), "class": None}]}, "prediction[0].class must be a string"),
        ({"predictions": [{**_prediction(5, 5), "confidence": "high"}]}, "prediction[0].confidence must be a finite number"),
        (
            {"predictions": [{"class": "basketball", "confidence": 0.9, "x": 5, "y": 5, "height": 10}]},
            "prediction[0].width must be a finite number",
        ),
    ],
)
def test_ball_seed_metrics_reject_malformed_hosted_responses(result, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        evaluate_results(
            [{"image": "a.jpg", "clip": "clip-a", "balls": []}],
            [result],
            confidence=0.4,
        )


def test_ball_seed_metrics_reject_empty_or_mismatched_inputs():
    with pytest.raises(ValueError, match="at least one labeled frame"):
        evaluate_results([], [], confidence=0.4)
    with pytest.raises(ValueError, match="result count"):
        evaluate_results([{"balls": []}], [], confidence=0.4)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]\n", "each record must be a JSON object"),
        ('{"image":"frame.jpg","balls":[[0,0,"bad",10]]}\n', "must be a finite number"),
        ("", "no labeled frames found"),
    ],
)
def test_ball_seed_ground_truth_rejects_malformed_or_empty_jsonl(tmp_path: Path, payload: str, message: str):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_ground_truth(labels)

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
