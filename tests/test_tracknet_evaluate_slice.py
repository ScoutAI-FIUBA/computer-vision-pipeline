from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TrackNetV3"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import (
    build_final_comparison_report,
    build_metrics_report,
    main as evaluate_main,
    select_best_checkpoint_report,
    write_validation_artifacts,
)
from modal_tracknet_finetune import build_run_layout, main as modal_main, normalize_run_id
from preprocess import preprocess_basketball_dataset


def _write_summary(path: Path, *, tp: float, fp1: float, fp2: float, fn: float, mean_error: float, median_error: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "TP": tp,
                "TN": 0.0,
                "FP1": fp1,
                "FP2": fp2,
                "FN": fn,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "miss_rate": 0.0,
                "p_at_10": 0.25,
                "p_at_40": 1.0,
                "mean_error": mean_error,
                "median_error": median_error,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _write_clip(root: Path, split: str, game: str, clip: str) -> None:
    import csv

    import cv2
    import numpy as np

    video_path = root / split / game / "video" / f"{clip}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 12))
    for _ in range(3):
        writer.write(np.zeros((12, 16, 3), dtype=np.uint8))
    writer.release()

    csv_path = root / split / game / "csv" / f"{clip}_ball.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Frame", "Visibility", "X", "Y"])
        writer.writeheader()
        writer.writerows(
            [
                {"Frame": 0, "Visibility": 1, "X": 5, "Y": 4},
                {"Frame": 1, "Visibility": 1, "X": 6, "Y": 5},
                {"Frame": 2, "Visibility": 0, "X": 0, "Y": 0},
            ]
        )


def test_build_metrics_report_includes_visible_frame_metrics_and_false_positives() -> None:
    report = build_metrics_report(
        {
            "TP": 7.0,
            "TN": 5.0,
            "FP1": 1.0,
            "FP2": 2.0,
            "FN": 2.0,
            "precision": 0.7,
            "recall": 0.777,
            "f1": 0.736,
            "p_at_10": 0.5,
            "p_at_40": 0.9,
            "mean_error": 12.0,
            "median_error": 11.0,
        },
        split="test",
        checkpoint="ckpts/TrackNet_best.pt",
        dataset_root="/cache/tracknet-finetune/runs/demo/adapted-data",
        evaluation_mode="weight",
    )

    assert report["metrics"]["visible_frames"] == 10.0
    assert report["metrics"]["detection_rate"] == 0.8
    assert report["metrics"]["p_at_20"] == 0.7
    assert report["metrics"]["false_positives_visibility_0"] == 2.0
    assert report["metrics"]["false_positive_rate"] == 2.0 / 7.0
    assert report["metrics"]["precision"] == 0.7
    assert report["metrics"]["recall"] == 0.777
    assert report["metrics"]["f1"] == 0.736
    assert report["checkpoint"] == "ckpts/TrackNet_best.pt"


def test_select_best_checkpoint_report_applies_guard_and_tiebreaks() -> None:
    baseline = {"checkpoint": "baseline.pt", "metrics": {"detection_rate": 0.81, "p_at_20": 0.7, "median_error": 11.0}}
    selected = select_best_checkpoint_report(
        baseline,
        [
            {"checkpoint": "epoch-1.pt", "metrics": {"detection_rate": 0.79, "p_at_20": 0.84, "median_error": 10.0}},
            {"checkpoint": "epoch-2.pt", "metrics": {"detection_rate": 0.80, "p_at_20": 0.84, "median_error": 9.0}},
            {"checkpoint": "epoch-3.pt", "metrics": {"detection_rate": 0.75, "p_at_20": 0.9, "median_error": 8.0}},
        ],
    )

    assert selected["selected_checkpoint"] == "epoch-2.pt"
    assert selected["selected_metrics"]["p_at_20"] == 0.84
    assert selected["rejected_checkpoints"] == ["epoch-3.pt"]


def test_build_final_comparison_report_outputs_side_by_side_deltas() -> None:
    baseline = {
        "checkpoint": "baseline.pt",
        "split": "test",
        "dataset_root": "/cache/run/adapted-data",
        "dataset_manifest_fingerprint": "manifest-123",
        "metrics": {
            "detection_rate": 0.8,
            "p_at_10": 0.44,
            "p_at_20": 0.72,
            "p_at_40": 0.91,
            "mean_error": 14.0,
            "median_error": 13.0,
            "false_positive_rate": 0.15,
            "precision": 0.7,
            "recall": 0.8,
            "f1": 0.747,
        },
    }
    finetuned = {
        "checkpoint": "finetuned.pt",
        "split": "test",
        "dataset_root": "/cache/run/adapted-data",
        "dataset_manifest_fingerprint": "manifest-123",
        "metrics": {
            "detection_rate": 0.81,
            "p_at_10": 0.51,
            "p_at_20": 0.79,
            "p_at_40": 0.95,
            "mean_error": 11.0,
            "median_error": 10.0,
            "false_positive_rate": 0.1,
            "precision": 0.75,
            "recall": 0.81,
            "f1": 0.779,
        },
    }

    comparison = build_final_comparison_report(baseline, finetuned)

    assert comparison["split"] == "test"
    assert comparison["delta"]["p_at_10"] == 0.07
    assert comparison["delta"]["p_at_20"] == 0.07
    assert comparison["delta"]["p_at_40"] == 0.04
    assert comparison["delta"]["detection_rate"] == 0.01
    assert comparison["delta"]["false_positive_rate"] == -0.05
    assert comparison["delta"]["precision"] == 0.05
    assert comparison["delta"]["recall"] == 0.01
    assert comparison["delta"]["f1"] == 0.032
    assert comparison["baseline"]["checkpoint"] == "baseline.pt"
    assert comparison["finetuned"]["checkpoint"] == "finetuned.pt"


def test_build_final_comparison_report_rejects_validation_reports_for_test_claim() -> None:
    baseline = {"checkpoint": "baseline.pt", "split": "validation", "metrics": {"detection_rate": 0.8, "p_at_20": 0.72}}
    finetuned = {"checkpoint": "finetuned.pt", "split": "test", "metrics": {"detection_rate": 0.81, "p_at_20": 0.79}}

    with pytest.raises(ValueError, match="test"):
        build_final_comparison_report(baseline, finetuned, split="test")


def test_build_final_comparison_report_rejects_mismatched_dataset_provenance() -> None:
    baseline = {
        "checkpoint": "baseline.pt",
        "split": "test",
        "dataset_root": "/cache/run-a/adapted-data",
        "dataset_manifest_fingerprint": "manifest-a",
        "metrics": {"detection_rate": 0.8, "p_at_20": 0.72},
    }
    finetuned = {
        "checkpoint": "finetuned.pt",
        "split": "test",
        "dataset_root": "/cache/run-b/adapted-data",
        "dataset_manifest_fingerprint": "manifest-b",
        "metrics": {"detection_rate": 0.81, "p_at_20": 0.79},
    }

    with pytest.raises(ValueError, match="dataset"):
        build_final_comparison_report(baseline, finetuned, split="test")


@pytest.mark.parametrize(
    "baseline, finetuned",
    [
        (
            {"checkpoint": "baseline.pt", "split": "test", "metrics": {"detection_rate": 0.8, "p_at_20": 0.72}},
            {"checkpoint": "finetuned.pt", "split": "test", "metrics": {"detection_rate": 0.81, "p_at_20": 0.79}},
        ),
        (
            {
                "checkpoint": "baseline.pt",
                "split": "test",
                "dataset_root": "/cache/run/adapted-data",
                "dataset_manifest_fingerprint": "manifest-123",
                "metrics": {"detection_rate": 0.8, "p_at_20": 0.72},
            },
            {
                "checkpoint": "finetuned.pt",
                "split": "test",
                "dataset_root": "/cache/run/adapted-data",
                "metrics": {"detection_rate": 0.81, "p_at_20": 0.79},
            },
        ),
    ],
)
def test_build_final_comparison_report_rejects_missing_dataset_provenance(baseline: dict, finetuned: dict) -> None:
    with pytest.raises(ValueError, match="dataset provenance"):
        build_final_comparison_report(baseline, finetuned, split="test")


def test_evaluate_compare_cli_rejects_validation_reports(tmp_path: Path) -> None:
    baseline_validation_report = tmp_path / "baseline-validation-report.json"
    finetuned_validation_report = tmp_path / "finetuned-validation-report.json"
    comparison_report = tmp_path / "comparison.json"

    baseline_validation_report.write_text(
        json.dumps({"checkpoint": "baseline.pt", "split": "validation", "metrics": {"detection_rate": 0.8, "p_at_20": 0.72}}),
        encoding="utf-8",
    )
    finetuned_validation_report.write_text(
        json.dumps({"checkpoint": "finetuned.pt", "split": "validation", "metrics": {"detection_rate": 0.81, "p_at_20": 0.79}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test"):
        evaluate_main([
            "compare",
            "--baseline-report",
            str(baseline_validation_report),
            "--finetuned-report",
            str(finetuned_validation_report),
            "--split",
            "test",
            "--output",
            str(comparison_report),
        ])


@pytest.mark.parametrize(
    "baseline, finetuned, expected_message",
    [
        (
            {
                "checkpoint": "baseline.pt",
                "split": "test",
                "metrics": {"detection_rate": 0.8, "p_at_20": 0.72},
            },
            {
                "checkpoint": "finetuned.pt",
                "split": "test",
                "metrics": {"detection_rate": 0.81, "p_at_20": 0.79},
            },
            "dataset_root",
        ),
        (
            {
                "checkpoint": "baseline.pt",
                "split": "test",
                "dataset_root": "/cache/run/adapted-data",
                "metrics": {"detection_rate": 0.8, "p_at_20": 0.72},
            },
            {
                "checkpoint": "finetuned.pt",
                "split": "test",
                "dataset_root": "/cache/run/adapted-data",
                "metrics": {"detection_rate": 0.81, "p_at_20": 0.79},
            },
            "dataset_manifest_fingerprint",
        ),
    ],
)
def test_evaluate_compare_cli_rejects_missing_dataset_provenance(
    tmp_path: Path,
    baseline: dict,
    finetuned: dict,
    expected_message: str,
) -> None:
    baseline_report = tmp_path / "baseline-test-report.json"
    finetuned_report = tmp_path / "finetuned-test-report.json"
    comparison_output = tmp_path / "final_comparison.json"

    baseline_report.write_text(json.dumps(baseline), encoding="utf-8")
    finetuned_report.write_text(json.dumps(finetuned), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        evaluate_main([
            "compare",
            "--baseline-report",
            str(baseline_report),
            "--finetuned-report",
            str(finetuned_report),
            "--split",
            "test",
            "--output",
            str(comparison_output),
        ])


def test_evaluate_compare_cli_persists_validated_dataset_provenance(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline-test-report.json"
    finetuned_report = tmp_path / "finetuned-test-report.json"
    comparison_output = tmp_path / "final_comparison.json"

    provenance_root = "/cache/tracknet-finetune/runs/demo/adapted-data"
    provenance_fingerprint = "manifest-audit-9"
    baseline_payload = {
        "checkpoint": "ckpts/TrackNet_best.pt",
        "split": "test",
        "dataset_root": provenance_root,
        "dataset_manifest_fingerprint": provenance_fingerprint,
        "metrics": {
            "detection_rate": 0.8,
            "p_at_10": 0.4,
            "p_at_20": 0.7,
            "p_at_40": 0.9,
            "mean_error": 14.0,
            "median_error": 13.0,
        },
    }
    finetuned_payload = {
        "checkpoint": "run/TrackNet_best.pt",
        "split": "test",
        "dataset_root": provenance_root,
        "dataset_manifest_fingerprint": provenance_fingerprint,
        "metrics": {
            "detection_rate": 0.81,
            "p_at_10": 0.5,
            "p_at_20": 0.8,
            "p_at_40": 0.95,
            "mean_error": 11.0,
            "median_error": 10.0,
        },
    }
    baseline_report.write_text(json.dumps(baseline_payload), encoding="utf-8")
    finetuned_report.write_text(json.dumps(finetuned_payload), encoding="utf-8")

    assert evaluate_main([
        "compare",
        "--baseline-report",
        str(baseline_report),
        "--finetuned-report",
        str(finetuned_report),
        "--split",
        "test",
        "--output",
        str(comparison_output),
    ]) == 0

    payload = json.loads(comparison_output.read_text(encoding="utf-8"))
    assert payload["dataset_root"] == provenance_root
    assert payload["dataset_manifest_fingerprint"] == provenance_fingerprint
    assert payload["split"] == "test"
    assert payload["baseline"]["checkpoint"] == "ckpts/TrackNet_best.pt"
    assert payload["finetuned"]["checkpoint"] == "run/TrackNet_best.pt"


@pytest.mark.parametrize(
    ("baseline_split", "candidate_split"),
    [("test", "validation"), ("validation", "test")],
)
def test_evaluate_select_best_cli_rejects_non_validation_reports(
    tmp_path: Path,
    baseline_split: str,
    candidate_split: str,
) -> None:
    baseline_report = tmp_path / "baseline-report.json"
    candidate_report = tmp_path / "candidate-report.json"
    output_report = tmp_path / "selection.json"
    baseline_report.write_text(
        json.dumps({"checkpoint": "baseline.pt", "split": baseline_split, "metrics": {"detection_rate": 0.8, "p_at_20": 0.72}}),
        encoding="utf-8",
    )
    candidate_report.write_text(
        json.dumps({"checkpoint": "finetuned.pt", "split": candidate_split, "metrics": {"detection_rate": 0.81, "p_at_20": 0.79}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation"):
        evaluate_main([
            "select-best",
            "--baseline-report",
            str(baseline_report),
            "--candidate-report",
            str(candidate_report),
            "--output",
            str(output_report),
        ])


def test_modal_main_rejects_validation_reports_wired_into_final_compare(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset-tracknet"
    reports_root = tmp_path / "reports"
    baseline_validation_report = reports_root / "baseline-validation-report.json"
    finetuned_validation_report = reports_root / "finetuned-validation-report.json"

    baseline_validation_report.parent.mkdir(parents=True, exist_ok=True)
    baseline_validation_report.write_text(
        json.dumps({"checkpoint": "baseline.pt", "split": "validation", "metrics": {"detection_rate": 0.8, "p_at_20": 0.72}}),
        encoding="utf-8",
    )
    finetuned_validation_report.write_text(
        json.dumps({"checkpoint": "finetuned.pt", "split": "validation", "metrics": {"detection_rate": 0.81, "p_at_20": 0.79}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="test"):
        modal_main([
            "--run-id",
            "bad-compare-inputs",
            "--raw-dataset-root",
            str(raw_root),
            "--volume-root",
            str(tmp_path / "modal-volume"),
            "--pretrained-tracknet",
            "ckpts/TrackNet_best.pt",
            "--baseline-validation-report",
            str(baseline_validation_report),
            "--candidate-validation-report",
            str(finetuned_validation_report),
            "--baseline-test-report",
            str(baseline_validation_report),
            "--finetuned-test-report",
            str(finetuned_validation_report),
            "--plan-output",
            str(reports_root / "modal-plan.json"),
            "--dry-run",
        ])


def test_write_validation_artifacts_persists_machine_readable_report(tmp_path: Path) -> None:
    report_path = write_validation_artifacts(
        tmp_path,
        split="validation",
        checkpoint="checkpoints/TrackNet_epoch_001.pt",
        dataset_root="/cache/tracknet-finetune/runs/demo/adapted-data",
        evaluation_mode="weight",
        summary={
            "TP": 5.0,
            "TN": 0.0,
            "FP1": 1.0,
            "FP2": 0.0,
            "FN": 2.0,
            "p_at_10": 0.3,
            "p_at_40": 0.9,
            "mean_error": 12.5,
            "median_error": 11.0,
        },
        output_name="epoch-001-validation.json",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["checkpoint"] == "checkpoints/TrackNet_epoch_001.pt"
    assert payload["metrics"]["p_at_20"] == 0.625


def test_evaluate_and_modal_clis_cover_non_gpu_smoke_path(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset-tracknet"
    adapted_root = tmp_path / "adapted-data"
    run_root = build_run_layout(tmp_path / "volume", "demo-run").run_root
    reports_root = run_root / "reports"
    _write_clip(raw_root, "validation", "game_01", "clip_001")
    _write_clip(raw_root, "test", "game_02", "clip_002")
    preprocess_basketball_dataset(raw_root, adapted_root, splits=("validation", "test"))

    baseline_summary = _write_summary(reports_root / "baseline-summary.json", tp=6, fp1=2, fp2=1, fn=2, mean_error=14.0, median_error=13.0)
    finetuned_summary = _write_summary(reports_root / "finetuned-summary.json", tp=7, fp1=1, fp2=1, fn=2, mean_error=11.0, median_error=10.0)
    baseline_validation_report = reports_root / "baseline-validation-report.json"
    finetuned_validation_report = reports_root / "finetuned-validation-report.json"
    baseline_test_report = reports_root / "baseline-test-report.json"
    finetuned_test_report = reports_root / "finetuned-test-report.json"
    selection_report = reports_root / "selection.json"
    comparison_report = reports_root / "comparison.json"
    modal_plan = reports_root / "modal-plan.json"

    assert evaluate_main([
        "metrics",
        "--summary-json",
        str(baseline_summary),
        "--split",
        "validation",
        "--checkpoint",
        "ckpts/TrackNet_best.pt",
        "--dataset-root",
        str(adapted_root),
        "--output",
        str(baseline_validation_report),
    ]) == 0
    assert evaluate_main([
        "metrics",
        "--summary-json",
        str(finetuned_summary),
        "--split",
        "validation",
        "--checkpoint",
        "run/TrackNet_best.pt",
        "--dataset-root",
        str(adapted_root),
        "--output",
        str(finetuned_validation_report),
    ]) == 0
    assert evaluate_main([
        "metrics",
        "--summary-json",
        str(baseline_summary),
        "--split",
        "test",
        "--checkpoint",
        "ckpts/TrackNet_best.pt",
        "--dataset-root",
        str(adapted_root),
        "--output",
        str(baseline_test_report),
    ]) == 0
    assert evaluate_main([
        "metrics",
        "--summary-json",
        str(finetuned_summary),
        "--split",
        "test",
        "--checkpoint",
        "run/TrackNet_best.pt",
        "--dataset-root",
        str(adapted_root),
        "--output",
        str(finetuned_test_report),
    ]) == 0
    assert evaluate_main([
        "select-best",
        "--baseline-report",
        str(baseline_validation_report),
        "--candidate-report",
        str(finetuned_validation_report),
        "--output",
        str(selection_report),
    ]) == 0
    assert evaluate_main([
        "compare",
        "--baseline-report",
        str(baseline_test_report),
        "--finetuned-report",
        str(finetuned_test_report),
        "--split",
        "test",
        "--output",
        str(comparison_report),
    ]) == 0
    assert modal_main([
        "--run-id",
        "demo-run",
        "--raw-dataset-root",
        str(raw_root),
        "--volume-root",
        str(tmp_path / "volume"),
        "--pretrained-tracknet",
        "ckpts/TrackNet_best.pt",
        "--baseline-validation-report",
        str(baseline_validation_report),
        "--candidate-validation-report",
        str(finetuned_validation_report),
        "--baseline-test-report",
        str(baseline_test_report),
        "--finetuned-test-report",
        str(finetuned_test_report),
        "--plan-output",
        str(modal_plan),
        "--dry-run",
    ]) == 0

    selection = json.loads(selection_report.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_report.read_text(encoding="utf-8"))
    modal = json.loads(modal_plan.read_text(encoding="utf-8"))
    layout = build_run_layout(tmp_path / "volume", "demo-run")

    assert selection["selected_checkpoint"] == "run/TrackNet_best.pt"
    assert comparison["delta"]["p_at_20"] > 0
    assert modal["layout"]["adapted_data_root"] == str(layout.adapted_data_root)
    assert modal["commands"][0][0:2] == [sys.executable, "TrackNetV3/basketball_data.py"]
    assert modal["commands"][1][0:2] == [sys.executable, "TrackNetV3/visualize_labels.py"]
    assert modal["commands"][2][0:2] == [sys.executable, "TrackNetV3/preprocess.py"]
    # Inline evaluate phase: 4 `evaluate.py metrics` commands now live
    # between train and select-best. Verify their --output paths line up
    # with the report inputs we pre-generated for selection and compare.
    metrics_commands = [
        cmd for cmd in modal["commands"]
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "metrics"]
    ]
    assert len(metrics_commands) == 4
    metrics_outputs = [cmd[cmd.index("--output") + 1] for cmd in metrics_commands]
    assert metrics_outputs == [
        str(baseline_validation_report),
        str(finetuned_validation_report),
        str(baseline_test_report),
        str(finetuned_test_report),
    ]


def test_modal_plan_keeps_raw_dataset_read_only_and_uses_volume_layout(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset-tracknet"
    reports_root = build_run_layout(tmp_path / "modal-volume", "slice-five").reports_root
    baseline_validation_report = reports_root / "baseline-validation-report.json"
    finetuned_validation_report = reports_root / "finetuned-validation-report.json"
    baseline_test_report = reports_root / "baseline-test-report.json"
    finetuned_test_report = reports_root / "finetuned-test-report.json"
    plan_output = reports_root / "modal-plan.json"

    assert modal_main([
        "--run-id",
        "slice-five",
        "--raw-dataset-root",
        str(raw_root),
        "--volume-root",
        str(tmp_path / "modal-volume"),
        "--pretrained-tracknet",
        "ckpts/TrackNet_best.pt",
        "--baseline-validation-report",
        str(baseline_validation_report),
        "--candidate-validation-report",
        str(finetuned_validation_report),
        "--baseline-test-report",
        str(baseline_test_report),
        "--finetuned-test-report",
        str(finetuned_test_report),
        "--plan-output",
        str(plan_output),
        "--dry-run",
    ]) == 0

    plan = json.loads(plan_output.read_text(encoding="utf-8"))

    assert plan["raw_dataset_root"] == str(raw_root)
    assert plan["raw_dataset_read_only"] is True
    assert plan["layout"]["run_root"].endswith("tracknet-finetune/runs/slice-five")
    assert plan["layout"]["visualizations_root"].endswith("visualizations")
    assert plan["commands"][0][-2:] == ["--report", str(Path(plan["layout"]["reports_root"]) / "validation-report.json")]
    assert plan["commands"][1][2:4] == [str(raw_root), str(Path(plan["layout"]["visualizations_root"]))]


def test_modal_dry_run_is_plan_only_when_reports_are_missing(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset-tracknet"
    reports_root = build_run_layout(tmp_path / "modal-volume", "missing-reports").reports_root
    plan_output = reports_root / "modal-plan.json"

    assert modal_main([
        "--run-id",
        "missing-reports",
        "--raw-dataset-root",
        str(raw_root),
        "--volume-root",
        str(tmp_path / "modal-volume"),
        "--pretrained-tracknet",
        "ckpts/TrackNet_best.pt",
        "--baseline-validation-report",
        str(reports_root / "baseline-validation-report.json"),
        "--candidate-validation-report",
        str(reports_root / "finetuned-validation-report.json"),
        "--baseline-test-report",
        str(reports_root / "baseline-test-report.json"),
        "--finetuned-test-report",
        str(reports_root / "finetuned-test-report.json"),
        "--plan-output",
        str(plan_output),
        "--dry-run",
    ]) == 0

    plan = json.loads(plan_output.read_text(encoding="utf-8"))

    assert plan["dry_run"] is True
    assert plan["ready_to_execute"] is False
    assert sorted(item["status"] for item in plan["prerequisites"]) == ["missing", "missing", "missing", "missing"]


def test_modal_real_run_generates_inline_report_paths(tmp_path: Path, monkeypatch) -> None:
    """`run_real` now generates the four report artifacts itself between
    train and select-best, so a non-dry call must NOT fail fast when the
    reports are absent. The plan still records the four outputs under
    the run layout."""
    layout = build_run_layout(tmp_path / "volume", "inline-real")
    plan_output = layout.run_root / "plans" / "modal-plan.json"

    # Stub subprocess.run so the test exercises the wrapper plan path
    # without invoking train.py (which needs GPU/torch).
    def fake_run(command, *args, **kwargs):
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        __import__("modal_tracknet_finetune", fromlist=["subprocess"]).subprocess,
        "run",
        fake_run,
    )

    # Note: the four `--*-report` flags are intentionally omitted so the
    # wrapper falls back to the run layout defaults.
    assert modal_main([
        "--run-id",
        "inline-real",
        "--raw-dataset-root",
        str(tmp_path / "dataset-tracknet"),
        "--volume-root",
        str(tmp_path / "volume"),
        "--pretrained-tracknet",
        "ckpts/TrackNet_best.pt",
        "--plan-output",
        str(plan_output),
    ]) == 0

    payload = json.loads(plan_output.read_text(encoding="utf-8"))
    metrics_commands = [
        cmd for cmd in payload["commands"]
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "metrics"]
    ]
    assert len(metrics_commands) == 4
    metrics_outputs = [cmd[cmd.index("--output") + 1] for cmd in metrics_commands]
    assert metrics_outputs == [
        str(layout.reports_root / "baseline-validation-report.json"),
        str(layout.reports_root / "candidate-validation-report.json"),
        str(layout.reports_root / "baseline-test-report.json"),
        str(layout.reports_root / "finetuned-test-report.json"),
    ]


def test_build_run_layout_sanitizes_run_id_and_keeps_paths_under_runs_root(tmp_path: Path) -> None:
    layout = build_run_layout(tmp_path / "volume", "../shared-run")

    assert normalize_run_id("../shared-run") == "shared-run"
    assert layout.run_root == (tmp_path / "volume" / "tracknet-finetune" / "runs" / "shared-run").resolve()
    assert layout.run_root != (tmp_path / "volume" / "tracknet-finetune" / "shared-run").resolve()


def test_modal_main_rejects_run_ids_that_cannot_be_normalized(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="run_id"):
        modal_main([
            "--run-id",
            "...",
            "--raw-dataset-root",
            str(tmp_path / "dataset-tracknet"),
            "--volume-root",
            str(tmp_path / "modal-volume"),
            "--pretrained-tracknet",
            "ckpts/TrackNet_best.pt",
            "--baseline-validation-report",
            str(tmp_path / "reports" / "baseline.json"),
            "--candidate-validation-report",
            str(tmp_path / "reports" / "finetuned.json"),
            "--baseline-test-report",
            str(tmp_path / "reports" / "baseline-test.json"),
            "--finetuned-test-report",
            str(tmp_path / "reports" / "finetuned-test.json"),
            "--plan-output",
            str(tmp_path / "reports" / "modal-plan.json"),
            "--dry-run",
        ])
