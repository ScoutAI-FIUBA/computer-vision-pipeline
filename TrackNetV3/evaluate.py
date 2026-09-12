from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from basketball_data import read_json_manifest
from finetune import compute_validation_metrics, select_best_checkpoint


DELTA_METRIC_KEYS = (
    "detection_rate",
    "p_at_10",
    "p_at_20",
    "p_at_40",
    "mean_error",
    "median_error",
    "false_positive_rate",
    "precision",
    "recall",
    "f1",
)


def _false_positive_rate(summary: dict) -> float | None:
    fp2 = float(summary.get("FP2", 0.0))
    negative_frames = float(summary.get("TN", 0.0)) + fp2
    if negative_frames <= 0:
        return None
    return fp2 / negative_frames


def _metric_delta(baseline_metrics: dict, finetuned_metrics: dict) -> dict:
    delta = {}
    for key in DELTA_METRIC_KEYS:
        baseline_value = baseline_metrics.get(key)
        finetuned_value = finetuned_metrics.get(key)
        if baseline_value is None or finetuned_value is None:
            continue
        delta[key] = round(float(finetuned_value) - float(baseline_value), 6)
    return delta


def build_metrics_report(
    summary: dict,
    *,
    split: str,
    checkpoint: str,
    dataset_root: str,
    evaluation_mode: str,
) -> dict:
    manifest = read_json_manifest(Path(dataset_root) / "manifest.json")
    metrics = compute_validation_metrics(summary=summary) | {
        "false_positives_visibility_0": float(summary.get("FP2", 0.0)),
        "false_positive_rate": _false_positive_rate(summary),
        "precision": summary.get("precision"),
        "recall": summary.get("recall"),
        "f1": summary.get("f1"),
        "miss_rate": summary.get("miss_rate"),
    }
    return {
        "split": split,
        "checkpoint": checkpoint,
        "dataset_root": dataset_root,
        "dataset_manifest_fingerprint": manifest.get("fingerprint") if manifest else None,
        "evaluation_mode": evaluation_mode,
        "metrics": metrics,
        "raw_summary": summary,
    }


def select_best_checkpoint_report(
    baseline_report: dict,
    candidate_reports: list[dict],
    *,
    detection_rate_guard: float = 0.02,
) -> dict:
    baseline_metrics = baseline_report["metrics"]
    selected_metrics = select_best_checkpoint(
        baseline_metrics,
        [candidate["metrics"] | {"checkpoint": candidate["checkpoint"]} for candidate in candidate_reports],
        detection_rate_guard=detection_rate_guard,
    )
    minimum_detection_rate = float(baseline_metrics["detection_rate"]) - detection_rate_guard
    rejected = [
        candidate["checkpoint"]
        for candidate in candidate_reports
        if float(candidate["metrics"]["detection_rate"]) < minimum_detection_rate
    ]
    return {
        "selection_metric": "p_at_20",
        "detection_rate_guard": detection_rate_guard,
        "baseline_checkpoint": baseline_report["checkpoint"],
        "selected_checkpoint": selected_metrics["checkpoint"],
        "selected_metrics": {key: value for key, value in selected_metrics.items() if key != "checkpoint"},
        "rejected_checkpoints": rejected,
    }


def build_final_comparison_report(baseline_report: dict, finetuned_report: dict, *, split: str | None = None) -> dict:
    resolved_split = split or finetuned_report.get("split") or baseline_report.get("split") or "test"
    _require_report_split(baseline_report, resolved_split, label="baseline")
    _require_report_split(finetuned_report, resolved_split, label="finetuned")
    _require_same_dataset_provenance(baseline_report, finetuned_report)
    baseline_metrics = baseline_report["metrics"]
    finetuned_metrics = finetuned_report["metrics"]
    return {
        "split": resolved_split,
        "dataset_root": baseline_report["dataset_root"],
        "dataset_manifest_fingerprint": baseline_report["dataset_manifest_fingerprint"],
        "baseline": {"checkpoint": baseline_report["checkpoint"], "metrics": baseline_metrics},
        "finetuned": {"checkpoint": finetuned_report["checkpoint"], "metrics": finetuned_metrics},
        "delta": _metric_delta(baseline_metrics, finetuned_metrics),
    }


def _require_report_split(report: dict, expected_split: str, *, label: str) -> None:
    actual_split = report.get("split")
    if actual_split != expected_split:
        raise ValueError(f"{label} report split {actual_split!r} does not match requested {expected_split!r}")


def _require_reports_split(reports: list[dict], expected_split: str, *, label: str) -> None:
    for index, report in enumerate(reports, start=1):
        _require_report_split(report, expected_split, label=f"{label} #{index}")


def _require_same_dataset_provenance(baseline_report: dict, finetuned_report: dict) -> None:
    baseline_root_raw = baseline_report.get("dataset_root")
    finetuned_root_raw = finetuned_report.get("dataset_root")
    baseline_manifest = baseline_report.get("dataset_manifest_fingerprint")
    finetuned_manifest = finetuned_report.get("dataset_manifest_fingerprint")

    if not baseline_root_raw or not finetuned_root_raw:
        raise ValueError("dataset provenance missing: both reports must include dataset_root")
    if not baseline_manifest or not finetuned_manifest:
        raise ValueError("dataset provenance missing: both reports must include dataset_manifest_fingerprint")

    baseline_root = Path(str(baseline_root_raw)).resolve()
    finetuned_root = Path(str(finetuned_root_raw)).resolve()
    if baseline_root != finetuned_root:
        raise ValueError(
            f"dataset provenance mismatch: baseline dataset_root {baseline_report.get('dataset_root')!r} != finetuned dataset_root {finetuned_report.get('dataset_root')!r}"
        )

    if baseline_manifest != finetuned_manifest:
        raise ValueError(
            "dataset provenance mismatch: baseline and finetuned dataset_manifest_fingerprint differ"
        )


def write_json_report(output_path: Path | str, payload: dict) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        Path(temporary_name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary_name, path)
    finally:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
    return path


def write_validation_artifacts(
    reports_root: Path | str,
    *,
    split: str,
    checkpoint: str,
    dataset_root: str,
    evaluation_mode: str,
    summary: dict,
    output_name: str,
) -> Path:
    return write_json_report(
        Path(reports_root) / output_name,
        build_metrics_report(
            summary,
            split=split,
            checkpoint=checkpoint,
            dataset_root=dataset_root,
            evaluation_mode=evaluation_mode,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TrackNet basketball evaluation helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metrics_parser = subparsers.add_parser("metrics")
    metrics_parser.add_argument("--summary-json", type=Path, required=True)
    metrics_parser.add_argument("--split", required=True)
    metrics_parser.add_argument("--checkpoint", required=True)
    metrics_parser.add_argument("--dataset-root", required=True)
    metrics_parser.add_argument("--evaluation-mode", default="weight")
    metrics_parser.add_argument("--output", type=Path, required=True)

    select_parser = subparsers.add_parser("select-best")
    select_parser.add_argument("--baseline-report", type=Path, required=True)
    select_parser.add_argument("--candidate-report", action="append", type=Path, required=True)
    select_parser.add_argument("--detection-rate-guard", type=float, default=0.02)
    select_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline-report", type=Path, required=True)
    compare_parser.add_argument("--finetuned-report", type=Path, required=True)
    compare_parser.add_argument("--split", default="test")
    compare_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "metrics":
        summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
        write_json_report(
            args.output,
            build_metrics_report(
                summary,
                split=args.split,
                checkpoint=args.checkpoint,
                dataset_root=args.dataset_root,
                evaluation_mode=args.evaluation_mode,
            ),
        )
        return 0

    if args.command == "select-best":
        baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        candidate_reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.candidate_report]
        _require_report_split(baseline_report, "validation", label="baseline")
        _require_reports_split(candidate_reports, "validation", label="candidate")
        write_json_report(
            args.output,
            select_best_checkpoint_report(
                baseline_report,
                candidate_reports,
                detection_rate_guard=args.detection_rate_guard,
            ),
        )
        return 0

    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    finetuned_report = json.loads(args.finetuned_report.read_text(encoding="utf-8"))
    write_json_report(args.output, build_final_comparison_report(baseline_report, finetuned_report, split=args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
