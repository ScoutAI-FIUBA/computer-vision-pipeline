from __future__ import annotations

import argparse
import json
from pathlib import Path

from basketball_data import RAW_SPLITS, main as validate_main, read_json_manifest
from evaluate import main as evaluate_main
from preprocess import main as preprocess_main


def run_local_smoke(
    *,
    raw_dataset_root: Path | str,
    run_root: Path | str,
    baseline_summary_json: Path | str,
    splits: tuple[str, ...] | list[str] = ("validation",),
) -> dict:
    raw_root = Path(raw_dataset_root)
    resolved_run_root = Path(run_root)
    reports_root = resolved_run_root / "reports"
    adapted_data_root = resolved_run_root / "adapted-data"
    validation_report = reports_root / "validation-report.json"
    baseline_report = reports_root / "baseline-validation.json"
    requested_splits = tuple(splits or RAW_SPLITS)
    if requested_splits != ("validation",):
        raise ValueError("smoke is validation-only; use splits=('validation',) to keep report names and content truthful")

    validate_args = [str(raw_root), "--report", str(validation_report)]
    preprocess_args = [str(raw_root), "--output-root", str(adapted_data_root)]
    for split in requested_splits:
        validate_args.extend(["--split", split])
        preprocess_args.extend(["--split", split])

    if validate_main(validate_args) != 0:
        raise ValueError(f"validation failed; see {validation_report}")
    if preprocess_main(preprocess_args) != 0:
        raise ValueError("preprocess failed")

    if evaluate_main(
        [
            "metrics",
            "--summary-json",
            str(baseline_summary_json),
            "--split",
            "validation",
            "--checkpoint",
            "baseline-reference",
            "--dataset-root",
            str(adapted_data_root),
            "--evaluation-mode",
            "report-only",
            "--output",
            str(baseline_report),
        ]
    ) != 0:
        raise ValueError("baseline evaluation report generation failed")

    manifest = read_json_manifest(adapted_data_root / "manifest.json") or {}
    return {
        "ok": True,
        "smoke_mode": "non-gpu",
        "baseline_eval_mode": "report-only",
        "requested_splits": list(requested_splits),
        "validation_report": str(validation_report),
        "adapted_data_root": str(adapted_data_root),
        "baseline_report": str(baseline_report),
        "manifest_fingerprint": manifest.get("fingerprint"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small local validation-only smoke path for TrackNet basketball validation, preprocess, and baseline-report plumbing.")
    parser.add_argument("raw_dataset_root", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-summary-json", type=Path, required=True)
    parser.add_argument("--split", action="append", dest="splits", choices=("validation",))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_local_smoke(
        raw_dataset_root=args.raw_dataset_root,
        run_root=args.run_root,
        baseline_summary_json=args.baseline_summary_json,
        splits=tuple(args.splits or ("validation",)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
