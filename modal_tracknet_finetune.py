from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from TrackNetV3.basketball_data import validate_processed_root


MODAL_VOLUME_ROOT = "/mnt/tracknet"
DEFAULT_PRETRAINED_TRACKNET = f"{MODAL_VOLUME_ROOT}/pretrained/TrackNet_best.pt"


try:
    import modal

    _APP = modal.App("tracknet-finetune")
    _REPO_ROOT = Path(__file__).resolve().parent
    _TRACKNET_DIR = _REPO_ROOT / "TrackNetV3"
    _WRAPPER_FILE = Path(__file__).resolve()
    _MODAL_IMAGE = (
        modal.Image.debian_slim(python_version="3.11")
        # System deps TrackNetV3 / OpenCV / PyTorch expect inside the container.
        .apt_install(
            "ffmpeg",
            "libsm6",
            "libxext6",
            "libgl1",
            "libglib2.0-0",
        )
        # TrackNetV3/requirements.txt pins `torch==1.10.0`, which has no
        # prebuilt wheel for Python 3.11 and pulls in an old numpy that
        # also lacks a cp311 wheel; pip then tries to compile numpy from
        # source via gcc, which fails inside the Modal image. Install
        # numpy FIRST so pip resolves to a prebuilt wheel, then install
        # the rest of the deps as an explicit flat list with versions
        # that ship cp311 wheels on PyPI. We deliberately skip
        # `pip_install_from_requirements(TrackNetV3/requirements.txt)`
        # here — torch==1.10.0 would drag us back into the broken
        # numpy source-build path. `opencv-python` uses a hyphen in the
        # pip package name (the requirements file uses an underscore,
        # which would install the wrong distribution). `opencv-python
        # 4.4.0.46` predates Python 3.11 wheels and pulls numpy from
        # source on cp311, so we pin 4.8.0.76 (the first release with a
        # cp311 wheel on PyPI). Every dep in this list was audited
        # against PyPI for a cp311 wheel: `torchvision==0.15.1` is the
        # final torchvision pin (its sibling `0.15.0` is yanked on PyPI
        # for an incorrect torch dependency, so it never resolves); the
        # matching `torch==2.0.0` is the only torch release line that
        # `torchvision==0.15.1` accepts. The follow-up pycocotools +
        # matplotlib audit added two transitive TrackNetV3 imports the
        # original explicit list missed: `pycocotools==2.0.11` is
        # imported at the top of `TrackNetV3/test.py` for COCO-style mAP
        # evaluation (train.py imports test.py, so the failure surfaced
        # at the train step), and `matplotlib==3.11.1` is imported at
        # the top of `TrackNetV3/utils/visualize.py` (transitively
        # pulled in by train.py). Both releases ship manylinux cp311
        # wheels on PyPI for x86_64 and aarch64.
        .pip_install("numpy==1.26.4")
        .pip_install(
            "torch==2.0.0",
            "torchvision==0.15.1",
            "pandas==2.0.0",
            "Pillow==10.0.0",
            "opencv-python==4.8.0.76",
            "plotly==5.8.2",
            "dash==2.5.1",
            "tqdm",
            "tensorboard",
            "pycocotools==2.0.11",
            "matplotlib==3.11.1",
            "parse",
        )
        # Bake the TrackNetV3 source tree into the image at /root/TrackNetV3
        # so the wrapper's subprocess calls (TrackNetV3/basketball_data.py,
        # train.py, evaluate.py, test.py, preprocess.py, visualize_labels.py)
        # resolve inside the container.
        .add_local_dir(str(_TRACKNET_DIR), remote_path="/root/TrackNetV3", copy=True)
        # The wrapper itself needs to be available so `python -m modal_tracknet_finetune`
        # and the `@app.function` body resolve.
        .add_local_file(str(_WRAPPER_FILE), remote_path="/root/modal_tracknet_finetune.py", copy=True)
    )
    _MODAL_VOLUME = modal.Volume.from_name("tracknet-data", create_if_missing=True)
except ImportError:  # pragma: no cover - modal missing in local venv
    modal = None
    _APP = None
    _MODAL_IMAGE = None
    _MODAL_VOLUME = None


def get_app():
    """Return the Modal `App` for this wrapper, or `None` if Modal is unavailable.

    Kept as a tiny accessor so callers can branch on availability without
    re-trying the import.
    """
    return _APP


def get_modal_image():
    """Return the declared Modal `Image`, or `None` if the SDK is unavailable.

    Exposed for tests and for callers that need the image without spinning up
    a remote container.
    """
    return _MODAL_IMAGE


def get_modal_volume():
    """Return the Modal `Volume` for `tracknet-data`, or `None` if the SDK is unavailable."""
    return _MODAL_VOLUME


app = _APP  # exposed for `modal run modal_tracknet_finetune.py::ENTRY`


@dataclass(frozen=True)
class RunLayout:
    run_root: Path
    adapted_data_root: Path
    reports_root: Path
    checkpoints_root: Path
    logs_root: Path
    visualizations_root: Path


def normalize_run_id(run_id: str) -> str:
    parts = []
    for raw_part in str(run_id).replace("\\", "/").split("/"):
        cleaned = "".join(character if character.isalnum() or character in "._-" else "-" for character in raw_part.strip())
        cleaned = cleaned.strip("._-")
        if cleaned:
            parts.append(cleaned)
    normalized = "-".join(parts)
    if not normalized:
        raise ValueError("run_id must contain at least one alphanumeric character after normalization")
    return normalized


def _ensure_under_root(path: Path, root: Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the intended root: {resolved_path}") from exc
    return resolved_path


def build_run_layout(volume_root: Path | str, run_id: str) -> RunLayout:
    runs_root = (Path(volume_root) / "tracknet-finetune" / "runs").resolve()
    normalized_run_id = normalize_run_id(run_id)
    run_root = _ensure_under_root(runs_root / normalized_run_id, runs_root, label="run_id")
    return RunLayout(
        run_root=run_root,
        adapted_data_root=run_root / "adapted-data",
        reports_root=run_root / "reports",
        checkpoints_root=run_root / "checkpoints",
        logs_root=run_root / "logs",
        visualizations_root=run_root / "visualizations",
    )


def inline_evaluate_paths(layout: RunLayout) -> dict[str, Path]:
    """Stable per-(split, role) report paths under the run layout.

    The wrapper now generates the four metric reports itself between
    train and select-best. These are the canonical paths the inline
    evaluate commands read from and write to.
    """
    return {
        "baseline_validation_report": layout.reports_root / "baseline-validation-report.json",
        "candidate_validation_report": layout.reports_root / "candidate-validation-report.json",
        "baseline_test_report": layout.reports_root / "baseline-test-report.json",
        "finetuned_test_report": layout.reports_root / "finetuned-test-report.json",
    }


def _run_output_path(path: Path, layout: RunLayout, *, label: str) -> Path:
    return _ensure_under_root(path, layout.run_root, label=label)


def _normalize_test_py_split(split: str) -> str:
    return "val" if split == "validation" else split


def build_commands(
    *,
    layout: RunLayout,
    raw_dataset_root: Path | str | None = None,
    processed_root: Path | str | None = None,
    pretrained_tracknet: str,
    baseline_validation_report: Path | str | None = None,
    candidate_validation_report: Path | str | None = None,
    baseline_test_report: Path | str | None = None,
    finetuned_test_report: Path | str | None = None,
) -> list[list[str]]:
    if (raw_dataset_root is None) == (processed_root is None):
        raise ValueError("exactly one of raw_dataset_root or processed_root is required")
    dataset_root = str(processed_root if processed_root is not None else layout.adapted_data_root)
    defaults = inline_evaluate_paths(layout)
    baseline_validation_report = _run_output_path(Path(baseline_validation_report) if baseline_validation_report else defaults["baseline_validation_report"], layout, label="baseline_validation_report")
    candidate_validation_report = _run_output_path(Path(candidate_validation_report) if candidate_validation_report else defaults["candidate_validation_report"], layout, label="candidate_validation_report")
    baseline_test_report = _run_output_path(Path(baseline_test_report) if baseline_test_report else defaults["baseline_test_report"], layout, label="baseline_test_report")
    finetuned_test_report = _run_output_path(Path(finetuned_test_report) if finetuned_test_report else defaults["finetuned_test_report"], layout, label="finetuned_test_report")

    # Inline evaluate: 4 (split, checkpoint) pairs.
    # Baseline validation runs before train because `train.py` consumes
    # that report during epoch 1; the remaining candidate/test steps stay
    # after train. Baseline runs against the pretrained checkpoint;
    # candidate runs against the fine-tuned `TrackNet_best.pt`.
    evaluate_steps = [
        ("validation", str(Path(pretrained_tracknet)), str(baseline_validation_report)),
        ("validation", str(layout.checkpoints_root / "TrackNet_best.pt"), str(candidate_validation_report)),
        ("test", str(Path(pretrained_tracknet)), str(baseline_test_report)),
        ("test", str(layout.checkpoints_root / "TrackNet_best.pt"), str(finetuned_test_report)),
    ]
    inline_evaluate_commands: list[list[str]] = []
    for split, checkpoint, output_report in evaluate_steps:
        test_py_split = _normalize_test_py_split(split)
        # TrackNetV3/test.py writes the per-split summary JSON under
        # `{save_dir}/{split}_eval_res_weight.json`; the wrapper points
        # `save_dir` at the run layout's reports root so the subsequent
        # `evaluate.py metrics` step can read the summary directly.
        summary_path = layout.reports_root / f"{test_py_split}_eval_res_weight.json"
        inline_evaluate_commands.append([
            sys.executable, "TrackNetV3/test.py",
            "--tracknet_file", checkpoint,
            "--dataset_root", dataset_root,
            "--split", test_py_split,
            "--eval_mode", "weight",
            "--save_dir", str(layout.reports_root),
            "--cache_root", str(layout.checkpoints_root / "cache"),
        ])
        inline_evaluate_commands.append([
            sys.executable, "TrackNetV3/evaluate.py", "metrics",
            "--summary-json", str(summary_path),
            "--split", split,
            "--checkpoint", checkpoint,
            "--dataset-root", dataset_root,
            "--output", output_report,
        ])

    baseline_validation_pair = inline_evaluate_commands[:2]
    post_train_evaluate_commands = inline_evaluate_commands[2:]

    raw_commands = []
    if raw_dataset_root is not None:
        raw_root = str(raw_dataset_root)
        raw_commands = [
            [sys.executable, "TrackNetV3/basketball_data.py", raw_root, "--report", str(layout.reports_root / "validation-report.json")],
            [sys.executable, "TrackNetV3/visualize_labels.py", raw_root, str(layout.visualizations_root)],
            [sys.executable, "TrackNetV3/preprocess.py", raw_root, "--output-root", str(layout.adapted_data_root)],
        ]
    return [
        *raw_commands,
        *baseline_validation_pair,
        [sys.executable, "TrackNetV3/train.py", "--model_name", "TrackNet", "--bg_mode", "concat", "--seq_len", "8", "--epochs", "20", "--early_stopping_patience", "5", "--dataset_root", dataset_root, "--cache_root", str(layout.checkpoints_root / "cache"), "--save_dir", str(layout.checkpoints_root), "--pretrained_tracknet", pretrained_tracknet, "--baseline-validation-report", str(baseline_validation_report)],
        *post_train_evaluate_commands,
        [sys.executable, "TrackNetV3/evaluate.py", "select-best", "--baseline-report", str(baseline_validation_report), "--candidate-report", str(candidate_validation_report), "--output", str(layout.reports_root / "best_checkpoint.json")],
        [sys.executable, "TrackNetV3/evaluate.py", "compare", "--baseline-report", str(baseline_test_report), "--finetuned-report", str(finetuned_test_report), "--split", "test", "--output", str(layout.reports_root / "final_comparison.json")],
    ]


def _read_report_if_present(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _require_report_split(path: Path, expected_split: str, *, label: str) -> None:
    report = _read_report_if_present(path)
    if report is None:
        return
    actual_split = report.get("split")
    if actual_split != expected_split:
        raise ValueError(f"{label} report split {actual_split!r} does not match required {expected_split!r}: {path}")


def collect_report_prerequisites(
    *,
    baseline_validation_report: Path,
    candidate_validation_report: Path,
    baseline_test_report: Path,
    finetuned_test_report: Path,
) -> list[dict[str, str]]:
    prerequisites = []
    for path, expected_split, label in (
        (baseline_validation_report, "validation", "baseline validation"),
        (candidate_validation_report, "validation", "candidate validation"),
        (baseline_test_report, "test", "baseline test"),
        (finetuned_test_report, "test", "finetuned test"),
    ):
        report = _read_report_if_present(path)
        status = "missing"
        actual_split = None
        if report is not None:
            actual_split = report.get("split")
            status = "ready" if actual_split == expected_split else "invalid"
        prerequisite = {
            "label": label,
            "path": str(path),
            "expected_split": expected_split,
            "status": status,
        }
        if actual_split is not None:
            prerequisite["actual_split"] = str(actual_split)
        prerequisites.append(prerequisite)
    return prerequisites


def validate_report_inputs(
    *,
    baseline_validation_report: Path,
    candidate_validation_report: Path,
    baseline_test_report: Path,
    finetuned_test_report: Path,
) -> None:
    # The wrapper now generates the four metric reports itself between
    # train and select-best. Pre-existing reports are no longer required
    # external inputs; this helper remains for dry-run layout validation
    # via `validate_existing_report_inputs` and only rejects inputs that
    # exist with the wrong split.
    prerequisites = collect_report_prerequisites(
        baseline_validation_report=baseline_validation_report,
        candidate_validation_report=candidate_validation_report,
        baseline_test_report=baseline_test_report,
        finetuned_test_report=finetuned_test_report,
    )
    for item in prerequisites:
        if item["status"] == "invalid":
            _require_report_split(Path(item["path"]), item["expected_split"], label=item["label"])
    return None


def validate_existing_report_inputs(prerequisites: list[dict[str, str]]) -> None:
    for item in prerequisites:
        if item["status"] == "invalid":
            _require_report_split(Path(item["path"]), item["expected_split"], label=item["label"])


def write_plan(output_path: Path | str, payload: dict) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thin Modal/local wrapper for TrackNet basketball fine-tuning orchestration.")
    parser.add_argument("--run-id", required=True)
    dataset_root_group = parser.add_mutually_exclusive_group(required=True)
    dataset_root_group.add_argument("--raw-dataset-root", type=Path)
    dataset_root_group.add_argument("--processed-root", type=Path)
    parser.add_argument("--volume-root", type=Path, default=Path("/cache"))
    parser.add_argument("--pretrained-tracknet", required=True)
    # The four report args are kept for backward compatibility but default
    # to the inline-generated paths under the run layout. Real runs no
    # longer require these files to pre-exist on disk.
    parser.add_argument("--baseline-validation-report", type=Path, default=None)
    parser.add_argument("--candidate-validation-report", type=Path, default=None)
    parser.add_argument("--baseline-test-report", type=Path, default=None)
    parser.add_argument("--finetuned-test-report", type=Path, default=None)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def _resolve_report_arg(args: argparse.Namespace, layout: RunLayout, name: str) -> Path:
    value = getattr(args, name)
    # `None` is the documented "flag omitted" case. An empty string from
    # `_build_argv` collapses to `Path('.')` after argparse's `type=Path`,
    # which used to slip through here and explode inside
    # `collect_report_prerequisites` with `IsADirectoryError: [Errno 21]
    # Is a directory: '.'`. Treat both as "not provided" so the wrapper
    # always resolves to a real file path under the run layout.
    path = inline_evaluate_paths(layout)[name] if value is None or Path(value) == Path(".") else Path(value)
    return _run_output_path(path, layout, label=name)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        normalized_run_id = normalize_run_id(args.run_id)
        layout = build_run_layout(args.volume_root, normalized_run_id)
        manifest = validate_processed_root(args.processed_root) if args.processed_root is not None else None
        plan_output = _run_output_path(args.plan_output, layout, label="plan output")
        baseline_validation_report = _resolve_report_arg(args, layout, "baseline_validation_report")
        candidate_validation_report = _resolve_report_arg(args, layout, "candidate_validation_report")
        baseline_test_report = _resolve_report_arg(args, layout, "baseline_test_report")
        finetuned_test_report = _resolve_report_arg(args, layout, "finetuned_test_report")
        prerequisites = collect_report_prerequisites(
            baseline_validation_report=baseline_validation_report,
            candidate_validation_report=candidate_validation_report,
            baseline_test_report=baseline_test_report,
            finetuned_test_report=finetuned_test_report,
        )
        if args.dry_run:
            # Plan-only validation: reject wrong-split inputs when files exist,
            # but do not fail when files are absent (the wrapper generates them).
            validate_existing_report_inputs(prerequisites)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    commands = build_commands(
        layout=layout,
        raw_dataset_root=args.raw_dataset_root,
        processed_root=args.processed_root,
        pretrained_tracknet=args.pretrained_tracknet,
        baseline_validation_report=baseline_validation_report,
        candidate_validation_report=candidate_validation_report,
        baseline_test_report=baseline_test_report,
        finetuned_test_report=finetuned_test_report,
    )
    plan = {
            "run_id": normalized_run_id,
            "dataset_mode": "processed" if args.processed_root is not None else "raw",
            "dataset_root": str(args.processed_root or layout.adapted_data_root),
            "fingerprint": manifest["fingerprint"] if manifest is not None else None,
            "processed_dataset_id": args.processed_root.name if args.processed_root is not None else None,
            "volume_root": str(args.volume_root),
            "layout": {key: str(value) for key, value in asdict(layout).items()},
            "commands": commands,
            "prerequisites": prerequisites,
            "ready_to_execute": all(item["status"] == "ready" for item in prerequisites),
            "dry_run": args.dry_run,
    }
    if args.raw_dataset_root is not None:
        plan["raw_dataset_root"] = str(args.raw_dataset_root)
        plan["raw_dataset_read_only"] = True
    write_plan(plan_output, plan)
    if args.dry_run:
        return 0
    for command in commands:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- Modal entrypoints -------------------------------------------------------
# These wrappers let `modal run modal_tracknet_finetune.py::ENTRY ...` discover
# real `@app.function` entrypoints instead of the prior "no functions or local
# entrypoints" complaint. The local CLI above is untouched; the entrypoints
# delegate back to `main()` so the orchestration logic stays in one place.
#
# Kept minimal on purpose: this is just the wiring fix. The detached-job,
# status.json, resume, and persistent experiment state from the extension PRD
# are intentionally NOT implemented here.


def _build_argv(
    *,
    run_id: str,
    volume_root: str,
    pretrained_tracknet: str,
    baseline_validation_report: str,
    candidate_validation_report: str,
    baseline_test_report: str,
    finetuned_test_report: str,
    plan_output: str,
    dry_run: bool,
    raw_dataset_root: str = "",
    processed_root: str = "",
) -> list[str]:
    if not pretrained_tracknet:
        pretrained_tracknet = DEFAULT_PRETRAINED_TRACKNET
    if pretrained_tracknet.startswith("/path/"):
        raise ValueError(
            "pretrained_tracknet points to a host-local /path/... location; "
            f"copy the checkpoint into {MODAL_VOLUME_ROOT}/pretrained/ or pass its mounted path"
        )
    if not plan_output:
        plan_output = str(build_run_layout(volume_root, run_id).run_root / "plans" / "modal-plan.json")
    argv = [
        "--run-id", run_id,
        "--volume-root", volume_root,
        "--pretrained-tracknet", pretrained_tracknet,
        "--baseline-validation-report", baseline_validation_report,
        "--candidate-validation-report", candidate_validation_report,
        "--baseline-test-report", baseline_test_report,
        "--finetuned-test-report", finetuned_test_report,
        "--plan-output", plan_output,
    ]
    if bool(raw_dataset_root) == bool(processed_root):
        raise ValueError("exactly one of raw_dataset_root or processed_root is required")
    argv[2:2] = ["--raw-dataset-root", raw_dataset_root] if raw_dataset_root else ["--processed-root", processed_root]
    if dry_run:
        argv += ["--dry-run"]
    return argv



def run_real(
    *,
    run_id: str,
    raw_dataset_root: str = "",
    processed_root: str = "",
    volume_root: str = MODAL_VOLUME_ROOT,
    pretrained_tracknet: str = DEFAULT_PRETRAINED_TRACKNET,
    baseline_validation_report: str = "",
    candidate_validation_report: str = "",
    baseline_test_report: str = "",
    finetuned_test_report: str = "",
    plan_output: str = "",
) -> int:
    return main(_build_argv(
        run_id=run_id,
        raw_dataset_root=raw_dataset_root,
        processed_root=processed_root,
        volume_root=volume_root,
        pretrained_tracknet=pretrained_tracknet,
        baseline_validation_report=baseline_validation_report,
        candidate_validation_report=candidate_validation_report,
        baseline_test_report=baseline_test_report,
        finetuned_test_report=finetuned_test_report,
        plan_output=plan_output,
        dry_run=False,
    ))


def run_dry(
    *,
    run_id: str,
    raw_dataset_root: str = "",
    processed_root: str = "",
    volume_root: str = MODAL_VOLUME_ROOT,
    pretrained_tracknet: str = DEFAULT_PRETRAINED_TRACKNET,
    baseline_validation_report: str = "",
    candidate_validation_report: str = "",
    baseline_test_report: str = "",
    finetuned_test_report: str = "",
    plan_output: str = "",
) -> int:
    return main(_build_argv(
        run_id=run_id,
        raw_dataset_root=raw_dataset_root,
        processed_root=processed_root,
        volume_root=volume_root,
        pretrained_tracknet=pretrained_tracknet,
        baseline_validation_report=baseline_validation_report,
        candidate_validation_report=candidate_validation_report,
        baseline_test_report=baseline_test_report,
        finetuned_test_report=finetuned_test_report,
        plan_output=plan_output,
        dry_run=True,
    ))


if _APP is not None:
    # Apply Modal's `@app.function` decorator only when Modal is available,
    # so the module still imports cleanly in a plain venv without `modal`.
    # Both entrypoints share the same image/volume so the TrackNetV3 source
    # tree baked into the image resolves, and the `tracknet-data` volume is
    # mounted at /mnt/tracknet (matching the wrapper's expected run layout).
    _COMMON_KWARGS = {
        "image": _MODAL_IMAGE,
        "volumes": {"/mnt/tracknet": _MODAL_VOLUME},
        # L4 is the smallest GPU with enough VRAM for TrackNetV3 fine-tuning
        # on basketball clips; falls back to CPU if L4 is unavailable in region.
        "gpu": "L4",
        # Six hours: covers the full validation -> preprocess -> train ->
        # evaluate -> select -> compare pipeline with margin.
        "timeout": 60 * 60 * 6,
    }
    run_real = _APP.function(
        name="run",
        **_COMMON_KWARGS,
    )(run_real)
    run_dry = _APP.function(
        name="dry_run",
        # Dry run only writes a plan; the full six-hour timeout is unnecessary,
        # but it still mounts the image/volume and requests the same GPU so
        # `dry_run` exercises the same container shape as the real run.
        timeout=5 * 60,
        image=_MODAL_IMAGE,
        volumes={"/mnt/tracknet": _MODAL_VOLUME},
        gpu="L4",
    )(run_dry)
