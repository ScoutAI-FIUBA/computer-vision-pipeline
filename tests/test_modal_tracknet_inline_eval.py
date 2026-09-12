"""Focused coverage for the inline-evaluate patch on `modal_tracknet_finetune`.

Follow-up patch on top of the archived
`tracknet-basketball-modal-finetuning` change. Proves only the new
behavior the patch adds:

- `build_commands` keeps exactly four `evaluate.py metrics` steps while
  moving the baseline-validation pair before the train step.
- The four report paths default to the run layout under
  `reports/{baseline,candidate}-{validation,test}-report.json`.
- The wrapper no longer raises ``missing prerequisite report inputs``
  when `run_real` is called without the four pre-existing reports.
- `run_real` (non-dry) runs the inline-evaluate command sequence and
  writes the plan without requiring report inputs.
- `run_dry` keeps validating the layout and writes the plan when
  reports already exist.
- The four report flags remain accepted for backward compatibility.
- When the four `--*-report` flags are NOT passed (or `_build_argv`
  forwards empty strings for them), the resolved values point to
  files (not directories) inside the run layout, and `run_real` does
  not raise `IsADirectoryError` when calling
  `collect_report_prerequisites`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import modal_tracknet_finetune as wrapper  # noqa: E402
from modal_tracknet_finetune import (  # noqa: E402
    RunLayout,
    build_commands,
    build_run_layout,
    inline_evaluate_paths,
    main as modal_main,
)


def _metrics_command(commands: list[list[str]]) -> list[list[str]]:
    """Return only the `evaluate.py metrics` commands from the plan."""
    return [
        cmd for cmd in commands
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "metrics"]
    ]


def _test_py_commands(commands: list[list[str]]) -> list[list[str]]:
    """Return only the upstream `TrackNetV3/test.py` commands."""
    return [
        cmd for cmd in commands
        if cmd[:2] == [sys.executable, "TrackNetV3/test.py"]
    ]


def test_build_commands_keeps_four_evaluate_metrics_steps_in_stable_order(tmp_path: Path) -> None:
    layout = build_run_layout(tmp_path / "volume", "inline-eval")
    raw_root = tmp_path / "dataset-tracknet"
    pretrained = str(tmp_path / "TrackNet_best.pt")

    commands = build_commands(
        layout=layout,
        raw_dataset_root=raw_root,
        pretrained_tracknet=pretrained,
        baseline_validation_report=layout.reports_root / "baseline-validation-report.json",
        candidate_validation_report=layout.reports_root / "candidate-validation-report.json",
        baseline_test_report=layout.reports_root / "baseline-test-report.json",
        finetuned_test_report=layout.reports_root / "finetuned-test-report.json",
    )

    # Phase ordering invariants: one baseline-validation metrics step now
    # runs before train; the remaining three metrics steps still sit after
    # train and before select-best/compare.
    train_index = next(
        index for index, cmd in enumerate(commands)
        if cmd[:2] == [sys.executable, "TrackNetV3/train.py"]
    )
    select_best_index = next(
        index for index, cmd in enumerate(commands)
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "select-best"]
    )
    compare_index = next(
        index for index, cmd in enumerate(commands)
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "compare"]
    )
    metrics = _metrics_command(commands)

    assert len(metrics) == 4, (
        "wrapper must insert exactly four `evaluate.py metrics` steps"
    )
    assert train_index < select_best_index < compare_index
    metric_indexes = [commands.index(cmd) for cmd in metrics]
    assert metric_indexes[0] < train_index
    assert metric_indexes[1:] == sorted(metric_indexes[1:])
    for metric_index in metric_indexes[1:]:
        assert train_index < metric_index < select_best_index

    # Each metrics command must pass the full set of required args.
    for cmd in metrics:
        for required in ("--summary-json", "--split", "--checkpoint", "--dataset-root", "--output"):
            assert required in cmd, f"missing {required} in metrics command: {cmd}"

    # The four metrics steps must match the expected (split, checkpoint, output)
    # triples for baseline/candidate on validation/test.
    expected_split_output = [
        ("validation", str(layout.reports_root / "baseline-validation-report.json"), str(Path(pretrained))),
        ("validation", str(layout.reports_root / "candidate-validation-report.json"), str(layout.checkpoints_root / "TrackNet_best.pt")),
        ("test", str(layout.reports_root / "baseline-test-report.json"), str(Path(pretrained))),
        ("test", str(layout.reports_root / "finetuned-test-report.json"), str(layout.checkpoints_root / "TrackNet_best.pt")),
    ]
    actual_split_output = []
    for cmd in metrics:
        split_index = cmd.index("--split") + 1
        output_index = cmd.index("--output") + 1
        checkpoint_index = cmd.index("--checkpoint") + 1
        actual_split_output.append(
            (cmd[split_index], cmd[output_index], cmd[checkpoint_index])
        )
    assert actual_split_output == expected_split_output

    # select-best and compare must read the inline-generated reports.
    select_best_cmd = commands[select_best_index]
    assert str(layout.reports_root / "baseline-validation-report.json") in select_best_cmd
    assert str(layout.reports_root / "candidate-validation-report.json") in select_best_cmd

    compare_cmd = commands[compare_index]
    assert str(layout.reports_root / "baseline-test-report.json") in compare_cmd
    assert str(layout.reports_root / "finetuned-test-report.json") in compare_cmd


def test_build_commands_moves_baseline_validation_pair_before_train_only(tmp_path: Path) -> None:
    """The baseline-validation summary+metrics pair must run before train.

    This follow-up fixes the runtime hole where `train.py` needs the
    baseline validation report during epoch 1, so the wrapper must place
    that pair before the `train.py` command while leaving the rest of the
    post-train evaluation tail intact.
    """
    layout = build_run_layout(tmp_path / "volume", "baseline-before-train")
    pretrained = str(tmp_path / "TrackNet_best.pt")

    commands = build_commands(
        layout=layout,
        raw_dataset_root=tmp_path / "dataset-tracknet",
        pretrained_tracknet=pretrained,
    )

    train_index = next(
        index for index, cmd in enumerate(commands)
        if cmd[:2] == [sys.executable, "TrackNetV3/train.py"]
    )
    baseline_summary_index = next(
        index for index, cmd in enumerate(commands)
        if cmd[:2] == [sys.executable, "TrackNetV3/test.py"]
        and cmd[cmd.index("--split") + 1] == "val"
        and cmd[cmd.index("--tracknet_file") + 1] == pretrained
    )
    baseline_metrics_index = next(
        index for index, cmd in enumerate(commands)
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "metrics"]
        and cmd[cmd.index("--split") + 1] == "validation"
        and cmd[cmd.index("--checkpoint") + 1] == pretrained
        and cmd[cmd.index("--output") + 1] == str(layout.reports_root / "baseline-validation-report.json")
    )
    candidate_validation_index = next(
        index for index, cmd in enumerate(commands)
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "metrics"]
        and cmd[cmd.index("--output") + 1] == str(layout.reports_root / "candidate-validation-report.json")
    )
    baseline_test_index = next(
        index for index, cmd in enumerate(commands)
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "metrics"]
        and cmd[cmd.index("--output") + 1] == str(layout.reports_root / "baseline-test-report.json")
    )
    compare_index = next(
        index for index, cmd in enumerate(commands)
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "compare"]
    )

    assert len(commands) == 14
    assert baseline_summary_index < baseline_metrics_index < train_index
    assert train_index < candidate_validation_index < baseline_test_index < compare_index


def test_build_commands_normalizes_only_test_py_validation_split(tmp_path: Path) -> None:
    """`TrackNetV3/test.py` must receive `val`, while logical report semantics stay put."""
    layout = build_run_layout(tmp_path / "volume", "split-normalization")
    pretrained = str(tmp_path / "TrackNet_best.pt")

    commands = build_commands(
        layout=layout,
        raw_dataset_root=tmp_path / "dataset-tracknet",
        pretrained_tracknet=pretrained,
    )

    test_py_commands = _test_py_commands(commands)
    assert len(commands) == 14
    assert len(test_py_commands) == 4
    assert [cmd[cmd.index("--split") + 1] for cmd in test_py_commands] == [
        "val",
        "val",
        "test",
        "test",
    ]
    assert [cmd[cmd.index("--tracknet_file") + 1] for cmd in test_py_commands] == [
        pretrained,
        str(layout.checkpoints_root / "TrackNet_best.pt"),
        pretrained,
        str(layout.checkpoints_root / "TrackNet_best.pt"),
    ]

    metrics = _metrics_command(commands)
    assert [cmd[cmd.index("--split") + 1] for cmd in metrics] == [
        "validation",
        "validation",
        "test",
        "test",
    ]
    assert [Path(cmd[cmd.index("--summary-json") + 1]).name for cmd in metrics] == [
        "val_eval_res_weight.json",
        "val_eval_res_weight.json",
        "test_eval_res_weight.json",
        "test_eval_res_weight.json",
    ]


def test_build_commands_passes_adapted_dataset_root_to_every_test_py_call(tmp_path: Path) -> None:
    layout = build_run_layout(tmp_path / "volume", "dataset-root-plumbing")

    commands = build_commands(
        layout=layout,
        raw_dataset_root=tmp_path / "dataset-tracknet",
        pretrained_tracknet=str(tmp_path / "TrackNet_best.pt"),
    )

    test_py_commands = _test_py_commands(commands)
    assert len(test_py_commands) == 4
    assert all("--dataset_root" in cmd for cmd in test_py_commands)
    assert [cmd[cmd.index("--dataset_root") + 1] for cmd in test_py_commands] == [
        str(layout.adapted_data_root),
        str(layout.adapted_data_root),
        str(layout.adapted_data_root),
        str(layout.adapted_data_root),
    ]


def test_build_commands_reports_default_to_run_layout(tmp_path: Path) -> None:
    """The wrapper no longer requires external report inputs.

    When the four `--*-report` flags are not passed, the inline evaluate
    steps must still emit the same four outputs under the run layout.
    """
    layout = build_run_layout(tmp_path / "volume", "defaults-test")
    raw_root = tmp_path / "dataset-tracknet"
    pretrained = str(tmp_path / "TrackNet_best.pt")

    # The current wrapper still accepts all four report args. We pass the
    # run-layout paths to keep the existing contract intact while proving
    # the inserted steps land at the run layout.
    commands = build_commands(
        layout=layout,
        raw_dataset_root=raw_root,
        pretrained_tracknet=pretrained,
        baseline_validation_report=layout.reports_root / "baseline-validation-report.json",
        candidate_validation_report=layout.reports_root / "candidate-validation-report.json",
        baseline_test_report=layout.reports_root / "baseline-test-report.json",
        finetuned_test_report=layout.reports_root / "finetuned-test-report.json",
    )

    metrics = _metrics_command(commands)
    inline_outputs = {cmd[cmd.index("--output") + 1] for cmd in metrics}
    expected = {
        str(layout.reports_root / "baseline-validation-report.json"),
        str(layout.reports_root / "candidate-validation-report.json"),
        str(layout.reports_root / "baseline-test-report.json"),
        str(layout.reports_root / "finetuned-test-report.json"),
    }
    assert inline_outputs == expected


def test_modal_real_run_no_longer_raises_on_missing_report_inputs(tmp_path: Path, monkeypatch) -> None:
    """`run_real` (non-dry) must NOT fail-fast on missing report inputs.

    The wrapper now generates the four report artifacts itself between
    train and select-best. Pre-existing report files are no longer a
    prerequisite for a real run.
    """
    plan_output = build_run_layout(tmp_path / "modal-volume", "inline-eval-real").run_root / "plans" / "modal-plan.json"
    plan_output.parent.mkdir(parents=True, exist_ok=True)

    # Stub `subprocess.run` so the real-run path doesn't try to invoke
    # train.py (which needs GPU/torch). The wrapper only writes the plan
    # and reports the command sequence; this test is about plan shape, not
    # subprocess execution.
    invoked_commands: list[list[str]] = []

    def fake_run(command, *args, **kwargs):
        invoked_commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    result = modal_main([
        "--run-id",
        "inline-eval-real",
        "--raw-dataset-root",
        str(tmp_path / "dataset-tracknet"),
        "--volume-root",
        str(tmp_path / "modal-volume"),
        "--pretrained-tracknet",
        str(tmp_path / "TrackNet_best.pt"),
        "--baseline-validation-report",
        str(build_run_layout(tmp_path / "modal-volume", "inline-eval-real").reports_root / "absent-baseline-val.json"),
        "--candidate-validation-report",
        str(build_run_layout(tmp_path / "modal-volume", "inline-eval-real").reports_root / "absent-candidate-val.json"),
        "--baseline-test-report",
        str(build_run_layout(tmp_path / "modal-volume", "inline-eval-real").reports_root / "absent-baseline-test.json"),
        "--finetuned-test-report",
        str(build_run_layout(tmp_path / "modal-volume", "inline-eval-real").reports_root / "absent-finetuned-test.json"),
        "--plan-output",
        str(plan_output),
    ])

    assert result == 0
    assert plan_output.exists()
    payload = json.loads(plan_output.read_text(encoding="utf-8"))
    assert payload["dry_run"] is False
    metrics_count = sum(
        1 for cmd in payload["commands"]
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "metrics"]
    )
    assert metrics_count == 4
    # And the real run actually attempted every command in the plan.
    assert len(invoked_commands) == len(payload["commands"])


def test_modal_dry_run_still_writes_plan_when_reports_exist(tmp_path: Path) -> None:
    """Dry-run remains plan-only and writes the plan when inputs are valid."""
    layout = build_run_layout(tmp_path / "volume", "inline-eval-dry")
    plan_output = layout.run_root / "plans" / "modal-plan.json"

    # Pre-write the four reports so dry-run can validate the layout.
    for label, split in (
        ("baseline-validation-report", "validation"),
        ("candidate-validation-report", "validation"),
        ("baseline-test-report", "test"),
        ("finetuned-test-report", "test"),
    ):
        report_path = layout.reports_root / f"{label}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"split": split}), encoding="utf-8")

    result = modal_main([
        "--run-id",
        "inline-eval-dry",
        "--raw-dataset-root",
        str(tmp_path / "dataset-tracknet"),
        "--volume-root",
        str(tmp_path / "volume"),
        "--pretrained-tracknet",
        str(tmp_path / "TrackNet_best.pt"),
        "--baseline-validation-report",
        str(layout.reports_root / "baseline-validation-report.json"),
        "--candidate-validation-report",
        str(layout.reports_root / "candidate-validation-report.json"),
        "--baseline-test-report",
        str(layout.reports_root / "baseline-test-report.json"),
        "--finetuned-test-report",
        str(layout.reports_root / "finetuned-test-report.json"),
        "--plan-output",
        str(plan_output),
        "--dry-run",
    ])

    assert result == 0
    assert plan_output.exists()
    payload = json.loads(plan_output.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    metrics_count = sum(
        1 for cmd in payload["commands"]
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "metrics"]
    )
    assert metrics_count == 4
    # When all four pre-existing reports are present, dry-run reports ready.
    assert payload["ready_to_execute"] is True


def test_modal_dry_run_rejects_wrong_split_in_existing_report(tmp_path: Path) -> None:
    """Dry-run still rejects reports that declare the wrong split."""
    layout = build_run_layout(tmp_path / "volume", "wrong-split-dry")
    plan_output = layout.run_root / "plans" / "modal-plan.json"

    bad_validation = layout.reports_root / "baseline-validation-report.json"
    bad_validation.parent.mkdir(parents=True, exist_ok=True)
    bad_validation.write_text(json.dumps({"split": "test"}), encoding="utf-8")

    with pytest.raises(SystemExit, match="split"):
        modal_main([
            "--run-id",
            "wrong-split-dry",
            "--raw-dataset-root",
            str(tmp_path / "dataset-tracknet"),
            "--volume-root",
            str(tmp_path / "volume"),
            "--pretrained-tracknet",
            str(tmp_path / "TrackNet_best.pt"),
            "--baseline-validation-report",
            str(bad_validation),
            "--candidate-validation-report",
            str(layout.reports_root / "candidate-validation-report.json"),
            "--baseline-test-report",
            str(layout.reports_root / "baseline-test-report.json"),
            "--finetuned-test-report",
            str(layout.reports_root / "finetuned-test-report.json"),
            "--plan-output",
            str(plan_output),
            "--dry-run",
        ])


def test_modal_wiring_entrypoints_still_register_inline_eval_layout(tmp_path: Path) -> None:
    """The Modal entrypoints (run/dry_run) continue to expose the
    inline-evaluate wrapper without raising on missing inputs."""
    dry_fn = wrapper.app.registered_functions["dry_run"].get_raw_f()
    plan_output = build_run_layout(tmp_path, "inline-eval-wiring").run_root / "plans" / "modal-plan.json"

    result = dry_fn(
        run_id="inline-eval-wiring",
        raw_dataset_root=str(tmp_path / "dataset-tracknet"),
        volume_root=str(tmp_path),
        pretrained_tracknet=str(tmp_path / "TrackNet_best.pt"),
        baseline_validation_report=str(build_run_layout(tmp_path, "inline-eval-wiring").reports_root / "absent-baseline-val.json"),
        candidate_validation_report=str(build_run_layout(tmp_path, "inline-eval-wiring").reports_root / "absent-candidate-val.json"),
        baseline_test_report=str(build_run_layout(tmp_path, "inline-eval-wiring").reports_root / "absent-baseline-test.json"),
        finetuned_test_report=str(build_run_layout(tmp_path, "inline-eval-wiring").reports_root / "absent-finetuned-test.json"),
        plan_output=str(plan_output),
    )

    assert result == 0
    assert plan_output.exists()
    payload = json.loads(plan_output.read_text(encoding="utf-8"))
    metrics_count = sum(
        1 for cmd in payload["commands"]
        if cmd[:3] == [sys.executable, "TrackNetV3/evaluate.py", "metrics"]
    )
    assert metrics_count == 4


def test_build_commands_with_no_report_overrides_uses_run_layout_for_every_metric(tmp_path: Path) -> None:
    """Triangulation: calling `build_commands` with `None` for all four
    report flags must still produce exactly the four inline evaluate
    steps with `--output` paths under the run layout, never external
    paths the user did not provide."""
    layout = build_run_layout(tmp_path / "volume", "triangulate-defaults")
    raw_root = tmp_path / "dataset-tracknet"
    pretrained = str(tmp_path / "TrackNet_best.pt")

    commands = build_commands(
        layout=layout,
        raw_dataset_root=raw_root,
        pretrained_tracknet=pretrained,
    )

    metrics = _metrics_command(commands)
    assert len(metrics) == 4
    expected_outputs = [
        str(layout.reports_root / "baseline-validation-report.json"),
        str(layout.reports_root / "candidate-validation-report.json"),
        str(layout.reports_root / "baseline-test-report.json"),
        str(layout.reports_root / "finetuned-test-report.json"),
    ]
    actual_outputs = [cmd[cmd.index("--output") + 1] for cmd in metrics]
    assert actual_outputs == expected_outputs


def test_build_commands_with_user_overrides_honors_user_paths(tmp_path: Path) -> None:
    """Triangulation: when the user explicitly passes the four report
    flags, `build_commands` must route the inline evaluate outputs to
    the user-provided paths (backward compatibility)."""
    layout = build_run_layout(tmp_path / "volume", "triangulate-overrides")
    raw_root = tmp_path / "dataset-tracknet"
    pretrained = str(tmp_path / "TrackNet_best.pt")

    user_baseline_validation = layout.reports_root / "user-baseline-validation.json"
    user_candidate_validation = layout.reports_root / "user-candidate-validation.json"
    user_baseline_test = layout.reports_root / "user-baseline-test.json"
    user_finetuned_test = layout.reports_root / "user-finetuned-test.json"

    commands = build_commands(
        layout=layout,
        raw_dataset_root=raw_root,
        pretrained_tracknet=pretrained,
        baseline_validation_report=user_baseline_validation,
        candidate_validation_report=user_candidate_validation,
        baseline_test_report=user_baseline_test,
        finetuned_test_report=user_finetuned_test,
    )

    metrics = _metrics_command(commands)
    actual_outputs = [cmd[cmd.index("--output") + 1] for cmd in metrics]
    assert actual_outputs == [
        str(user_baseline_validation),
        str(user_candidate_validation),
        str(user_baseline_test),
        str(user_finetuned_test),
    ]


def test_resolve_report_arg_treats_dot_path_as_unset(tmp_path: Path) -> None:
    """`_resolve_report_arg` must NOT return `'.'` when the underlying
    CLI value collapsed to the current directory.

    Background: when `_build_argv` forwards an empty string for one of
    the four `--*-report` flags, argparse's `type=Path` converts it to
    `PosixPath('.')`. The wrapper used to treat that as a user-supplied
    path, which made `collect_report_prerequisites` blow up with
    `IsADirectoryError` trying to read `.` as JSON.

    After the fix, `_resolve_report_arg` must detect that the value
    collapses to the run layout instead and return the per-run report
    path. The returned path must be a file (not a directory) and must
    sit under `layout.reports_root`.
    """
    layout = build_run_layout(tmp_path / "volume", "dot-paths-test")

    # Simulate what argparse produces when `_build_argv` forwards `""`.
    fake_args = SimpleNamespace(
        baseline_validation_report=Path("."),
        candidate_validation_report=Path("."),
        baseline_test_report=Path("."),
        finetuned_test_report=Path("."),
    )

    resolved = {
        "baseline_validation_report": wrapper._resolve_report_arg(
            fake_args, layout, "baseline_validation_report"
        ),
        "candidate_validation_report": wrapper._resolve_report_arg(
            fake_args, layout, "candidate_validation_report"
        ),
        "baseline_test_report": wrapper._resolve_report_arg(
            fake_args, layout, "baseline_test_report"
        ),
        "finetuned_test_report": wrapper._resolve_report_arg(
            fake_args, layout, "finetuned_test_report"
        ),
    }

    expected = inline_evaluate_paths(layout)
    for key, path in resolved.items():
        assert path == expected[key], (
            f"when the CLI arg collapsed to '.', {key!r} must resolve "
            f"to the run-layout default {expected[key]!r}, got {path!r}"
        )
        # The resolved path must point at a file inside the run layout,
        # never at the current directory.
        assert str(path) != "."
        assert str(path).startswith(str(layout.reports_root) + "/") or \
            str(path) == str(layout.reports_root / expected_default_name(key))
        # And it must be a regular-file target, not a directory that
        # already exists on disk.
        assert path.parent == layout.reports_root
        assert path.suffix == ".json"


def expected_default_name(key: str) -> str:
    return {
        "baseline_validation_report": "baseline-validation-report.json",
        "candidate_validation_report": "candidate-validation-report.json",
        "baseline_test_report": "baseline-test-report.json",
        "finetuned_test_report": "finetuned-test-report.json",
    }[key]


def test_collect_report_prerequisites_does_not_raise_when_args_collapse_to_dot(
    tmp_path: Path,
) -> None:
    """`collect_report_prerequisites` is the function that originally
    raised `IsADirectoryError: [Errno 21] Is a directory: '.'` when the
    four flag defaults collapsed to the current directory. After the
    fix, calling it with run-layout paths must NOT raise, and every
    prerequisite must be reported as `missing` (since the wrapper
    generates the files itself)."""
    layout = build_run_layout(tmp_path / "volume", "prereq-collect-test")

    resolved = inline_evaluate_paths(layout)

    # The call that used to blow up.
    prerequisites = wrapper.collect_report_prerequisites(
        baseline_validation_report=resolved["baseline_validation_report"],
        candidate_validation_report=resolved["candidate_validation_report"],
        baseline_test_report=resolved["baseline_test_report"],
        finetuned_test_report=resolved["finetuned_test_report"],
    )

    assert len(prerequisites) == 4
    assert all(item["status"] == "missing" for item in prerequisites)
    # And none of the resolved paths is the current directory.
    for item in prerequisites:
        assert item["path"] != "."
        assert Path(item["path"]).parent == layout.reports_root


def test_run_real_does_not_raise_isadirectoryerror_when_report_flags_omitted(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end: `run_real` invoked the way a Modal caller would —
    without the four `--*-report` flags — must NOT raise
    `IsADirectoryError`. The wrapper must resolve each missing flag to
    its run-layout file path before `collect_report_prerequisites`
    tries to read it.

    We exercise `main()` with the same argv a Modal caller would build
    (i.e. empty strings for the four report flags, since `_build_argv`
    passes through empty defaults as positional CLI args). The fix
    happens inside `main()`, so the `Modal`-wrapped `run_real` is not
    on the path here.
    """
    plan_output = build_run_layout(tmp_path / "modal-volume", "isadir-bug-repro").run_root / "plans" / "modal-plan.json"
    plan_output.parent.mkdir(parents=True, exist_ok=True)

    # Stub subprocess.run so the real run doesn't try to invoke
    # train.py (which needs GPU/torch). The fix is about argument
    # resolution before subprocess execution, so stubbing is safe.
    def fake_run(command, *args, **kwargs):
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    # Mirror exactly what `_build_argv` produces when the caller
    # omits all four `--*-report` flags: the args are present but
    # empty, so argparse's `type=Path` collapses each one to `'.'`.
    result = modal_main([
        "--run-id",
        "isadir-bug-repro",
        "--raw-dataset-root",
        str(tmp_path / "dataset-tracknet"),
        "--volume-root",
        str(tmp_path / "modal-volume"),
        "--pretrained-tracknet",
        str(tmp_path / "TrackNet_best.pt"),
        "--baseline-validation-report", "",
        "--candidate-validation-report", "",
        "--baseline-test-report", "",
        "--finetuned-test-report", "",
        "--plan-output",
        str(plan_output),
    ])

    assert result == 0
    assert plan_output.exists()
    payload = json.loads(plan_output.read_text(encoding="utf-8"))
    # Every prerequisite must point at a path inside the run layout,
    # never at the literal string ".".
    for item in payload["prerequisites"]:
        assert item["path"] != "."
        assert Path(item["path"]).parent.exists() is False or \
            Path(item["path"]).parent.is_dir()
        assert item["status"] == "missing"


def _write_processed_root(root: Path, *, fingerprint: str = "shared-fingerprint") -> Path:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"fingerprint": fingerprint, "generated_splits": ["train", "val", "test"]}),
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        match_root = root / split / "match01"
        (match_root / "video").mkdir(parents=True)
        (match_root / "csv").mkdir()
        (match_root / "frame" / "clip01").mkdir(parents=True)
        (match_root / "video" / "clip01.mp4").touch()
        (match_root / "csv" / "clip01_ball.csv").touch()
        (match_root / "frame" / "clip01" / "0.png").touch()
    return root


@pytest.mark.parametrize("dataset_args", [[], ["--raw-dataset-root", "/raw", "--processed-root", "/processed"]])
def test_modal_requires_exactly_one_dataset_root_before_command_construction(
    tmp_path: Path, monkeypatch, capsys, dataset_args: list[str]
) -> None:
    build_commands_called = False

    def record_build_commands(**kwargs):
        nonlocal build_commands_called
        build_commands_called = True
        return []

    monkeypatch.setattr(wrapper, "build_commands", record_build_commands)

    with pytest.raises(SystemExit):
        modal_main([
            "--run-id", "dataset-mode",
            *dataset_args,
            "--pretrained-tracknet", str(tmp_path / "TrackNet_best.pt"),
            "--plan-output", str(tmp_path / "plan.json"),
            "--dry-run",
        ])

    error_output = capsys.readouterr().err
    assert "--raw-dataset-root" in error_output
    assert "--processed-root" in error_output
    assert build_commands_called is False


def test_processed_root_plan_skips_raw_stages_and_retains_provenance(tmp_path: Path) -> None:
    processed_root = _write_processed_root(tmp_path / "shared" / "basketball-v1")
    layout = build_run_layout(tmp_path / "volume", "shared-run")

    commands = build_commands(
        layout=layout,
        processed_root=processed_root,
        pretrained_tracknet=str(tmp_path / "TrackNet_best.pt"),
    )

    assert len(commands) == 11
    assert all("basketball_data.py" not in command for command in commands)
    assert all("visualize_labels.py" not in command for command in commands)
    assert all("preprocess.py" not in command for command in commands)

    consumers = [
        command for command in commands
        if command[:2] in (
            [sys.executable, "TrackNetV3/test.py"],
            [sys.executable, "TrackNetV3/train.py"],
        ) or command[:3] == [sys.executable, "TrackNetV3/evaluate.py", "metrics"]
    ]
    assert len(consumers) == 9
    assert [
        command[command.index("--dataset_root") + 1]
        for command in consumers if "--dataset_root" in command
    ] == [str(processed_root)] * 5
    assert [
        command[command.index("--dataset-root") + 1]
        for command in consumers if "--dataset-root" in command
    ] == [str(processed_root)] * 4
    assert [
        command[command.index("--cache_root") + 1]
        for command in _test_py_commands(commands)
    ] == [str(layout.checkpoints_root / "cache")] * 4


def test_processed_root_rejects_plan_and_report_outputs_outside_the_run_root(tmp_path: Path) -> None:
    processed_root = _write_processed_root(tmp_path / "shared" / "basketball-v3")
    layout = build_run_layout(tmp_path / "volume", "isolated-run")

    with pytest.raises(SystemExit, match="plan output escapes"):
        modal_main([
            "--run-id", "isolated-run",
            "--processed-root", str(processed_root),
            "--volume-root", str(tmp_path / "volume"),
            "--pretrained-tracknet", str(tmp_path / "TrackNet_best.pt"),
            "--plan-output", str(processed_root / "plan.json"),
            "--dry-run",
        ])

    with pytest.raises(ValueError, match="baseline_validation_report escapes"):
        build_commands(
            layout=layout,
            processed_root=processed_root,
            pretrained_tracknet=str(tmp_path / "TrackNet_best.pt"),
            baseline_validation_report=processed_root / "report.json",
        )


def test_processed_root_plan_metadata_and_outputs_stay_in_run_root(tmp_path: Path) -> None:
    processed_root = _write_processed_root(tmp_path / "shared" / "basketball-v2", fingerprint="fingerprint-v2")
    plan_output = build_run_layout(tmp_path / "volume", "isolated-run").run_root / "plans" / "processed-plan.json"

    result = modal_main([
        "--run-id", "isolated-run",
        "--processed-root", str(processed_root),
        "--volume-root", str(tmp_path / "volume"),
        "--pretrained-tracknet", str(tmp_path / "TrackNet_best.pt"),
        "--plan-output", str(plan_output),
        "--dry-run",
    ])

    assert result == 0
    payload = json.loads(plan_output.read_text(encoding="utf-8"))
    run_root = Path(payload["layout"]["run_root"])
    assert payload["dataset_mode"] == "processed"
    assert payload["dataset_root"] == str(processed_root)
    assert payload["fingerprint"] == "fingerprint-v2"
    assert payload["processed_dataset_id"] == "basketball-v2"
    assert all(
        Path(path).is_relative_to(run_root)
        for key, path in payload["layout"].items()
        if key != "run_root"
    )


@pytest.mark.parametrize(
    ("setup", "error"),
    [
        (lambda root: None, "unreadable manifest"),
        (lambda root: _write_processed_root(root, fingerprint=" "), "missing fingerprint"),
    ],
)
def test_processed_root_errors_before_subprocess_execution(
    tmp_path: Path, monkeypatch, setup, error: str
) -> None:
    processed_root = tmp_path / "shared" / "invalid"
    setup(processed_root)
    subprocess_called = False

    def record_subprocess(*args, **kwargs):
        nonlocal subprocess_called
        subprocess_called = True

    monkeypatch.setattr(wrapper.subprocess, "run", record_subprocess)

    with pytest.raises(SystemExit, match=error):
        modal_main([
            "--run-id", "invalid-shared",
            "--processed-root", str(processed_root),
            "--pretrained-tracknet", str(tmp_path / "TrackNet_best.pt"),
            "--plan-output", str(tmp_path / "plan.json"),
        ])

    assert subprocess_called is False


@pytest.mark.parametrize(
    ("raw_root", "processed_root", "expected_flag", "expected_root"),
    [
        ("/raw", "", "--raw-dataset-root", "/raw"),
        ("", "/mnt/tracknet/processed/basketball-v3", "--processed-root", "/mnt/tracknet/processed/basketball-v3"),
    ],
)
def test_build_argv_forwards_only_the_selected_dataset_root(
    raw_root: str, processed_root: str, expected_flag: str, expected_root: str
) -> None:
    argv = wrapper._build_argv(
        run_id="wrapper-mode",
        raw_dataset_root=raw_root,
        processed_root=processed_root,
        volume_root="/cache",
        pretrained_tracknet="/models/TrackNet_best.pt",
        baseline_validation_report="",
        candidate_validation_report="",
        baseline_test_report="",
        finetuned_test_report="",
        plan_output="/cache/plan.json",
        dry_run=True,
    )

    assert argv[argv.index(expected_flag) + 1] == expected_root
    assert ("--raw-dataset-root" in argv) is (expected_flag == "--raw-dataset-root")
    assert ("--processed-root" in argv) is (expected_flag == "--processed-root")


def test_build_argv_defaults_plan_output_under_the_run_root() -> None:
    argv = wrapper._build_argv(
        run_id="default-plan",
        raw_dataset_root="/raw",
        volume_root="/cache",
        pretrained_tracknet="/models/TrackNet_best.pt",
        baseline_validation_report="",
        candidate_validation_report="",
        baseline_test_report="",
        finetuned_test_report="",
        plan_output="",
        dry_run=True,
    )

    assert argv[argv.index("--plan-output") + 1] == "/cache/tracknet-finetune/runs/default-plan/plans/modal-plan.json"


def test_modal_entrypoints_default_to_the_persistent_volume() -> None:
    assert wrapper.MODAL_VOLUME_ROOT == "/mnt/tracknet"
    assert wrapper.DEFAULT_PRETRAINED_TRACKNET == "/mnt/tracknet/pretrained/TrackNet_best.pt"


def test_modal_entrypoint_rejects_host_local_checkpoint_path() -> None:
    with pytest.raises(ValueError, match="host-local"):
        wrapper._build_argv(
            run_id="unsafe-checkpoint",
            raw_dataset_root="/raw",
            volume_root="/mnt/tracknet",
            pretrained_tracknet="/path/to/TrackNet_best.pt",
            baseline_validation_report="",
            candidate_validation_report="",
            baseline_test_report="",
            finetuned_test_report="",
            plan_output="",
            dry_run=True,
        )


def test_modal_entrypoint_fills_checkpoint_default_and_mounts_outputs() -> None:
    argv = wrapper._build_argv(
        run_id="mounted-run",
        raw_dataset_root="/raw",
        volume_root="/mnt/tracknet",
        pretrained_tracknet="",
        baseline_validation_report="",
        candidate_validation_report="",
        baseline_test_report="",
        finetuned_test_report="",
        plan_output="",
        dry_run=True,
    )

    assert argv[argv.index("--pretrained-tracknet") + 1] == "/mnt/tracknet/pretrained/TrackNet_best.pt"
    assert argv[argv.index("--plan-output") + 1].startswith("/mnt/tracknet/tracknet-finetune/runs/")
