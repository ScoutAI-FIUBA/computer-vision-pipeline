"""Focused coverage for the train-command follow-up patches on `modal_tracknet_finetune`.

Follow-up patch on top of the archived
`tracknet-basketball-modal-finetuning` change. Proves only the new
behavior the patch adds:

- `build_commands` includes `--bg_mode concat` in the train command so
  the train step matches the official `TrackNet_best.pt` checkpoint
  shape (`bg_mode='concat'`).
- `build_commands` includes `--seq_len 8` in the train command so the
  train step matches the official `TrackNet_best.pt` checkpoint shape
  (`seq_len=8`).
- `build_commands` includes `--epochs 20` and `--early_stopping_patience 5`
  for the requested Modal fine-tuning duration and validation stopping.
- The rest of the train command content/order stays unchanged.

The official pretrained checkpoint `TrackNet_best.pt` was saved with
`bg_mode='concat'` and `seq_len=8`, but the prior `build_commands(...)`
`train.py` invocation only passed `--model_name TrackNet`, missing
`--bg_mode concat` and `--seq_len 8`. The pretrained checkpoint
compatibility check in `TrackNetV3/finetune.py::_assert_checkpoint_matches`
rejected the mismatch with `TrainingStateError: incompatible checkpoint:
bg_mode='concat' != ''`. This patch wires the two missing flags through
the wrapper so the train step matches the checkpoint shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modal_tracknet_finetune import (  # noqa: E402
    build_commands,
    build_run_layout,
)


def _train_command(commands: list[list[str]]) -> list[str]:
    """Return the single `train.py` command from the plan."""
    matches = [
        cmd for cmd in commands
        if cmd[:2] == [sys.executable, "TrackNetV3/train.py"]
    ]
    assert len(matches) == 1, (
        f"plan must contain exactly one train.py command, got {len(matches)}"
    )
    return matches[0]


def test_train_command_passes_bg_mode_concat(tmp_path: Path) -> None:
    """The train command must pass `--bg_mode concat` so it matches the
    official `TrackNet_best.pt` checkpoint shape."""
    layout = build_run_layout(tmp_path / "volume", "bg-mode-flag")
    pretrained = str(tmp_path / "TrackNet_best.pt")

    commands = build_commands(
        layout=layout,
        raw_dataset_root=tmp_path / "dataset-tracknet",
        pretrained_tracknet=pretrained,
    )

    train_cmd = _train_command(commands)
    bg_mode_index = train_cmd.index("--bg_mode")
    assert train_cmd[bg_mode_index + 1] == "concat", (
        f"train command must pass `--bg_mode concat` to match the "
        f"official TrackNet_best.pt checkpoint, got `{train_cmd[bg_mode_index + 1]!r}`"
    )


def test_train_command_passes_seq_len_8(tmp_path: Path) -> None:
    """The train command must pass `--seq_len 8` so it matches the
    official `TrackNet_best.pt` checkpoint shape."""
    layout = build_run_layout(tmp_path / "volume", "seq-len-flag")
    pretrained = str(tmp_path / "TrackNet_best.pt")

    commands = build_commands(
        layout=layout,
        raw_dataset_root=tmp_path / "dataset-tracknet",
        pretrained_tracknet=pretrained,
    )

    train_cmd = _train_command(commands)
    seq_len_index = train_cmd.index("--seq_len")
    assert train_cmd[seq_len_index + 1] == "8", (
        f"train command must pass `--seq_len 8` to match the official "
        f"TrackNet_best.pt checkpoint, got `{train_cmd[seq_len_index + 1]!r}`"
    )


def test_train_command_passes_epochs_and_early_stopping(tmp_path: Path) -> None:
    """The train command carries the requested duration and patience."""
    layout = build_run_layout(tmp_path / "volume", "epochs-flag")
    pretrained = str(tmp_path / "TrackNet_best.pt")

    commands = build_commands(
        layout=layout,
        raw_dataset_root=tmp_path / "dataset-tracknet",
        pretrained_tracknet=pretrained,
    )

    train_cmd = _train_command(commands)
    epochs_index = train_cmd.index("--epochs")
    assert train_cmd[epochs_index + 1] == "20", (
        f"train command must pass `--epochs 20` for the Modal flow, "
        f"got `{train_cmd[epochs_index + 1]!r}`"
    )
    patience_index = train_cmd.index("--early_stopping_patience")
    assert train_cmd[patience_index + 1] == "5"


def test_train_command_preserves_existing_order_around_epochs(tmp_path: Path) -> None:
    """The focused train command keeps its existing argument order."""
    layout = build_run_layout(tmp_path / "volume", "train-order")
    raw_root = tmp_path / "dataset-tracknet"
    pretrained = str(tmp_path / "TrackNet_best.pt")

    plan = build_commands(
        layout=layout,
        raw_dataset_root=raw_root,
        pretrained_tracknet=pretrained,
    )

    train_cmd = _train_command(plan)
    assert train_cmd == [
        sys.executable,
        "TrackNetV3/train.py",
        "--model_name",
        "TrackNet",
        "--bg_mode",
        "concat",
        "--seq_len",
        "8",
        "--epochs",
        "20",
        "--early_stopping_patience",
        "5",
        "--dataset_root",
        str(layout.adapted_data_root),
        "--cache_root",
        str(layout.checkpoints_root / "cache"),
        "--save_dir",
        str(layout.checkpoints_root),
        "--pretrained_tracknet",
        pretrained,
        "--baseline-validation-report",
        str(layout.reports_root / "baseline-validation-report.json"),
    ]
