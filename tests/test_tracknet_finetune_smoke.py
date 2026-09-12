from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TrackNetV3"))

from preprocess import main as preprocess_main
from preprocess import preprocess_basketball_dataset
from smoke import run_local_smoke
from visualize_labels import render_label_samples


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 12))
    for _ in range(3):
        writer.write(np.zeros((12, 16, 3), dtype=np.uint8))
    writer.release()


def _write_labels(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Frame", "Visibility", "X", "Y"])
        writer.writeheader()
        writer.writerows(
            [
                {"Frame": 0, "Visibility": 1, "X": 5, "Y": 4},
                {"Frame": 1, "Visibility": 0, "X": 0, "Y": 0},
                {"Frame": 2, "Visibility": 1, "X": 8, "Y": 6},
            ]
        )


def _write_clip(root: Path, split: str, game: str, clip: str) -> None:
    _write_video(root / split / game / "video" / f"{clip}.mp4")
    _write_labels(root / split / game / "csv" / f"{clip}_ball.csv")


def _write_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
{
  "TP": 2,
  "TN": 0,
  "FP1": 0,
  "FP2": 1,
  "FN": 0,
  "precision": 0.666667,
  "recall": 1.0,
  "f1": 0.8,
  "miss_rate": 0.0,
  "p_at_10": 1.0,
  "p_at_40": 1.0,
  "mean_error": 6.0,
  "median_error": 6.0
}
""".strip(),
        encoding="utf-8",
    )


def test_render_label_samples_creates_overlay_artifacts(tmp_path: Path) -> None:
    video_path = tmp_path / "train" / "game_01" / "video" / "clip_001.mp4"
    csv_path = tmp_path / "train" / "game_01" / "csv" / "clip_001_ball.csv"
    _write_video(video_path)
    _write_labels(csv_path)

    outputs = render_label_samples(video_path, csv_path, tmp_path / "viz", sample_count=2)

    assert len(outputs) == 2
    assert outputs[0].exists()
    rendered = cv2.imread(str(outputs[0]))
    assert int(rendered[4, 5, 1]) > 0


def test_preprocess_basketball_dataset_creates_split_isolated_outputs(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset-tracknet"
    output_root = tmp_path / "adapted-data"
    _write_clip(raw_root, "train", "game_01", "clip_001")
    _write_clip(raw_root, "validation", "game_02", "clip_002")
    _write_clip(raw_root, "test", "game_03", "clip_003")

    report = preprocess_basketball_dataset(raw_root, output_root)

    assert report["rebuilt_splits"] == ["train", "val", "test"]
    assert (output_root / "train" / "match01" / "video" / "clip_001.mp4").exists()
    assert (output_root / "val" / "match02" / "frame" / "clip_002" / "0.png").exists()
    assert (output_root / "test" / "match03" / "median.npz").exists()
    assert not (output_root / "validation").exists()
    assert (raw_root / "validation" / "game_02" / "video" / "clip_002.mp4").exists()
    assert (raw_root / "train" / "game_01" / "video" / "clip_001.mp4").exists()


def test_preprocess_basketball_dataset_rebuilds_only_changed_split_and_cli_smoke(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset-tracknet"
    output_root = tmp_path / "adapted-data"
    _write_clip(raw_root, "train", "game_01", "clip_001")
    _write_clip(raw_root, "validation", "game_02", "clip_002")

    first = preprocess_basketball_dataset(raw_root, output_root, splits=("train", "validation"))
    train_marker = output_root / "train" / "keep.txt"
    val_marker = output_root / "val" / "stale.txt"
    train_marker.write_text("keep", encoding="utf-8")
    val_marker.write_text("remove", encoding="utf-8")
    (raw_root / "validation" / "game_02" / "csv" / "clip_002_ball.csv").write_text(
        "Frame,Visibility,X,Y\n0,1,5,4\n1,1,7,5\n2,1,8,6\n",
        encoding="utf-8",
    )

    exit_code = preprocess_main([str(raw_root), "--output-root", str(output_root), "--split", "train", "--split", "validation"])

    assert first["rebuilt_splits"] == ["train", "val"]
    assert exit_code == 0
    assert train_marker.exists()
    assert not val_marker.exists()
    assert (output_root / "manifest.json").exists()


def test_run_local_smoke_wires_validation_preprocess_and_baseline_report(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset-tracknet"
    _write_clip(raw_root, "validation", "game_02", "clip_002")
    summary_path = tmp_path / "baseline-summary.json"
    _write_summary(summary_path)

    result = run_local_smoke(
        raw_dataset_root=raw_root,
        run_root=tmp_path / "run",
        baseline_summary_json=summary_path,
        splits=("validation",),
    )

    assert result["baseline_eval_mode"] == "report-only"
    assert Path(result["validation_report"]).exists()
    assert Path(result["baseline_report"]).exists()
    assert (Path(result["adapted_data_root"]) / "val" / "match02" / "frame" / "clip_002" / "0.png").exists()


def test_run_local_smoke_fails_before_preprocess_when_validation_blocks_dataset(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset-tracknet"
    _write_video(raw_root / "validation" / "game_02" / "video" / "clip_002.mp4")
    summary_path = tmp_path / "baseline-summary.json"
    _write_summary(summary_path)

    with pytest.raises(ValueError, match="validation failed"):
        run_local_smoke(
            raw_dataset_root=raw_root,
            run_root=tmp_path / "run",
            baseline_summary_json=summary_path,
            splits=("validation",),
        )

    assert not (tmp_path / "run" / "adapted-data").exists()


@pytest.mark.parametrize("splits", [("test",), ("validation", "test")])
def test_run_local_smoke_rejects_non_validation_splits(tmp_path: Path, splits: tuple[str, ...]) -> None:
    raw_root = tmp_path / "dataset-tracknet"
    _write_clip(raw_root, "validation", "game_02", "clip_002")
    _write_clip(raw_root, "test", "game_03", "clip_003")
    summary_path = tmp_path / "baseline-summary.json"
    _write_summary(summary_path)

    with pytest.raises(ValueError, match="validation-only"):
        run_local_smoke(
            raw_dataset_root=raw_root,
            run_root=tmp_path / "run",
            baseline_summary_json=summary_path,
            splits=splits,
        )
