from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TrackNetV3"))

from basketball_data import (
    build_file_manifest,
    build_preprocess_manifest,
    detect_match_name_collisions,
    determine_rebuild_splits,
    generated_split_name,
    validate_dataset,
    validate_processed_root,
    write_validation_report,
)
from preprocess import preprocess_basketball_dataset


def _write_video(path: Path, *, width: int = 16, height: int = 12, frames: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    for _ in range(frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


def _write_labels(path: Path, rows: list[dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Frame", "Visibility", "X", "Y"])
        writer.writeheader()
        writer.writerows(rows)


def _write_clip(root: Path, split: str, game: str, clip: str, *, rows: list[dict[str, int]] | None = None) -> None:
    _write_video(root / split / game / "video" / f"{clip}.mp4")
    _write_labels(
        root / split / game / "csv" / f"{clip}_ball.csv",
        rows or [
            {"Frame": 0, "Visibility": 1, "X": 5, "Y": 4},
            {"Frame": 1, "Visibility": 0, "X": 0, "Y": 0},
            {"Frame": 2, "Visibility": 1, "X": 8, "Y": 6},
        ],
    )


def _write_processed_tree(root: Path, *, manifest: dict | None = None) -> dict:
    manifest = manifest or {"fingerprint": "processed-fingerprint", "generated_splits": ["train", "val", "test"]}
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for split in manifest.get("generated_splits", []):
        match_root = root / split / "match01"
        (match_root / "video").mkdir(parents=True, exist_ok=True)
        (match_root / "csv").mkdir()
        (match_root / "frame" / "clip01").mkdir(parents=True)
        (match_root / "video" / "clip01.mp4").touch()
        (match_root / "csv" / "clip01_ball.csv").touch()
        (match_root / "frame" / "clip01" / "0.png").touch()
    return manifest


def test_validate_processed_root_returns_valid_manifest(tmp_path: Path) -> None:
    manifest = _write_processed_tree(tmp_path / "processed")

    assert validate_processed_root(tmp_path / "processed") == manifest


def test_validate_processed_root_rejects_malformed_manifest(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    (root / "manifest.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"processed.*manifest"):
        validate_processed_root(root)


@pytest.mark.parametrize(
    ("manifest", "error"),
    [
        ({"fingerprint": "  ", "generated_splits": ["train", "val", "test"]}, "fingerprint"),
        ({"fingerprint": "fingerprint", "generated_splits": ["train", "val"]}, "test generated split"),
    ],
)
def test_validate_processed_root_rejects_incomplete_manifest(tmp_path: Path, manifest: dict, error: str) -> None:
    root = tmp_path / "processed"
    _write_processed_tree(root, manifest=manifest)

    with pytest.raises(ValueError, match=error):
        validate_processed_root(root)


def test_validate_processed_root_rejects_missing_split_directory(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    _write_processed_tree(root)
    shutil.rmtree(root / "val")

    with pytest.raises(ValueError, match="val split directory"):
        validate_processed_root(root)


def test_validate_processed_root_rejects_splits_without_materialized_tracknet_data(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"fingerprint": "processed-fingerprint", "generated_splits": ["train", "val", "test"]}),
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        (root / split).mkdir()

    with pytest.raises(ValueError, match="train split contents"):
        validate_processed_root(root)


def test_validate_processed_root_rejects_split_missing_a_required_tracknet_layout_directory(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    _write_processed_tree(root)
    shutil.rmtree(root / "train" / "match01" / "video")

    with pytest.raises(ValueError, match="train split contents"):
        validate_processed_root(root)


def test_validate_processed_root_rejects_raw_like_root(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "train").mkdir(parents=True)

    with pytest.raises(ValueError, match=r"raw.*manifest"):
        validate_processed_root(raw_root)


def test_generated_split_name_maps_validation_to_val_only_for_generated_layout() -> None:
    assert generated_split_name("validation") == "val"
    assert generated_split_name("train") == "train"


def test_validate_dataset_reports_pairs_stats_and_json_report(tmp_path: Path) -> None:
    _write_clip(tmp_path, "train", "game_01", "clip_001")
    _write_clip(tmp_path, "validation", "game_02", "clip_002")

    report = validate_dataset(tmp_path)

    assert report["ok"] is True
    assert len(report["pairs"]) == 2
    assert report["stats"]["splits"]["train"]["clips"] == 1
    assert report["stats"]["splits"]["validation"]["generated_name"] == "val"

    output_path = tmp_path / "report.json"
    write_validation_report(report, output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["stats"]["total_clips"] == 2


def test_validate_dataset_reports_missing_video_bad_csv_and_split_leakage(tmp_path: Path) -> None:
    _write_clip(tmp_path, "train", "game_01", "clip_001")
    _write_clip(tmp_path, "validation", "game_01", "clip_001")
    _write_video(tmp_path / "test" / "game_04" / "video" / "clip_004.mp4")
    _write_labels(
        tmp_path / "test" / "game_04" / "csv" / "clip_004_ball.csv",
        [{"Frame": 9, "Visibility": 1, "X": 99, "Y": 99}],
    )
    _write_labels(
        tmp_path / "test" / "game_03" / "csv" / "clip_003_ball.csv",
        [{"Frame": 9, "Visibility": 1, "X": 99, "Y": 99}],
    )

    report = validate_dataset(tmp_path)
    reasons = {error["reason"] for error in report["errors"]}

    assert report["ok"] is False
    assert "clip appears in multiple splits" in reasons
    assert "missing video for clip" in reasons
    assert "frame index exceeds video length" in reasons


def test_build_file_manifest_changes_when_source_file_changes(tmp_path: Path) -> None:
    video_path = tmp_path / "train" / "game_01" / "video" / "clip_001.mp4"
    csv_path = tmp_path / "train" / "game_01" / "csv" / "clip_001_ball.csv"
    _write_video(video_path)
    _write_labels(csv_path, [{"Frame": 0, "Visibility": 1, "X": 1, "Y": 1}])

    before = build_file_manifest([video_path, csv_path])
    csv_path.write_text("Frame,Visibility,X,Y\n0,1,2,2\n", encoding="utf-8")
    after = build_file_manifest([video_path, csv_path])

    assert before["fingerprint"] != after["fingerprint"]


def test_determine_rebuild_splits_only_marks_changed_generated_split(tmp_path: Path) -> None:
    train_video = tmp_path / "train" / "game_01" / "video" / "clip_001.mp4"
    train_csv = tmp_path / "train" / "game_01" / "csv" / "clip_001_ball.csv"
    val_video = tmp_path / "validation" / "game_02" / "video" / "clip_002.mp4"
    val_csv = tmp_path / "validation" / "game_02" / "csv" / "clip_002_ball.csv"
    _write_video(train_video)
    _write_labels(train_csv, [{"Frame": 0, "Visibility": 1, "X": 1, "Y": 1}])
    _write_video(val_video)
    _write_labels(val_csv, [{"Frame": 0, "Visibility": 1, "X": 2, "Y": 2}])

    previous = build_preprocess_manifest(
        source_root=tmp_path,
        source_paths=[train_video, train_csv, val_video, val_csv],
        requested_splits=["train", "validation"],
        preprocess_params={"height": 288, "width": 512, "bg_mode": "", "seq_len": 8},
    )

    val_csv.write_text("Frame,Visibility,X,Y\n0,1,3,3\n", encoding="utf-8")

    current = build_preprocess_manifest(
        source_root=tmp_path,
        source_paths=[train_video, train_csv, val_video, val_csv],
        requested_splits=["train", "validation"],
        preprocess_params={"height": 288, "width": 512, "bg_mode": "", "seq_len": 8},
    )

    assert determine_rebuild_splits(previous, current) == ["val"]


def test_detect_match_name_collisions_flags_distinct_games_that_normalize_together(tmp_path: Path) -> None:
    _write_clip(tmp_path, "train", "game_01", "clip_001")
    _write_clip(tmp_path, "train", "court1", "clip_002")

    report = validate_dataset(tmp_path, splits=("train",))
    collisions = detect_match_name_collisions(report["pairs"])

    assert collisions == [
        {
            "generated_split": "train",
            "match_name": "match01",
            "games": ["court1", "game_01"],
        }
    ]


def test_preprocess_basketball_dataset_fails_before_writing_when_match_names_collide(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset-tracknet"
    output_root = tmp_path / "adapted-data"
    _write_clip(raw_root, "train", "game_01", "clip_001")
    _write_clip(raw_root, "train", "court1", "clip_002")

    with pytest.raises(ValueError, match="normalized match collision"):
        preprocess_basketball_dataset(raw_root, output_root, splits=("train",))

    assert not output_root.exists()
