from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from basketball_data import (
    DEFAULT_IMAGE_FORMAT,
    RAW_SPLITS,
    build_preprocess_manifest,
    detect_match_name_collisions,
    determine_rebuild_splits,
    generated_split_name,
    normalize_match_name,
    read_json_manifest,
    validate_dataset,
    write_json_manifest,
)


def preprocess_basketball_dataset(
    raw_dataset_root: Path | str,
    output_root: Path | str,
    *,
    splits: tuple[str, ...] | list[str] = RAW_SPLITS,
    seq_len: int = 8,
    bg_mode: str = "",
    height: int = 288,
    width: int = 512,
    image_format: str = DEFAULT_IMAGE_FORMAT,
) -> dict:
    raw_root = Path(raw_dataset_root)
    adapted_root = Path(output_root)
    requested_splits = tuple(splits)
    validation_report = validate_dataset(raw_root, splits=requested_splits)
    if validation_report["errors"]:
        raise ValueError(json.dumps(validation_report["errors"], indent=2))

    source_paths = [Path(pair[path_key]) for pair in validation_report["pairs"] for path_key in ("video_path", "csv_path")]
    collisions = detect_match_name_collisions(validation_report["pairs"])
    if collisions:
        raise ValueError(json.dumps({"reason": "normalized match collision", "collisions": collisions}, indent=2))
    current_manifest = build_preprocess_manifest(
        source_root=raw_root,
        source_paths=source_paths,
        requested_splits=requested_splits,
        preprocess_params={
            "seq_len": seq_len,
            "bg_mode": bg_mode,
            "height": height,
            "width": width,
            "image_format": image_format,
        },
    )
    manifest_path = adapted_root / "manifest.json"
    previous_manifest = read_json_manifest(manifest_path)
    rebuilt_splits = determine_rebuild_splits(previous_manifest, current_manifest)

    pairs_by_split: dict[str, list[dict]] = defaultdict(list)
    for pair in validation_report["pairs"]:
        pairs_by_split[generated_split_name(pair["split"])].append(pair)

    for split in rebuilt_splits:
        shutil.rmtree(adapted_root / split, ignore_errors=True)
        _materialize_split(adapted_root, split, pairs_by_split.get(split, []), image_format=image_format)

    write_json_manifest(current_manifest, manifest_path)
    return {
        "ok": True,
        "raw_dataset_root": str(raw_root),
        "output_root": str(adapted_root),
        "requested_splits": list(requested_splits),
        "rebuilt_splits": rebuilt_splits,
        "manifest": current_manifest,
    }


def _materialize_split(adapted_root: Path, generated_split: str, pairs: list[dict], *, image_format: str) -> None:
    match_dirs: dict[Path, list[Path]] = defaultdict(list)

    for pair in pairs:
        match_dir = adapted_root / generated_split / normalize_match_name(pair["game"])
        video_path = Path(pair["video_path"])
        csv_path = Path(pair["csv_path"])
        clip = pair["clip"]

        (match_dir / "video").mkdir(parents=True, exist_ok=True)
        (match_dir / "csv").mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, match_dir / "video" / video_path.name)
        shutil.copy2(csv_path, match_dir / "csv" / csv_path.name)
        rally_dir = match_dir / "frame" / clip
        _write_clip_frames(video_path, rally_dir, image_format=image_format)
        match_dirs[match_dir].append(rally_dir)

    for match_dir, rally_dirs in match_dirs.items():
        _write_match_median(match_dir, rally_dirs)


def _write_clip_frames(video_path: Path, rally_dir: Path, *, image_format: str) -> None:
    rally_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
            cv2.imwrite(str(rally_dir / f"{index}.{image_format}"), frame)
            index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"video has no readable frames: {video_path}")
    median = np.median(np.array(frames), axis=0)[..., ::-1]
    np.savez(rally_dir / "median.npz", median=median)


def _write_match_median(match_dir: Path, rally_dirs: list[Path]) -> None:
    medians = [np.load(rally_dir / "median.npz")["median"] for rally_dir in rally_dirs]
    median = np.median(np.array(medians), axis=0)
    np.savez(match_dir / "median.npz", median=median)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TrackNet-compatible basketball preprocess outputs without mutating raw data.")
    parser.add_argument("raw_dataset_root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", action="append", dest="splits", choices=RAW_SPLITS)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--bg-mode", default="")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = preprocess_basketball_dataset(
            args.raw_dataset_root,
            args.output_root,
            splits=tuple(args.splits or RAW_SPLITS),
            seq_len=args.seq_len,
            bg_mode=args.bg_mode,
            height=args.height,
            width=args.width,
        )
    except ValueError as exc:
        print(exc)
        return 1

    print(json.dumps({"rebuilt_splits": report["rebuilt_splits"], "output_root": report["output_root"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
