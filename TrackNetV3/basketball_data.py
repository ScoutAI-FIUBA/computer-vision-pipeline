from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


RAW_SPLITS = ("train", "validation", "test")
REQUIRED_LABEL_COLUMNS = ("Frame", "Visibility", "X", "Y")
DEFAULT_IMAGE_FORMAT = "png"


@dataclass(frozen=True)
class ClipPair:
    split: str
    game: str
    clip: str
    video_path: Path
    csv_path: Path


def generated_split_name(split: str) -> str:
    return "val" if split == "validation" else split


def resolve_label_csv_path(match_dir: Path | str, clip: str, split: str) -> Path:
    match_root = Path(match_dir)
    generated_split = generated_split_name(split)
    candidates = [match_root / "csv" / f"{clip}_ball.csv"]
    if generated_split == "test":
        candidates.insert(0, match_root / "corrected_csv" / f"{clip}_ball.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def normalize_match_name(game: str) -> str:
    digits = "".join(re.findall(r"\d+", game))
    if not digits:
        raise ValueError(f"cannot derive match name from game {game!r}")
    return f"match{int(digits):02d}"


def detect_match_name_collisions(pairs: Iterable[ClipPair | dict]) -> list[dict]:
    collisions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair in pairs:
        split = generated_split_name(pair["split"] if isinstance(pair, dict) else pair.split)
        game = pair["game"] if isinstance(pair, dict) else pair.game
        collisions[(split, normalize_match_name(game))].add(game)

    return [
        {
            "generated_split": generated_split,
            "match_name": match_name,
            "games": sorted(games),
        }
        for (generated_split, match_name), games in sorted(collisions.items())
        if len(games) > 1
    ]


def _iter_real_files(directory: Path, suffix: str) -> Iterable[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == suffix and ":Zone.Identifier" not in path.name
    )


def _clip_stem_from_csv(path: Path) -> str:
    name = path.name
    return name[: -len("_ball.csv")] if name.endswith("_ball.csv") else path.stem


def _read_video_info(video_path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("cannot open video")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return frame_count, width, height
    finally:
        capture.release()


def build_file_manifest(paths: Iterable[Path]) -> dict:
    entries = []
    for path in sorted(Path(path) for path in paths):
        stat = path.stat()
        entries.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = json.dumps(entries, sort_keys=True).encode("utf-8")
    return {"entries": entries, "fingerprint": hashlib.sha256(payload).hexdigest()}


def build_preprocess_manifest(
    *,
    source_root: Path | str,
    source_paths: Iterable[Path],
    requested_splits: Iterable[str],
    preprocess_params: dict,
) -> dict:
    root = Path(source_root)
    requested = [split for split in requested_splits]
    split_paths: dict[str, list[Path]] = defaultdict(list)
    normalized_paths = [Path(path) for path in source_paths]

    for path in normalized_paths:
        relative_parts = path.relative_to(root).parts
        split_paths[generated_split_name(relative_parts[0])].append(path)

    split_manifests = {
        split: build_file_manifest(paths)
        for split, paths in sorted(split_paths.items())
    }
    payload = {
        "source_root": str(root),
        "requested_splits": requested,
        "generated_splits": [generated_split_name(split) for split in requested],
        "preprocess_params": preprocess_params,
        "split_fingerprints": {split: manifest["fingerprint"] for split, manifest in split_manifests.items()},
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return payload | {"fingerprint": fingerprint, "splits": split_manifests}


def determine_rebuild_splits(previous_manifest: dict | None, current_manifest: dict) -> list[str]:
    current_splits = current_manifest["generated_splits"]
    if previous_manifest is None:
        return current_splits
    if previous_manifest.get("preprocess_params") != current_manifest.get("preprocess_params"):
        return current_splits
    if previous_manifest.get("requested_splits") != current_manifest.get("requested_splits"):
        return current_splits

    previous_fingerprints = previous_manifest.get("split_fingerprints", {})
    current_fingerprints = current_manifest.get("split_fingerprints", {})
    return [split for split in current_splits if previous_fingerprints.get(split) != current_fingerprints.get(split)]


def read_json_manifest(path: Path | str) -> dict | None:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_processed_root(root: Path | str) -> dict:
    processed_root = Path(root)
    manifest_path = processed_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid processed root {processed_root}: unreadable manifest") from exc

    if not isinstance(manifest.get("fingerprint"), str) or not manifest["fingerprint"].strip():
        raise ValueError(f"invalid processed root {processed_root}: missing fingerprint")
    generated_splits = manifest.get("generated_splits")
    if not isinstance(generated_splits, list):
        raise ValueError(f"invalid processed root {processed_root}: missing generated_splits")
    for split in ("train", "val", "test"):
        if split not in generated_splits:
            raise ValueError(f"invalid processed root {processed_root}: missing {split} generated split")
        split_root = processed_root / split
        if not split_root.is_dir():
            raise ValueError(f"invalid processed root {processed_root}: missing {split} split directory")
        if not any(
            all((match_dir / directory).is_dir() and any(path.is_file() for path in (match_dir / directory).rglob("*")) for directory in ("video", "csv", "frame"))
            for match_dir in split_root.iterdir() if match_dir.is_dir()
        ):
            raise ValueError(f"invalid processed root {processed_root}: missing {split} split contents")
    return manifest


def write_json_manifest(manifest: dict, path: Path | str) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def write_validation_report(report: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def validate_dataset(raw_dataset_root: Path | str, splits: Iterable[str] = RAW_SPLITS) -> dict:
    root = Path(raw_dataset_root)
    requested_splits = [split for split in splits]
    errors: list[dict] = []
    pairs: list[ClipPair] = []
    split_stats: dict[str, dict] = {}
    seen_cross_split: dict[tuple[str, str], str] = {}

    for split in requested_splits:
        split_dir = root / split
        games = sorted(path for path in split_dir.iterdir() if path.is_dir()) if split_dir.exists() else []
        split_clip_count = 0
        split_stats[split] = {"games": len(games), "clips": 0, "generated_name": generated_split_name(split)}

        for game_dir in games:
            video_dir = game_dir / "video"
            csv_dir = game_dir / "csv"
            videos = {path.stem: path for path in _iter_real_files(video_dir, ".mp4")}
            labels = {_clip_stem_from_csv(path): path for path in _iter_real_files(csv_dir, ".csv")}
            stems = sorted(set(videos) | set(labels))

            for stem in stems:
                video_path = videos.get(stem)
                csv_path = labels.get(stem)
                if video_path is None:
                    errors.append({"split": split, "game": game_dir.name, "file": f"{stem}_ball.csv", "reason": "missing video for clip"})
                    continue
                if csv_path is None:
                    errors.append({"split": split, "game": game_dir.name, "file": f"{stem}.mp4", "reason": "missing csv for clip"})
                    continue

                pair = ClipPair(split=split, game=game_dir.name, clip=stem, video_path=video_path, csv_path=csv_path)
                pairs.append(pair)
                split_clip_count += 1
                leakage_key = (game_dir.name, stem)
                previous_split = seen_cross_split.get(leakage_key)
                if previous_split is not None and previous_split != split:
                    errors.append({"split": split, "game": game_dir.name, "file": stem, "reason": "clip appears in multiple splits"})
                else:
                    seen_cross_split[leakage_key] = split
                errors.extend(_validate_clip_pair(pair))

        split_stats[split]["clips"] = split_clip_count

    return {
        "ok": not errors,
        "root": str(root),
        "pairs": [pair.__dict__ | {"video_path": str(pair.video_path), "csv_path": str(pair.csv_path)} for pair in pairs],
        "errors": errors,
        "stats": {
            "requested_splits": requested_splits,
            "total_clips": len(pairs),
            "splits": split_stats,
            "manifest": build_file_manifest([path for pair in pairs for path in (pair.video_path, pair.csv_path)]),
        },
    }


def _validate_clip_pair(pair: ClipPair) -> list[dict]:
    errors: list[dict] = []
    try:
        frame_count, width, height = _read_video_info(pair.video_path)
    except ValueError as exc:
        return [{"split": pair.split, "game": pair.game, "file": pair.video_path.name, "reason": str(exc)}]

    with pair.csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(column not in reader.fieldnames for column in REQUIRED_LABEL_COLUMNS):
            return [{"split": pair.split, "game": pair.game, "file": pair.csv_path.name, "reason": "missing required label columns"}]

        last_frame = -1
        for row_index, row in enumerate(reader, start=2):
            try:
                frame = int(row["Frame"])
                visibility = int(row["Visibility"])
                x = float(row["X"])
                y = float(row["Y"])
            except (TypeError, ValueError):
                errors.append({"split": pair.split, "game": pair.game, "file": pair.csv_path.name, "reason": "label row contains non-numeric values", "row": row_index})
                continue

            if frame <= last_frame:
                errors.append({"split": pair.split, "game": pair.game, "file": pair.csv_path.name, "reason": "frame values must be strictly increasing", "row": row_index})
            last_frame = frame

            if frame < 0:
                errors.append({"split": pair.split, "game": pair.game, "file": pair.csv_path.name, "reason": "frame index must be non-negative", "row": row_index})
            if frame >= frame_count:
                errors.append({"split": pair.split, "game": pair.game, "file": pair.csv_path.name, "reason": "frame index exceeds video length", "row": row_index})
            if visibility not in (0, 1):
                errors.append({"split": pair.split, "game": pair.game, "file": pair.csv_path.name, "reason": "visibility must be 0 or 1", "row": row_index})
            if visibility == 1 and not (0 <= x < width and 0 <= y < height):
                errors.append({"split": pair.split, "game": pair.game, "file": pair.csv_path.name, "reason": "visible coordinates fall outside video bounds", "row": row_index})

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate basketball TrackNet dataset layout without mutating raw data.")
    parser.add_argument("raw_dataset_root", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--split", action="append", dest="splits", choices=RAW_SPLITS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    report = validate_dataset(args.raw_dataset_root, splits=args.splits or RAW_SPLITS)
    if args.report is not None:
        write_validation_report(report, args.report)
    if report["errors"]:
        print(json.dumps(report["errors"], indent=2))
        return 1
    print(json.dumps(report["stats"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
