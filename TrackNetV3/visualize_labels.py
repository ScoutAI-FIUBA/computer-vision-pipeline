from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from basketball_data import validate_dataset


def render_label_samples(video_path: Path | str, csv_path: Path | str, output_dir: Path | str, *, sample_count: int = 3) -> list[Path]:
    video_path = Path(video_path)
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["Visibility"]) == 1:
                rows.append((int(row["Frame"]), int(float(row["X"])), int(float(row["Y"]))))
    rows = rows[:sample_count]

    capture = cv2.VideoCapture(str(video_path))
    outputs: list[Path] = []
    try:
        for frame_index, x, y in rows:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            output_path = output_dir / f"{video_path.stem}_frame_{frame_index:04d}.png"
            cv2.imwrite(str(output_path), frame)
            outputs.append(output_path)
    finally:
        capture.release()

    return outputs


def render_dataset_samples(raw_dataset_root: Path | str, output_dir: Path | str, *, max_clips: int = 3, sample_count: int = 3) -> list[Path]:
    report = validate_dataset(raw_dataset_root)
    if report["errors"]:
        raise ValueError("dataset validation failed")

    outputs: list[Path] = []
    for pair in report["pairs"][:max_clips]:
        outputs.extend(
            render_label_samples(pair["video_path"], pair["csv_path"], Path(output_dir) / pair["split"] / pair["game"], sample_count=sample_count)
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Render sampled label overlays without mutating the raw dataset.")
    parser.add_argument("raw_dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-clips", type=int, default=3)
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()

    render_dataset_samples(args.raw_dataset_root, args.output_dir, max_clips=args.max_clips, sample_count=args.sample_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
