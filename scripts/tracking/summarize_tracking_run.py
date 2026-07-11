#!/usr/bin/env python3
"""Print a concise technical summary for one tracks JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def load_summary(path: Path) -> dict[str, object]:
    frame_indexes: set[int] = set()
    track_ids: set[int] = set()
    track_frames: dict[int, set[int]] = defaultdict(set)
    total_rows = 0
    detected_rows = 0
    predicted_rows = 0
    other_state_rows = 0
    state_rows_seen = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                blob = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid track JSONL record at line {line_number}: {error}") from error

            total_rows += 1
            frame_index = int(blob["frame_index"])
            frame_indexes.add(frame_index)

            if "track_state" in blob:
                state_rows_seen += 1
                state = blob.get("track_state")
                if state == "detected":
                    detected_rows += 1
                elif state == "predicted":
                    predicted_rows += 1
                else:
                    other_state_rows += 1

            track_id = blob.get("track_id")
            if track_id is None:
                continue

            track_id = int(track_id)
            track_ids.add(track_id)
            track_frames[track_id].add(frame_index)

    track_spans = {track_id: len(frames) for track_id, frames in track_frames.items()}
    longest_tracks = sorted(
        track_frames.items(),
        key=lambda item: (-len(item[1]), min(item[1]), item[0]),
    )[:5]

    return {
        "total_rows": total_rows,
        "sampled_frames": len(frame_indexes),
        "min_frame": min(frame_indexes) if frame_indexes else None,
        "max_frame": max(frame_indexes) if frame_indexes else None,
        "unique_track_ids": len(track_ids),
        "detected_rows": detected_rows if state_rows_seen else None,
        "predicted_rows": predicted_rows if state_rows_seen else None,
        "other_state_rows": other_state_rows if state_rows_seen else None,
        "one_frame_tracks": sum(1 for span in track_spans.values() if span == 1),
        "short_lived_tracks": sum(1 for span in track_spans.values() if span <= 2),
        "average_track_span": mean(track_spans.values()) if track_spans else None,
        "median_track_span": median(track_spans.values()) if track_spans else None,
        "longest_tracks": [
            {
                "track_id": track_id,
                "sampled_frames": len(frames),
                "first_frame": min(frames),
                "last_frame": max(frames),
            }
            for track_id, frames in longest_tracks
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print technical summary metrics for one tracking run JSONL.")
    parser.add_argument("tracks_jsonl", type=Path, help="Path to tracks JSONL from track_players_video.py or video_pipeline.py")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = load_summary(args.tracks_jsonl)

    print("Tracking run summary")
    print(f"Tracks file: {args.tracks_jsonl}")
    print(f"Rows: {summary['total_rows']}")
    print(f"Sampled frames: {summary['sampled_frames']}")
    if summary["min_frame"] is None:
        print("Frame range: n/a")
    else:
        print(f"Frame range: {summary['min_frame']}..{summary['max_frame']}")
    print(f"Unique track IDs: {summary['unique_track_ids']}")

    if summary["detected_rows"] is None:
        print("Observed rows: n/a")
        print("Predicted rows: n/a")
    else:
        print(f"Observed rows: {summary['detected_rows']}")
        print(f"Predicted rows: {summary['predicted_rows']}")
        if summary["other_state_rows"]:
            print(f"Other track_state rows: {summary['other_state_rows']}")

    print(f"One-frame tracks: {summary['one_frame_tracks']}")
    print(f"Short-lived tracks (<=2 sampled frames): {summary['short_lived_tracks']}")

    if summary["average_track_span"] is None:
        print("Average track span (sampled frames): n/a")
        print("Median track span (sampled frames): n/a")
    else:
        print(f"Average track span (sampled frames): {summary['average_track_span']:.2f}")
        print(f"Median track span (sampled frames): {summary['median_track_span']:.2f}")

    print("Longest tracks:")
    for item in summary["longest_tracks"]:
        print(
            f"- track_id={item['track_id']} sampled_frames={item['sampled_frames']} "
            f"frame_range={item['first_frame']}..{item['last_frame']}"
        )


if __name__ == "__main__":
    main()
