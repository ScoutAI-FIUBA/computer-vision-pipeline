#!/usr/bin/env python3
"""Summarize manual tracking-review events into baseline counts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


EVENT_ALIASES = {
    "id_switch": "id_switch",
    "switch": "id_switch",
    "full_loss": "full_loss",
    "loss": "full_loss",
    "track_loss": "full_loss",
    "successful_recovery": "successful_recovery",
    "recovery": "successful_recovery",
}


@dataclass
class ReviewEvent:
    clip: str
    event: str
    frame_index: int | None
    track_id: int | None
    notes: str | None


def normalize_event(raw_event: object) -> str:
    if not isinstance(raw_event, str):
        raise ValueError("Review event must include a string event field")
    event = EVENT_ALIASES.get(raw_event.strip().lower())
    if event is None:
        supported = ", ".join(sorted(EVENT_ALIASES))
        raise ValueError(f"Unsupported event '{raw_event}'. Supported values: {supported}")
    return event


def parse_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def load_review_events(path: Path, clip_filter: str | None) -> list[ReviewEvent]:
    events: list[ReviewEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                blob = json.loads(line)
                clip = str(blob.get("clip") or blob.get("video") or "default")
                event = normalize_event(blob.get("event") or blob.get("type"))
                review_event = ReviewEvent(
                    clip=clip,
                    event=event,
                    frame_index=parse_optional_int(blob.get("frame_index")),
                    track_id=parse_optional_int(blob.get("track_id")),
                    notes=None if blob.get("notes") is None else str(blob.get("notes")),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid review JSONL record at line {line_number}: {error}") from error
            if clip_filter is None or review_event.clip == clip_filter:
                events.append(review_event)
    return events


def load_track_summary(path: Path) -> dict[str, int | None]:
    frame_indexes: set[int] = set()
    track_ids: set[int] = set()
    predicted_rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                blob = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid track JSONL record at line {line_number}: {error}") from error

            frame_indexes.add(int(blob["frame_index"]))
            if blob.get("track_id") is not None:
                track_ids.add(int(blob["track_id"]))
            if blob.get("track_state") == "predicted":
                predicted_rows += 1

    return {
        "sampled_frames": len(frame_indexes),
        "unique_track_ids": len(track_ids),
        "predicted_rows": predicted_rows,
        "min_frame": min(frame_indexes) if frame_indexes else None,
        "max_frame": max(frame_indexes) if frame_indexes else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count manual tracking-review events for the current baseline workflow."
    )
    parser.add_argument(
        "review_jsonl",
        type=Path,
        help="Manual review JSONL. One line per event with event=id_switch|full_loss|successful_recovery.",
    )
    parser.add_argument(
        "--tracks-jsonl",
        type=Path,
        help="Optional tracks.jsonl from track_players_video.py or video_pipeline.py for clip context.",
    )
    parser.add_argument(
        "--clip",
        help="Optional clip name filter when the review JSONL contains multiple clips.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    events = load_review_events(args.review_jsonl, args.clip)
    counts = Counter(event.event for event in events)
    total_occlusion_outcomes = counts["full_loss"] + counts["successful_recovery"]
    recovery_rate = (
        (counts["successful_recovery"] / total_occlusion_outcomes) * 100.0
        if total_occlusion_outcomes
        else None
    )

    print("Tracking baseline review")
    print(f"Review file: {args.review_jsonl}")
    print(f"Clip filter: {args.clip or 'all'}")

    if args.tracks_jsonl is not None:
        summary = load_track_summary(args.tracks_jsonl)
        print(f"Tracks file: {args.tracks_jsonl}")
        print(
            "Track context: "
            f"sampled_frames={summary['sampled_frames']} "
            f"unique_track_ids={summary['unique_track_ids']} "
            f"predicted_rows={summary['predicted_rows']}"
        )
        if summary["min_frame"] is not None:
            print(f"Track frame range: {summary['min_frame']}..{summary['max_frame']}")

    print(f"Review events: {len(events)}")
    print(f"ID switches: {counts['id_switch']}")
    print(f"Full track losses: {counts['full_loss']}")
    print(f"Successful recoveries: {counts['successful_recovery']}")
    print(f"Unrecovered losses: {max(0, counts['full_loss'] - counts['successful_recovery'])}")
    if recovery_rate is None:
        print("Occlusion recovery rate: n/a (no loss/recovery events recorded)")
    else:
        print(
            "Occlusion recovery rate: "
            f"{counts['successful_recovery']}/{total_occlusion_outcomes} ({recovery_rate:.1f}%)"
        )


if __name__ == "__main__":
    main()
