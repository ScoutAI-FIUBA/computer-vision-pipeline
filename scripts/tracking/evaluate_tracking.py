#!/usr/bin/env python3
"""Summarize manual tracking-review events into baseline counts.

Default output is the original single-run report. Optional: --per-clip, --json,
--validate, and --baseline/--candidate comparison (see docs/tracking-evaluation.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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


EVENT_KEYS = ("id_switch", "full_loss", "successful_recovery")


@dataclass
class ReviewEvent:
    clip: str
    event: str
    frame_index: int | None
    track_id: int | None
    notes: str | None
    stable_player_id: str | None = None


@dataclass
class ReviewFile:
    events: list[ReviewEvent] = field(default_factory=list)
    declared_clips: set[str] = field(default_factory=set)  # clips seen in events or clip_review records
    pending_clip_reviews: int = 0  # template records with any field still "pending"


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


def parse_review_record(blob: object) -> tuple[ReviewEvent | None, str]:
    """Parse one JSONL object. Returns (event, clip); event is None for `clip_review` template records."""
    if not isinstance(blob, dict):
        raise ValueError("record must be a JSON object")
    clip = str(blob.get("clip") or blob.get("video") or "default")
    if blob.get("record_type") == "clip_review":
        return None, clip
    event = normalize_event(blob.get("event") or blob.get("type"))
    frame_index = parse_optional_int(blob.get("frame_index"))
    if frame_index is not None and frame_index < 0:
        raise ValueError("frame_index must be >= 0")
    stable = blob.get("stable_player_id")
    if stable is not None and (not isinstance(stable, str) or not stable.strip()):
        raise ValueError("stable_player_id must be a non-empty string when present")
    return ReviewEvent(
        clip=clip,
        event=event,
        frame_index=frame_index,
        track_id=parse_optional_int(blob.get("track_id")),
        notes=None if blob.get("notes") is None else str(blob.get("notes")),
        stable_player_id=stable,
    ), clip


def load_review(path: Path, clip_filter: str | None) -> ReviewFile:
    """Load a review file, raising ValueError on the first bad line (line-numbered)."""
    review = ReviewFile()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                blob = json.loads(line)
                review_event, clip = parse_review_record(blob)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid review JSONL record at line {line_number}: {error}") from error
            if clip_filter is not None and clip != clip_filter:
                continue
            review.declared_clips.add(clip)
            if review_event is None:
                if any(v == "pending" for v in blob.values()):
                    review.pending_clip_reviews += 1
            else:
                review.events.append(review_event)
    return review


def load_review_events(path: Path, clip_filter: str | None) -> list[ReviewEvent]:
    return load_review(path, clip_filter).events


def validate_review_file(path: Path) -> list[str]:
    """Return every line-numbered problem in the review file (empty list when valid)."""
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parse_review_record(json.loads(line))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                errors.append(f"line {line_number}: {error}")
    return errors


def count_events(events: list[ReviewEvent]) -> dict:
    counts = Counter(event.event for event in events)
    outcomes = counts["full_loss"] + counts["successful_recovery"]
    stable_ids = sorted({e.stable_player_id for e in events if e.stable_player_id})
    return {
        "review_events": len(events),
        "id_switch": counts["id_switch"],
        "full_loss": counts["full_loss"],
        "successful_recovery": counts["successful_recovery"],
        "unrecovered_losses": max(0, counts["full_loss"] - counts["successful_recovery"]),
        "recovery_rate": counts["successful_recovery"] / outcomes if outcomes else None,
        "events_with_stable_player_id": sum(1 for e in events if e.stable_player_id),
        "stable_player_ids": stable_ids,
    }


def summarize_by_clip(review: ReviewFile) -> dict[str, dict]:
    grouped: dict[str, list[ReviewEvent]] = defaultdict(list)
    for clip in review.declared_clips:
        grouped[clip]
    for event in review.events:
        grouped[event.clip].append(event)
    return {clip: count_events(grouped[clip]) for clip in sorted(grouped)}


def compare_reviews(baseline: ReviewFile, candidate: ReviewFile) -> dict:
    """Deltas are candidate - baseline. Lower is better for id_switch/full_loss, higher for recoveries.

    Totals cover only clips present on both sides; one-sided clips are listed separately.
    """
    base, cand = summarize_by_clip(baseline), summarize_by_clip(candidate)
    clips = {}
    totals = {"baseline": Counter(), "candidate": Counter(), "delta": Counter()}
    for clip in sorted(set(base) | set(cand)):
        entry: dict = {"baseline": base.get(clip), "candidate": cand.get(clip)}
        if clip not in base or clip not in cand:
            entry["only_in"] = "candidate" if clip not in base else "baseline"
            entry["delta"] = None
        else:
            entry["delta"] = {k: cand[clip][k] - base[clip][k] for k in EVENT_KEYS}
            for key in EVENT_KEYS:
                totals["baseline"][key] += base[clip][key]
                totals["candidate"][key] += cand[clip][key]
                totals["delta"][key] += entry["delta"][key]
        clips[clip] = entry
    return {
        "clips": clips,
        "total_shared_clips": {name: {k: totals[name][k] for k in EVENT_KEYS} for name in totals},
        "one_sided_clips": sorted(c for c, e in clips.items() if "only_in" in e),
    }


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
        nargs="?",
        help="Manual review JSONL. One line per event with event=id_switch|full_loss|successful_recovery. "
        "Optional stable_player_id is the canonical identity; track_id is only event provenance.",
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
    parser.add_argument("--per-clip", action="store_true", help="Also print a per-clip breakdown.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate the review JSONL contract, list every line-numbered error, and exit (1 if invalid).",
    )
    parser.add_argument("--baseline", type=Path, help="Baseline review JSONL (use with --candidate).")
    parser.add_argument("--candidate", type=Path, help="Candidate review JSONL (use with --baseline).")
    return parser


def format_rate(counts: dict) -> str:
    outcomes = counts["full_loss"] + counts["successful_recovery"]
    if counts["recovery_rate"] is None:
        return "n/a (no loss/recovery events recorded)"
    return f"{counts['successful_recovery']}/{outcomes} ({counts['recovery_rate'] * 100.0:.1f}%)"


def print_comparison(result: dict, baseline: Path, candidate: Path) -> None:
    print("Tracking baseline vs candidate review")
    print(f"Baseline: {baseline}")
    print(f"Candidate: {candidate}")
    print("Deltas are candidate - baseline (lower is better for id_switch/full_loss, higher for successful_recovery).")
    for clip, entry in result["clips"].items():
        if entry["delta"] is None:
            print(f"Clip {clip}: only in {entry['only_in']} review, not compared")
            continue
        parts = [
            f"{key}={entry['baseline'][key]}->{entry['candidate'][key]} ({entry['delta'][key]:+d})"
            for key in EVENT_KEYS
        ]
        print(f"Clip {clip}: " + " ".join(parts))
    total = result["total_shared_clips"]
    print("Total (clips on both sides): " + " ".join(
        f"{key}={total['baseline'][key]}->{total['candidate'][key]} ({total['delta'][key]:+d})"
        for key in EVENT_KEYS
    ))
    if result["one_sided_clips"]:
        print("WARNING: clips present on one side only: " + ", ".join(result["one_sided_clips"]))


def run_compare(args: argparse.Namespace) -> int:
    for label, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        if path is None:
            raise ValueError("--baseline and --candidate must be given together")
    result = compare_reviews(load_review(args.baseline, args.clip), load_review(args.candidate, args.clip))
    if args.json:
        print(json.dumps({"baseline": str(args.baseline), "candidate": str(args.candidate), **result}, indent=2))
    else:
        print_comparison(result, args.baseline, args.candidate)
    return 0


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        if args.baseline or args.candidate:
            if args.review_jsonl is not None:
                parser.error("review_jsonl cannot be combined with --baseline/--candidate")
            raise SystemExit(run_compare(args))
        if args.review_jsonl is None:
            parser.error("review_jsonl is required (or use --baseline and --candidate)")
        if args.validate:
            errors = validate_review_file(args.review_jsonl)
            for error in errors:
                print(f"INVALID {args.review_jsonl}: {error}", file=sys.stderr)
            if not errors:
                print(f"Review file OK: {args.review_jsonl}")
            raise SystemExit(1 if errors else 0)
        review = load_review(args.review_jsonl, args.clip)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)

    events = review.events
    counts = count_events(events)
    summary = load_track_summary(args.tracks_jsonl) if args.tracks_jsonl is not None else None

    if args.json:
        print(json.dumps({
            "review_file": str(args.review_jsonl),
            "clip_filter": args.clip,
            "tracks_file": None if args.tracks_jsonl is None else str(args.tracks_jsonl),
            "track_context": summary,
            "aggregate": counts,
            "clips": summarize_by_clip(review),
            "pending_clip_reviews": review.pending_clip_reviews,
        }, indent=2))
        return

    print("Tracking baseline review")
    print(f"Review file: {args.review_jsonl}")
    print(f"Clip filter: {args.clip or 'all'}")

    if summary is not None:
        print(f"Tracks file: {args.tracks_jsonl}")
        print(
            "Track context: "
            f"sampled_frames={summary['sampled_frames']} "
            f"unique_track_ids={summary['unique_track_ids']} "
            f"predicted_rows={summary['predicted_rows']}"
        )
        if summary["min_frame"] is not None:
            print(f"Track frame range: {summary['min_frame']}..{summary['max_frame']}")

    print(f"Review events: {counts['review_events']}")
    print(f"ID switches: {counts['id_switch']}")
    print(f"Full track losses: {counts['full_loss']}")
    print(f"Successful recoveries: {counts['successful_recovery']}")
    print(f"Unrecovered losses: {counts['unrecovered_losses']}")
    print(f"Occlusion recovery rate: {format_rate(counts)}")

    if args.per_clip:
        print("Per clip:")
        for clip, clip_counts in summarize_by_clip(review).items():
            print(
                f"  {clip}: events={clip_counts['review_events']} id_switch={clip_counts['id_switch']} "
                f"full_loss={clip_counts['full_loss']} successful_recovery={clip_counts['successful_recovery']} "
                f"unrecovered={clip_counts['unrecovered_losses']} recovery_rate={format_rate(clip_counts)}"
            )
        if counts["stable_player_ids"]:
            print("Stable player ids in events: " + ", ".join(counts["stable_player_ids"]))
        if review.pending_clip_reviews:
            print(f"Pending clip_review records: {review.pending_clip_reviews}")


if __name__ == "__main__":
    main()
