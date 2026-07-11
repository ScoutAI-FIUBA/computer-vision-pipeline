#!/usr/bin/env python3
"""Render projected player tracks on a top-down FIBA court."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.court.build_court_homography import draw_fiba_court


COURT_LENGTH_M = 28.0
COURT_WIDTH_M = 15.0


@dataclass
class ProjectedTrackRecord:
    frame_index: int
    track_id: int | None
    class_name: str
    confidence: float | None
    court_xy_m: tuple[float, float]


def parse_record(line: str) -> ProjectedTrackRecord:
    blob = json.loads(line)
    court_xy_m = blob.get("court_xy_m")
    if not isinstance(court_xy_m, list) or len(court_xy_m) < 2:
        raise ValueError("Missing court_xy_m in projected track record")
    return ProjectedTrackRecord(
        frame_index=int(blob["frame_index"]),
        track_id=None if blob.get("track_id") is None else int(blob["track_id"]),
        class_name=str(blob.get("class_name", "player")),
        confidence=None if blob.get("confidence") is None else float(blob["confidence"]),
        court_xy_m=(float(court_xy_m[0]), float(court_xy_m[1])),
    )


def load_projected_tracks(path: Path) -> list[ProjectedTrackRecord]:
    records: list[ProjectedTrackRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(parse_record(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid JSONL record at line {line_number}: {error}") from error
    return records


def point_in_bounds(x_m: float, y_m: float) -> bool:
    return 0.0 <= x_m <= COURT_LENGTH_M and 0.0 <= y_m <= COURT_WIDTH_M


def track_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (90, 90, 90)
    palette = (
        (255, 99, 71),
        (65, 105, 225),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (231, 76, 60),
    )
    r, g, b = palette[track_id % len(palette)]
    return (b, g, r)


def meter_to_px(x_m: float, y_m: float, pixels_per_meter: int) -> tuple[int, int]:
    return int(round(x_m * pixels_per_meter)), int(round(y_m * pixels_per_meter))


def draw_summary_png(records: list[ProjectedTrackRecord], output_path: Path, pixels_per_meter: int, point_radius: int) -> int:
    import cv2

    court = draw_fiba_court(pixels_per_meter=pixels_per_meter)
    plotted = 0
    counts_by_track: dict[int | None, int] = defaultdict(int)

    for record in records:
        x_m, y_m = record.court_xy_m
        if not point_in_bounds(x_m, y_m):
            continue
        x_px, y_px = meter_to_px(x_m, y_m, pixels_per_meter)
        color = track_color(record.track_id)
        counts_by_track[record.track_id] += 1
        alpha = 0.25 if counts_by_track[record.track_id] > 1 else 0.75
        overlay = court.copy()
        cv2.circle(overlay, (x_px, y_px), point_radius, color, -1)
        court = cv2.addWeighted(overlay, alpha, court, 1.0 - alpha, 0.0)
        plotted += 1

    cv2.putText(
        court,
        f"Projected player positions: {plotted} points across {len({record.track_id for record in records})} tracks",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), court):
        raise RuntimeError(f"Could not write summary PNG: {output_path}")
    return plotted


def draw_frame_overlay(frame_records: list[ProjectedTrackRecord], pixels_per_meter: int, point_radius: int) -> Any:
    import cv2

    court = draw_fiba_court(pixels_per_meter=pixels_per_meter)
    plotted = 0
    for record in frame_records:
        x_m, y_m = record.court_xy_m
        if not point_in_bounds(x_m, y_m):
            continue
        x_px, y_px = meter_to_px(x_m, y_m, pixels_per_meter)
        color = track_color(record.track_id)
        cv2.circle(court, (x_px, y_px), point_radius, color, -1)
        label = "?" if record.track_id is None else str(record.track_id)
        cv2.putText(court, label, (x_px + point_radius + 2, y_px - point_radius - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        plotted += 1

    cv2.putText(
        court,
        f"frame tracks: {plotted}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return court


def write_debug_video(
    records: list[ProjectedTrackRecord],
    output_path: Path,
    pixels_per_meter: int,
    point_radius: int,
    output_fps: float,
    sample_every: int,
) -> tuple[int, int]:
    import cv2

    if output_fps <= 0:
        raise ValueError("--output-fps must be greater than 0")

    grouped: dict[int, list[ProjectedTrackRecord]] = defaultdict(list)
    for record in records:
        grouped[record.frame_index].append(record)

    if not grouped:
        raise ValueError("No projected track records available for video output")

    first_frame = draw_frame_overlay(grouped[min(grouped)], pixels_per_meter, point_radius)
    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not write debug video: {output_path}")

    frames_written = 0
    plotted = 0
    try:
        for frame_index in sorted(grouped):
            if sample_every > 1 and frame_index % sample_every != 0:
                continue
            frame = draw_frame_overlay(grouped[frame_index], pixels_per_meter, point_radius)
            cv2.putText(
                frame,
                f"frame={frame_index}",
                (12, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frames_written += 1
            plotted += sum(1 for record in grouped[frame_index] if point_in_bounds(*record.court_xy_m))
    finally:
        writer.release()
    return frames_written, plotted


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__ or "Render projected player tracks on a top-down FIBA court.")
    parser.add_argument("projected_tracks_jsonl", type=Path, help="Projected track JSONL from track_players_video.py or video_pipeline.py")
    parser.add_argument("--output-png", type=Path, help="Path for the top-down summary PNG")
    parser.add_argument("--output-video", type=Path, help="Optional path for a simple top-down MP4")
    parser.add_argument("--pixels-per-meter", type=int, default=40, help="Court render scale. Default: 40")
    parser.add_argument("--point-radius", type=int, default=6, help="Circle radius for plotted player positions. Default: 6")
    parser.add_argument("--output-fps", type=float, default=5.0, help="FPS for --output-video. Default: 5")
    parser.add_argument("--sample-every", type=int, default=1, help="Write every Nth sampled frame to --output-video. Default: 1")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect the JSONL and report what would be written")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.pixels_per_meter < 10:
        raise SystemExit("--pixels-per-meter must be at least 10")
    if args.point_radius < 1:
        raise SystemExit("--point-radius must be at least 1")
    if args.sample_every < 1:
        raise SystemExit("--sample-every must be at least 1")
    if not args.projected_tracks_jsonl.exists():
        raise SystemExit(f"Projected track JSONL not found: {args.projected_tracks_jsonl}")

    output_png = args.output_png or Path("outputs") / f"{args.projected_tracks_jsonl.stem}_court_debug.png"
    output_video = args.output_video

    records = load_projected_tracks(args.projected_tracks_jsonl)
    if not records:
        raise SystemExit("No projected track records found in input JSONL")

    frame_indexes = {record.frame_index for record in records}
    track_ids = {record.track_id for record in records}
    in_bounds = sum(1 for record in records if point_in_bounds(*record.court_xy_m))

    print(f"Input JSONL: {args.projected_tracks_jsonl}")
    print(f"Projected track records: {len(records)}")
    print(f"Frames with projected tracks: {len(frame_indexes)}")
    print(f"Track IDs: {len(track_ids)}")
    print(f"In-bounds projected positions: {in_bounds}")
    print(f"Summary PNG: {output_png}")
    if output_video is not None:
        print(f"Debug video: {output_video}")

    if args.dry_run:
        return

    plotted_png = draw_summary_png(records, output_png, args.pixels_per_meter, args.point_radius)
    print(f"PNG plotted points: {plotted_png}")

    if output_video is not None:
        frames_written, plotted_video = write_debug_video(
            records,
            output_video,
            args.pixels_per_meter,
            args.point_radius,
            args.output_fps,
            args.sample_every,
        )
        print(f"Video frames written: {frames_written}")
        print(f"Video plotted points: {plotted_video}")


if __name__ == "__main__":
    main()
