#!/usr/bin/env python3
"""Unified video pipeline entrypoint for player detection, tracking, and JSONL output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.court.detect_court_keypoints import DEFAULT_MODEL_ID as COURT_MODEL_ID  # noqa: E402
from scripts.shared.env_config import get_roboflow_api_key  # noqa: E402
from scripts.tracking.track_players_video import (  # noqa: E402
    DEFAULT_DEEPOCSORT_CONFIG_PATH,
    DEFAULT_REID_CONFIG_PATH,
    inspect_video,
    process_video,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the video pipeline with player detection, tracking, and per-inference JSONL output."
    )
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Path for the annotated tracked video")
    parser.add_argument("--tracks-output", type=Path, help="Path for JSONL track records")
    parser.add_argument("--json-output", type=Path, help="Path for per-inference raw JSONL records")
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=True, help="Write per-inference raw JSONL by default")
    parser.add_argument(
        "--court-keypoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run court keypoint inference on sampled frames and overlay/export the results.",
    )
    parser.add_argument("--court-json-output", type=Path, help="Path for per-frame court keypoint JSONL records")
    parser.add_argument("--projected-tracks-output", type=Path, help="Path for per-track projected court coordinate JSONL records")
    parser.add_argument(
        "--court-model-id",
        default=COURT_MODEL_ID,
        help="Roboflow court model ID. Defaults to COURT_MODEL_ID.",
    )
    parser.add_argument(
        "--court-confidence",
        type=float,
        default=0.7,
        help="Minimum court keypoint confidence to keep. Default: 0.7",
    )
    parser.add_argument("--frame-stride", type=int, default=30, help="Infer every N frames. Default: 30")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to read")
    parser.add_argument("--dry-run", action="store_true", help="Print video info and estimated calls without inference")
    parser.add_argument("--confidence", type=float, default=0.0, help="Minimum confidence to track and draw")
    parser.add_argument(
        "--tracker-backend",
        choices=("bytetrack", "deepocsort"),
        default="bytetrack",
        help="Tracking backend. Default: bytetrack",
    )
    parser.add_argument(
        "--tracker-config",
        type=Path,
        default=DEFAULT_DEEPOCSORT_CONFIG_PATH,
        help=(
            "Deep OC-SORT tracker config file. This is the main place to tune tracker behavior; "
            "the --deepocsort-* flags only override values from this file. "
            f"Default: {DEFAULT_DEEPOCSORT_CONFIG_PATH}"
        ),
    )
    parser.add_argument(
        "--reid-config",
        type=Path,
        default=DEFAULT_REID_CONFIG_PATH,
        help=(
            "ReID crop/quality config file for the lightweight appearance reassignment path. "
            f"Default: {DEFAULT_REID_CONFIG_PATH}"
        ),
    )
    parser.add_argument(
        "--deepocsort-reid-weights",
        default=None,
        help=(
            "Override reid_weights from --tracker-config when --tracker-backend deepocsort is selected."
        ),
    )
    parser.add_argument(
        "--deepocsort-device",
        default=None,
        help="Override device from --tracker-config when --tracker-backend deepocsort is selected.",
    )
    parser.add_argument(
        "--deepocsort-half",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override half from --tracker-config when --tracker-backend deepocsort is selected.",
    )
    parser.add_argument(
        "--sampled-output-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write only inferred frames by default. Use --no-sampled-output-only to include skipped frames unannotated.",
    )
    parser.add_argument("--output-fps", type=float, help="Output video FPS. Defaults to input FPS / frame stride for sampled output.")
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Average each tracked box/label over this many sampled detections. Default: 1 (disabled)",
    )
    parser.add_argument(
        "--reuse-stale-tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse the last smoothed tracked boxes on skipped frames. Off by default to avoid stale overlays.",
    )
    parser.add_argument(
        "--persist-tracks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep recent track IDs alive briefly with predicted boxes on missed sampled detections. Default: on",
    )
    parser.add_argument(
        "--max-missing-sampled-frames",
        type=int,
        default=1,
        help="Maximum sampled-frame gap to keep a track alive with prediction. Default: 1",
    )
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    if args.output_fps is not None and args.output_fps <= 0:
        raise SystemExit("--output-fps must be greater than 0")
    if args.smoothing_window < 1:
        raise SystemExit("--smoothing-window must be at least 1")
    if args.max_missing_sampled_frames < 0:
        raise SystemExit("--max-missing-sampled-frames must be at least 0")
    if args.court_confidence < 0 or args.court_confidence > 1:
        raise SystemExit("--court-confidence must be between 0 and 1")

    args.output = args.output or Path("outputs") / f"{args.video.stem}_pipeline.mp4"
    args.tracks_output = args.tracks_output or Path("outputs") / f"{args.video.stem}_tracks.jsonl"
    args.json_output = args.json_output or Path("outputs") / f"{args.video.stem}_inference.jsonl"
    args.court_json_output = args.court_json_output or Path("outputs") / f"{args.video.stem}_court_keypoints.jsonl"
    args.projected_tracks_output = args.projected_tracks_output or Path("outputs") / f"{args.video.stem}_projected_tracks.jsonl"

    try:
        info = inspect_video(args.video, args.frame_stride, args.max_frames)
        if args.dry_run:
            estimated_calls = info["estimated_roboflow_calls"] * (2 if args.court_keypoints else 1)
            print(f"Video: {args.video}")
            print(f"FPS: {info['fps'] or 'unknown'}")
            print(f"Frame count: {info['frame_count'] or 'unknown'}")
            print(f"Frames to read: {info['frames_to_read'] or 'unknown'}")
            print(f"Estimated Roboflow calls: {estimated_calls}")
            print(f"Sampled output only: {args.sampled_output_only}")
            print(f"Save JSONL: {args.save_json}")
            print(f"Court keypoints: {args.court_keypoints}")
            if args.court_keypoints:
                print(f"Court model ID: {args.court_model_id}")
            return
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    api_key = get_roboflow_api_key()
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")

    try:
        stats = process_video(
            args.video,
            args.output,
            args.tracks_output,
            api_key,
            json_output=args.json_output if args.save_json else None,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            sampled_output_only=args.sampled_output_only,
            confidence=args.confidence,
            output_fps=args.output_fps,
            tracker_backend=args.tracker_backend,
            tracker_config=args.tracker_config,
            reid_config=args.reid_config,
            deepocsort_reid_weights=args.deepocsort_reid_weights,
            deepocsort_device=args.deepocsort_device,
            deepocsort_half=args.deepocsort_half,
            smoothing_window=args.smoothing_window,
            reuse_stale_tracks=args.reuse_stale_tracks,
            persist_tracks=args.persist_tracks,
            max_missing_sampled_frames=args.max_missing_sampled_frames,
            court_keypoints=args.court_keypoints,
            court_json_output=args.court_json_output if args.court_keypoints else None,
            court_model_id=args.court_model_id,
            court_confidence=args.court_confidence,
            projected_tracks_output=args.projected_tracks_output if args.court_keypoints else None,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(f"Frames read: {stats['frames_read']}/{stats['frame_count'] or 'unknown'}")
    print(f"Roboflow calls: {stats['roboflow_calls']}")
    print(f"Output FPS: {stats['output_fps']}")
    print(f"Tracker backend: {stats['tracker_backend']}")
    if stats["tracker_config"] is not None:
        print(f"Tracker config: {stats['tracker_config']}")
    print(f"ReID config: {stats['reid_config']}")
    print(f"Track records: {stats['track_records']}")
    if args.save_json:
        print(f"Inference records: {stats['inference_records']}")
    if args.court_keypoints:
        print(f"Court frames: {stats['court_frames']}")
        print(f"Court Roboflow calls: {stats['court_roboflow_calls']}")
        print(f"Court keypoint records: {stats['court_inference_records']}")
        print(f"Projected track records: {stats['projected_track_records']}")
        print(f"Projected track frames: {stats['projected_track_frames']}")
    print(f"Tracked video saved to: {args.output}")
    print(f"Track JSONL saved to: {args.tracks_output}")
    if args.save_json:
        print(f"Inference JSONL saved to: {args.json_output}")
    if args.court_keypoints:
        print(f"Court JSONL saved to: {args.court_json_output}")
        print(f"Projected track JSONL saved to: {args.projected_tracks_output}")


if __name__ == "__main__":
    main()
