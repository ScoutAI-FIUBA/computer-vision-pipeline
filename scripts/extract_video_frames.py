#!/usr/bin/env python3
"""Extract sampled frames from a local video into a local directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.detect_players_video import open_video, video_info  # noqa: E402


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_stride: int = 30,
    max_frames: int | None = None,
) -> dict:
    import cv2

    capture = open_video(video_path)
    _, frame_count, frames_to_read = video_info(capture, max_frames)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    saved_frames = 0

    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_stride == 0:
                output_path = output_dir / f"frame_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Failed to write frame image: {output_path}")
                saved_frames += 1

            frame_index += 1
    finally:
        capture.release()

    return {
        "frames_read": frame_index,
        "frame_count": frame_count or None,
        "frames_to_read": frames_to_read,
        "saved_frames": saved_frames,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sampled frames from a local video.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--frame-stride", type=int, default=30, help="Save every Nth frame. Default: 30")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to read")
    parser.add_argument("--output-dir", type=Path, help="Directory for extracted frame images")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")

    output_dir = args.output_dir or Path("outputs") / f"{args.video.stem}_frames"

    try:
        stats = extract_frames(
            args.video,
            output_dir,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(f"Frames read: {stats['frames_read']}/{stats['frame_count'] or 'unknown'}")
    print(f"Frames eligible: {stats['frames_to_read']}")
    print(f"Frames saved: {stats['saved_frames']}")
    print(f"Output directory: {stats['output_dir']}")


if __name__ == "__main__":
    main()
