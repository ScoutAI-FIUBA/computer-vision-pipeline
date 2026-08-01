"""Local CLI for the pipeline.

    python -m pipeline.cli path/to/clip.mp4 --max-frames 90
    python -m pipeline.cli path/to/clip.mp4 --dry-run

`--dry-run` only probes the video and estimates detector calls, so it works on a
machine with no GPU and no SAM2 installed. Use it to sanity-check cost before
spending anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from pipeline.config import DEFAULT_TARGET_FPS, PipelineConfig
from pipeline.runner import run_pipeline
from pipeline.video import frame_step_for, probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run basketball tracking on a local video.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for tracks.jsonl and results.json. Defaults to outputs/<video stem>",
    )
    parser.add_argument("--frame-stride", type=int, help="Run the detector every N frames")
    parser.add_argument("--max-frames", type=int, help="Stop after N frames")
    parser.add_argument("--confidence", type=float, help="Minimum detector confidence")
    parser.add_argument(
        "--target-fps",
        type=float,
        help="Process the video at this rate instead of its native one. Biggest cost lever.",
    )
    parser.add_argument("--sam2-model-id", help="SAM2 checkpoint, e.g. facebook/sam2-hiera-small")
    parser.add_argument(
        "--annotated-video",
        action="store_true",
        help="Also write annotated.mp4. Costs extra runtime; off by default.",
    )
    parser.add_argument(
        "--team-embedder-backend",
        choices=("siglip", "colour", "none"),
        help="Team ID embedder backend",
    )
    parser.add_argument(
        "--team-embedder-model-id",
        help="SigLIP model id for team embeddings, e.g. vit_base_patch16_siglip_224",
    )
    parser.add_argument("--team-count", type=int, help="Number of teams to cluster")
    parser.add_argument(
        "--min-team-samples-per-track",
        type=int,
        help="Skip tracks with fewer team-embedding samples than this",
    )
    parser.add_argument(
        "--team-assignment-margin",
        type=float,
        help="Leave a track unassigned when both team clusters score too similarly",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe the video and estimate detector calls without running anything",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1

    stride = args.frame_stride or PipelineConfig.__dataclass_fields__["frame_stride"].default

    if args.dry_run:
        info = probe(args.video)
        target = args.target_fps if args.target_fps is not None else DEFAULT_TARGET_FPS
        step = frame_step_for(info.fps, target)

        available = math.ceil(info.frame_count / step) if info.frame_count > 0 else 0
        frames = available if args.max_frames is None else min(available, args.max_frames)
        estimated_calls = 0 if frames <= 0 else math.ceil(frames / stride)

        # Measured on an L4 with hiera-small. Recalculate this if you change GPU
        # or checkpoint — it is the number that tells you whether a run is
        # affordable before you start it.
        throughput_fps = 3.3
        estimated_seconds = frames / throughput_fps if frames else 0.0

        print(
            json.dumps(
                {
                    "video": str(args.video),
                    "source_fps": info.fps,
                    "frame_count": info.frame_count,
                    "duration_seconds": info.duration_seconds,
                    "target_fps": target,
                    "frame_step": step,
                    "frames_to_process": frames,
                    "frame_stride": stride,
                    "estimated_detector_calls": estimated_calls,
                    "estimated_gpu_seconds": round(estimated_seconds, 1),
                    "estimated_gpu_minutes": round(estimated_seconds / 60, 1),
                },
                indent=2,
            )
        )
        return 0

    config = PipelineConfig.from_env(
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        confidence=args.confidence,
        target_fps=args.target_fps,
        sam2_model_id=args.sam2_model_id,
        team_embedder_backend=args.team_embedder_backend,
        team_embedder_model_id=args.team_embedder_model_id,
        team_count=args.team_count,
        min_team_samples_per_track=args.min_team_samples_per_track,
        team_assignment_margin=args.team_assignment_margin,
        write_annotated_video=True if args.annotated_video else None,
    )

    output_dir = args.output_dir or Path("outputs") / args.video.stem
    results = run_pipeline(args.video, config, output_dir)
    print(json.dumps(results.to_dict(), indent=2))
    print(f"\nOutputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
