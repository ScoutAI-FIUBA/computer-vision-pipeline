#!/usr/bin/env python3
"""Run sampled-frame court keypoint detection + FIBA homography on a video.

This script reuses the shared labeled template from
``scripts/court_template.py`` via ``scripts/build_court_homography.py``. That
template is the authoritative label -> court-coordinate map for this repo.

The script:

- Reads a video, runs the court keypoint model on sampled frames, builds a
  homography per inferred frame, and writes a side-by-side debug video:
  original frame on the left, FIBA 2D court with projected keypoints on the
  right.
- Reuses ``parse_keypoints`` from ``scripts/detect_court_keypoints``,
  ``infer_frame`` / ``roboflow_client`` from
  ``scripts/detect_court_keypoints_video``, the ``compute_homography`` /
  ``draw_fiba_court`` / ``draw_projected_keypoints`` math from
  ``scripts/build_court_homography``, and video plumbing from
  ``scripts/detect_players_video``.
- Defaults to ``--sampled-output-only true`` so only inferred frames are
  written (the homography per frame is the only one we have, so showing
  un-inferred frames would be misleading).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from env_config import COURT_MODEL_ID, get_roboflow_api_key


def _import_homography_helpers():
    from build_court_homography import (
        compute_homography,
        draw_fiba_court,
        draw_projected_keypoints,
    )
    from court_template import canonical_point_map

    return canonical_point_map, compute_homography, draw_fiba_court, draw_projected_keypoints


def _import_keypoint_parser():
    from detect_court_keypoints import parse_keypoints
    return parse_keypoints


def _import_video_inference():
    from detect_court_keypoints_video import infer_frame, roboflow_client
    return infer_frame, roboflow_client


def _import_video_plumbing():
    from detect_players_video import (
        default_output_fps,
        inspect_video,
        open_video,
        video_info,
    )
    return default_output_fps, inspect_video, open_video, video_info


def composite_frame(
    frame,
    homography_status: dict[str, Any],
    pixels_per_meter: int = 40,
):
    """Build a side-by-side composite: original frame | FIBA 2D court.

    ``homography_status`` keys:
      - ``status``: ``"ok"`` | ``"insufficient"`` | ``"error"``
      - ``frame_index``: int
      - ``matched_count``: int
      - ``inliers_count``: int
      - ``reprojection_mean_m``: float
    """
    import cv2
    import numpy as np

    _, _, draw_fiba_court, draw_projected_keypoints = _import_homography_helpers()

    height, width = frame.shape[:2]
    court = draw_fiba_court(pixels_per_meter=pixels_per_meter)
    if homography_status.get("status") == "ok":
        court_annotated = draw_projected_keypoints(
            court,
            homography_status["matrix"],
            homography_status["matched"],
            pixels_per_meter=pixels_per_meter,
        )
    else:
        court_annotated = court
    court_resized = cv2.resize(court_annotated, (width, height), interpolation=cv2.INTER_AREA)

    sep = np.full((height, 4, 3), 32, dtype=np.uint8)
    composite = np.hstack([frame, sep, court_resized])

    label = (
        f"frame={homography_status.get('frame_index', '?')} "
        f"status={homography_status.get('status', '?')}"
    )
    if homography_status.get("status") == "ok":
        label += (
            f"  mode={homography_status.get('geometry_mode', '?')}"
            f"  matched={homography_status.get('matched_count', 0)}"
            f"  inliers={homography_status.get('inliers_count', 0)}"
            f"  mean_err={homography_status.get('reprojection_mean_m', 0.0):.3f}m"
        )
    elif homography_status.get("status") == "insufficient":
        label += (
            f"  mode={homography_status.get('geometry_mode', '?')}"
            f"  matched={homography_status.get('matched_count', 0)}"
        )

    cv2.putText(
        composite,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        composite,
        "FIBA court (28m x 15m) - authoritative label template",
        (12, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 255),
        1,
        cv2.LINE_AA,
    )
    return composite


def process_video(
    video_path: Path,
    output_path: Path,
    api_key: str,
    model_id: str,
    frame_stride: int = 30,
    max_frames: int | None = None,
    min_confidence: float = 0.7,
    sampled_output_only: bool = True,
    output_fps: float | None = None,
    min_points: int = 4,
    reproj_threshold: float = 8.0,
    geometry_mode: str = "strict",
) -> dict:
    import cv2

    canonical_point_map, compute_homography, _, _ = _import_homography_helpers()
    parse_keypoints = _import_keypoint_parser()
    infer_frame, roboflow_client = _import_video_inference()
    default_output_fps, _, open_video, video_info = _import_video_plumbing()
    point_map = canonical_point_map(geometry_mode)

    capture = open_video(video_path)
    fps, frame_count, frames_to_read = video_info(capture, max_frames)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    effective_output_fps = default_output_fps(fps, frame_stride, sampled_output_only, output_fps)

    composite_width = 2 * width + 4
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        effective_output_fps,
        (composite_width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    frame_index = 0
    calls = 0
    frames_written = 0
    frames_with_homography = 0
    inliers_total = 0
    error_total = 0.0
    error_count = 0

    try:
        client = roboflow_client(api_key)
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            inferred = False
            status: dict[str, Any] = {
                "frame_index": frame_index,
                "status": "skipped",
                "matched_count": 0,
                "inliers_count": 0,
                "reprojection_mean_m": 0.0,
                "geometry_mode": geometry_mode,
            }
            if frame_index % frame_stride == 0:
                result = infer_frame(client, frame, f"_{frame_index}.jpg", model_id)
                calls += 1
                inferred = True
                keypoints = parse_keypoints(result, min_confidence, width, height)
                status["matched_count"] = sum(1 for point in keypoints if point.get("label") in point_map)
                try:
                    homography = compute_homography(
                        keypoints,
                        min_confidence=min_confidence,
                        min_points=min_points,
                        reproj_threshold=reproj_threshold,
                        geometry_mode=geometry_mode,
                        point_map=point_map,
                    )
                    frames_with_homography += 1
                    inliers_total += homography.inliers_count
                    error_total += homography.reprojection_mean_m
                    error_count += 1
                    status = {
                        "frame_index": frame_index,
                        "status": "ok",
                        "matched_count": len(homography.matched),
                        "inliers_count": homography.inliers_count,
                        "reprojection_mean_m": homography.reprojection_mean_m,
                        "geometry_mode": homography.geometry_mode,
                        "matrix": homography.matrix,
                        "matched": homography.matched,
                        "used_labels": [entry["label"] for entry in homography.matched],
                        "skipped": homography.skipped,
                    }
                except ValueError as error:
                    status["status"] = "insufficient"
                    status["error"] = str(error)

            if inferred or not sampled_output_only:
                writer.write(composite_frame(frame, status))
                frames_written += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {
        "model_id": model_id,
        "frames_read": frame_index,
        "frame_count": frame_count or None,
        "frames_to_read": frames_to_read,
        "roboflow_calls": calls,
        "frames_written": frames_written,
        "frames_with_homography": frames_with_homography,
        "mean_inliers": (inliers_total / frames_with_homography) if frames_with_homography else 0.0,
        "mean_reprojection_error_m": (error_total / error_count) if error_count else 0.0,
        "output_fps": effective_output_fps,
        "geometry_mode": geometry_mode,
        "geometry_points_available": sorted(point_map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "Court homography video.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Path for the annotated video (default outputs/<video-stem>_homography.mp4)")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Roboflow model ID. Defaults to COURT_MODEL_ID or basketball-court-detection-2/22",
    )
    parser.add_argument("--frame-stride", type=int, default=30, help="Infer every N frames. Default: 30")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to read")
    parser.add_argument("--dry-run", action="store_true", help="Print video info and estimated calls without inference")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Minimum keypoint confidence to use for homography. Default: 0.7",
    )
    parser.add_argument(
        "--sampled-output-only",
        dest="sampled_output_only",
        action="store_true",
        default=True,
        help="Write only frames sent to Roboflow. Default: true",
    )
    parser.add_argument(
        "--no-sampled-output-only",
        dest="sampled_output_only",
        action="store_false",
        help="Write every frame; only inferred frames get the homography overlay",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        help="Output video FPS. Defaults to input FPS / frame stride for sampled output",
    )
    parser.add_argument("--min-points", type=int, default=4, help="Minimum matched points to fit a homography (default 4)")
    parser.add_argument("--reproj-threshold", type=float, default=8.0, help="RANSAC reprojection threshold in pixels (default 8.0)")
    parser.add_argument(
        "--geometry-mode",
        choices=("strict", "debug"),
        default="strict",
        help="Canonical geometry filter mode. Both modes currently use the same authoritative mapped labels.",
    )
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    if args.confidence < 0 or args.confidence > 1:
        raise SystemExit("--confidence must be between 0 and 1")
    if args.output_fps is not None and args.output_fps <= 0:
        raise SystemExit("--output-fps must be greater than 0")
    if args.min_points < 4:
        raise SystemExit("--min-points must be at least 4")
    if args.reproj_threshold <= 0:
        raise SystemExit("--reproj-threshold must be greater than 0")

    api_key = get_roboflow_api_key()
    if not api_key and not args.dry_run:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")
    model_id = args.model_id or COURT_MODEL_ID
    args.output = args.output or Path("outputs") / f"{args.video.stem}_homography.mp4"

    _, inspect_video, _, _ = _import_video_plumbing()

    try:
        info = inspect_video(args.video, args.frame_stride, args.max_frames)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    if args.dry_run:
        print(f"Video: {args.video}")
        print(f"FPS: {info['fps'] or 'unknown'}")
        print(f"Frame count: {info['frame_count'] or 'unknown'}")
        print(f"Frames to read: {info['frames_to_read'] or 'unknown'}")
        print(f"Estimated Roboflow calls: {info['estimated_roboflow_calls']}")
        print(f"Model ID: {model_id}")
        print("Output (planned): side-by-side composite per inferred frame (original | FIBA 2D court)")
        return

    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")

    if args.geometry_mode == "debug":
        print("Geometry mode: debug (same authoritative mapped labels as strict).")
    else:
        print("Geometry mode: strict (authoritative mapped labels only).")

    try:
        stats = process_video(
            args.video,
            args.output,
            api_key,
            model_id=model_id,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            min_confidence=args.confidence,
            sampled_output_only=args.sampled_output_only,
            output_fps=args.output_fps,
            min_points=args.min_points,
            reproj_threshold=args.reproj_threshold,
            geometry_mode=args.geometry_mode,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(f"Model ID: {stats['model_id']}")
    print(f"Geometry mode: {stats['geometry_mode']}")
    print(f"Frames read: {stats['frames_read']}/{stats['frame_count'] or 'unknown'}")
    print(f"Roboflow calls: {stats['roboflow_calls']}")
    print(f"Frames with homography: {stats['frames_with_homography']}")
    print(f"Mean inliers per homography frame: {stats['mean_inliers']:.2f}")
    print(f"Mean reprojection error (m): {stats['mean_reprojection_error_m']:.3f}")
    print(f"Output FPS: {stats['output_fps']}")
    print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
