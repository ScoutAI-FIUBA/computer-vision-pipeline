#!/usr/bin/env python3
"""Run Roboflow player detection plus Supervision ByteTrack on sampled video frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.detect_players_video import (  # noqa: E402
    MODEL_ID,
    default_output_fps,
    infer_frame,
    inspect_video,
    open_video,
    roboflow_client,
    video_info,
)


def detections_from_result(result: dict, min_confidence: float):
    import supervision as sv

    detections = sv.Detections.from_inference(result)
    if detections.confidence is not None:
        detections = detections[detections.confidence >= min_confidence]
    return detections


def track_ids_for(detections) -> list[int | None]:
    track_ids = getattr(detections, "tracker_id", None)
    if track_ids is None:
        track_ids = detections.data.get("tracker_id")
    if track_ids is None:
        track_ids = [None] * len(detections)
    return [None if track_id is None else int(track_id) for track_id in track_ids]


def labels_for(detections) -> list[str]:
    class_names = detections.data.get("class_name", ["object"] * len(detections))
    confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
    labels = []
    for track_id, class_name, confidence in zip(track_ids_for(detections), class_names, confidences):
        prefix = f"#{track_id} " if track_id is not None else ""
        labels.append(f"{prefix}{class_name} {confidence:.2f}" if confidence is not None else f"{prefix}{class_name}")
    return labels


def draw_tracks(frame, detections):
    import supervision as sv

    annotated = sv.BoxAnnotator().annotate(scene=frame, detections=detections)
    return sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=labels_for(detections))


def write_track_records(handle, frame_index: int, detections) -> int:
    class_names = detections.data.get("class_name", ["object"] * len(detections))
    confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
    count = 0
    for track_id, class_name, confidence, xyxy in zip(track_ids_for(detections), class_names, confidences, detections.xyxy):
        handle.write(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "class_name": str(class_name),
                    "confidence": None if confidence is None else float(confidence),
                    "xyxy": [float(value) for value in xyxy],
                }
            )
            + "\n"
        )
        count += 1
    return count


def process_video(
    video_path: Path,
    output_path: Path,
    tracks_output: Path,
    api_key: str,
    frame_stride: int = 30,
    max_frames: int | None = None,
    sampled_output_only: bool = True,
    confidence: float = 0.0,
    output_fps: float | None = None,
) -> dict:
    import cv2
    import supervision as sv

    capture = open_video(video_path)
    fps, frame_count, frames_to_read = video_info(capture, max_frames)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks_output.parent.mkdir(parents=True, exist_ok=True)
    output_fps = default_output_fps(fps, frame_stride, sampled_output_only, output_fps)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    frame_index = 0
    calls = 0
    track_records = 0

    try:
        client = roboflow_client(api_key)
        tracker = sv.ByteTrack()
        with tracks_output.open("w", encoding="utf-8") as tracks_file:
            while max_frames is None or frame_index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % frame_stride == 0:
                    result = infer_frame(client, frame, f"_{frame_index}.jpg")
                    calls += 1
                    detections = detections_from_result(result, confidence)
                    detections = tracker.update_with_detections(detections)
                    track_records += write_track_records(tracks_file, frame_index, detections)
                    writer.write(draw_tracks(frame, detections))
                elif not sampled_output_only:
                    writer.write(frame)

                frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {
        "model_id": MODEL_ID,
        "frames_read": frame_index,
        "frame_count": frame_count or None,
        "frames_to_read": frames_to_read,
        "roboflow_calls": calls,
        "output_fps": output_fps,
        "track_records": track_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Track basketball players in sampled video frames with ByteTrack.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Path for the tracked video")
    parser.add_argument("--tracks-output", type=Path, help="Path for JSONL track records")
    parser.add_argument("--frame-stride", type=int, default=30, help="Infer every N frames. Default: 30")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to read")
    parser.add_argument("--dry-run", action="store_true", help="Print video info and estimated calls without inference")
    parser.add_argument("--confidence", type=float, default=0.0, help="Minimum confidence to track and draw")
    parser.add_argument(
        "--sampled-output-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write only inferred frames by default. Use --no-sampled-output-only to include skipped frames unannotated.",
    )
    parser.add_argument("--output-fps", type=float, help="Output video FPS. Defaults to input FPS / frame stride for sampled output.")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    if args.output_fps is not None and args.output_fps <= 0:
        raise SystemExit("--output-fps must be greater than 0")

    args.output = args.output or Path("outputs") / f"{args.video.stem}_tracked.mp4"
    args.tracks_output = args.tracks_output or Path("outputs") / f"{args.video.stem}_tracks.jsonl"

    try:
        info = inspect_video(args.video, args.frame_stride, args.max_frames)
        if args.dry_run:
            print(f"Video: {args.video}")
            print(f"FPS: {info['fps'] or 'unknown'}")
            print(f"Frame count: {info['frame_count'] or 'unknown'}")
            print(f"Frames to read: {info['frames_to_read'] or 'unknown'}")
            print(f"Estimated Roboflow calls: {info['estimated_roboflow_calls']}")
            print(f"Sampled output only: {args.sampled_output_only}")
            return
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as error:
        raise SystemExit("Missing dependency: python-dotenv. Run `python -m pip install -r requirements.txt`.") from error

    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")

    try:
        stats = process_video(
            args.video,
            args.output,
            args.tracks_output,
            api_key,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            sampled_output_only=args.sampled_output_only,
            confidence=args.confidence,
            output_fps=args.output_fps,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(f"Frames read: {stats['frames_read']}/{stats['frame_count'] or 'unknown'}")
    print(f"Roboflow calls: {stats['roboflow_calls']}")
    print(f"Output FPS: {stats['output_fps']}")
    print(f"Track records: {stats['track_records']}")
    print(f"Tracked video saved to: {args.output}")
    print(f"Track JSONL saved to: {args.tracks_output}")


if __name__ == "__main__":
    main()
