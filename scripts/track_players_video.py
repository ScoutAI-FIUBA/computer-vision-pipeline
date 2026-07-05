#!/usr/bin/env python3
"""Run Roboflow player detection plus Supervision ByteTrack on sampled video frames."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.annotation_helpers import annotate_detections, class_names_for, detections_from_inference, format_label  # noqa: E402
from scripts.detect_players_video import (  # noqa: E402
    MODEL_ID,
    default_output_fps,
    inspect_video,
    open_video,
    roboflow_client,
    video_info,
)
from scripts.detect_court_keypoints import (  # noqa: E402
    DEFAULT_MODEL_ID as COURT_MODEL_ID,
    draw_with_annotators,
    parse_keypoints,
    to_keypoints,
)
from scripts.env_config import get_roboflow_api_key  # noqa: E402


@dataclass
class TrackSample:
    xyxy: list[float]
    confidence: float | None
    class_name: str


class TrackSmoother:
    def __init__(self, window_size: int = 3) -> None:
        self.history: dict[int, deque[TrackSample]] = defaultdict(lambda: deque(maxlen=window_size))
        self.last_detections = None

    def update_with_detections(self, detections):
        import numpy as np
        import supervision as sv

        if len(detections) == 0:
            self.last_detections = detections
            return detections

        track_ids = track_ids_for(detections)
        class_names = detections.data.get("class_name", ["object"] * len(detections))
        confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)

        smoothed_xyxy = []
        smoothed_confidences = []
        smoothed_class_names = []
        smoothed_track_ids = []

        for track_id, class_name, confidence, xyxy in zip(track_ids, class_names, confidences, detections.xyxy):
            if track_id is None:
                smoothed_xyxy.append([float(value) for value in xyxy])
                smoothed_confidences.append(None if confidence is None else float(confidence))
                smoothed_class_names.append(str(class_name))
                smoothed_track_ids.append(None)
                continue

            sample = TrackSample(
                xyxy=[float(value) for value in xyxy],
                confidence=None if confidence is None else float(confidence),
                class_name=str(class_name),
            )
            history = self.history[track_id]
            history.append(sample)

            smoothed_xyxy.append(np.mean([item.xyxy for item in history], axis=0))
            smoothed_class_names.append(Counter(item.class_name for item in history).most_common(1)[0][0])

            recent_confidences = [item.confidence for item in history if item.confidence is not None]
            smoothed_confidences.append(
                sum(recent_confidences) / len(recent_confidences) if recent_confidences else None
            )
            smoothed_track_ids.append(track_id)

        smoothed_detections = sv.Detections(
            xyxy=np.asarray(smoothed_xyxy, dtype=float),
            confidence=(
                None
                if any(confidence is None for confidence in smoothed_confidences)
                else np.asarray(smoothed_confidences, dtype=float)
            ),
            class_id=detections.class_id.copy() if detections.class_id is not None else None,
            tracker_id=(
                None
                if any(track_id is None for track_id in smoothed_track_ids)
                else np.asarray(smoothed_track_ids, dtype=int)
            ),
            data={"class_name": smoothed_class_names, "tracker_id": smoothed_track_ids},
            metadata=detections.metadata,
        )
        self.last_detections = smoothed_detections
        return smoothed_detections

    def current_detections(self):
        return self.last_detections


def detections_from_result(result: dict, min_confidence: float):
    return detections_from_inference(result, min_confidence=min_confidence)


def infer_frame_with_model(client, frame, suffix: str, model_id: str) -> dict:
    import cv2

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        if not cv2.imwrite(str(temp_path), frame):
            raise RuntimeError("Failed to write temporary frame image")
        return client.infer(str(temp_path), model_id=model_id)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def track_ids_for(detections) -> list[int | None]:
    track_ids = getattr(detections, "tracker_id", None)
    if track_ids is None:
        track_ids = detections.data.get("tracker_id")
    if track_ids is None:
        track_ids = [None] * len(detections)
    return [None if track_id is None else int(track_id) for track_id in track_ids]


def labels_for(detections) -> list[str]:
    confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
    labels = []
    for track_id, class_name, confidence in zip(track_ids_for(detections), class_names_for(detections), confidences):
        prefix = f"#{track_id} " if track_id is not None else ""
        labels.append(format_label(class_name, confidence, prefix=prefix))
    return labels


def draw_tracks(frame, detections):
    return annotate_detections(frame, detections, labels=labels_for(detections))


def write_track_records(handle, frame_index: int, detections) -> int:
    confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
    count = 0
    for track_id, class_name, confidence, xyxy in zip(
        track_ids_for(detections), class_names_for(detections), confidences, detections.xyxy
    ):
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


def write_inference_record(handle, frame_index: int, timestamp_seconds: float | None, result: dict) -> None:
    handle.write(
        json.dumps(
            {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "model_id": MODEL_ID,
                "predictions": result.get("predictions", []),
            }
        )
        + "\n"
    )


def write_court_record(
    handle,
    frame_index: int,
    timestamp_seconds: float | None,
    model_id: str,
    keypoints: list[dict],
) -> None:
    handle.write(
        json.dumps(
            {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "model_id": model_id,
                "keypoints": keypoints,
            }
        )
        + "\n"
    )


def process_video(
    video_path: Path,
    output_path: Path,
    tracks_output: Path,
    api_key: str,
    json_output: Path | None = None,
    frame_stride: int = 30,
    max_frames: int | None = None,
    sampled_output_only: bool = True,
    confidence: float = 0.0,
    output_fps: float | None = None,
    smoothing_window: int = 3,
    reuse_stale_tracks: bool = False,
    court_keypoints: bool = False,
    court_json_output: Path | None = None,
    court_model_id: str = COURT_MODEL_ID,
    court_confidence: float = 0.7,
) -> dict:
    import cv2
    import supervision as sv

    capture = open_video(video_path)
    fps, frame_count, frames_to_read = video_info(capture, max_frames)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks_output.parent.mkdir(parents=True, exist_ok=True)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
    if court_json_output is not None:
        court_json_output.parent.mkdir(parents=True, exist_ok=True)
    output_fps = default_output_fps(fps, frame_stride, sampled_output_only, output_fps)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    frame_index = 0
    calls = 0
    track_records = 0
    inference_records = 0
    court_inference_records = 0
    court_frames = 0

    try:
        client = roboflow_client(api_key)
        tracker = sv.ByteTrack()
        smoother = TrackSmoother(window_size=smoothing_window)
        with tracks_output.open("w", encoding="utf-8") as tracks_file:
            json_context = json_output.open("w", encoding="utf-8") if json_output is not None else nullcontext()
            court_context = court_json_output.open("w", encoding="utf-8") if court_json_output is not None else nullcontext()
            with json_context as json_file, court_context as court_file:
                while max_frames is None or frame_index < max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break

                    if frame_index % frame_stride == 0:
                        timestamp_seconds = None if fps <= 0 else frame_index / fps
                        result = infer_frame_with_model(client, frame, f"_{frame_index}.jpg", MODEL_ID)
                        calls += 1
                        if json_file is not None:
                            write_inference_record(json_file, frame_index, timestamp_seconds, result)
                            inference_records += 1
                        detections = detections_from_result(result, confidence)
                        detections = tracker.update_with_detections(detections)
                        detections = smoother.update_with_detections(detections)
                        track_records += write_track_records(tracks_file, frame_index, detections)
                        annotated_frame = draw_tracks(frame, detections)

                        if court_keypoints:
                            court_result = infer_frame_with_model(client, frame, f"_court_{frame_index}.jpg", court_model_id)
                            calls += 1
                            court_keypoints_list = parse_keypoints(court_result, court_confidence, width, height)
                            if court_file is not None:
                                write_court_record(
                                    court_file,
                                    frame_index,
                                    timestamp_seconds,
                                    court_model_id,
                                    court_keypoints_list,
                                )
                                court_inference_records += 1
                            annotated_frame = draw_with_annotators(annotated_frame, to_keypoints(court_keypoints_list))
                            court_frames += 1

                        writer.write(annotated_frame)
                    elif not sampled_output_only:
                        stale_detections = smoother.current_detections() if reuse_stale_tracks else None
                        writer.write(draw_tracks(frame, stale_detections) if stale_detections is not None else frame)

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
        "inference_records": inference_records,
        "court_model_id": court_model_id if court_keypoints else None,
        "court_roboflow_calls": court_frames,
        "court_frames": court_frames,
        "court_inference_records": court_inference_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Track basketball players in sampled video frames with ByteTrack.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Path for the tracked video")
    parser.add_argument("--tracks-output", type=Path, help="Path for JSONL track records")
    parser.add_argument("--json-output", type=Path, help="Path for per-inference raw JSONL records")
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
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    if args.output_fps is not None and args.output_fps <= 0:
        raise SystemExit("--output-fps must be greater than 0")
    if args.smoothing_window < 1:
        raise SystemExit("--smoothing-window must be at least 1")

    args.output = args.output or Path("outputs") / f"{args.video.stem}_tracked.mp4"
    args.tracks_output = args.tracks_output or Path("outputs") / f"{args.video.stem}_tracks.jsonl"
    args.json_output = args.json_output or Path("outputs") / f"{args.video.stem}_inference.jsonl"

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

    api_key = get_roboflow_api_key()
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")

    try:
        stats = process_video(
            args.video,
            args.output,
            args.tracks_output,
            api_key,
            json_output=args.json_output,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            sampled_output_only=args.sampled_output_only,
            confidence=args.confidence,
            output_fps=args.output_fps,
            smoothing_window=args.smoothing_window,
            reuse_stale_tracks=args.reuse_stale_tracks,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(f"Frames read: {stats['frames_read']}/{stats['frame_count'] or 'unknown'}")
    print(f"Roboflow calls: {stats['roboflow_calls']}")
    print(f"Output FPS: {stats['output_fps']}")
    print(f"Track records: {stats['track_records']}")
    print(f"Inference records: {stats['inference_records']}")
    print(f"Tracked video saved to: {args.output}")
    print(f"Track JSONL saved to: {args.tracks_output}")
    print(f"Inference JSONL saved to: {args.json_output}")


if __name__ == "__main__":
    main()
