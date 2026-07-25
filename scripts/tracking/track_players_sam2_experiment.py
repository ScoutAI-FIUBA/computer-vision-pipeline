#!/usr/bin/env python3
"""Run detector-seeded SAM2 tracking on a local video clip."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.shared.env_config import get_roboflow_api_key  # noqa: E402
from scripts.tracking.track_players_video import (  # noqa: E402
    MODEL_ID,
    default_output_fps,
    detections_from_result,
    draw_tracks,
    infer_frame_with_model,
    inspect_video,
    open_video,
    roboflow_client,
    video_info,
    write_inference_record,
    write_track_records,
)


DEFAULT_SAM2_MODEL_ID = "facebook/sam2-hiera-large"
DEFAULT_MATCH_IOU = 0.15
SAM2_ALLOWED_CLASSES = ("player",)


def load_sam2_predictor(model_id: str):
    try:
        import torch
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except ImportError as error:
        raise RuntimeError(
            "SAM2 tracking requires optional dependencies. Install PyTorch first, then run `python3 -m pip install -r requirements-sam2.txt`."
        ) from error

    predictor = SAM2VideoPredictor.from_pretrained(model_id)
    autocast_context = nullcontext()
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    return torch, predictor, autocast_context


def xyxy_iou(left_xyxy: list[float], right_xyxy: list[float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left_xyxy
    right_x1, right_y1, right_x2, right_y2 = right_xyxy
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    inter_area = inter_w * inter_h
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def mask_to_xyxy(mask_logits) -> list[float] | None:
    import numpy as np

    mask = mask_logits > 0.0
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask_array = np.asarray(mask)
    if mask_array.ndim > 2:
        mask_array = mask_array.squeeze()
    if mask_array.ndim != 2 or not mask_array.any():
        return None

    ys, xs = np.where(mask_array)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def rows_to_detections(rows: list[dict]):
    import numpy as np
    import supervision as sv

    if not rows:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=float),
            confidence=np.empty((0,), dtype=float),
            class_id=np.empty((0,), dtype=int),
            tracker_id=np.empty((0,), dtype=int),
            data={
                "confidence": [],
                "class_id": [],
                "class_name": [],
                "tracker_id": [],
                "track_state": [],
                "missing_sampled_frames": [],
            },
        )

    confidences = [row["confidence"] for row in rows]
    return sv.Detections(
        xyxy=np.asarray([row["xyxy"] for row in rows], dtype=float),
        confidence=np.asarray([0.0 if value is None else float(value) for value in confidences], dtype=float),
        class_id=np.asarray([0 for _ in rows], dtype=int),
        tracker_id=np.asarray([int(row["track_id"]) for row in rows], dtype=int),
        data={
            "confidence": confidences,
            "class_id": [0 for _ in rows],
            "class_name": [row["class_name"] for row in rows],
            "tracker_id": [int(row["track_id"]) for row in rows],
            "track_state": [row["track_state"] for row in rows],
            "missing_sampled_frames": [0 for _ in rows],
        },
    )


def detection_rows(detections) -> list[dict]:
    class_names = detections.data.get("class_name", ["player"] * len(detections))
    confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
    rows = []
    for xyxy, confidence, class_name in zip(detections.xyxy, confidences, class_names):
        rows.append(
            {
                "xyxy": [float(value) for value in xyxy],
                "confidence": None if confidence is None else float(confidence),
                "class_name": str(class_name),
                "track_id": None,
                "track_state": "detected",
            }
        )
    return rows


def assign_track_ids(
    current_rows: list[dict],
    active_rows: list[dict],
    next_track_id: int,
    min_match_iou: float,
) -> tuple[list[dict], int]:
    available_active_ids = {int(row["track_id"]) for row in active_rows}
    active_by_id = {int(row["track_id"]): row for row in active_rows}

    for row in current_rows:
        best_track_id = None
        best_iou = -1.0
        for track_id in available_active_ids:
            candidate = active_by_id[track_id]
            score = xyxy_iou(row["xyxy"], candidate["xyxy"])
            if score < min_match_iou or score <= best_iou:
                continue
            best_track_id = track_id
            best_iou = score

        if best_track_id is None:
            row["track_id"] = next_track_id
            next_track_id += 1
            continue

        row["track_id"] = best_track_id
        available_active_ids.remove(best_track_id)

    return current_rows, next_track_id


def propagate_rows(
    predictor,
    inference_state,
    *,
    start_frame_idx: int,
    stop_frame_idx: int,
    confidence_by_track_id: dict[int, float | None],
    class_name_by_track_id: dict[int, str],
) -> dict[int, list[dict]]:
    propagated: dict[int, list[dict]] = {}
    if stop_frame_idx <= start_frame_idx:
        return propagated

    max_frame_num_to_track = stop_frame_idx - start_frame_idx + 1
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=False,
    ):
        if out_frame_idx <= start_frame_idx or out_frame_idx > stop_frame_idx:
            continue

        frame_rows = []
        for mask_index, obj_id in enumerate(out_obj_ids):
            track_id = int(obj_id)
            xyxy = mask_to_xyxy(out_mask_logits[mask_index])
            if xyxy is None:
                continue
            frame_rows.append(
                {
                    "xyxy": xyxy,
                    "confidence": confidence_by_track_id.get(track_id),
                    "class_name": class_name_by_track_id.get(track_id, "player"),
                    "track_id": track_id,
                    "track_state": "propagated",
                }
            )
        propagated[out_frame_idx] = frame_rows

    return propagated


def process_video(
    video_path: Path,
    output_path: Path,
    tracks_output: Path,
    api_key: str,
    *,
    json_output: Path | None,
    frame_stride: int,
    max_frames: int | None,
    confidence: float,
    output_fps: float | None,
    sam2_model_id: str,
    sam2_match_iou: float,
) -> dict[str, object]:
    import cv2
    import numpy as np

    torch, predictor, autocast_context = load_sam2_predictor(sam2_model_id)

    capture = open_video(video_path)
    fps, frame_count, frames_to_read = video_info(capture, max_frames)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks_output.parent.mkdir(parents=True, exist_ok=True)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)

    effective_output_fps = default_output_fps(fps, 1, False, output_fps)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), effective_output_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    client = roboflow_client(api_key)
    next_track_id = 1
    next_inference_frame = 0
    cached_rows_by_frame: dict[int, list[dict]] = {}
    confidence_by_track_id: dict[int, float | None] = {}
    class_name_by_track_id: dict[int, str] = {}
    calls = 0
    frame_index = 0
    track_records = 0
    inference_records = 0

    try:
        with torch.inference_mode(), autocast_context:
            inference_state = predictor.init_state(
                video_path=str(video_path),
                offload_video_to_cpu=not torch.cuda.is_available(),
                offload_state_to_cpu=not torch.cuda.is_available(),
                async_loading_frames=False,
            )
            with tracks_output.open("w", encoding="utf-8") as tracks_file:
                json_context = json_output.open("w", encoding="utf-8") if json_output is not None else nullcontext()
                with json_context as json_file:
                    while max_frames is None or frame_index < max_frames:
                        ok, frame = capture.read()
                        if not ok:
                            break

                        current_rows = cached_rows_by_frame.pop(frame_index, [])
                        if frame_index >= next_inference_frame:
                            timestamp_seconds = None if fps <= 0 else frame_index / fps
                            result = infer_frame_with_model(client, frame, f"_{frame_index}.jpg", MODEL_ID)
                            calls += 1
                            if json_file is not None:
                                write_inference_record(json_file, frame_index, timestamp_seconds, result)
                                inference_records += 1

                            detections = detections_from_result(result, confidence, allowed_classes=SAM2_ALLOWED_CLASSES)
                            sampled_rows, next_track_id = assign_track_ids(
                                detection_rows(detections),
                                current_rows,
                                next_track_id,
                                sam2_match_iou,
                            )
                            cached_rows_by_frame.clear()
                            if sampled_rows:
                                predictor.reset_state(inference_state)
                                for row in sampled_rows:
                                    track_id = int(row["track_id"])
                                    confidence_by_track_id[track_id] = row["confidence"]
                                    class_name_by_track_id[track_id] = row["class_name"]
                                    predictor.add_new_points_or_box(
                                        inference_state=inference_state,
                                        frame_idx=frame_index,
                                        obj_id=track_id,
                                        box=np.asarray(row["xyxy"], dtype=np.float32),
                                    )

                                current_rows = sampled_rows
                                segment_stop_frame = min(
                                    (frame_count - 1) if frame_count > 0 else frame_index,
                                    (max_frames - 1) if max_frames is not None else ((frame_count - 1) if frame_count > 0 else frame_index),
                                    frame_index + frame_stride,
                                )
                                cached_rows_by_frame.update(
                                    propagate_rows(
                                        predictor,
                                        inference_state,
                                        start_frame_idx=frame_index,
                                        stop_frame_idx=segment_stop_frame,
                                        confidence_by_track_id=confidence_by_track_id,
                                        class_name_by_track_id=class_name_by_track_id,
                                    )
                                )
                            next_inference_frame = frame_index + frame_stride

                        detections = rows_to_detections(current_rows)
                        track_records += write_track_records(tracks_file, frame_index, detections)
                        writer.write(draw_tracks(frame, detections))
                        frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {
        "model_id": MODEL_ID,
        "sam2_model_id": sam2_model_id,
        "frames_read": frame_index,
        "frame_count": frame_count or None,
        "frames_to_read": frames_to_read,
        "roboflow_calls": calls,
        "output_fps": effective_output_fps,
        "track_records": track_records,
        "inference_records": inference_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detector-seeded SAM2 tracking on a local clip using only player boxes as SAM2 seeds."
    )
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Path for the annotated tracked video")
    parser.add_argument("--tracks-output", type=Path, help="Path for JSONL track records")
    parser.add_argument("--json-output", type=Path, help="Path for per-inference raw JSONL records")
    parser.add_argument("--frame-stride", type=int, default=15, help="Run detector every N frames before re-seeding SAM2. Default: 15")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to read")
    parser.add_argument("--dry-run", action="store_true", help="Print video info and estimated detector calls without loading SAM2")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        help="Minimum detector confidence to keep for player detections before seeding SAM2",
    )
    parser.add_argument("--output-fps", type=float, help="Output video FPS. Defaults to input FPS.")
    parser.add_argument(
        "--sam2-model-id",
        default=DEFAULT_SAM2_MODEL_ID,
        help=f"SAM2 Hugging Face model ID. Default: {DEFAULT_SAM2_MODEL_ID}",
    )
    parser.add_argument(
        "--sam2-match-iou",
        type=float,
        default=DEFAULT_MATCH_IOU,
        help=f"Minimum IoU to keep an existing ID when refreshing from detector boxes. Default: {DEFAULT_MATCH_IOU}",
    )
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    if args.output_fps is not None and args.output_fps <= 0:
        raise SystemExit("--output-fps must be greater than 0")
    if args.sam2_match_iou < 0 or args.sam2_match_iou > 1:
        raise SystemExit("--sam2-match-iou must be between 0 and 1")

    args.output = args.output or Path("outputs") / f"{args.video.stem}_sam2.mp4"
    args.tracks_output = args.tracks_output or Path("outputs") / f"{args.video.stem}_sam2_tracks.jsonl"
    args.json_output = args.json_output or Path("outputs") / f"{args.video.stem}_sam2_inference.jsonl"

    try:
        info = inspect_video(args.video, args.frame_stride, args.max_frames)
        if args.dry_run:
            print(f"Video: {args.video}")
            print(f"FPS: {info['fps'] or 'unknown'}")
            print(f"Frame count: {info['frame_count'] or 'unknown'}")
            print(f"Frames to read: {info['frames_to_read'] or 'unknown'}")
            print(f"Estimated Roboflow calls: {info['estimated_roboflow_calls']}")
            print(f"SAM2 model ID: {args.sam2_model_id}")
            print(f"SAM2 seed classes: {', '.join(SAM2_ALLOWED_CLASSES)}")
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
            confidence=args.confidence,
            output_fps=args.output_fps,
            sam2_model_id=args.sam2_model_id,
            sam2_match_iou=args.sam2_match_iou,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(f"Frames read: {stats['frames_read']}/{stats['frame_count'] or 'unknown'}")
    print(f"Roboflow calls: {stats['roboflow_calls']}")
    print(f"Output FPS: {stats['output_fps']}")
    print(f"SAM2 model ID: {stats['sam2_model_id']}")
    print(f"SAM2 seed classes: {', '.join(SAM2_ALLOWED_CLASSES)}")
    print(f"Track records: {stats['track_records']}")
    print(f"Inference records: {stats['inference_records']}")
    print(f"Tracked video saved to: {args.output}")
    print(f"Track JSONL saved to: {args.tracks_output}")
    print(f"Inference JSONL saved to: {args.json_output}")


if __name__ == "__main__":
    main()
