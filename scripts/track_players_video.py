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

from scripts.annotation_helpers import (  # noqa: E402
    annotate_detections,
    class_names_for,
    detections_from_inference,
    format_label,
    label_prefix,
)
from scripts.detect_players_video import (  # noqa: E402
    MODEL_ID,
    default_output_fps,
    inspect_video,
    open_video,
    roboflow_client,
    video_info,
)
from scripts.build_court_homography import compute_homography, project_points  # noqa: E402
from scripts.court_template import canonical_point_map  # noqa: E402
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


@dataclass
class PersistedTrackSample:
    xyxy: list[float]
    class_name: str
    confidence: float | None
    frame_index: int
    track_state: str = "detected"
    missing_sampled_frames: int = 0
    appearance: list[float] | None = None


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


class TrackPersistence:
    def __init__(self, max_missing_sampled_frames: int = 1) -> None:
        self.max_missing_sampled_frames = max_missing_sampled_frames
        self.history: dict[int, deque[PersistedTrackSample]] = defaultdict(lambda: deque(maxlen=2))
        self.last_detections = None

    def update_with_detections(self, detections, frame_index: int, frame):
        import numpy as np
        import supervision as sv

        states = ["detected"] * len(detections)
        reid_applied = [False] * len(detections)
        reassigned_from_track_ids = [None] * len(detections)
        reid_scores = [None] * len(detections)
        if self.max_missing_sampled_frames < 1:
            persisted = clone_detections_with_metadata(
                detections,
                track_states=states,
                missing_sampled_frames=[0] * len(detections),
                reid_applied=reid_applied,
                reassigned_from_track_id=reassigned_from_track_ids,
                reid_score=reid_scores,
            )
            self.last_detections = persisted
            return persisted

        rows = build_detection_rows(detections, frame)
        self.reassign_recent_tracks(rows, frame_index)

        present_track_ids: set[int] = set()

        xyxy_rows: list[list[float]] = []
        confidence_rows: list[float | None] = []
        class_name_rows: list[str] = []
        class_id_rows: list[int | None] = []
        track_id_rows: list[int | None] = []
        track_state_rows: list[str] = []
        missing_rows: list[int] = []
        reid_applied_rows: list[bool] = []
        reassigned_from_rows: list[int | None] = []
        reid_score_rows: list[float | None] = []

        for row in rows:
            xyxy_rows.append(row["xyxy"])
            confidence_rows.append(row["confidence"])
            class_name_rows.append(row["class_name"])
            class_id_rows.append(row["class_id"])
            track_id_rows.append(row["track_id"])
            track_state_rows.append("detected")
            missing_rows.append(0)
            reid_applied_rows.append(bool(row["reid_applied"]))
            reassigned_from_rows.append(row["reassigned_from_track_id"])
            reid_score_rows.append(row["reid_score"])

            track_id = row["track_id"]
            if track_id is None:
                continue

            present_track_ids.add(track_id)
            self.history[track_id].append(
                PersistedTrackSample(
                    xyxy=row["xyxy"],
                    class_name=row["class_name"],
                    confidence=row["confidence"],
                    frame_index=frame_index,
                    appearance=row["appearance"],
                )
            )

        for track_id, samples in self.history.items():
            if track_id in present_track_ids or not samples:
                continue

            last_sample = samples[-1]
            missing_sampled_frames = frame_index - last_sample.frame_index
            if missing_sampled_frames < 1 or missing_sampled_frames > self.max_missing_sampled_frames:
                continue

            xyxy_rows.append(predict_xyxy(samples, missing_sampled_frames))
            confidence_rows.append(None)
            class_name_rows.append(last_sample.class_name)
            class_id_rows.append(None)
            track_id_rows.append(track_id)
            track_state_rows.append("predicted")
            missing_rows.append(missing_sampled_frames)
            reid_applied_rows.append(False)
            reassigned_from_rows.append(None)
            reid_score_rows.append(None)

        persisted = sv.Detections(
            xyxy=np.asarray(xyxy_rows, dtype=float) if xyxy_rows else np.empty((0, 4), dtype=float),
            confidence=(
                np.asarray([np.nan if confidence is None else confidence for confidence in confidence_rows], dtype=float)
                if confidence_rows
                else None
            ),
            class_id=None if any(class_id is None for class_id in class_id_rows) else np.asarray(class_id_rows, dtype=int),
            tracker_id=(
                None
                if any(track_id is None for track_id in track_id_rows)
                else np.asarray(track_id_rows, dtype=int)
            ),
            data={
                "confidence": confidence_rows,
                "class_id": class_id_rows,
                "class_name": class_name_rows,
                "tracker_id": track_id_rows,
                "track_state": track_state_rows,
                "missing_sampled_frames": missing_rows,
                "reid_applied": reid_applied_rows,
                "reassigned_from_track_id": reassigned_from_rows,
                "reid_score": reid_score_rows,
            },
            metadata=detections.metadata,
        )
        self.last_detections = persisted
        return persisted

    def current_detections(self):
        return self.last_detections

    def reassign_recent_tracks(self, rows: list[dict], frame_index: int) -> None:
        missing_candidates = []
        for track_id, samples in self.history.items():
            if not samples:
                continue
            last_sample = samples[-1]
            missing_sampled_frames = frame_index - last_sample.frame_index
            if missing_sampled_frames < 1 or missing_sampled_frames > self.max_missing_sampled_frames:
                continue
            if last_sample.appearance is None:
                continue
            missing_candidates.append(
                {
                    "track_id": track_id,
                    "xyxy": predict_xyxy(samples, missing_sampled_frames),
                    "appearance": last_sample.appearance,
                }
            )

        if not missing_candidates:
            return

        reserved_track_ids = {row["track_id"] for row in rows if row["track_id"] is not None and row["track_id"] in self.history}
        claimed_track_ids: set[int] = set()
        for row in rows:
            track_id = row["track_id"]
            if row["appearance"] is None:
                continue
            if track_id is not None and track_id in self.history:
                continue

            best_candidate = None
            best_score = -1.0
            for candidate in missing_candidates:
                candidate_track_id = candidate["track_id"]
                if candidate_track_id in reserved_track_ids or candidate_track_id in claimed_track_ids:
                    continue
                if not is_reid_spatial_match(row["xyxy"], candidate["xyxy"]):
                    continue
                score = appearance_similarity(row["appearance"], candidate["appearance"])
                if score < 0.88 or score <= best_score:
                    continue
                best_candidate = candidate
                best_score = score

            if best_candidate is None:
                continue

            claimed_track_ids.add(best_candidate["track_id"])
            row["reid_applied"] = True
            row["reassigned_from_track_id"] = track_id
            row["reid_score"] = round(best_score, 4)
            row["track_id"] = best_candidate["track_id"]


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


def confidences_for(detections) -> list[float | None]:
    confidences = getattr(detections, "confidence", None)
    if confidences is None:
        confidences = detections.data.get("confidence")
    if confidences is None:
        confidences = [None] * len(detections)
    return [None if confidence is None or confidence != confidence else float(confidence) for confidence in confidences]


def track_states_for(detections) -> list[str]:
    states = detections.data.get("track_state")
    if states is None:
        states = ["detected"] * len(detections)
    return [str(state) for state in states]


def class_ids_for(detections) -> list[int | None]:
    class_ids = getattr(detections, "class_id", None)
    if class_ids is None:
        class_ids = detections.data.get("class_id")
    if class_ids is None:
        return [None] * len(detections)
    return [None if class_id is None else int(class_id) for class_id in class_ids]


def missing_sampled_frames_for(detections) -> list[int]:
    values = detections.data.get("missing_sampled_frames")
    if values is None:
        values = [0] * len(detections)
    return [int(value) for value in values]


def reid_applied_for(detections) -> list[bool]:
    values = detections.data.get("reid_applied")
    if values is None:
        values = [False] * len(detections)
    return [bool(value) for value in values]


def reassigned_from_track_ids_for(detections) -> list[int | None]:
    values = detections.data.get("reassigned_from_track_id")
    if values is None:
        values = [None] * len(detections)
    return [None if value is None else int(value) for value in values]


def reid_scores_for(detections) -> list[float | None]:
    values = detections.data.get("reid_score")
    if values is None:
        values = [None] * len(detections)
    return [None if value is None else float(value) for value in values]


def build_detection_rows(detections, frame) -> list[dict]:
    rows = []
    confidences = confidences_for(detections)
    for track_id, class_id, class_name, confidence, xyxy in zip(
        track_ids_for(detections),
        class_ids_for(detections),
        class_names_for(detections),
        confidences,
        detections.xyxy,
    ):
        xyxy_list = [float(value) for value in xyxy]
        rows.append(
            {
                "track_id": track_id,
                "class_id": class_id,
                "class_name": str(class_name),
                "confidence": None if confidence is None else float(confidence),
                "xyxy": xyxy_list,
                "appearance": extract_torso_appearance(frame, xyxy_list),
                "reid_applied": False,
                "reassigned_from_track_id": None,
                "reid_score": None,
            }
        )
    return rows


def extract_torso_appearance(frame, xyxy) -> list[float] | None:
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 12 or box_height < 24:
        return None

    torso_x1 = max(0, min(width, int(round(x1 + box_width * 0.2))))
    torso_x2 = max(0, min(width, int(round(x2 - box_width * 0.2))))
    torso_y1 = max(0, min(height, int(round(y1 + box_height * 0.15))))
    torso_y2 = max(0, min(height, int(round(y1 + box_height * 0.65))))
    if torso_x2 - torso_x1 < 8 or torso_y2 - torso_y1 < 12:
        return None

    crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    hsv_hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
    lab_hist = cv2.calcHist([lab], [1, 2], None, [4, 4], [0, 256, 0, 256]).flatten()
    features = np.concatenate([hsv_hist, lab_hist]).astype(np.float32)
    total = float(features.sum())
    if total <= 0:
        return None

    features /= total
    norm = float(np.linalg.norm(features))
    if norm <= 0:
        return None

    return (features / norm).tolist()


def appearance_similarity(left: list[float], right: list[float]) -> float:
    import numpy as np

    return float(np.clip(np.dot(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)), 0.0, 1.0))


def bbox_center(xyxy) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))


def is_reid_spatial_match(current_xyxy, candidate_xyxy) -> bool:
    import math

    current_center = bbox_center(current_xyxy)
    candidate_center = bbox_center(candidate_xyxy)
    center_distance = math.dist(current_center, candidate_center)
    current_height = current_xyxy[3] - current_xyxy[1]
    candidate_height = candidate_xyxy[3] - candidate_xyxy[1]
    max_distance = max(35.0, min(current_height, candidate_height) * 0.6)
    height_ratio = max(current_height, candidate_height) / max(1.0, min(current_height, candidate_height))
    return center_distance <= max_distance and height_ratio <= 1.35


def predict_xyxy(samples: deque[PersistedTrackSample], missing_sampled_frames: int) -> list[float]:
    if len(samples) < 2:
        return list(samples[-1].xyxy)

    previous_sample = samples[-2]
    last_sample = samples[-1]
    frame_delta = last_sample.frame_index - previous_sample.frame_index
    if frame_delta <= 0:
        return list(last_sample.xyxy)

    velocity = [
        (last_value - previous_value) / frame_delta
        for previous_value, last_value in zip(previous_sample.xyxy, last_sample.xyxy)
    ]
    return [float(last_value + velocity_value * missing_sampled_frames) for last_value, velocity_value in zip(last_sample.xyxy, velocity)]


def clone_detections_with_metadata(
    detections,
    track_states: list[str],
    missing_sampled_frames: list[int],
    reid_applied: list[bool] | None = None,
    reassigned_from_track_id: list[int | None] | None = None,
    reid_score: list[float | None] | None = None,
):
    import supervision as sv

    detection_count = len(detections)

    return sv.Detections(
        xyxy=detections.xyxy.copy(),
        confidence=detections.confidence.copy() if detections.confidence is not None else None,
        class_id=detections.class_id.copy() if detections.class_id is not None else None,
        tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None,
        data={
            **detections.data,
            "confidence": confidences_for(detections),
            "class_id": class_ids_for(detections),
            "track_state": track_states,
            "missing_sampled_frames": missing_sampled_frames,
            "tracker_id": track_ids_for(detections),
            "class_name": class_names_for(detections),
            "reid_applied": reid_applied if reid_applied is not None else [False] * detection_count,
            "reassigned_from_track_id": (
                reassigned_from_track_id if reassigned_from_track_id is not None else [None] * detection_count
            ),
            "reid_score": reid_score if reid_score is not None else [None] * detection_count,
        },
        metadata=detections.metadata,
    )


def labels_for(detections) -> list[str]:
    prefixes = [label_prefix(track_id) for track_id in track_ids_for(detections)]
    class_names = class_names_for(detections)
    confidences = confidences_for(detections)
    metadata_items = []
    for state, missing_sampled_frames, reid_applied in zip(
        track_states_for(detections),
        missing_sampled_frames_for(detections),
        reid_applied_for(detections),
    ):
        metadata_items.append(
            [
                f"[{state}]" if state != "detected" else None,
                f"gap:{missing_sampled_frames}" if missing_sampled_frames else None,
                "reid" if reid_applied else None,
            ]
        )
    return [
        format_label(class_name, confidence, prefix=prefix, metadata=metadata)
        for class_name, confidence, prefix, metadata in zip(class_names, confidences, prefixes, metadata_items)
    ]


def draw_tracks(frame, detections):
    return annotate_detections(frame, detections, labels=labels_for(detections))


def write_track_records(handle, frame_index: int, detections) -> int:
    confidences = confidences_for(detections)
    count = 0
    for track_id, class_name, confidence, xyxy, track_state, missing_sampled_frames, reid_applied, reassigned_from_track_id, reid_score in zip(
        track_ids_for(detections),
        class_names_for(detections),
        confidences,
        detections.xyxy,
        track_states_for(detections),
        missing_sampled_frames_for(detections),
        reid_applied_for(detections),
        reassigned_from_track_ids_for(detections),
        reid_scores_for(detections),
    ):
        handle.write(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "track_state": track_state,
                    "missing_sampled_frames": missing_sampled_frames,
                    "class_name": str(class_name),
                    "confidence": None if confidence is None else float(confidence),
                    "xyxy": [float(value) for value in xyxy],
                    "reid_applied": reid_applied,
                    "reassigned_from_track_id": reassigned_from_track_id,
                    "reid_score": reid_score,
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


def foot_point_for_bbox(xyxy) -> list[float]:
    x1, _, x2, y2 = xyxy
    return [float((x1 + x2) / 2.0), float(y2)]


def compute_frame_homography(keypoints: list[dict], min_confidence: float):
    return compute_homography(
        keypoints,
        min_confidence=min_confidence,
        min_points=4,
        reproj_threshold=8.0,
        geometry_mode="strict",
        point_map=canonical_point_map("strict"),
    )


def write_projected_track_records(handle, frame_index: int, detections, homography_result) -> int:
    confidences = confidences_for(detections)
    image_points = [foot_point_for_bbox(xyxy) for xyxy in detections.xyxy]
    projected_points = project_points(homography_result.matrix, image_points)
    count = 0
    for track_id, class_name, confidence, xyxy, image_point, projected_point, track_state, missing_sampled_frames, reid_applied, reassigned_from_track_id, reid_score in zip(
        track_ids_for(detections),
        class_names_for(detections),
        confidences,
        detections.xyxy,
        image_points,
        projected_points,
        track_states_for(detections),
        missing_sampled_frames_for(detections),
        reid_applied_for(detections),
        reassigned_from_track_ids_for(detections),
        reid_scores_for(detections),
    ):
        handle.write(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "track_state": track_state,
                    "missing_sampled_frames": missing_sampled_frames,
                    "class_name": str(class_name),
                    "confidence": None if confidence is None else float(confidence),
                    "xyxy": [float(value) for value in xyxy],
                    "image_foot_xy": image_point,
                    "court_xy_m": [float(projected_point[0]), float(projected_point[1])],
                    "reid_applied": reid_applied,
                    "reassigned_from_track_id": reassigned_from_track_id,
                    "reid_score": reid_score,
                    "homography_frame_status": "ok",
                    "homography_inliers_count": homography_result.inliers_count,
                    "homography_reprojection_mean_m": homography_result.reprojection_mean_m,
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
    json_output: Path | None = None,
    frame_stride: int = 30,
    max_frames: int | None = None,
    sampled_output_only: bool = True,
    confidence: float = 0.0,
    output_fps: float | None = None,
    smoothing_window: int = 3,
    reuse_stale_tracks: bool = False,
    persist_tracks: bool = True,
    max_missing_sampled_frames: int = 1,
    court_keypoints: bool = False,
    court_json_output: Path | None = None,
    court_model_id: str = COURT_MODEL_ID,
    court_confidence: float = 0.7,
    projected_tracks_output: Path | None = None,
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
    if projected_tracks_output is not None:
        projected_tracks_output.parent.mkdir(parents=True, exist_ok=True)
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
    projected_track_records = 0
    projected_track_frames = 0

    try:
        client = roboflow_client(api_key)
        tracker = sv.ByteTrack()
        smoother = TrackSmoother(window_size=smoothing_window)
        persister = TrackPersistence(max_missing_sampled_frames=max_missing_sampled_frames if persist_tracks else 0)
        with tracks_output.open("w", encoding="utf-8") as tracks_file:
            json_context = json_output.open("w", encoding="utf-8") if json_output is not None else nullcontext()
            court_context = court_json_output.open("w", encoding="utf-8") if court_json_output is not None else nullcontext()
            projected_context = (
                projected_tracks_output.open("w", encoding="utf-8")
                if projected_tracks_output is not None
                else nullcontext()
            )
            with json_context as json_file, court_context as court_file, projected_context as projected_file:
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
                        detections = persister.update_with_detections(detections, frame_index, frame)
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
                            try:
                                homography_result = compute_frame_homography(court_keypoints_list, court_confidence)
                            except (RuntimeError, ValueError):
                                homography_result = None
                            if homography_result is not None and projected_file is not None and len(detections) > 0:
                                projected_track_records += write_projected_track_records(
                                    projected_file,
                                    frame_index,
                                    detections,
                                    homography_result,
                                )
                                projected_track_frames += 1
                            court_frames += 1

                        writer.write(annotated_frame)
                    elif not sampled_output_only:
                        stale_detections = persister.current_detections() if reuse_stale_tracks else None
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
        "projected_track_records": projected_track_records,
        "projected_track_frames": projected_track_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Track basketball players in sampled video frames with ByteTrack.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, help="Path for the tracked video")
    parser.add_argument("--tracks-output", type=Path, help="Path for JSONL track records")
    parser.add_argument("--json-output", type=Path, help="Path for per-inference raw JSONL records")
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

    args.output = args.output or Path("outputs") / f"{args.video.stem}_tracked.mp4"
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
            print(f"Court keypoints: {args.court_keypoints}")
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
    print(f"Track records: {stats['track_records']}")
    print(f"Inference records: {stats['inference_records']}")
    if args.court_keypoints:
        print(f"Court frames: {stats['court_frames']}")
        print(f"Court Roboflow calls: {stats['court_roboflow_calls']}")
        print(f"Court keypoint records: {stats['court_inference_records']}")
        print(f"Projected track records: {stats['projected_track_records']}")
        print(f"Projected track frames: {stats['projected_track_frames']}")
    print(f"Tracked video saved to: {args.output}")
    print(f"Track JSONL saved to: {args.tracks_output}")
    print(f"Inference JSONL saved to: {args.json_output}")
    if args.court_keypoints:
        print(f"Court JSONL saved to: {args.court_json_output}")
        print(f"Projected track JSONL saved to: {args.projected_tracks_output}")


if __name__ == "__main__":
    main()
