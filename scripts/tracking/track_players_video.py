#!/usr/bin/env python3
"""Run Roboflow player detection plus a pluggable tracker on sampled video frames."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import tomllib
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.shared.annotation_helpers import (  # noqa: E402
    annotate_detections,
    class_names_for,
    detections_from_inference,
    format_label,
    label_prefix,
)
from scripts.tracking.detect_players_video import (  # noqa: E402
    MODEL_ID,
    default_output_fps,
    inspect_video,
    open_video,
    roboflow_client,
    video_info,
)
from scripts.court.build_court_homography import compute_homography, project_points  # noqa: E402
from scripts.court.court_template import canonical_point_map  # noqa: E402
from scripts.court.detect_court_keypoints import (  # noqa: E402
    DEFAULT_MODEL_ID as COURT_MODEL_ID,
    draw_with_annotators,
    parse_keypoints,
    to_keypoints,
)
from scripts.shared.env_config import get_roboflow_api_key  # noqa: E402


DEFAULT_TRACKER_BACKEND = "bytetrack"
DEFAULT_DEEPOCSORT_CONFIG_PATH = Path("config") / "deepocsort.toml"
DEFAULT_REID_CONFIG_PATH = Path("config") / "reid.toml"
DEFAULT_DEEPOCSORT_REID_WEIGHTS = "osnet_x0_25_msmt17.pt"
DEFAULT_DEEPOCSORT_DEVICE = "cpu"
TRACKER_BACKEND_CHOICES = (DEFAULT_TRACKER_BACKEND, "deepocsort")

DEEPOCSORT_RUNTIME_DEFAULTS = {
    "reid_weights": DEFAULT_DEEPOCSORT_REID_WEIGHTS,
    "device": DEFAULT_DEEPOCSORT_DEVICE,
    "half": False,
}

DEEPOCSORT_TUNING_DEFAULTS = {
    "det_thresh": 0.3,
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3,
    "delta_t": 3,
    "inertia": 0.2,
    "w_association_emb": 0.5,
    "alpha_fixed_emb": 0.95,
    "aw_param": 0.5,
    "embedding_off": False,
    "cmc_off": False,
    "aw_off": False,
}

DEEPOCSORT_CONFIG_TYPES = {
    "reid_weights": str,
    "device": str,
    "half": bool,
    "det_thresh": (int, float),
    "max_age": int,
    "min_hits": int,
    "iou_threshold": (int, float),
    "delta_t": int,
    "inertia": (int, float),
    "w_association_emb": (int, float),
    "alpha_fixed_emb": (int, float),
    "aw_param": (int, float),
    "embedding_off": bool,
    "cmc_off": bool,
    "aw_off": bool,
}

REID_CROP_DEFAULTS = {
    "enabled": True,
    "x_margin_fraction": 0.2,
    "y_start_fraction": 0.15,
    "y_end_fraction": 0.65,
}

REID_OVERLAP_DEFAULTS = {
    "heavy_overlap_fraction": 0.35,
    "min_visible_fraction": 0.4,
}

REID_QUALITY_DEFAULTS = {
    "min_box_width": 12,
    "min_box_height": 24,
    "min_crop_width": 8,
    "min_crop_height": 12,
}

REID_CONFIG_TYPES = {
    "enabled": bool,
    "x_margin_fraction": (int, float),
    "y_start_fraction": (int, float),
    "y_end_fraction": (int, float),
    "heavy_overlap_fraction": (int, float),
    "min_visible_fraction": (int, float),
    "min_box_width": int,
    "min_box_height": int,
    "min_crop_width": int,
    "min_crop_height": int,
}


def load_deepocsort_config(config_path: Path) -> dict[str, object]:
    try:
        with config_path.open("rb") as config_file:
            raw_config = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise RuntimeError(f"Deep OC-SORT config file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise RuntimeError(f"Could not parse Deep OC-SORT config file {config_path}: {error}") from error

    deepocsort_config = raw_config.get("deepocsort")
    if deepocsort_config is None:
        deepocsort_config = {}
    if not isinstance(deepocsort_config, dict):
        raise RuntimeError(f"Deep OC-SORT config file must define a [deepocsort] table: {config_path}")

    runtime_config = deepocsort_config.get("runtime") or {}
    tuning_config = deepocsort_config.get("tuning") or {}
    if not isinstance(runtime_config, dict):
        raise RuntimeError(f"[deepocsort.runtime] must be a table in {config_path}")
    if not isinstance(tuning_config, dict):
        raise RuntimeError(f"[deepocsort.tuning] must be a table in {config_path}")

    merged = {
        **DEEPOCSORT_RUNTIME_DEFAULTS,
        **DEEPOCSORT_TUNING_DEFAULTS,
        **runtime_config,
        **tuning_config,
    }
    for key, expected_type in DEEPOCSORT_CONFIG_TYPES.items():
        value = merged[key]
        if not isinstance(value, expected_type):
            expected_names = (
                ", ".join(sorted(item.__name__ for item in expected_type))
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            raise RuntimeError(
                f"Deep OC-SORT config value {key!r} in {config_path} must be {expected_names}, got {type(value).__name__}"
            )

    return merged


def resolve_deepocsort_settings(
    config_path: Path,
    *,
    deepocsort_reid_weights: str | None,
    deepocsort_device: str | None,
    deepocsort_half: bool | None,
) -> dict[str, object]:
    settings = load_deepocsort_config(config_path)
    if deepocsort_reid_weights is not None:
        settings["reid_weights"] = deepocsort_reid_weights
    if deepocsort_device is not None:
        settings["device"] = deepocsort_device
    if deepocsort_half is not None:
        settings["half"] = deepocsort_half
    return settings


def load_reid_config(config_path: Path) -> dict[str, object]:
    try:
        with config_path.open("rb") as config_file:
            raw_config = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise RuntimeError(f"ReID config file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise RuntimeError(f"Could not parse ReID config file {config_path}: {error}") from error

    reid_config = raw_config.get("reid")
    if reid_config is None:
        reid_config = {}
    if not isinstance(reid_config, dict):
        raise RuntimeError(f"ReID config file must define a [reid] table: {config_path}")

    crop_config = reid_config.get("crop") or {}
    overlap_config = reid_config.get("overlap") or {}
    quality_config = reid_config.get("quality") or {}
    if not isinstance(crop_config, dict):
        raise RuntimeError(f"[reid.crop] must be a table in {config_path}")
    if not isinstance(overlap_config, dict):
        raise RuntimeError(f"[reid.overlap] must be a table in {config_path}")
    if not isinstance(quality_config, dict):
        raise RuntimeError(f"[reid.quality] must be a table in {config_path}")

    merged = {
        **REID_CROP_DEFAULTS,
        **REID_OVERLAP_DEFAULTS,
        **REID_QUALITY_DEFAULTS,
        **crop_config,
        **overlap_config,
        **quality_config,
    }
    for key, expected_type in REID_CONFIG_TYPES.items():
        value = merged[key]
        if not isinstance(value, expected_type):
            expected_names = (
                ", ".join(sorted(item.__name__ for item in expected_type))
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            raise RuntimeError(
                f"ReID config value {key!r} in {config_path} must be {expected_names}, got {type(value).__name__}"
            )

    if not 0.0 <= float(merged["x_margin_fraction"]) < 0.5:
        raise RuntimeError(f"ReID config x_margin_fraction in {config_path} must be between 0.0 and 0.5")
    if not 0.0 <= float(merged["y_start_fraction"]) < float(merged["y_end_fraction"]) <= 1.0:
        raise RuntimeError(
            f"ReID config y_start_fraction/y_end_fraction in {config_path} must satisfy 0.0 <= start < end <= 1.0"
        )
    if not 0.0 <= float(merged["heavy_overlap_fraction"]) <= 1.0:
        raise RuntimeError(f"ReID config heavy_overlap_fraction in {config_path} must be between 0.0 and 1.0")
    if not 0.0 <= float(merged["min_visible_fraction"]) <= 1.0:
        raise RuntimeError(f"ReID config min_visible_fraction in {config_path} must be between 0.0 and 1.0")

    return merged


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
    def __init__(self, max_missing_sampled_frames: int = 1, reid_config: dict[str, object] | None = None) -> None:
        self.max_missing_sampled_frames = max_missing_sampled_frames
        self.reid_config = reid_config or {**REID_CROP_DEFAULTS, **REID_OVERLAP_DEFAULTS, **REID_QUALITY_DEFAULTS}
        self.history: dict[int, deque[PersistedTrackSample]] = defaultdict(lambda: deque(maxlen=2))
        self.last_detections = None

    def update_with_detections(self, detections, frame_index: int, frame):
        import numpy as np
        import supervision as sv

        states = ["detected"] * len(detections)
        if self.max_missing_sampled_frames < 1:
            persisted = clone_detections_with_metadata(
                detections,
                track_states=states,
                missing_sampled_frames=[0] * len(detections),
            )
            self.last_detections = persisted
            return persisted

        rows = build_detection_rows(detections, frame, self.reid_config)
        self.reassign_recent_tracks(rows, frame_index)

        present_track_ids: set[int] = set()

        xyxy_rows: list[list[float]] = []
        confidence_rows: list[float | None] = []
        class_name_rows: list[str] = []
        class_id_rows: list[int | None] = []
        track_id_rows: list[int | None] = []
        track_state_rows: list[str] = []
        missing_rows: list[int] = []

        for row in rows:
            xyxy_rows.append(row["xyxy"])
            confidence_rows.append(row["confidence"])
            class_name_rows.append(row["class_name"])
            class_id_rows.append(row["class_id"])
            track_id_rows.append(row["track_id"])
            track_state_rows.append("detected")
            missing_rows.append(0)

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


class TrackerAdapter:
    def __init__(
        self,
        backend: str = DEFAULT_TRACKER_BACKEND,
        tracker_config: Path = DEFAULT_DEEPOCSORT_CONFIG_PATH,
        deepocsort_reid_weights: str | None = None,
        deepocsort_device: str | None = None,
        deepocsort_half: bool | None = None,
    ) -> None:
        self.backend = backend

        if backend == "bytetrack":
            import supervision as sv

            self._tracker = sv.ByteTrack()
            return

        if backend == "deepocsort":
            try:
                from boxmot.trackers.registry import create_tracker
            except ImportError as error:
                raise RuntimeError(
                    "deepocsort requires a BoxMOT build that exposes boxmot.trackers.registry.create_tracker. Install dependencies from requirements.txt to enable it."
                ) from error

            resolved_settings = resolve_deepocsort_settings(
                tracker_config,
                deepocsort_reid_weights=deepocsort_reid_weights,
                deepocsort_device=deepocsort_device,
                deepocsort_half=deepocsort_half,
            )
            tracker_kwargs = {key: resolved_settings[key] for key in DEEPOCSORT_TUNING_DEFAULTS}
            self._tracker = create_tracker(
                "deepocsort",
                reid_weights=str(resolved_settings["reid_weights"]),
                device=str(resolved_settings["device"]),
                half=bool(resolved_settings["half"]),
                tracker_kwargs=tracker_kwargs,
            )
            return

        raise RuntimeError(
            f"Unsupported tracker backend: {backend}. Supported backends: {', '.join(TRACKER_BACKEND_CHOICES)}"
        )

    def update(self, detections, frame):
        if self.backend == "bytetrack":
            del frame
            return self._tracker.update_with_detections(detections)

        if self.backend == "deepocsort":
            return boxmot_detections_to_supervision(
                self._tracker.update(supervision_detections_to_boxmot(detections), frame),
                detections,
            )

        raise RuntimeError(f"Unsupported tracker backend during update: {self.backend}")


def detections_from_result(result: dict, min_confidence: float):
    return detections_from_inference(result, min_confidence=min_confidence)


def supervision_detections_to_boxmot(detections):
    import numpy as np

    if len(detections) == 0:
        return np.empty((0, 6), dtype=np.float32)

    confidences = [0.0 if confidence is None else float(confidence) for confidence in confidences_for(detections)]
    class_ids = [0 if class_id is None else int(class_id) for class_id in class_ids_for(detections)]
    rows = [
        [float(x1), float(y1), float(x2), float(y2), confidence, float(class_id)]
        for (x1, y1, x2, y2), confidence, class_id in zip(detections.xyxy, confidences, class_ids)
    ]
    return np.asarray(rows, dtype=np.float32)


def boxmot_detections_to_supervision(tracks, source_detections):
    import numpy as np
    import supervision as sv

    track_array = np.asarray(tracks, dtype=float)
    if track_array.size == 0:
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
            },
            metadata=source_detections.metadata,
        )
    if track_array.ndim == 1:
        track_array = track_array.reshape(1, -1)

    source_class_names = class_names_for(source_detections)
    source_class_ids = class_ids_for(source_detections)

    xyxy = track_array[:, :4].astype(float, copy=False)
    tracker_ids = track_array[:, 4].astype(int, copy=False)
    confidences = track_array[:, 5].astype(float, copy=False)
    class_ids = track_array[:, 6].astype(int, copy=False)

    det_indices = track_array[:, 7].astype(int, copy=False) if track_array.shape[1] > 7 else None
    class_names = []
    original_class_ids = []
    for row_index, class_id in enumerate(class_ids):
        det_index = None if det_indices is None else int(det_indices[row_index])
        if det_index is not None and 0 <= det_index < len(source_class_names):
            class_names.append(str(source_class_names[det_index]))
            original_class_ids.append(source_class_ids[det_index])
            continue
        class_names.append(str(class_id))
        original_class_ids.append(int(class_id))

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidences,
        class_id=(
            None
            if any(class_id is None for class_id in original_class_ids)
            else np.asarray(original_class_ids, dtype=int)
        ),
        tracker_id=tracker_ids,
        data={
            "confidence": confidences.tolist(),
            "class_id": original_class_ids,
            "class_name": class_names,
            "tracker_id": tracker_ids.tolist(),
        },
        metadata=source_detections.metadata,
    )


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


def build_detection_rows(detections, frame, reid_config: dict[str, object]) -> list[dict]:
    rows = []
    confidences = confidences_for(detections)
    boxes = [[float(value) for value in xyxy] for xyxy in detections.xyxy]
    for track_id, class_id, class_name, confidence, xyxy in zip(
        track_ids_for(detections),
        class_ids_for(detections),
        class_names_for(detections),
        confidences,
        boxes,
    ):
        xyxy_list = list(xyxy)
        rows.append(
            {
                "track_id": track_id,
                "class_id": class_id,
                "class_name": str(class_name),
                "confidence": None if confidence is None else float(confidence),
                "xyxy": xyxy_list,
                "appearance": extract_torso_appearance(
                    frame,
                    xyxy_list,
                    [other_box for other_box in boxes if other_box is not xyxy],
                    reid_config,
                ),
                "reid_applied": False,
                "reassigned_from_track_id": None,
                "reid_score": None,
            }
        )
    return rows


def extract_torso_appearance(frame, xyxy, other_boxes, reid_config: dict[str, object]) -> list[float] | None:
    import cv2
    import numpy as np

    if not bool(reid_config["enabled"]):
        return None

    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < int(reid_config["min_box_width"]) or box_height < int(reid_config["min_box_height"]):
        return None

    torso_xyxy = torso_crop_xyxy(frame.shape, xyxy, reid_config)
    if torso_xyxy is None:
        return None

    visible_regions = [torso_xyxy]
    for other_xyxy in other_boxes:
        overlap = intersect_xyxy(torso_xyxy, other_xyxy)
        if overlap is None:
            continue
        next_visible_regions = []
        for region in visible_regions:
            next_visible_regions.extend(subtract_xyxy(region, overlap))
        visible_regions = next_visible_regions
        if not visible_regions:
            return None

    torso_area = xyxy_area(torso_xyxy)
    visible_area = sum(xyxy_area(region) for region in visible_regions)
    if torso_area <= 0 or visible_area <= 0:
        return None

    overlap_fraction = 1.0 - (visible_area / torso_area)
    if overlap_fraction >= float(reid_config["heavy_overlap_fraction"]):
        return None

    best_visible_region = max(visible_regions, key=xyxy_area)
    if xyxy_area(best_visible_region) / torso_area < float(reid_config["min_visible_fraction"]):
        return None

    torso_x1, torso_y1, torso_x2, torso_y2 = best_visible_region
    if torso_x2 - torso_x1 < int(reid_config["min_crop_width"]) or torso_y2 - torso_y1 < int(reid_config["min_crop_height"]):
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


def torso_crop_xyxy(frame_shape, xyxy, reid_config: dict[str, object]) -> tuple[int, int, int, int] | None:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1

    x_margin_fraction = float(reid_config["x_margin_fraction"])
    y_start_fraction = float(reid_config["y_start_fraction"])
    y_end_fraction = float(reid_config["y_end_fraction"])
    torso_x1 = max(0, min(width, int(round(x1 + box_width * x_margin_fraction))))
    torso_x2 = max(0, min(width, int(round(x2 - box_width * x_margin_fraction))))
    torso_y1 = max(0, min(height, int(round(y1 + box_height * y_start_fraction))))
    torso_y2 = max(0, min(height, int(round(y1 + box_height * y_end_fraction))))
    if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
        return None
    return (torso_x1, torso_y1, torso_x2, torso_y2)


def intersect_xyxy(left_xyxy, right_xyxy) -> tuple[int, int, int, int] | None:
    x1 = max(int(left_xyxy[0]), int(right_xyxy[0]))
    y1 = max(int(left_xyxy[1]), int(right_xyxy[1]))
    x2 = min(int(left_xyxy[2]), int(right_xyxy[2]))
    y2 = min(int(left_xyxy[3]), int(right_xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def subtract_xyxy(source_xyxy, cut_xyxy) -> list[tuple[int, int, int, int]]:
    overlap = intersect_xyxy(source_xyxy, cut_xyxy)
    if overlap is None:
        return [tuple(int(value) for value in source_xyxy)]

    source_x1, source_y1, source_x2, source_y2 = (int(value) for value in source_xyxy)
    cut_x1, cut_y1, cut_x2, cut_y2 = overlap
    pieces = [
        (source_x1, source_y1, source_x2, cut_y1),
        (source_x1, cut_y2, source_x2, source_y2),
        (source_x1, cut_y1, cut_x1, cut_y2),
        (cut_x2, cut_y1, source_x2, cut_y2),
    ]
    return [piece for piece in pieces if piece[2] > piece[0] and piece[3] > piece[1]]


def xyxy_area(xyxy) -> int:
    return max(0, int(xyxy[2]) - int(xyxy[0])) * max(0, int(xyxy[3]) - int(xyxy[1]))


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
):
    import supervision as sv

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
        },
        metadata=detections.metadata,
    )


def labels_for(detections) -> list[str]:
    prefixes = [label_prefix(track_id) for track_id in track_ids_for(detections)]
    class_names = class_names_for(detections)
    confidences = confidences_for(detections)
    metadata_items = []
    for state, missing_sampled_frames in zip(track_states_for(detections), missing_sampled_frames_for(detections)):
        metadata_items.append(
            [
                f"[{state}]" if state != "detected" else None,
                f"gap:{missing_sampled_frames}" if missing_sampled_frames else None,
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
    for track_id, class_name, confidence, xyxy, track_state, missing_sampled_frames in zip(
        track_ids_for(detections),
        class_names_for(detections),
        confidences,
        detections.xyxy,
        track_states_for(detections),
        missing_sampled_frames_for(detections),
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
    for track_id, class_name, confidence, xyxy, image_point, projected_point, track_state, missing_sampled_frames in zip(
        track_ids_for(detections),
        class_names_for(detections),
        confidences,
        detections.xyxy,
        image_points,
        projected_points,
        track_states_for(detections),
        missing_sampled_frames_for(detections),
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
    tracker_backend: str = DEFAULT_TRACKER_BACKEND,
    tracker_config: Path = DEFAULT_DEEPOCSORT_CONFIG_PATH,
    reid_config: Path = DEFAULT_REID_CONFIG_PATH,
    deepocsort_reid_weights: str | None = None,
    deepocsort_device: str | None = None,
    deepocsort_half: bool | None = None,
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
        resolved_reid_config = load_reid_config(reid_config)
        client = roboflow_client(api_key)
        tracker = TrackerAdapter(
            backend=tracker_backend,
            tracker_config=tracker_config,
            deepocsort_reid_weights=deepocsort_reid_weights,
            deepocsort_device=deepocsort_device,
            deepocsort_half=deepocsort_half,
        )
        smoother = TrackSmoother(window_size=smoothing_window)
        persister = TrackPersistence(
            max_missing_sampled_frames=max_missing_sampled_frames if persist_tracks else 0,
            reid_config=resolved_reid_config,
        )
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
                        detections = tracker.update(detections, frame)
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
        "tracker_backend": tracker_backend,
        "tracker_config": str(tracker_config) if tracker_backend == "deepocsort" else None,
        "reid_config": str(reid_config),
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
    parser = argparse.ArgumentParser(description="Track basketball players in sampled video frames.")
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
        "--tracker-backend",
        choices=TRACKER_BACKEND_CHOICES,
        default=DEFAULT_TRACKER_BACKEND,
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
