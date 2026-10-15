"""Minimal make/miss event extraction from detector-frame action classes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

BALL_IN_BASKET_CLASS_ID = 1
JUMP_SHOT_CLASS_ID = 5
LAYUP_DUNK_CLASS_ID = 6
BALL_CLASS_ID = 0

BALL_IN_BASKET_CLASS_NAMES = {"ball_in_basket"}
JUMP_SHOT_CLASS_NAMES = {"jump_shot"}
LAYUP_DUNK_CLASS_NAMES = {"layup_dunk"}
BALL_CLASS_NAMES = {"ball"}
# A shot box is a false positive when it is cut off by the image edge and the detector sees
# the ball only far away from it. Distance alone is not enough: the detector's ball can itself
# be wrong, and confident mid-court shot boxes must survive that.
SHOT_BALL_MAX_DISTANCE_BOX_HEIGHTS = 2.0
SHOT_BOX_EDGE_TOLERANCE_PX = 1.0
# A ball-in-basket box is a false positive when the detector sees the ball confidently far away
# (a pass in a player's hands). A ball hidden in the net is often missed or low-confidence, so
# only confident balls count.
BASKET_CONTRADICTING_BALL_MIN_CONFIDENCE = 0.5
BASKET_CONTRADICTING_BALL_MIN_DISTANCE_BASKET_WIDTHS = 10.0
SAME_EVENT_WINDOW_SECONDS = 1.0
OPENING_CONFIRMATION_WINDOW_SECONDS = 0.3
POST_MADE_SUPPRESSION_SECONDS = 2.0
MISSED_SHOT_TIMEOUT_SECONDS = 5.0
LOW_CONFIDENCE_BURST_THRESHOLD = 0.5
# A weak first box must not swallow the confident boxes that follow it (a real shot often starts
# with one weak box); two consecutive confident boxes rule out a lone confident false positive.
SUPPRESSED_BURST_RELEASE_CONFIDENT_FRAMES = 2
# The event type is a confidence-weighted vote over the last burst of shot boxes. A single stray
# box of the other class (a jump shot read once as a layup) must not relabel the whole event.
SHOT_TYPE_BURST_GAP_SECONDS = 0.3
# Used only when a timestamp is missing: 0.3 s at 30 fps.
SHOT_TYPE_BURST_GAP_FRAMES_FALLBACK = 9


@dataclass
class ShotEventCandidate:
    event_id: int
    shot_type: str
    start_frame_index: int
    start_timestamp_seconds: float | None
    start_image_xy: tuple[float, float] | None
    start_shot_xyxy: tuple[float, float, float, float] | None
    latest_shot_frame_index: int
    latest_shot_timestamp_seconds: float | None
    latest_shot_image_xy: tuple[float, float] | None
    latest_shot_xyxy: tuple[float, float, float, float] | None
    shot_signal_active: bool = True
    inactive_inference_count: int = 0
    ball_in_basket_frame_index: int | None = None
    ball_in_basket_timestamp_seconds: float | None = None
    # (frame_index, timestamp_seconds, shot_type, confidence) for every frame with a shot signal.
    shot_frames: list[tuple[int, float | None, str, float | None]] = field(default_factory=list)


@dataclass
class ShotEventCooldown:
    until_timestamp_seconds: float | None
    shot_signal_active: bool = False


@dataclass
class SuppressedShotBurst:
    latest_shot_frame_index: int
    latest_shot_timestamp_seconds: float | None
    shot_signal_active: bool = True
    # Current run of consecutive confident shot frames, kept so a release loses nothing:
    # (frame_index, timestamp_seconds, shot_type, confidence, center_xy, xyxy).
    confident_run: list[tuple] = field(default_factory=list)


def shot_signal_trace_record(
    *,
    frame_index: int,
    timestamp_seconds: float | None,
    signals: dict[str, object],
    candidate: ShotEventCandidate | ShotEventCooldown | SuppressedShotBurst | None,
    resolved_event: dict | None,
) -> dict:
    shot_type_signal = shot_type_for_frame(signals)
    if isinstance(candidate, ShotEventCandidate):
        event_state = "active"
        reason = "tracking_shot_signal" if shot_type_signal is not None else "awaiting_resolution"
    elif isinstance(candidate, ShotEventCooldown):
        event_state = "cooldown"
        reason = f"resolved_{resolved_event['outcome']}" if resolved_event is not None else "cooldown"
    elif isinstance(candidate, SuppressedShotBurst):
        event_state = "idle"
        reason = "suppressing_low_confidence_burst"
    else:
        event_state = "idle"
        reason = f"resolved_{resolved_event['outcome']}" if resolved_event is not None else "idle"

    record = {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "has_jump_shot": signals["has_jump_shot"],
        "has_ball_in_basket": signals["has_ball_in_basket"],
        "has_layup_dunk": signals["has_layup_dunk"],
        "shot_type_signal": shot_type_signal,
        "event_state": event_state,
        "reason": reason,
    }
    if isinstance(candidate, ShotEventCandidate):
        record["event_id"] = candidate.event_id
    return record


def shot_raw_prediction_record(
    *,
    frame_index: int,
    timestamp_seconds: float | None,
    model_id: str,
    result: object,
) -> dict:
    predictions = result.get("predictions", []) if isinstance(result, dict) else []
    return {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "model_id": model_id,
        "predictions": predictions if isinstance(predictions, list) else [],
    }


def normalize_prediction_class_name(value: object | None) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def prediction_has_class_id(prediction: dict, target_class_id: int) -> bool:
    value = prediction.get("class_id")
    if value is None:
        return False
    try:
        return int(value) == target_class_id
    except (TypeError, ValueError):
        return False


def prediction_matches(prediction: dict, *, class_id: int, class_names: set[str]) -> bool:
    if prediction_has_class_id(prediction, class_id):
        return True
    return normalize_prediction_class_name(prediction.get("class")) in class_names


def prediction_confidence(prediction: dict) -> float | None:
    value = prediction.get("confidence")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def comparable_prediction_confidence(prediction: dict) -> float:
    return prediction_confidence(prediction) or 0.0


def prediction_xyxy(prediction: dict) -> tuple[float, float, float, float] | None:
    keys = ("x", "y", "width", "height")
    if any(prediction.get(key) is None for key in keys):
        return None
    try:
        center_x = float(prediction["x"])
        center_y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
    except (TypeError, ValueError):
        return None
    half_width = width / 2.0
    half_height = height / 2.0
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def prediction_center_xy(prediction: dict) -> tuple[float, float] | None:
    xyxy = prediction_xyxy(prediction)
    if xyxy is None:
        return None
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def best_shot_prediction(result: object) -> tuple[str | None, dict | None]:
    predictions = result.get("predictions", []) if isinstance(result, dict) else []
    best_by_type: dict[str, dict] = {}
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        if prediction_matches(prediction, class_id=LAYUP_DUNK_CLASS_ID, class_names=LAYUP_DUNK_CLASS_NAMES):
            shot_type = "layup_dunk"
        elif prediction_matches(prediction, class_id=JUMP_SHOT_CLASS_ID, class_names=JUMP_SHOT_CLASS_NAMES):
            shot_type = "jump_shot"
        else:
            continue
        current = best_by_type.get(shot_type)
        if current is None or comparable_prediction_confidence(prediction) > comparable_prediction_confidence(current):
            best_by_type[shot_type] = prediction
    if "layup_dunk" in best_by_type:
        return "layup_dunk", best_by_type["layup_dunk"]
    if "jump_shot" in best_by_type:
        return "jump_shot", best_by_type["jump_shot"]
    return None, None


def _shot_prediction_is_truncated_and_far_from_ball(
    prediction: dict, ball_centers: list[tuple[float, float]]
) -> bool:
    """True for an edge-truncated shot box when every ball detected this frame is far from it.

    A frame with no ball detection never rejects a shot, so occluded balls cost no recall. The
    right edge cannot be checked because the frame width is not part of the detector result.
    """
    if not ball_centers:
        return False
    xyxy = prediction_xyxy(prediction)
    center = prediction_center_xy(prediction)
    if xyxy is None or center is None:
        return False
    if xyxy[0] > SHOT_BOX_EDGE_TOLERANCE_PX and xyxy[1] > SHOT_BOX_EDGE_TOLERANCE_PX:
        return False
    max_distance = SHOT_BALL_MAX_DISTANCE_BOX_HEIGHTS * max(xyxy[3] - xyxy[1], 1.0)
    nearest = min(
        ((center[0] - ball_x) ** 2 + (center[1] - ball_y) ** 2) ** 0.5 for ball_x, ball_y in ball_centers
    )
    return nearest > max_distance


def _ball_detections(predictions: list) -> list[tuple[tuple[float, float], float]]:
    return [
        (center, comparable_prediction_confidence(prediction))
        for prediction in predictions
        if isinstance(prediction, dict)
        and prediction_matches(prediction, class_id=BALL_CLASS_ID, class_names=BALL_CLASS_NAMES)
        for center in [prediction_center_xy(prediction)]
        if center is not None
    ]


def reject_shot_predictions_far_from_ball(predictions: list) -> list:
    ball_centers = [center for center, _ in _ball_detections(predictions)]
    return [
        prediction
        for prediction in predictions
        if not (
            isinstance(prediction, dict)
            and (
                prediction_matches(prediction, class_id=JUMP_SHOT_CLASS_ID, class_names=JUMP_SHOT_CLASS_NAMES)
                or prediction_matches(prediction, class_id=LAYUP_DUNK_CLASS_ID, class_names=LAYUP_DUNK_CLASS_NAMES)
            )
            and _shot_prediction_is_truncated_and_far_from_ball(prediction, ball_centers)
        )
    ]


def _basket_is_contradicted_by_confident_ball(
    prediction: dict, confident_ball_centers: list[tuple[float, float]]
) -> bool:
    """True when a confident ball is detected this frame and every one is far from the basket box."""
    if not confident_ball_centers:
        return False
    xyxy = prediction_xyxy(prediction)
    center = prediction_center_xy(prediction)
    if xyxy is None or center is None:
        return False
    min_distance = BASKET_CONTRADICTING_BALL_MIN_DISTANCE_BASKET_WIDTHS * max(xyxy[2] - xyxy[0], 1.0)
    nearest = min(
        ((center[0] - ball_x) ** 2 + (center[1] - ball_y) ** 2) ** 0.5 for ball_x, ball_y in confident_ball_centers
    )
    return nearest > min_distance


def reject_ball_in_basket_predictions_contradicted_by_ball(predictions: list) -> list:
    confident_ball_centers = [
        center
        for center, confidence in _ball_detections(predictions)
        if confidence >= BASKET_CONTRADICTING_BALL_MIN_CONFIDENCE
    ]
    return [
        prediction
        for prediction in predictions
        if not (
            isinstance(prediction, dict)
            and prediction_matches(
                prediction, class_id=BALL_IN_BASKET_CLASS_ID, class_names=BALL_IN_BASKET_CLASS_NAMES
            )
            and _basket_is_contradicted_by_confident_ball(prediction, confident_ball_centers)
        )
    ]


def shot_frame_signals(result: object) -> dict[str, object]:
    predictions = result.get("predictions", []) if isinstance(result, dict) else []
    predictions = reject_shot_predictions_far_from_ball(predictions)
    predictions = reject_ball_in_basket_predictions_contradicted_by_ball(predictions)
    result = {"predictions": predictions}
    shot_type_signal, shot_prediction = best_shot_prediction(result)
    has_jump_shot = any(
        prediction_matches(prediction, class_id=JUMP_SHOT_CLASS_ID, class_names=JUMP_SHOT_CLASS_NAMES)
        for prediction in predictions
        if isinstance(prediction, dict)
    )
    has_layup_dunk = any(
        prediction_matches(prediction, class_id=LAYUP_DUNK_CLASS_ID, class_names=LAYUP_DUNK_CLASS_NAMES)
        for prediction in predictions
        if isinstance(prediction, dict)
    )
    has_ball_in_basket = any(
        prediction_matches(
            prediction,
            class_id=BALL_IN_BASKET_CLASS_ID,
            class_names=BALL_IN_BASKET_CLASS_NAMES,
        )
        for prediction in predictions
        if isinstance(prediction, dict)
    )
    return {
        "has_jump_shot": has_jump_shot,
        "has_layup_dunk": has_layup_dunk,
        "has_ball_in_basket": has_ball_in_basket,
        "shot_type_signal": shot_type_signal,
        "shot_signal_confidence": None if shot_prediction is None else prediction_confidence(shot_prediction),
        "shot_signal_xyxy": None if shot_prediction is None else prediction_xyxy(shot_prediction),
        "shot_signal_center_xy": None if shot_prediction is None else prediction_center_xy(shot_prediction),
    }


def shot_type_for_frame(signals: dict[str, object]) -> str | None:
    if signals.get("shot_type_signal") in {"jump_shot", "layup_dunk"}:
        return str(signals["shot_type_signal"])
    if signals["has_layup_dunk"]:
        return "layup_dunk"
    if signals["has_jump_shot"]:
        return "jump_shot"
    return None


def _is_same_burst(
    previous: tuple[int, float | None, str, float | None], current: tuple[int, float | None, str, float | None]
) -> bool:
    gap_seconds = elapsed_seconds(previous[1], current[1])
    if gap_seconds is None:
        return current[0] - previous[0] < SHOT_TYPE_BURST_GAP_FRAMES_FALLBACK
    return gap_seconds < SHOT_TYPE_BURST_GAP_SECONDS


def vote_shot_type(candidate: ShotEventCandidate) -> str:
    """Confidence-weighted vote over the last burst of shot boxes; ties go to jump_shot."""
    frames = sorted(candidate.shot_frames, key=lambda entry: entry[0])
    if not frames:
        return candidate.shot_type
    burst = [frames[-1]]
    for entry in reversed(frames[:-1]):
        if not _is_same_burst(entry, burst[0]):
            break
        burst.insert(0, entry)
    # A box without a confidence still counts, with the smallest weight that breaks no tie.
    weights = {"jump_shot": 0.0, "layup_dunk": 0.0}
    for _, _, shot_type, confidence in burst:
        weights[shot_type] += confidence if confidence is not None else 1e-6
    return "layup_dunk" if weights["layup_dunk"] > weights["jump_shot"] else "jump_shot"


def _record_shot_frame(
    candidate: ShotEventCandidate,
    frame_index: int,
    timestamp_seconds: float | None,
    shot_type: str,
    signals: dict[str, object],
) -> None:
    confidence = signals.get("shot_signal_confidence")
    candidate.shot_frames.append(
        (frame_index, timestamp_seconds, shot_type, None if confidence is None else float(confidence))
    )


def finalize_shot_event(
    candidate: ShotEventCandidate,
    *,
    outcome: str,
    resolved_frame_index: int,
    resolved_timestamp_seconds: float | None,
    resolution_reason: str,
) -> dict:
    end_frame_index = (
        candidate.ball_in_basket_frame_index
        if candidate.ball_in_basket_frame_index is not None
        else candidate.latest_shot_frame_index
    )
    end_timestamp_seconds = (
        candidate.ball_in_basket_timestamp_seconds
        if candidate.ball_in_basket_timestamp_seconds is not None
        else candidate.latest_shot_timestamp_seconds
    )
    start_frame_index = candidate.start_frame_index
    start_timestamp_seconds = candidate.start_timestamp_seconds
    start_image_xy = candidate.start_image_xy
    start_shot_xyxy = candidate.start_shot_xyxy
    if outcome == "made":
        start_frame_index = candidate.latest_shot_frame_index
        start_timestamp_seconds = candidate.latest_shot_timestamp_seconds
        start_image_xy = candidate.latest_shot_image_xy
        start_shot_xyxy = candidate.latest_shot_xyxy
    return {
        "event_id": candidate.event_id,
        "shot_type": vote_shot_type(candidate),
        "outcome": outcome,
        "start_frame_index": start_frame_index,
        "end_frame_index": end_frame_index,
        "resolved_frame_index": resolved_frame_index,
        "start_timestamp_seconds": start_timestamp_seconds,
        "end_timestamp_seconds": end_timestamp_seconds,
        "resolved_timestamp_seconds": resolved_timestamp_seconds,
        "ball_in_basket_frame_index": candidate.ball_in_basket_frame_index,
        "ball_in_basket_timestamp_seconds": candidate.ball_in_basket_timestamp_seconds,
        "start_image_xy": None if start_image_xy is None else [float(start_image_xy[0]), float(start_image_xy[1])],
        "start_shot_xyxy": None if start_shot_xyxy is None else [float(value) for value in start_shot_xyxy],
        "resolution_reason": resolution_reason,
    }


def drop_single_frame_miss(event: dict) -> dict | None:
    # A single shot box is detector noise (e.g. a pass read as a shot); a real attempt spans several frames.
    if event["end_frame_index"] == event["start_frame_index"]:
        return None
    return event


def elapsed_seconds(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start


def should_suppress_shot_burst(candidate: ShotEventCandidate | None, signals: dict[str, object]) -> bool:
    confidence = signals.get("shot_signal_confidence")
    return (
        candidate is None
        and shot_type_for_frame(signals) is not None
        and confidence is not None
        and float(confidence) < LOW_CONFIDENCE_BURST_THRESHOLD
    )


def update_shot_event_candidate(
    candidate: ShotEventCandidate | ShotEventCooldown | SuppressedShotBurst | None,
    *,
    next_event_id: int,
    frame_index: int,
    timestamp_seconds: float | None,
    signals: dict[str, object],
) -> tuple[ShotEventCandidate | ShotEventCooldown | None, dict | None, int]:
    shot_type = shot_type_for_frame(signals)
    has_shot_signal = shot_type is not None

    if isinstance(candidate, ShotEventCooldown):
        if elapsed_seconds(candidate.until_timestamp_seconds, timestamp_seconds) is None:
            return None, None, next_event_id
        candidate.shot_signal_active = has_shot_signal
        if timestamp_seconds < candidate.until_timestamp_seconds:
            return candidate, None, next_event_id
        if candidate.shot_signal_active:
            return candidate, None, next_event_id
        candidate = None

    if isinstance(candidate, SuppressedShotBurst):
        if not has_shot_signal:
            candidate.shot_signal_active = False
            candidate.confident_run = []
            return candidate, None, next_event_id
        shot_gap_seconds = elapsed_seconds(candidate.latest_shot_timestamp_seconds, timestamp_seconds)
        if candidate.shot_signal_active or (shot_gap_seconds is not None and shot_gap_seconds < SAME_EVENT_WINDOW_SECONDS):
            confidence = signals.get("shot_signal_confidence")
            if confidence is not None and float(confidence) < LOW_CONFIDENCE_BURST_THRESHOLD:
                candidate.confident_run = []
            else:
                candidate.confident_run.append(
                    (
                        frame_index,
                        timestamp_seconds,
                        shot_type,
                        None if confidence is None else float(confidence),
                        signals.get("shot_signal_center_xy"),
                        signals.get("shot_signal_xyxy"),
                    )
                )
            if len(candidate.confident_run) < SUPPRESSED_BURST_RELEASE_CONFIDENT_FRAMES:
                candidate.shot_signal_active = True
                candidate.latest_shot_frame_index = frame_index
                candidate.latest_shot_timestamp_seconds = timestamp_seconds
                return candidate, None, next_event_id
            # Release: open the candidate as if it had started at the run's first frame; the
            # current frame is then handled below like any later frame of a live candidate.
            first, *rest = candidate.confident_run
            candidate = ShotEventCandidate(
                event_id=next_event_id,
                shot_type=first[2],
                start_frame_index=first[0],
                start_timestamp_seconds=first[1],
                start_image_xy=first[4],
                start_shot_xyxy=first[5],
                latest_shot_frame_index=first[0],
                latest_shot_timestamp_seconds=first[1],
                latest_shot_image_xy=first[4],
                latest_shot_xyxy=first[5],
                shot_frames=[(entry[0], entry[1], entry[2], entry[3]) for entry in [first, *rest[:-1]]],
            )
            next_event_id += 1
        else:
            candidate = None

    if should_suppress_shot_burst(candidate, signals):
        return (
            SuppressedShotBurst(
                latest_shot_frame_index=frame_index,
                latest_shot_timestamp_seconds=timestamp_seconds,
            ),
            None,
            next_event_id,
        )

    if candidate is None and shot_type is not None:
        candidate = ShotEventCandidate(
            event_id=next_event_id,
            shot_type=shot_type,
            start_frame_index=frame_index,
            start_timestamp_seconds=timestamp_seconds,
            start_image_xy=signals.get("shot_signal_center_xy"),
            start_shot_xyxy=signals.get("shot_signal_xyxy"),
            latest_shot_frame_index=frame_index,
            latest_shot_timestamp_seconds=timestamp_seconds,
            latest_shot_image_xy=signals.get("shot_signal_center_xy"),
            latest_shot_xyxy=signals.get("shot_signal_xyxy"),
        )
        next_event_id += 1

    if candidate is None:
        return None, None, next_event_id

    is_rising_edge = has_shot_signal and not candidate.shot_signal_active
    previous_latest_frame_index = candidate.latest_shot_frame_index
    previous_latest_timestamp_seconds = candidate.latest_shot_timestamp_seconds

    if not has_shot_signal:
        candidate.shot_signal_active = False

    if signals["has_ball_in_basket"]:
        if has_shot_signal:
            candidate.latest_shot_frame_index = frame_index
            candidate.latest_shot_timestamp_seconds = timestamp_seconds
            candidate.latest_shot_image_xy = signals.get("shot_signal_center_xy")
            candidate.latest_shot_xyxy = signals.get("shot_signal_xyxy")
            _record_shot_frame(candidate, frame_index, timestamp_seconds, shot_type, signals)
        candidate.ball_in_basket_frame_index = frame_index
        candidate.ball_in_basket_timestamp_seconds = timestamp_seconds
        cooldown_until = (
            None if timestamp_seconds is None else timestamp_seconds + POST_MADE_SUPPRESSION_SECONDS
        )
        return (
            ShotEventCooldown(until_timestamp_seconds=cooldown_until, shot_signal_active=has_shot_signal),
            finalize_shot_event(
                candidate,
                outcome="made",
                resolved_frame_index=frame_index,
                resolved_timestamp_seconds=timestamp_seconds,
                resolution_reason="ball_in_basket",
            ),
            next_event_id,
        )

    if is_rising_edge:
        shot_gap_seconds = elapsed_seconds(previous_latest_timestamp_seconds, timestamp_seconds)
        if (
            shot_gap_seconds is not None
            and previous_latest_frame_index == candidate.start_frame_index
            and OPENING_CONFIRMATION_WINDOW_SECONDS < shot_gap_seconds < SAME_EVENT_WINDOW_SECONDS
        ):
            candidate.start_frame_index = frame_index
            candidate.start_timestamp_seconds = timestamp_seconds
            candidate.start_image_xy = signals.get("shot_signal_center_xy")
            candidate.start_shot_xyxy = signals.get("shot_signal_xyxy")
            candidate.latest_shot_frame_index = frame_index
            candidate.latest_shot_timestamp_seconds = timestamp_seconds
            candidate.latest_shot_image_xy = signals.get("shot_signal_center_xy")
            candidate.latest_shot_xyxy = signals.get("shot_signal_xyxy")
            _record_shot_frame(candidate, frame_index, timestamp_seconds, shot_type, signals)
            candidate.shot_signal_active = True
            return candidate, None, next_event_id
        if shot_gap_seconds is not None and shot_gap_seconds >= SAME_EVENT_WINDOW_SECONDS:
            resolved = drop_single_frame_miss(
                finalize_shot_event(
                    candidate,
                    outcome="missed",
                    resolved_frame_index=frame_index,
                    resolved_timestamp_seconds=timestamp_seconds,
                    resolution_reason="new_shot_started",
                )
            )
            replacement = ShotEventCandidate(
                event_id=next_event_id,
                shot_type=shot_type,
                start_frame_index=frame_index,
                start_timestamp_seconds=timestamp_seconds,
                start_image_xy=signals.get("shot_signal_center_xy"),
                start_shot_xyxy=signals.get("shot_signal_xyxy"),
                latest_shot_frame_index=frame_index,
                latest_shot_timestamp_seconds=timestamp_seconds,
                latest_shot_image_xy=signals.get("shot_signal_center_xy"),
                latest_shot_xyxy=signals.get("shot_signal_xyxy"),
            )
            _record_shot_frame(replacement, frame_index, timestamp_seconds, shot_type, signals)
            return replacement, resolved, next_event_id + 1

    if shot_type is not None:
        candidate.shot_signal_active = True
        candidate.latest_shot_frame_index = frame_index
        candidate.latest_shot_timestamp_seconds = timestamp_seconds
        candidate.latest_shot_image_xy = signals.get("shot_signal_center_xy")
        candidate.latest_shot_xyxy = signals.get("shot_signal_xyxy")
        _record_shot_frame(candidate, frame_index, timestamp_seconds, shot_type, signals)
        return candidate, None, next_event_id

    inactive_gap_seconds = elapsed_seconds(candidate.latest_shot_timestamp_seconds, timestamp_seconds)
    if inactive_gap_seconds is not None and inactive_gap_seconds >= MISSED_SHOT_TIMEOUT_SECONDS:
        return (
            None,
            drop_single_frame_miss(
                finalize_shot_event(
                    candidate,
                    outcome="missed",
                    resolved_frame_index=frame_index,
                    resolved_timestamp_seconds=timestamp_seconds,
                    resolution_reason="shot_timeout",
                )
            ),
            next_event_id,
        )

    return candidate, None, next_event_id


def flush_open_shot_event(
    candidate: ShotEventCandidate | ShotEventCooldown | SuppressedShotBurst | None,
    *,
    frame_index: int,
    timestamp_seconds: float | None,
) -> dict | None:
    if candidate is None or isinstance(candidate, (ShotEventCooldown, SuppressedShotBurst)):
        return None
    return finalize_shot_event(
        candidate,
        outcome="unknown",
        resolved_frame_index=frame_index,
        resolved_timestamp_seconds=timestamp_seconds,
        resolution_reason="end_of_clip",
    )


def write_shot_event_record(handle, event: dict) -> None:
    handle.write(json.dumps(event) + "\n")


def write_shot_signal_record(handle, record: dict) -> None:
    handle.write(json.dumps(record) + "\n")


def write_shot_raw_prediction_record(handle, record: dict) -> None:
    handle.write(json.dumps(record) + "\n")
