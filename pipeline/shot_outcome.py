"""Heuristic outcome refinement: upgrade missed/unknown shots whose ball passes through the rim.

A made shot normally needs a ball-in-basket detection. When the detector misses it, the ball
trajectory around the detected rim can still show the ball coming from above, crossing the rim
zone and leaving below it. Only that pattern upgrades an event; anything ambiguous keeps the
existing outcome (prefer unknown over made).

A "made" event resolved only by a ball-in-basket detection is downgraded to unknown (not missed:
no evidence is not proof of a miss) when no ball was seen near the rim around that detection.

Event fields added here (only when the refinement runs):
    outcome_source: "ball_in_basket" | "rim_trajectory" | "shot_detector" (missed/unknown).
    rim_pass_frame_index / rim_exit_frame_index: the first in-zone ball point and the exit point,
        None unless the outcome was upgraded. ball_in_basket_frame_index is never faked.
    ball_near_rim_evidence: only on ball_in_basket events whose rim is known around the detection.
"""

from __future__ import annotations

import bisect
import json
import math
import os
from pathlib import Path

from pipeline.shot_events import (
    BALL_CLASS_NAMES,
    BALL_CLASS_ID,
    comparable_prediction_confidence,
    prediction_center_xy,
    prediction_matches,
    prediction_xyxy,
)

RIM_CLASS_ID = 9
RIM_CLASS_NAMES = {"rim"}
RIM_FALLBACK_FRAMES = 30
BALL_BOX_MIN_CONFIDENCE = 0.6
# The window starts at the event's first shot frame and ends this long after its last one.
WINDOW_AFTER_LAST_SHOT_SECONDS = 4.0
DEFAULT_FPS = 30.0
ABOVE_MAX_V = -0.6
ABOVE_MAX_ABS_U = 1.5
ZONE_MAX_ABS_U = 0.5
ZONE_MIN_V = -0.5
ZONE_MAX_V = 0.35
# Fragile: the measured exit of a made three-pointer was v ~0.38, so do not raise it.
BELOW_MIN_V = 0.35
BELOW_MAX_ABS_U = 0.5
# A second ball elsewhere in the frame (a held or dribbled ball) must not count as the shot ball
# leaving the zone sideways; points farther than this from the rim are ignored.
RELEVANT_MAX_ABS_U = 2.0
RELEVANT_MAX_ABS_V = 3.0
ABOVE_TO_ZONE_MAX_FRAMES = 30
ZONE_TO_EXIT_MAX_FRAMES = 45
# The detector's ball-in-basket class also fires on a bare rim/net. A basket detection only stands as
# "made" when a ball was seen this close to the rim (Euclidean, in rim widths) shortly before it.
BASKET_BALL_MAX_RIM_WIDTHS = 2.0
# Frames, not seconds: the clips run at 60 fps and a longer after-window reaches a ball at the net
# that is the effect of the detection, not evidence for it.
BASKET_BALL_WINDOW_BEFORE_FRAMES = 60
BASKET_BALL_WINDOW_AFTER_FRAMES = 10
UPGRADABLE_REASONS = {"shot_timeout", "new_shot_started", "end_of_clip"}


def _rim_boxes_by_frame(raw_rows: list[dict]) -> dict[int, tuple[float, float, float]]:
    """Best (highest-confidence) rim as (center_x, center_y, width) per frame."""
    best: dict[int, tuple[float, tuple[float, float, float]]] = {}
    for row in raw_rows:
        for prediction in row.get("predictions") or []:
            if not isinstance(prediction, dict) or not prediction_matches(
                prediction, class_id=RIM_CLASS_ID, class_names=RIM_CLASS_NAMES
            ):
                continue
            xyxy = prediction_xyxy(prediction)
            center = prediction_center_xy(prediction)
            if xyxy is None or center is None or xyxy[2] - xyxy[0] <= 0:
                continue
            confidence = comparable_prediction_confidence(prediction)
            frame = int(row["frame_index"])
            if frame not in best or confidence > best[frame][0]:
                best[frame] = (confidence, (center[0], center[1], xyxy[2] - xyxy[0]))
    return {frame: value for frame, (_, value) in best.items()}


def _rim_for_frame(rims: dict[int, tuple[float, float, float]], rim_frames: list[int], frame: int):
    if frame in rims:
        return rims[frame]
    position = bisect.bisect_left(rim_frames, frame)
    candidates = [rim_frames[i] for i in (position - 1, position) if 0 <= i < len(rim_frames)]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda candidate: (abs(candidate - frame), candidate))
    return rims[nearest] if abs(nearest - frame) <= RIM_FALLBACK_FRAMES else None


def _ball_points(raw_rows: list[dict], ball_rows: list[dict]) -> list[tuple[int, float, float]]:
    points: list[tuple[int, float, float]] = []
    for row in ball_rows:
        center = row.get("center_xy")
        if row.get("track_state") == "detected" and center is not None:
            points.append((int(row["frame_index"]), float(center[0]), float(center[1])))
    for row in raw_rows:
        for prediction in row.get("predictions") or []:
            if (
                isinstance(prediction, dict)
                and prediction_matches(prediction, class_id=BALL_CLASS_ID, class_names=BALL_CLASS_NAMES)
                and comparable_prediction_confidence(prediction) >= BALL_BOX_MIN_CONFIDENCE
            ):
                center = prediction_center_xy(prediction)
                if center is not None:
                    points.append((int(row["frame_index"]), center[0], center[1]))
    points.sort(key=lambda point: point[0])
    return points


def _frames_per_second(raw_rows: list[dict]) -> float:
    timed = [(row["frame_index"], row["timestamp_seconds"]) for row in raw_rows if row.get("timestamp_seconds") is not None]
    if len(timed) >= 2 and timed[-1][1] > timed[0][1] and timed[-1][0] > timed[0][0]:
        return (timed[-1][0] - timed[0][0]) / (timed[-1][1] - timed[0][1])
    return DEFAULT_FPS


def _in_zone(u: float, v: float) -> bool:
    return abs(u) <= ZONE_MAX_ABS_U and ZONE_MIN_V <= v <= ZONE_MAX_V


def find_rim_pass(
    normalized: list[tuple[int, float, float]], window_start: int, window_end: int
) -> tuple[int, int] | None:
    """(first in-zone frame, exit frame) of a ball coming from above, through the zone, out below."""
    points = [point for point in normalized if window_start <= point[0] <= window_end]
    for above_index, (above_frame, above_u, above_v) in enumerate(points):
        if not (above_v <= ABOVE_MAX_V and abs(above_u) <= ABOVE_MAX_ABS_U):
            continue
        for zone_index in range(above_index + 1, len(points)):
            zone_frame, zone_u, zone_v = points[zone_index]
            if zone_frame - above_frame > ABOVE_TO_ZONE_MAX_FRAMES:
                break
            if zone_frame <= above_frame or not _in_zone(zone_u, zone_v):
                continue
            for exit_frame, exit_u, exit_v in points[zone_index + 1 :]:
                if _in_zone(exit_u, exit_v):
                    continue
                # Only the first point out of the zone counts: any other exit is not a make.
                if (
                    exit_frame - zone_frame <= ZONE_TO_EXIT_MAX_FRAMES
                    and exit_v >= BELOW_MIN_V
                    and abs(exit_u) <= BELOW_MAX_ABS_U
                ):
                    return zone_frame, exit_frame
                break
    return None


def nearest_ball_rim_distance(
    all_normalized: list[tuple[int, float, float]], window_start: int, window_end: int
) -> float | None:
    """Smallest ball-to-rim distance in rim widths among the points inside the frame window."""
    distances = [math.hypot(u, v) for frame, u, v in all_normalized if window_start <= frame <= window_end]
    return min(distances) if distances else None


def refine_shot_events(events: list[dict], raw_rows: list[dict], ball_rows: list[dict]) -> list[dict]:
    rims = _rim_boxes_by_frame(raw_rows)
    rim_frames = sorted(rims)
    points = _ball_points(raw_rows, ball_rows)
    fps = _frames_per_second(raw_rows)
    window_frames = round(WINDOW_AFTER_LAST_SHOT_SECONDS * fps)
    all_normalized: list[tuple[int, float, float]] = []
    for frame, x, y in points:
        rim = _rim_for_frame(rims, rim_frames, frame)
        if rim is not None:
            all_normalized.append((frame, (x - rim[0]) / rim[2], (y - rim[1]) / rim[2]))
    normalized = [
        point for point in all_normalized if abs(point[1]) <= RELEVANT_MAX_ABS_U and abs(point[2]) <= RELEVANT_MAX_ABS_V
    ]

    refined = []
    for event in events:
        event = dict(event)
        event["outcome_source"] = "ball_in_basket" if event.get("resolution_reason") == "ball_in_basket" else "shot_detector"
        event["rim_pass_frame_index"] = None
        event["rim_exit_frame_index"] = None
        if event.get("outcome") in {"missed", "unknown"} and event.get("resolution_reason") in UPGRADABLE_REASONS:
            window_start = int(event["start_frame_index"])
            window_end = int(event["end_frame_index"]) + window_frames
            if event["resolution_reason"] == "new_shot_started":
                # The next shot's ball path must not decide this event.
                window_end = min(window_end, int(event["resolved_frame_index"]) - 1)
            rim_pass = find_rim_pass(normalized, window_start, window_end)
            if rim_pass is not None:
                event.update(
                    outcome="made",
                    resolution_reason="rim_trajectory",
                    outcome_source="rim_trajectory",
                    rim_pass_frame_index=rim_pass[0],
                    rim_exit_frame_index=rim_pass[1],
                )
        if (
            event["outcome_source"] == "ball_in_basket"
            and event.get("outcome") == "made"
            and event.get("ball_in_basket_frame_index") is not None
        ):
            basket_frame = int(event["ball_in_basket_frame_index"])
            window_start, window_end = basket_frame - BASKET_BALL_WINDOW_BEFORE_FRAMES, basket_frame + BASKET_BALL_WINDOW_AFTER_FRAMES
            # Without any rim box there is no evidence either way, so the event is left alone.
            if any(_rim_for_frame(rims, rim_frames, frame) is not None for frame in range(window_start, window_end + 1)):
                distance = nearest_ball_rim_distance(all_normalized, window_start, window_end)
                near = distance is not None and distance <= BASKET_BALL_MAX_RIM_WIDTHS
                event["ball_near_rim_evidence"] = near
                if not near:
                    event.update(outcome="unknown", resolution_reason="ball_in_basket_unconfirmed")
        refined.append(event)
    return refined


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def refine_shot_events_file(events_path: Path, raw_predictions_path: Path, ball_tracks_path: Path) -> None:
    """Rewrite shot_events.jsonl with refined outcomes; the replace keeps readers from seeing a partial file."""
    events = _load_jsonl(events_path)
    refined = refine_shot_events(events, _load_jsonl(raw_predictions_path), _load_jsonl(ball_tracks_path))
    temporary_path = events_path.with_name(events_path.name + ".tmp")
    temporary_path.write_text("".join(json.dumps(event) + "\n" for event in refined), encoding="utf-8")
    os.replace(temporary_path, events_path)
