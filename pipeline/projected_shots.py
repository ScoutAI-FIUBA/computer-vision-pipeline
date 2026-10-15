"""Join shot events with projected player positions into machine and image artifacts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from pipeline.court_projection import project_points


COURT_LENGTH_M = 28.0
COURT_WIDTH_M = 15.0


def draw_fiba_court(*, pixels_per_meter: int = 40):
    import cv2
    import numpy as np

    width_px = int(round(COURT_LENGTH_M * pixels_per_meter))
    height_px = int(round(COURT_WIDTH_M * pixels_per_meter))
    image = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    color_court = (40, 40, 40)
    color_paint = (60, 60, 60)
    line_thickness = 2

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        return int(round(x_m * pixels_per_meter)), int(round(y_m * pixels_per_meter))

    def arc_radius_m(center: tuple[float, float], edge: tuple[float, float]) -> float:
        return math.hypot(edge[0] - center[0], edge[1] - center[1])

    cv2.rectangle(image, to_px(0.0, 0.0), to_px(COURT_LENGTH_M, COURT_WIDTH_M), color_court, line_thickness)
    cv2.line(image, to_px(COURT_LENGTH_M / 2.0, 0.0), to_px(COURT_LENGTH_M / 2.0, COURT_WIDTH_M), color_court, line_thickness)
    cv2.circle(image, to_px(COURT_LENGTH_M / 2.0, COURT_WIDTH_M / 2.0), int(round(1.8 * pixels_per_meter)), color_court, line_thickness)

    left_hoop = (1.575, COURT_WIDTH_M / 2.0)
    right_hoop = (COURT_LENGTH_M - 1.575, COURT_WIDTH_M / 2.0)
    lane_half_width = 2.45
    lane_depth = 5.8

    for hoop in (left_hoop, right_hoop):
        cv2.circle(image, to_px(*hoop), int(round(0.225 * pixels_per_meter)), color_court, line_thickness)

    cv2.rectangle(image, to_px(0.0, (COURT_WIDTH_M / 2.0) - lane_half_width), to_px(lane_depth, (COURT_WIDTH_M / 2.0) + lane_half_width), color_paint, line_thickness)
    cv2.rectangle(image, to_px(COURT_LENGTH_M - lane_depth, (COURT_WIDTH_M / 2.0) - lane_half_width), to_px(COURT_LENGTH_M, (COURT_WIDTH_M / 2.0) + lane_half_width), color_paint, line_thickness)

    left_free_throw = (5.8, COURT_WIDTH_M / 2.0)
    right_free_throw = (COURT_LENGTH_M - 5.8, COURT_WIDTH_M / 2.0)
    free_throw_radius_px = int(round(1.8 * pixels_per_meter))
    cv2.circle(image, to_px(*left_free_throw), free_throw_radius_px, color_court, line_thickness)
    cv2.circle(image, to_px(*right_free_throw), free_throw_radius_px, color_court, line_thickness)

    three_radius_px = int(round(6.75 * pixels_per_meter))
    for hoop, start, end in ((left_hoop, -68, 68), (right_hoop, 112, 248)):
        cv2.ellipse(image, to_px(*hoop), (three_radius_px, three_radius_px), 0, start, end, color_court, line_thickness)

    corner_offset = 0.9
    corner_distance = 6.6
    cv2.line(image, to_px(corner_offset, 0.0), to_px(corner_offset, corner_distance), color_court, line_thickness)
    cv2.line(image, to_px(corner_offset, COURT_WIDTH_M), to_px(corner_offset, COURT_WIDTH_M - corner_distance), color_court, line_thickness)
    cv2.line(image, to_px(COURT_LENGTH_M - corner_offset, 0.0), to_px(COURT_LENGTH_M - corner_offset, corner_distance), color_court, line_thickness)
    cv2.line(image, to_px(COURT_LENGTH_M - corner_offset, COURT_WIDTH_M), to_px(COURT_LENGTH_M - corner_offset, COURT_WIDTH_M - corner_distance), color_court, line_thickness)

    return image


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_confirmed_numbers_by_track_id(path: Path | None) -> dict[int, str]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    confirmed: dict[int, str] = {}
    for track in payload.get("tracks", []):
        if track.get("status") != "confirmed":
            continue
        if track.get("track_id") is None or track.get("number") in (None, "", "unknown"):
            continue
        confirmed[int(track["track_id"])] = str(track["number"])
    return confirmed


def _load_team_ids_by_track_id(path: Path | None) -> dict[int, int]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = payload.get("assignments_by_track_id", {})
    return {int(track_id): int(team_id) for track_id, team_id in assignments.items()}


def _group_projected_tracks_by_frame(projected_tracks: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in projected_tracks:
        if row.get("class_name", "player") != "player":
            continue
        frame_index = row.get("frame_index")
        if frame_index is None:
            continue
        grouped.setdefault(int(frame_index), []).append(row)
    return grouped


def _point_in_xyxy(point_xy: tuple[float, float], xyxy: list[float] | tuple[float, float, float, float] | None) -> bool:
    if xyxy is None or len(xyxy) != 4:
        return False
    x, y = point_xy
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return x1 <= x <= x2 and y1 <= y <= y2


def _distance(point_a: tuple[float, float], point_b: tuple[float, float] | list[float] | None) -> float:
    if point_b is None or len(point_b) < 2:
        return float("inf")
    dx = float(point_a[0]) - float(point_b[0])
    dy = float(point_a[1]) - float(point_b[1])
    return (dx * dx + dy * dy) ** 0.5


def _bbox_center_xy(xyxy: list[float] | tuple[float, float, float, float] | None) -> tuple[float, float] | None:
    if xyxy is None or len(xyxy) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _shot_anchor_image_xy(event: dict) -> tuple[float, float] | None:
    start_image_xy = event.get("start_image_xy")
    if isinstance(start_image_xy, list) and len(start_image_xy) >= 2:
        return float(start_image_xy[0]), float(start_image_xy[1])
    return _bbox_center_xy(event.get("start_shot_xyxy"))


def _project_shot_anchor_court_xy(projected_rows: list[dict], shot_image_xy: tuple[float, float] | None) -> tuple[list[float] | None, str | None]:
    if shot_image_xy is None:
        return None, "missing_shot_anchor"

    image_points: list[tuple[float, float]] = []
    court_points: list[tuple[float, float]] = []
    for row in projected_rows:
        image_foot_xy = row.get("image_foot_xy")
        court_xy_m = row.get("court_xy_m")
        if image_foot_xy is None or len(image_foot_xy) < 2 or court_xy_m is None or len(court_xy_m) < 2:
            continue
        image_points.append((float(image_foot_xy[0]), float(image_foot_xy[1])))
        court_points.append((float(court_xy_m[0]), float(court_xy_m[1])))

    if len({point for point in image_points}) < 4 or len({point for point in court_points}) < 4:
        return None, "projected_shot_anchor_requires_4_projected_players"

    import cv2
    import numpy as np

    homography, _ = cv2.findHomography(
        np.asarray(image_points, dtype=np.float32),
        np.asarray(court_points, dtype=np.float32),
        0,
    )
    if homography is None:
        return None, "projected_shot_anchor_homography_failed"

    projected_point = project_points(homography, [shot_image_xy])[0]
    return [float(projected_point[0]), float(projected_point[1])], "shot_anchor_homography_from_projected_players"


def _nearest_projected_frame(frame_indexes: list[int], target_frame_index: int, max_frame_gap: int) -> int | None:
    best_frame_index = None
    best_gap = None
    for frame_index in frame_indexes:
        gap = abs(int(frame_index) - int(target_frame_index))
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_frame_index = int(frame_index)
    return best_frame_index


def _snapshot_row(row: dict, *, team_ids_by_track_id: dict[int, int], confirmed_numbers_by_track_id: dict[int, str]) -> dict:
    track_id = None if row.get("track_id") is None else int(row["track_id"])
    return {
        "track_id": track_id,
        "stable_player_id": row.get("stable_player_id"),
        "team_id": None if track_id is None else team_ids_by_track_id.get(track_id),
        "confirmed_number": None if track_id is None else confirmed_numbers_by_track_id.get(track_id),
        "track_state": row.get("track_state"),
        "image_foot_xy": [float(value) for value in row.get("image_foot_xy", [])[:2]],
        "court_xy_m": [float(value) for value in row.get("court_xy_m", [])[:2]],
        "xyxy": [float(value) for value in row.get("xyxy", [])[:4]],
    }


def _select_shooter_row(projected_rows: list[dict], shot_image_xy: tuple[float, float] | None) -> tuple[dict | None, str]:
    if shot_image_xy is None:
        return None, "missing_shot_anchor"
    if not projected_rows:
        return None, "no_projected_players"

    eligible_rows = [
        row
        for row in projected_rows
        if isinstance(row.get("stable_player_id"), str) and row["stable_player_id"].strip()
    ]
    if not eligible_rows:
        return None, "no_stable_player_id"

    best_row = None
    best_score = None
    for row in eligible_rows:
        bbox_center = _bbox_center_xy(row.get("xyxy"))
        score = (
            0 if _point_in_xyxy(shot_image_xy, row.get("xyxy")) else 1,
            _distance(shot_image_xy, bbox_center),
            _distance(shot_image_xy, row.get("image_foot_xy")),
            int(row.get("track_id", 10**9)) if row.get("track_id") is not None else 10**9,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_row = row
    return best_row, "nearest_player_bbox_to_shot_anchor"


ACTION_WINDOW_SECONDS = 2.0
ACTION_MIN_BOX_IOU = 0.3
ACTION_MIN_BALL_CONFIDENCE = 0.4
ACTION_BALL_X_MARGIN_RATIO = 0.1
ACTION_BALL_ABOVE_BOX_RATIO = 0.25
ACTION_BALL_BELOW_TOP_RATIO = 0.6
ACTION_MIN_VOTE_SCORE = 1.0
ACTION_MIN_WINNER_RATIO = 2.0
ACTION_COURT_MARGIN_M = 0.5
# The detector labels a player with the ball and the shooting pose with these classes.
ACTION_CLASS_IDS = {4, 5, 6, 7}
ACTION_CLASS_NAMES = {"player_in_possession", "player_jump_shot", "player_layup_dunk", "player_shot_block"}
BALL_CLASS_ID = 0
# Shooting-pose frames further apart than this belong to different attempts (e.g. an earlier miss before a putback).
ACTION_RELEASE_MAX_GAP_SECONDS = 0.25


def _on_court(row: dict) -> bool:
    court_xy = row.get("court_xy_m")
    if not isinstance(court_xy, (list, tuple)) or len(court_xy) < 2:
        return False
    margin = ACTION_COURT_MARGIN_M
    return (
        -margin <= float(court_xy[0]) <= COURT_LENGTH_M + margin
        and -margin <= float(court_xy[1]) <= COURT_WIDTH_M + margin
    )


def _iou(box_a: tuple[float, float, float, float], box_b: list[float] | tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _raw_prediction_fps(raw_rows_by_frame: dict[int, dict]) -> float | None:
    timed = [
        (frame, row["timestamp_seconds"])
        for frame, row in sorted(raw_rows_by_frame.items())
        if row.get("timestamp_seconds") is not None
    ]
    if len(timed) < 2 or timed[-1][1] <= timed[0][1]:
        return None
    return (timed[-1][0] - timed[0][0]) / (timed[-1][1] - timed[0][1])


@dataclass(frozen=True)
class ActionVote:
    row: dict
    best_frame: int
    release_frame: int
    release_row: dict
    release_xyxy: list[float]
    release_timestamp_seconds: float | None


def _made_anchor_frame_index(event: dict) -> int | None:
    # A trajectory-made event has no basket detection; the rim pass is the frame the made shot is anchored to.
    for key in ("ball_in_basket_frame_index", "rim_pass_frame_index"):
        if event.get(key) is not None:
            return event[key]
    return None


def _action_box_vote(
    event: dict,
    raw_rows_by_frame: dict[int, dict],
    projected_by_frame: dict[int, list[dict]],
) -> ActionVote | None:
    """Vote for the shooter among the detector's shooting-pose boxes before a made basket.

    A box votes for the projected player it overlaps most, but only when a ball detection sits at
    that box's hands or overhead in the same frame. That ball check is what separates the shooter
    from a teammate in the same pose with the ball far above his head. The winner needs a clear
    margin; otherwise the shooter is left unknown instead of guessed.

    The release is the first frame of the winner's last run of qualifying frames, so a putback is
    not anchored to an earlier miss.
    """
    from pipeline.shot_events import (
        prediction_confidence,
        prediction_has_class_id,
        normalize_prediction_class_name,
        prediction_xyxy,
    )

    basket_frame = _made_anchor_frame_index(event)
    fps = _raw_prediction_fps(raw_rows_by_frame)
    if basket_frame is None or fps is None:
        return None
    basket_frame = int(basket_frame)
    window_start = basket_frame - round(ACTION_WINDOW_SECONDS * fps)

    scores: dict[str, float] = {}
    best_box: dict[str, tuple[float, int, dict]] = {}
    boxes_by_frame: dict[str, dict[int, tuple[float, tuple[float, float, float, float], dict]]] = {}
    for frame in range(window_start, basket_frame + 1):
        predictions = (raw_rows_by_frame.get(frame) or {}).get("predictions") or []
        balls = []
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            if prediction_has_class_id(prediction, BALL_CLASS_ID) or normalize_prediction_class_name(
                prediction.get("class")
            ) == "ball":
                ball_xyxy = prediction_xyxy(prediction)
                if ball_xyxy is not None and (prediction_confidence(prediction) or 0.0) >= ACTION_MIN_BALL_CONFIDENCE:
                    balls.append(((ball_xyxy[0] + ball_xyxy[2]) / 2.0, (ball_xyxy[1] + ball_xyxy[3]) / 2.0))
        if not balls:
            continue
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            is_action = any(prediction_has_class_id(prediction, class_id) for class_id in ACTION_CLASS_IDS) or (
                normalize_prediction_class_name(prediction.get("class")) in ACTION_CLASS_NAMES
            )
            box = prediction_xyxy(prediction)
            if not is_action or box is None:
                continue
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            if not any(
                x1 - ACTION_BALL_X_MARGIN_RATIO * width <= ball_x <= x2 + ACTION_BALL_X_MARGIN_RATIO * width
                and y1 - ACTION_BALL_ABOVE_BOX_RATIO * height <= ball_y <= y1 + ACTION_BALL_BELOW_TOP_RATIO * height
                for ball_x, ball_y in balls
            ):
                continue
            candidates = [
                row
                for row in projected_by_frame.get(frame, [])
                if isinstance(row.get("stable_player_id"), str)
                and row["stable_player_id"].strip()
                and _on_court(row)
                and _iou(box, row.get("xyxy") or [0, 0, 0, 0]) >= ACTION_MIN_BOX_IOU
            ]
            if not candidates:
                continue
            row = max(candidates, key=lambda candidate: _iou(box, candidate["xyxy"]))
            confidence = prediction_confidence(prediction) or 0.0
            stable_id = row["stable_player_id"]
            scores[stable_id] = scores.get(stable_id, 0.0) + confidence
            if stable_id not in best_box or confidence > best_box[stable_id][0]:
                best_box[stable_id] = (confidence, frame, row)
            frame_boxes = boxes_by_frame.setdefault(stable_id, {})
            if frame not in frame_boxes or confidence > frame_boxes[frame][0]:
                frame_boxes[frame] = (confidence, box, row)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < ACTION_MIN_VOTE_SCORE:
        return None
    if len(ranked) > 1 and ranked[0][1] < ACTION_MIN_WINNER_RATIO * ranked[1][1]:
        return None
    winner = ranked[0][0]
    _, best_frame, row = best_box[winner]
    max_gap = max(1, round(ACTION_RELEASE_MAX_GAP_SECONDS * fps))
    frames = sorted(boxes_by_frame[winner])
    release_frame = frames[0]
    for previous, current in zip(frames, frames[1:]):
        if current - previous > max_gap:
            release_frame = current
    _, release_box, release_row = boxes_by_frame[winner][release_frame]
    return ActionVote(
        row=row,
        best_frame=best_frame,
        release_frame=release_frame,
        release_row=release_row,
        release_xyxy=[float(value) for value in release_box],
        release_timestamp_seconds=(raw_rows_by_frame.get(release_frame) or {}).get("timestamp_seconds"),
    )


def build_projected_shots(
    shot_events: list[dict],
    projected_tracks: list[dict],
    *,
    max_frame_gap: int = 6,
    team_ids_by_track_id: dict[int, int] | None = None,
    confirmed_numbers_by_track_id: dict[int, str] | None = None,
    raw_prediction_rows: list[dict] | None = None,
) -> dict:
    """Build shot projections; track-keyed team/number fields are compatibility enrichment.

    ``stable_player_id`` is the canonical player identity. Raw track IDs remain in the
    artifact only as frame-local provenance because the existing team/number APIs are
    track-keyed and do not provide stable-player keyed lookups yet.
    """
    team_ids_by_track_id = team_ids_by_track_id or {}
    confirmed_numbers_by_track_id = confirmed_numbers_by_track_id or {}
    projected_by_frame = _group_projected_tracks_by_frame(projected_tracks)
    available_frame_indexes = sorted(projected_by_frame)
    raw_rows_by_frame = (
        None
        if raw_prediction_rows is None
        else {int(row["frame_index"]): row for row in raw_prediction_rows if row.get("frame_index") is not None}
    )

    shots: list[dict] = []
    for event in shot_events:
        target_frame_index = int(event["start_frame_index"])
        matched_frame_index = _nearest_projected_frame(available_frame_indexes, target_frame_index, max_frame_gap)
        snapshot_rows = []
        shooter_row = None
        start_fields: dict = {"start_frame_source": "shot_detector"}
        attribution_method = "no_projected_frame_within_gap"
        if matched_frame_index is not None:
            frame_rows = projected_by_frame.get(matched_frame_index, [])
            snapshot_rows = [
                _snapshot_row(
                    row,
                    team_ids_by_track_id=team_ids_by_track_id,
                    confirmed_numbers_by_track_id=confirmed_numbers_by_track_id,
                )
                for row in sorted(frame_rows, key=lambda row: (row.get("track_id") is None, row.get("track_id")))
            ]
            shot_anchor_image_xy = _shot_anchor_image_xy(event)
            shooter_row, attribution_method = _select_shooter_row(
                frame_rows,
                shot_anchor_image_xy,
            )
            if raw_rows_by_frame is not None and _made_anchor_frame_index(event) is not None:
                vote = _action_box_vote(event, raw_rows_by_frame, projected_by_frame)
                if vote is not None:
                    shooter_row = vote.release_row
                    attribution_method = "action_box_vote"
                    snapshot_rows = [
                        _snapshot_row(
                            row,
                            team_ids_by_track_id=team_ids_by_track_id,
                            confirmed_numbers_by_track_id=confirmed_numbers_by_track_id,
                        )
                        for row in sorted(
                            projected_by_frame.get(vote.release_frame, []),
                            key=lambda row: (row.get("track_id") is None, row.get("track_id")),
                        )
                    ]
                    matched_frame_index = vote.release_frame
                    target_frame_index = vote.release_frame
                    center = _bbox_center_xy(vote.release_xyxy)
                    start_fields = {
                        "start_frame_index": vote.release_frame,
                        "start_timestamp_seconds": vote.release_timestamp_seconds,
                        "start_shot_xyxy": vote.release_xyxy,
                        "start_image_xy": None if center is None else [center[0], center[1]],
                        "detector_start_frame_index": event["start_frame_index"],
                        "detector_start_timestamp_seconds": event.get("start_timestamp_seconds"),
                        "start_frame_source": "action_box_release",
                    }
                elif shooter_row is not None:
                    # The anchor rule would have guessed; an unclear vote leaves the shooter unknown instead.
                    shooter_row = None
                    attribution_method = "ambiguous"
            shot_origin_court_xy_m = None
            shot_origin_projection_method = None
            if attribution_method == "ambiguous":
                # The backend drops shots without an origin; keep an approximate one so the make stays on the chart as unattributed.
                shot_box = event.get("start_shot_xyxy")
                foot_anchor = (
                    ((float(shot_box[0]) + float(shot_box[2])) / 2.0, float(shot_box[3]))
                    if isinstance(shot_box, (list, tuple)) and len(shot_box) == 4
                    else shot_anchor_image_xy
                )
                origin_xy, origin_method = _project_shot_anchor_court_xy(frame_rows, foot_anchor)
                if origin_xy is not None:
                    shot_origin_court_xy_m = origin_xy
                    shot_origin_projection_method = origin_method
        else:
            shot_origin_court_xy_m = None
            shot_origin_projection_method = None

        shooter_track_id = None if shooter_row is None or shooter_row.get("track_id") is None else int(shooter_row["track_id"])
        shooter_stable_player_id = None if shooter_row is None else shooter_row.get("stable_player_id")
        if shooter_row is not None:
            shot_origin_court_xy_m = [float(value) for value in shooter_row.get("court_xy_m", [])[:2]]
            shot_origin_projection_method = "shooter_foot"
        shots.append(
            {
                **event,
                **start_fields,
                "matched_projected_frame_index": matched_frame_index,
                "matched_projected_frame_gap": None
                if matched_frame_index is None
                else abs(int(matched_frame_index) - target_frame_index),
                "shot_origin_court_xy_m": shot_origin_court_xy_m,
                "shot_origin_projection_method": shot_origin_projection_method,
                "shooter_track_id": shooter_track_id,
                "shooter_stable_player_id": shooter_stable_player_id,
                "shooter_team_id": None if shooter_track_id is None else team_ids_by_track_id.get(shooter_track_id),
                "shooter_confirmed_number": None
                if shooter_track_id is None
                else confirmed_numbers_by_track_id.get(shooter_track_id),
                "shooter_attribution_method": attribution_method,
                "projected_player_snapshot": snapshot_rows,
            }
        )

    return {
        "schema_version": 1,
        "heuristics": {
            "event_anchor": "made shots with a clear shooting-pose vote: first frame of the winner's last shooting-pose run; otherwise start_frame_index/start_image_xy from shot_events.jsonl",
            "frame_matching": "nearest projected frame by minimum temporal distance",
            "shot_origin_projection": "use the attributed shooter's projected foot/court point; ambiguous made shots project the detector shot box foot point through a homography from projected players when at least 4 are available",
            "shooter_selection": "made shots with raw detector predictions: vote among shooting-pose boxes with the ball at their hands in the 2s before the basket, ambiguous when unclear; otherwise prefer containing bbox, then nearest bbox center and foot point",
            "team_number_enrichment": "compatibility-only lookup keyed by raw track_id; stable_player_id remains canonical",
        },
        "shots": shots,
    }


def _court_point_to_px(court_xy_m: list[float] | tuple[float, float] | None, pixels_per_meter: int) -> tuple[int, int] | None:
    if court_xy_m is None or len(court_xy_m) < 2:
        return None
    x_m = float(court_xy_m[0])
    y_m = float(court_xy_m[1])
    if not (0.0 <= x_m <= COURT_LENGTH_M and 0.0 <= y_m <= COURT_WIDTH_M):
        return None
    return int(round(x_m * pixels_per_meter)), int(round(y_m * pixels_per_meter))


def write_projected_shots_image(
    artifact: dict,
    output_path: Path,
    *,
    pixels_per_meter: int = 40,
    point_radius: int = 7,
) -> int:
    import cv2

    court = draw_fiba_court(pixels_per_meter=pixels_per_meter)
    plotted = 0
    for shot in artifact.get("shots", []):
        point = _court_point_to_px(shot.get("shot_origin_court_xy_m"), pixels_per_meter)
        if point is None:
            continue
        outcome = str(shot.get("outcome") or "unknown").lower()
        color = (0, 180, 0) if outcome == "made" else (0, 0, 255) if outcome == "missed" else (128, 128, 128)
        cv2.circle(court, point, point_radius, color, -1)
        cv2.circle(court, point, point_radius + 2, (255, 255, 255), 1)
        label = f"{shot.get('event_id', '?')}:{outcome[:1].upper()}"
        cv2.putText(court, label, (point[0] + point_radius + 3, point[1] - point_radius - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        plotted += 1

    cv2.putText(court, "Projected shot chart", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(court, "Made = green  Missed = red  Unknown = gray", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(court, f"Plotted shots: {plotted}/{len(artifact.get('shots', []))}", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), court):
        raise RuntimeError(f"Could not write projected shots image: {output_path}")
    return plotted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "Export projected shot events.")
    parser.add_argument("shot_events_jsonl", type=Path, help="Path to shot_events.jsonl")
    parser.add_argument("projected_tracks_jsonl", type=Path, help="Path to projected_tracks.jsonl")
    parser.add_argument("--output", type=Path, help="Path for projected_shots.json")
    parser.add_argument("--output-image", type=Path, help="Optional path for a projected_shots PNG")
    parser.add_argument("--teams", type=Path, help="Optional teams.json for team enrichment")
    parser.add_argument("--numbers", type=Path, help="Optional numbers.json for jersey-number enrichment")
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=6,
        help="Compatibility flag only; shots now always use the nearest projected frame. Default: 6",
    )
    args = parser.parse_args(argv)

    if args.max_frame_gap < 0:
        raise SystemExit("--max-frame-gap must be at least 0")

    output_path = args.output or args.shot_events_jsonl.with_name("projected_shots.json")
    artifact = build_projected_shots(
        _load_jsonl(args.shot_events_jsonl),
        _load_jsonl(args.projected_tracks_jsonl),
        max_frame_gap=args.max_frame_gap,
        team_ids_by_track_id=_load_team_ids_by_track_id(args.teams),
        confirmed_numbers_by_track_id=_load_confirmed_numbers_by_track_id(args.numbers),
    )
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    if args.output_image is not None:
        write_projected_shots_image(artifact, args.output_image)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
