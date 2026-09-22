"""Conservative possession evidence from existing player and ball rows."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping


POSSESSION_CONTRACT_VERSION = 1
POSSESSION_STATUSES = {"confirmed", "unknown", "lost", "unassigned"}
AUTHORITATIVE_PLAYER_STATES = {"tracked", "detected"}
AUTHORITATIVE_BALL_STATES = {"detected", "fallback"}


def _finite_box(value) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in box) or box[0] >= box[2] or box[1] >= box[3]:
        return None
    return box


def _finite_point(value) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        point = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(item) for item in point) else None


def _timestamp(row: dict) -> float | None:
    value = row.get("timestamp_seconds")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _identity(row: dict) -> tuple[bool, str | None]:
    track_id = row.get("track_id", row.get("raw_track_id"))
    if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 1:
        return False, None
    if "stable_player_id" not in row:
        return True, None
    stable_id = row.get("stable_player_id")
    if not isinstance(stable_id, str) or not stable_id.strip():
        return False, None
    if str(row.get("status", "resolved")).strip().lower() != "resolved":
        return False, None
    return True, stable_id.strip()


def _team_id(row: dict, track_id: int, team_ids_by_track_id: Mapping[int, int]) -> int | None:
    team_id, _conflict = _team_evidence(row, track_id, team_ids_by_track_id)
    return team_id


def _team_evidence(
    row: dict, track_id: int, team_ids_by_track_id: Mapping[int, int]
) -> tuple[int | None, bool]:
    values = []
    for value in (row.get("team_id"), team_ids_by_track_id.get(track_id)):
        if value is None:
            continue
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            return None, True
    if len(set(values)) > 1:
        return None, True
    return (values[0] if values else None), False


def _record(
    *,
    frame_index: int,
    timestamp_seconds: float | None,
    status: str,
    player_track_id: int | None = None,
    stable_player_id: str | None = None,
    team_id: int | None = None,
    confidence: float = 0.0,
    evidence_frame_indices: list[int] | None = None,
    source_records: list[str] | None = None,
    rejection_reason: str | None = None,
) -> dict:
    return {
        "schema_version": POSSESSION_CONTRACT_VERSION,
        "frame_index": int(frame_index),
        "timestamp_seconds": timestamp_seconds,
        "status": status,
        "player_track_id": player_track_id,
        "stable_player_id": stable_player_id,
        "team_id": team_id,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
        "evidence_frame_indices": list(evidence_frame_indices or []),
        "source_records": list(source_records or []),
        "rejection_reason": rejection_reason,
    }


def resolve_possession_events(
    player_rows: Iterable[dict],
    ball_rows: Iterable[dict],
    *,
    fps: float | None = None,
    confirmation_seconds: float = 0.5,
    near_margin_px: float = 8.0,
    frame_step: int = 1,
    team_ids_by_track_id: Mapping[int, int] | None = None,
) -> list[dict]:
    """Emit one conservative status per player-analysis frame.

    A ball row must exist at the exact shared frame. Its valid center must be
    inside one authoritative player box or its fixed pixel expansion. A holder
    is confirmed only after the same valid identity persists for the configured
    elapsed time; gaps reset confirmation rather than being interpolated.
    """
    if confirmation_seconds < 0 or near_margin_px < 0 or type(frame_step) is not int or frame_step < 1:
        raise ValueError("confirmation_seconds and near_margin_px must be non-negative; frame_step must be positive")
    team_ids_by_track_id = team_ids_by_track_id or {}
    players_by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in player_rows:
        if isinstance(row, dict) and isinstance(row.get("frame_index"), int):
            players_by_frame[int(row["frame_index"])].append(row)
    balls_by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in ball_rows:
        if isinstance(row, dict) and isinstance(row.get("frame_index"), int):
            balls_by_frame[int(row["frame_index"])].append(row)

    output: list[dict] = []
    pending: list[dict] = []
    for frame_index in sorted(players_by_frame):
        timestamp = next((_timestamp(row) for row in players_by_frame[frame_index] if _timestamp(row) is not None), None)
        ball_candidates = balls_by_frame.get(frame_index)
        if not ball_candidates:
            output.append(_record(
                frame_index=frame_index, timestamp_seconds=timestamp, status="lost",
                rejection_reason="ball_evidence_missing",
            ))
            pending = []
            continue
        if len(ball_candidates) != 1:
            output.append(_record(
                frame_index=frame_index, timestamp_seconds=timestamp, status="unknown",
                rejection_reason="ambiguous_ball_evidence",
            ))
            pending = []
            continue

        ball = ball_candidates[0]
        if str(ball.get("track_state", "")).strip().lower() == "lost":
            output.append(_record(
                frame_index=frame_index, timestamp_seconds=timestamp, status="lost",
                source_records=[f"ball_tracks.jsonl#frame={frame_index}"],
                rejection_reason="ball_track_lost",
            ))
            pending = []
            continue
        if str(ball.get("track_state", "")).strip().lower() not in AUTHORITATIVE_BALL_STATES:
            output.append(_record(
                frame_index=frame_index, timestamp_seconds=timestamp, status="unknown",
                source_records=[f"ball_tracks.jsonl#frame={frame_index}"],
                rejection_reason="ball_state_not_authoritative",
            ))
            pending = []
            continue
        center = _finite_point(ball.get("center_xy"))
        ball_box = _finite_box(ball.get("xyxy"))
        if center is None or ball_box is None or any(
            abs(actual - expected) > 1e-4
            for actual, expected in zip(center, ((ball_box[0] + ball_box[2]) / 2, (ball_box[1] + ball_box[3]) / 2))
        ):
            output.append(_record(
                frame_index=frame_index, timestamp_seconds=timestamp, status="unknown",
                source_records=[f"ball_tracks.jsonl#frame={frame_index}"],
                rejection_reason="ball_geometry_invalid",
            ))
            pending = []
            continue

        candidates = []
        for row in players_by_frame[frame_index]:
            if str(row.get("class_name", "player")).strip().lower() != "player":
                continue
            if str(row.get("track_state", "")).strip().lower() not in AUTHORITATIVE_PLAYER_STATES:
                continue
            box = _finite_box(row.get("xyxy", row.get("bbox_xyxy")))
            if box is None:
                continue
            expanded = (box[0] - near_margin_px, box[1] - near_margin_px, box[2] + near_margin_px, box[3] + near_margin_px)
            if expanded[0] <= center[0] <= expanded[2] and expanded[1] <= center[1] <= expanded[3]:
                valid_identity, stable_id = _identity(row)
                raw_id = row.get("track_id", row.get("raw_track_id"))
                candidates.append((row, int(raw_id) if valid_identity else None, stable_id, valid_identity))

        source = [f"tracks.jsonl#frame={frame_index}", f"ball_tracks.jsonl#frame={frame_index}"]
        if len(candidates) != 1:
            output.append(_record(
                frame_index=frame_index, timestamp_seconds=timestamp, status="unknown",
                source_records=source,
                rejection_reason="ambiguous_player_candidates" if len(candidates) > 1 else "no_authoritative_player_match",
            ))
            pending = []
            continue

        row, track_id, stable_id, valid_identity = candidates[0]
        if not valid_identity:
            output.append(_record(
                frame_index=frame_index, timestamp_seconds=timestamp, status="unassigned",
                source_records=source,
                rejection_reason="player_identity_unassigned",
            ))
            pending = []
            continue

        if pending and frame_index == pending[-1]["frame_index"] + frame_step:
            previous = pending[-1]
            same_identity = previous["track_id"] == track_id and previous["stable_id"] == stable_id
            if same_identity:
                pending.append({"frame_index": frame_index, "timestamp": timestamp, "track_id": track_id, "stable_id": stable_id, "row": row, "ball": ball})
            else:
                pending = [{"frame_index": frame_index, "timestamp": timestamp, "track_id": track_id, "stable_id": stable_id, "row": row, "ball": ball}]
        else:
            pending = [{"frame_index": frame_index, "timestamp": timestamp, "track_id": track_id, "stable_id": stable_id, "row": row, "ball": ball}]

        first, latest = pending[0], pending[-1]
        elapsed = None
        if first["timestamp"] is not None and latest["timestamp"] is not None:
            elapsed = latest["timestamp"] - first["timestamp"]
        elif fps and fps > 0:
            elapsed = (latest["frame_index"] - first["frame_index"]) / float(fps)
        confidence_values = [value for value in (row.get("confidence"), ball.get("confidence")) if isinstance(value, (int, float))]
        confidence = min(confidence_values, default=1.0)
        if elapsed is None or elapsed < confirmation_seconds:
            output.append(_record(
                frame_index=frame_index, timestamp_seconds=timestamp, status="unknown",
                source_records=source, rejection_reason="confirmation_pending", confidence=confidence,
            ))
            continue

        team_evidence = [
            _team_evidence(item["row"], track_id, team_ids_by_track_id) for item in pending
        ]
        team_conflict = any(conflict for _team_id_value, conflict in team_evidence)
        team_values = {team_id_value for team_id_value, _conflict in team_evidence}
        team_values.discard(None)
        team_id = next(iter(team_values)) if len(team_values) == 1 and not team_conflict else None
        output.append(_record(
            frame_index=frame_index, timestamp_seconds=timestamp, status="confirmed",
            player_track_id=track_id, stable_player_id=stable_id, team_id=team_id,
            confidence=confidence, evidence_frame_indices=[item["frame_index"] for item in pending],
            source_records=[
                f"tracks.jsonl#frame={item['frame_index']}" for item in pending
            ] + [f"ball_tracks.jsonl#frame={item['frame_index']}" for item in pending],
        ))
    return output
