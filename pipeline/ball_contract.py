"""Offline contracts for ball-track and shot evidence artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BALL_CONTRACT_VERSION = 1
BALL_STATES = {"detected", "fallback", "lost"}
BALL_REQUIRED_FIELDS = (
    "frame_index",
    "timestamp_seconds",
    "track_id",
    "track_state",
    "class_name",
    "confidence",
    "xyxy",
    "center_xy",
)


@dataclass
class BallTrackValidation:
    valid: bool
    errors: list[str]
    warnings: list[str]
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


def _finite_numbers(values, length: int) -> bool:
    return isinstance(values, (list, tuple)) and len(values) == length and all(
        isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values
    )


def _strict_frame_index(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def validate_ball_track_rows(
    rows: Iterable[dict],
    *,
    expected_frame_indices: Iterable[int] | None = None,
    frame_step: int | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> BallTrackValidation:
    """Validate rows without filling gaps or changing the source evidence."""
    rows = list(rows)
    errors: list[str] = []
    warnings: list[str] = []
    frames: dict[int, dict] = {}
    state_counts = {state: 0 for state in sorted(BALL_STATES)}
    track_ids: set[int] = set()
    track_id_changes: list[dict] = []
    previous_track_id = None
    previous_frame = None

    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {position}: must be an object")
            continue
        missing = [field for field in BALL_REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append(f"row {position}: missing required fields: {', '.join(missing)}")
            continue
        frame_index = _strict_frame_index(row["frame_index"])
        if frame_index is None:
            errors.append(f"row {position}: frame_index must be an integer")
            continue
        if frame_index in frames:
            errors.append(f"frame {frame_index}: duplicate ball evidence")
        frames[frame_index] = row
        timestamp = row["timestamp_seconds"]
        if timestamp is not None and (
            not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp))
        ):
            errors.append(f"frame {frame_index}: timestamp_seconds must be finite or null")
        state = row["track_state"]
        if state not in BALL_STATES:
            errors.append(f"frame {frame_index}: unsupported track_state {state!r}")
            continue
        state_counts[state] += 1
        track_id = row["track_id"]
        if track_id is not None:
            if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 1:
                errors.append(f"frame {frame_index}: track_id must be a positive integer or null")
            else:
                track_ids.add(track_id)
        if state == "lost":
            if row["xyxy"] is not None or row["center_xy"] is not None:
                errors.append(f"frame {frame_index}: lost evidence must not claim geometry")
            if row["confidence"] is not None:
                errors.append(f"frame {frame_index}: lost evidence must not claim confidence")
        else:
            if not _finite_numbers(row["xyxy"], 4) or not _finite_numbers(row["center_xy"], 2):
                errors.append(f"frame {frame_index}: {state} evidence has invalid geometry")
            elif row["xyxy"][0] >= row["xyxy"][2] or row["xyxy"][1] >= row["xyxy"][3]:
                errors.append(f"frame {frame_index}: geometry must have positive area")
            elif frame_width is not None and frame_height is not None and (
                row["xyxy"][0] < 0 or row["xyxy"][1] < 0
                or row["xyxy"][2] > frame_width or row["xyxy"][3] > frame_height
            ):
                errors.append(f"frame {frame_index}: geometry is outside frame bounds")
            else:
                expected_center = ((row["xyxy"][0] + row["xyxy"][2]) / 2, (row["xyxy"][1] + row["xyxy"][3]) / 2)
                if any(abs(float(actual) - expected) > 1e-4 for actual, expected in zip(row["center_xy"], expected_center)):
                    errors.append(f"frame {frame_index}: center_xy does not match xyxy")
            confidence = row["confidence"]
            if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
                errors.append(f"frame {frame_index}: confidence must be between 0 and 1")

        if previous_frame is not None and frame_step is not None and frame_index - previous_frame != frame_step:
            errors.append(f"frame cadence break: expected {previous_frame + frame_step}, got {frame_index}")
        if previous_track_id is not None and track_id is not None and track_id != previous_track_id:
            track_id_changes.append({"from": previous_track_id, "to": track_id, "frame_index": frame_index})
        previous_frame, previous_track_id = frame_index, track_id

    expected = None if expected_frame_indices is None else list(expected_frame_indices)
    missing_frames = [] if expected is None else [frame for frame in expected if frame not in frames]
    if missing_frames:
        errors.append(f"missing ball evidence for {len(missing_frames)} configured frames")
    if track_id_changes:
        warnings.append("ball track_id changed; inspect the evidence boundary before attribution")
    observed_count = len(frames)
    expected_count = len(expected) if expected is not None else observed_count
    metadata = {
        "contract_version": BALL_CONTRACT_VERSION,
        "frame_count": observed_count,
        "expected_frame_count": expected_count,
        "missing_frame_count": len(missing_frames),
        "missing_frames": missing_frames,
        "state_counts": state_counts,
        "lost_frame_count": state_counts["lost"],
        "lost_coverage": (state_counts["lost"] / expected_count if expected_count else 0.0),
        "coverage": (observed_count / expected_count if expected_count else 0.0),
        "track_ids": sorted(track_ids),
        "track_id_changes": track_id_changes,
    }
    return BallTrackValidation(not errors, errors, warnings, metadata)


def load_ball_track_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shot_ball_evidence(
    event: dict,
    ball_rows: Iterable[dict],
    *,
    expected_frame_indices: Iterable[int] | None = None,
) -> dict:
    start = int(event["start_frame_index"])
    end = int(event.get("end_frame_index", start))
    window = [row for row in ball_rows if start <= row["frame_index"] <= end]
    expected = (
        [frame for frame in expected_frame_indices if start <= frame <= end]
        if expected_frame_indices is not None
        else list(range(start, end + 1))
    )
    observed_frames = {row["frame_index"] for row in window}
    missing = [frame for frame in expected if frame not in observed_frames]
    drawable = [row for row in window if row.get("track_state") in {"detected", "fallback"} and row.get("xyxy") is not None]
    lost_count = sum(row.get("track_state") == "lost" for row in window)
    complete_observation = bool(expected) and not missing and not lost_count and len(drawable) == len(expected)
    if complete_observation:
        status = "observed"
    elif not window:
        status = "missing"
    elif missing:
        status = "unknown"
    elif not drawable:
        status = "lost"
    else:
        status = "unknown"
    return {
        "status": status,
        "frame_window": [start, end],
        "evidence_frame_indices": [int(row["frame_index"]) for row in window],
        "track_ids": sorted({int(row["track_id"]) for row in drawable if row.get("track_id") is not None}),
        "observed_frame_count": len(drawable),
        "lost_frame_count": lost_count,
        "missing_frame_count": len(missing),
        "coverage_status": "complete" if complete_observation else "partial" if window else "missing",
    }


def attach_shot_evidence(
    events: Iterable[dict],
    ball_rows: Iterable[dict],
    *,
    expected_frame_indices: Iterable[int] | None = None,
) -> list[dict]:
    """Add optional evidence references; never infer a shooter or team."""
    ball_rows = list(ball_rows)
    expected_frame_indices = (
        None if expected_frame_indices is None else list(expected_frame_indices)
    )
    return [
        {
            **event,
            "ball_evidence": shot_ball_evidence(
                event, ball_rows, expected_frame_indices=expected_frame_indices
            ),
            "player_identity": {"status": "unknown", "track_id": None, "team_id": None},
        }
        for event in events
    ]
