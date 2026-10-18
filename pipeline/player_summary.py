"""Per-player aggregates and 5 Hz court paths derived from stable-id stats rows."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

SUMMARY_FILENAME = "player_summary.json"
PATHS_FILENAME = "player_paths.jsonl"
SUMMARY_SCHEMA_VERSION = 1
PATH_HZ = 5
# Faster than any player: a step above this is a projection jump, not movement.
MAX_STEP_SPEED_MPS = 9.0
TEAM_AGREEMENT_MIN = 0.8
COURT_LENGTH_M, COURT_WIDTH_M, COURT_MARGIN_M = 28.0, 15.0, 0.5


def _on_court(point: Any) -> bool:
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return False
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    return (-COURT_MARGIN_M <= x <= COURT_LENGTH_M + COURT_MARGIN_M
            and -COURT_MARGIN_M <= y <= COURT_WIDTH_M + COURT_MARGIN_M)


def _jersey_by_player(numbers: Mapping[str, Any] | None) -> dict[str, dict]:
    jerseys: dict[str, dict] = {}
    for player in (numbers or {}).get("players", []) or []:
        stable_id = player.get("stable_player_id")
        if not stable_id:
            continue
        status = player.get("status")
        if status not in ("confirmed", "ambiguous", "unknown"):
            status = "unknown"
        number = player.get("number")
        jerseys[str(stable_id)] = {
            "number": str(number) if status == "confirmed" and number not in (None, "unknown") else None,
            "status": status,
            "support_count": int(player.get("support_count") or 0),
        }
    return jerseys


def build_player_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    numbers: Mapping[str, Any] | None,
    ball_rows: Iterable[Mapping[str, Any]],
    fps: float,
    frame_count: int,
) -> tuple[dict, list[dict]]:
    fps = float(fps) if fps and fps > 0 else 0.0
    tick_frames = fps / PATH_HZ if fps else 0.0
    by_player: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unassigned = 0
    for row in rows:
        stable_id = row.get("stable_player_id")
        if not stable_id:
            unassigned += 1
            continue
        by_player[str(stable_id)].append(row)

    jerseys = _jersey_by_player(numbers)
    players, paths = [], []
    for stable_id in sorted(by_player):
        player_rows = sorted(by_player[stable_id], key=lambda row: int(row["frame_index"]))
        frames = {int(row["frame_index"]) for row in player_rows}
        teams = Counter(int(row["team_id"]) for row in player_rows if row.get("team_id") is not None)
        team_id, agreement = None, 0.0
        if teams:
            top_team, top_count = teams.most_common(1)[0]
            agreement = top_count / sum(teams.values())
            team_id = top_team if agreement >= TEAM_AGREEMENT_MIN else None

        points: list[list] = []
        seen_ticks: set[int] = set()
        for row in player_rows:
            if not tick_frames or not _on_court(row.get("court_xy_m")):
                continue
            frame_index = int(row["frame_index"])
            tick = int(frame_index // tick_frames)
            if tick in seen_ticks:
                continue
            seen_ticks.add(tick)
            x, y = row["court_xy_m"]
            points.append([frame_index, round(float(x), 3), round(float(y), 3)])

        distance = 0.0
        for (f0, x0, y0), (f1, x1, y1) in zip(points, points[1:]):
            step = math.hypot(x1 - x0, y1 - y0)
            if step / ((f1 - f0) / fps) <= MAX_STEP_SPEED_MPS:
                distance += step

        track_ids = sorted({int(row.get("raw_track_id", row.get("track_id")))
                            for row in player_rows
                            if row.get("raw_track_id", row.get("track_id")) is not None})
        players.append({
            "stable_player_id": stable_id,
            "team_id": team_id,
            "team_agreement": round(agreement, 4),
            "track_ids": track_ids,
            "frames_observed": len(frames),
            "seconds_observed": round(len(frames) / fps, 3) if fps else 0.0,
            "first_frame": min(frames),
            "last_frame": max(frames),
            "distance_m": round(distance, 2),
            "avg_xy_m": [round(sum(p[1] for p in points) / len(points), 2),
                         round(sum(p[2] for p in points) / len(points), 2)] if points else None,
            "jersey": jerseys.get(stable_id, {"number": None, "status": "unknown", "support_count": 0}),
        })
        if points:
            paths.append({"stable_player_id": stable_id, "team_id": team_id, "hz": PATH_HZ, "points": points})

    ball_frames: dict[int, bool] = {}
    for row in ball_rows:
        frame = int(row["frame_index"])
        ball_frames[frame] = ball_frames.get(frame, False) or row.get("track_state") == "detected"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "fps": fps,
        "frame_count": int(frame_count),
        "duration_seconds": round(int(frame_count) / fps, 3) if fps else 0.0,
        "players": players,
        "ball": {"detected_frames": sum(ball_frames.values()), "frames": len(ball_frames)},
        "unassigned_rows": unassigned,
    }
    return summary, paths


def write_player_summary(output_dir: Path, summary: dict, paths: list[dict]) -> tuple[str, str]:
    output_dir = Path(output_dir)
    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / PATHS_FILENAME).write_text(
        "".join(json.dumps(path) + "\n" for path in paths), encoding="utf-8"
    )
    return SUMMARY_FILENAME, PATHS_FILENAME
