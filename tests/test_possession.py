from __future__ import annotations

import json

from pipeline.possession import POSSESSION_CONTRACT_VERSION, resolve_possession_events
from pipeline.runner import _write_possession_events


def ball(frame, center=(15.0, 15.0), state="detected", confidence=0.9):
    if state == "lost":
        return {
            "frame_index": frame,
            "timestamp_seconds": frame / 10,
            "track_state": "lost",
            "center_xy": None,
            "confidence": None,
        }
    return {
        "frame_index": frame,
        "timestamp_seconds": frame / 10,
        "track_state": state,
        "center_xy": list(center),
        "xyxy": [center[0] - 1, center[1] - 1, center[0] + 1, center[1] + 1],
        "confidence": confidence,
    }


def player(frame, track_id=7, box=(10.0, 10.0, 20.0, 20.0), **extra):
    return {
        "frame_index": frame,
        "track_id": track_id,
        "track_state": "tracked",
        "class_name": "player",
        "xyxy": list(box),
        "confidence": 0.8,
        **extra,
    }


def test_persistent_unique_holder_confirms_after_temporal_window():
    events = resolve_possession_events(
        [player(frame) for frame in (0, 1, 2)],
        [ball(frame) for frame in (0, 1, 2)],
        fps=4,
        team_ids_by_track_id={7: 1},
    )

    assert [event["status"] for event in events] == ["unknown", "unknown", "confirmed"]
    assert events[-1]["player_track_id"] == 7
    assert events[-1]["team_id"] == 1
    assert events[-1]["evidence_frame_indices"] == [0, 1, 2]


def test_ambiguous_candidates_never_confirm():
    events = resolve_possession_events(
        [player(0, 7), player(0, 8), player(1, 7), player(1, 8)],
        [ball(0), ball(1)],
        fps=4,
    )

    assert [event["status"] for event in events] == ["unknown", "unknown"]
    assert all(event["rejection_reason"] == "ambiguous_player_candidates" for event in events)


def test_missing_or_lost_ball_preserves_gap_and_requires_new_confirmation():
    events = resolve_possession_events(
        [player(frame) for frame in (0, 1, 2, 3, 4, 5)],
        [ball(0), ball(1), ball(2, state="lost"), ball(3), ball(4), ball(5)],
        fps=4,
    )

    assert [event["status"] for event in events] == ["unknown", "unknown", "lost", "unknown", "unknown", "confirmed"]
    assert events[2]["rejection_reason"] == "ball_track_lost"


def test_frame_gap_breaks_confirmation_even_when_elapsed_window_is_long_enough():
    events = resolve_possession_events(
        [player(0), player(10)],
        [ball(0), ball(10)],
        fps=10,
    )

    assert [event["status"] for event in events] == ["unknown", "unknown"]
    assert events[-1]["rejection_reason"] == "confirmation_pending"


def test_invalid_identity_is_unassigned_and_conflicting_team_is_null():
    events = resolve_possession_events(
        [
            player(0, stable_player_id=""),
            player(1, stable_player_id=""),
            player(2, stable_player_id=""),
        ],
        [ball(0), ball(1), ball(2)],
        fps=4,
        team_ids_by_track_id={7: 1},
    )
    assert events[-1]["status"] == "unassigned"
    assert events[-1]["team_id"] is None

    conflicting = resolve_possession_events(
        [player(0, team_id=1), player(1, team_id=2), player(2, team_id=2)],
        [ball(0), ball(1), ball(2)],
        fps=4,
    )
    assert conflicting[-1]["status"] == "confirmed"
    assert conflicting[-1]["team_id"] is None

    mapped_conflict = resolve_possession_events(
        [player(0, team_id=1), player(1, team_id=2), player(2, team_id=1)],
        [ball(0), ball(1), ball(2)],
        fps=2,
        team_ids_by_track_id={7: 1},
    )
    assert mapped_conflict[-1]["status"] == "confirmed"
    assert mapped_conflict[-1]["team_id"] is None


def test_runner_writer_emits_jsonl_and_keeps_results_reference(tmp_path):
    tracks = [player(frame) for frame in (0, 1, 2)]
    balls = [ball(frame) for frame in (0, 1, 2)]
    path = _write_possession_events(tmp_path, tracks, balls, fps=4, team_ids_by_track_id={7: 1})

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert path.name == "possession_events.jsonl"
    assert rows[-1]["schema_version"] == POSSESSION_CONTRACT_VERSION
    assert rows[-1]["status"] == "confirmed"


def test_runner_writer_preserves_uncertain_transitions(tmp_path):
    tracks = [
        player(0), player(1), player(2),
        player(3, 7), player(3, 8),
        player(4), player(5, stable_player_id=""),
        player(6), player(7), player(8),
    ]
    balls = [ball(0), ball(1), ball(2), ball(3), ball(5), ball(6), ball(7), ball(8)]

    path = _write_possession_events(tmp_path, tracks, balls, fps=4)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["status"] for row in rows] == [
        "unknown", "unknown", "confirmed", "unknown", "lost", "unassigned",
        "unknown", "unknown", "confirmed",
    ]
    assert rows[3]["rejection_reason"] == "ambiguous_player_candidates"
    assert rows[4]["rejection_reason"] == "ball_evidence_missing"
    assert rows[6]["rejection_reason"] == "confirmation_pending"
