from __future__ import annotations

from pipeline.possession_transitions import resolve_possession_transitions


def _row(frame, player, team, *, status="confirmed", evidence=True, terminal=False):
    return {
        "schema_version": 1,
        "frame_index": frame,
        "status": status,
        "player_track_id": player,
        "stable_player_id": player,
        "team_id": team,
        "evidence_frame_indices": [frame] if evidence else [],
        "source_records": [f"possession_events.jsonl#frame={frame}"],
        "rejection_reason": "terminal_loss" if terminal else None,
    }


def test_same_team_transfer_is_pass_with_source_provenance():
    events = resolve_possession_transitions([_row(10, "a", 1), _row(15, "b", 1)])

    assert events[0]["event_type"] == "pass"
    assert events[0]["status"] == "confirmed"
    assert events[0]["source_records"] == [
        "possession_events.jsonl#frame=10",
        "possession_events.jsonl#frame=15",
    ]


def test_opponent_transfer_is_steal():
    events = resolve_possession_transitions([_row(10, "a", 1), _row(12, "b", 2)])

    assert events[0]["event_type"] == "steal"
    assert events[0]["status"] == "confirmed"


def test_same_player_and_ambiguous_team_transitions_are_not_events():
    assert resolve_possession_transitions([_row(10, "a", 1), _row(12, "a", 1)]) == []
    assert resolve_possession_transitions([_row(10, "a", None), _row(12, "b", 1)]) == []


def test_stable_identity_ignores_raw_track_change():
    first = _row(10, "raw-1", 1)
    second = _row(12, "raw-2", 1)
    first["stable_player_id"] = second["stable_player_id"] = "stable-a"

    assert resolve_possession_transitions([first, second]) == []


def test_raw_identity_is_used_when_stable_id_is_absent():
    first = _row(10, 1, 1)
    second = _row(12, 2, 1)
    first.pop("stable_player_id")
    second.pop("stable_player_id")

    assert resolve_possession_transitions([first, second])[0]["event_type"] == "pass"


def test_duplicate_authoritative_frame_fails_closed():
    duplicate = _row(10, "b", 1)
    events = resolve_possession_transitions([_row(10, "a", 1), duplicate, _row(12, "c", 1)])

    assert events == []
    assert all(event["frame_gap"] > 0 for event in events)


def test_duplicate_authoritative_frame_between_holders_blocks_bridge_transfer():
    events = resolve_possession_transitions([_row(5, "a", 1), _row(10, "b", 1), _row(10, "c", 1), _row(12, "d", 1)])

    assert events == []


def test_unknown_gap_is_uncertain_loss_and_explicit_terminal_loss_is_confirmed():
    uncertain = resolve_possession_transitions([_row(10, "a", 1), _row(12, None, None, status="unknown")])
    terminal = resolve_possession_transitions([_row(10, "a", 1), _row(12, None, None, status="lost", terminal=True)])

    assert uncertain[0]["event_type"] == "lost_possession"
    assert uncertain[0]["status"] == "uncertain"
    assert terminal[0]["event_type"] == "turnover"
    assert terminal[0]["status"] == "confirmed"


def test_gap_limit_prevents_transfer_and_marks_uncertain_loss():
    events = resolve_possession_transitions([_row(10, "a", 1), _row(20, "b", 1)], max_frame_gap=5)

    assert len(events) == 1
    assert events[0]["event_type"] == "lost_possession"
    assert events[0]["status"] == "uncertain"
