from __future__ import annotations

from pipeline.possession import resolve_possession_events
from tests.test_possession import ball, player


def _confirmed(track_id):
    return resolve_possession_events(
        [player(frame, track_id=track_id) for frame in (0, 1, 2)],
        [ball(frame) for frame in (0, 1, 2)],
        fps=4,
        team_ids_by_track_id={0: 1, 7: 1},
    )


def test_raw_track_zero_can_hold_the_ball():
    events = _confirmed(0)
    assert [event["status"] for event in events] == ["unknown", "unknown", "confirmed"]
    assert events[-1]["player_track_id"] == 0
    assert events[-1]["team_id"] == 1


def test_negative_track_id_is_still_unassigned():
    events = _confirmed(-1)
    assert all(event["status"] == "unassigned" for event in events)


def test_track_zero_with_stable_id_requires_resolved_stable_id():
    events = resolve_possession_events(
        [player(frame, track_id=0, stable_player_id="p-0", status="resolved") for frame in (0, 1, 2)],
        [ball(frame) for frame in (0, 1, 2)],
        fps=4,
    )
    assert events[-1]["status"] == "confirmed"
    assert events[-1]["stable_player_id"] == "p-0"
