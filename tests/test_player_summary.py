from __future__ import annotations

import json

import pytest

from pipeline.player_summary import build_player_summary, write_player_summary


def _row(frame, x, y, *, player="player-001", team=1, track=5):
    return {"frame_index": frame, "stable_player_id": player, "team_id": team,
            "raw_track_id": track, "court_xy_m": [x, y]}


def _build(rows, *, numbers=None, ball_rows=(), fps=30.0, frame_count=300):
    return build_player_summary(rows, numbers=numbers, ball_rows=ball_rows, fps=fps, frame_count=frame_count)


def test_resamples_to_five_hz_first_observation_per_tick():
    # fps 30 -> 6-frame ticks; frames 0..11 give two ticks.
    rows = [_row(frame, frame * 0.1, 5.0) for frame in range(12)]
    summary, paths = _build(rows)

    assert paths[0]["hz"] == 5
    assert paths[0]["points"] == [[0, 0.0, 5.0], [6, 0.6, 5.0]]
    player = summary["players"][0]
    assert player["frames_observed"] == 12
    assert player["seconds_observed"] == pytest.approx(0.4)
    assert player["distance_m"] == pytest.approx(0.6)
    assert (player["first_frame"], player["last_frame"]) == (0, 11)
    assert summary["duration_seconds"] == pytest.approx(10.0)


def test_teleport_step_is_not_counted_as_distance():
    # 3 m in 0.2 s (15 m/s) is a projection jump; the following 1 m step is real.
    rows = [_row(0, 2.0, 5.0), _row(6, 5.0, 5.0), _row(12, 6.0, 5.0)]
    summary, _ = _build(rows)

    assert summary["players"][0]["distance_m"] == pytest.approx(1.0)


def test_off_court_points_are_dropped_from_paths_and_distance():
    rows = [_row(0, 2.0, 5.0), _row(6, 40.0, 5.0), _row(12, 2.5, 5.0)]
    summary, paths = _build(rows)

    assert [point[0] for point in paths[0]["points"]] == [0, 12]
    assert summary["players"][0]["frames_observed"] == 3


def test_team_majority_requires_agreement_threshold():
    agree = [_row(i, 1.0, 1.0, team=1) for i in range(4)] + [_row(4, 1.0, 1.0, team=0)]
    split = [_row(i, 1.0, 1.0, player="player-002", team=1) for i in range(3)] + [
        _row(i, 1.0, 1.0, player="player-002", team=0) for i in range(3, 5)]
    summary, _ = _build(agree + split)
    by_id = {player["stable_player_id"]: player for player in summary["players"]}

    assert by_id["player-001"]["team_id"] == 1
    assert by_id["player-001"]["team_agreement"] == pytest.approx(0.8)
    assert by_id["player-002"]["team_id"] is None
    assert by_id["player-002"]["team_agreement"] == pytest.approx(0.6)


def test_rows_without_stable_id_are_counted_not_summarised():
    rows = [_row(0, 1.0, 1.0), {"frame_index": 1, "track_id": 9, "court_xy_m": [1.0, 1.0]},
            {"frame_index": 2, "stable_player_id": None, "court_xy_m": [1.0, 1.0]}]
    summary, _ = _build(rows)

    assert summary["unassigned_rows"] == 2
    assert [player["stable_player_id"] for player in summary["players"]] == ["player-001"]


def test_jersey_statuses_and_track_ids():
    rows = [_row(0, 1.0, 1.0, player="player-001", track=9), _row(1, 1.0, 1.0, player="player-001", track=3),
            _row(0, 1.0, 1.0, player="player-002"), _row(0, 1.0, 1.0, player="player-003"),
            _row(0, 1.0, 1.0, player="player-004")]
    numbers = {"players": [
        {"stable_player_id": "player-001", "number": "7", "status": "confirmed", "support_count": 4},
        {"stable_player_id": "player-002", "number": "unknown", "status": "ambiguous", "support_count": 1},
        {"stable_player_id": "player-003", "number": "unknown", "status": "unknown", "support_count": 0},
    ]}
    summary, _ = _build(rows, numbers=numbers)
    by_id = {player["stable_player_id"]: player for player in summary["players"]}

    assert by_id["player-001"]["track_ids"] == [3, 9]
    assert by_id["player-001"]["jersey"] == {"number": "7", "status": "confirmed", "support_count": 4}
    assert by_id["player-002"]["jersey"] == {"number": None, "status": "ambiguous", "support_count": 1}
    assert by_id["player-003"]["jersey"]["status"] == "unknown"
    assert by_id["player-004"]["jersey"] == {"number": None, "status": "unknown", "support_count": 0}


def test_ball_counts_and_empty_input_and_write(tmp_path):
    ball = [{"frame_index": 0, "track_state": "detected"}, {"frame_index": 1, "track_state": "lost"},
            {"frame_index": 2, "track_state": "detected"}]
    summary, paths = _build([], ball_rows=ball)

    assert summary["players"] == [] and paths == []
    assert summary["ball"] == {"detected_frames": 2, "frames": 3}
    assert summary["schema_version"] == 1

    write_player_summary(tmp_path, summary, paths)
    assert json.loads((tmp_path / "player_summary.json").read_text())["fps"] == 30.0
    assert (tmp_path / "player_paths.jsonl").read_text() == ""
