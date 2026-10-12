from __future__ import annotations

import json

import pytest

from pipeline.ball_contract import attach_shot_evidence, validate_ball_track_rows
from pipeline.possession import resolve_possession_events


def row(frame, state="detected", track_id=1):
    geometry = None if state == "lost" else [10.0, 10.0, 20.0, 20.0]
    return {
        "frame_index": frame,
        "timestamp_seconds": frame / 30,
        "track_id": track_id,
        "track_state": state,
        "class_name": "ball",
        "confidence": None if state == "lost" else 0.9,
        "xyxy": geometry,
        "center_xy": None if geometry is None else [15.0, 15.0],
    }


def player_row(frame, track_id=7, box=(8.0, 8.0, 22.0, 22.0), **extra):
    return {
        "frame_index": frame,
        "track_id": track_id,
        "track_state": "tracked",
        "class_name": "player",
        "xyxy": list(box),
        **extra,
    }


def test_dense_track_validates_configured_cadence_and_counts_states():
    result = validate_ball_track_rows([row(0), row(1), row(2)], expected_frame_indices=[0, 1, 2], frame_step=1)
    assert result.valid
    assert result.metadata["state_counts"] == {"detected": 3, "fallback": 0, "lost": 0}
    assert result.metadata["coverage"] == 1.0


def test_sparse_lost_track_reports_missing_coverage_without_interpolation():
    result = validate_ball_track_rows([row(0), row(2, "lost")], expected_frame_indices=[0, 1, 2], frame_step=1)
    assert not result.valid
    assert result.metadata["missing_frames"] == [1]
    assert result.metadata["lost_frame_count"] == 1
    assert result.metadata["lost_coverage"] == 1 / 3


def test_geometry_state_contract_rejects_lost_geometry_and_bad_box():
    malformed = {**row(0, "lost"), "xyxy": [1, 1, 2, 2], "center_xy": [1.5, 1.5]}
    invalid_box = {**row(1), "xyxy": [3, 3, 2, 2]}
    result = validate_ball_track_rows([malformed, invalid_box])
    assert not result.valid
    assert any("lost evidence must not claim geometry" in error for error in result.errors)
    assert any("positive area" in error for error in result.errors)


def test_frame_index_rejects_bool_and_non_integral_float():
    assert not validate_ball_track_rows([row(True)]).valid
    assert not validate_ball_track_rows([row(1.5)]).valid


def test_shot_contract_references_ball_evidence_and_keeps_identity_unknown():
    events = attach_shot_evidence(
        [{"event_id": 1, "start_frame_index": 1, "end_frame_index": 3}],
        [row(1), row(2, "lost"), row(3)],
        expected_frame_indices=[1, 2, 3],
    )
    assert events[0]["ball_evidence"]["status"] == "unknown"
    assert events[0]["ball_evidence"]["coverage_status"] == "partial"
    assert events[0]["ball_evidence"]["evidence_frame_indices"] == [1, 2, 3]
    assert events[0]["player_identity"] == {"status": "unknown", "track_id": None, "stable_player_id": None, "team_id": None}


def test_shot_contract_does_not_hide_an_unrepresented_frame():
    event = attach_shot_evidence(
        [{"event_id": 2, "start_frame_index": 1, "end_frame_index": 3}],
        [row(1), row(3)],
    )[0]

    assert event["ball_evidence"]["status"] == "unknown"
    assert event["ball_evidence"]["missing_frame_count"] == 1


def test_shot_contract_fails_closed_for_malformed_shot_anchor_box():
    event = attach_shot_evidence(
        [{
            "event_id": 9,
            "start_frame_index": 1,
            "end_frame_index": 1,
            "start_image_xy": [15, 15],
            "start_shot_xyxy": [15, "bad", 20, 20],
        }],
        [row(1)],
        expected_frame_indices=[1],
        player_rows=[player_row(1)],
    )[0]

    assert event["player_identity"] == {"status": "unknown", "track_id": None, "stable_player_id": None, "team_id": None}


@pytest.mark.parametrize(
    "box",
    [
        [float("nan"), 10, 20, 20],
        [10, float("inf"), 20, 20],
        [10, 10, 10, 20],
        [10, 20, 20, 10],
    ],
)
def test_shot_contract_rejects_nonfinite_or_invalid_area_anchor_box(box):
    event = attach_shot_evidence(
        [{
            "event_id": 11,
            "start_frame_index": 1,
            "end_frame_index": 1,
            "start_image_xy": [15, 15],
            "start_shot_xyxy": box,
        }],
        [row(1)],
        expected_frame_indices=[1],
        player_rows=[player_row(1)],
    )[0]

    assert event["player_identity"] == {"status": "unknown", "track_id": None, "stable_player_id": None, "team_id": None}


def test_shot_contract_rejects_nonfinite_player_geometry():
    event = attach_shot_evidence(
        [{"event_id": 12, "start_frame_index": 1, "end_frame_index": 1, "start_image_xy": [15, 15]}],
        [row(1)],
        expected_frame_indices=[1],
        player_rows=[player_row(1, box=(10, 10, float("inf"), 20))],
    )[0]

    assert event["player_identity"] == {"status": "unknown", "track_id": None, "stable_player_id": None, "team_id": None}


@pytest.mark.parametrize("state", ["LOST", " unknown ", "unresolved", "other"])
def test_shot_contract_rejects_non_authoritative_player_states(state):
    event = attach_shot_evidence(
        [{"event_id": 13, "start_frame_index": 1, "end_frame_index": 1, "start_image_xy": [15, 15]}],
        [row(1)],
        expected_frame_indices=[1],
        player_rows=[player_row(1, track_state=state)],
    )[0]

    assert event["player_identity"] == {"status": "unknown", "track_id": None, "stable_player_id": None, "team_id": None}


@pytest.mark.parametrize("state", [" TRACKED ", "DETECTED"])
def test_shot_contract_accepts_normalized_authoritative_player_states(state):
    event = attach_shot_evidence(
        [{"event_id": 14, "start_frame_index": 1, "end_frame_index": 1, "start_image_xy": [15, 15]}],
        [row(1)],
        expected_frame_indices=[1],
        player_rows=[player_row(1, track_state=state, stable_player_id="player-7")],
    )[0]

    assert event["player_identity"]["status"] == "assigned"


def test_shot_contract_does_not_assign_raw_track_without_stable_identity():
    event = attach_shot_evidence(
        [{"event_id": 3, "start_frame_index": 1, "end_frame_index": 3, "start_image_xy": [15, 15]}],
        [row(1), row(2), row(3)],
        expected_frame_indices=[1, 2, 3],
        player_rows=[player_row(1), player_row(2), player_row(3)],
        team_ids_by_track_id={7: 1},
    )[0]

    assert event["player_identity"] == {"status": "unknown", "track_id": None, "stable_player_id": None, "team_id": None}


def test_shot_contract_assigns_unique_player_and_present_team_from_geometry():
    event = attach_shot_evidence(
        [{"event_id": 3, "start_frame_index": 1, "end_frame_index": 3, "start_image_xy": [15, 15]}],
        [row(1), row(2), row(3)],
        expected_frame_indices=[1, 2, 3],
        player_rows=[player_row(1, stable_player_id="player-7"), player_row(2, stable_player_id="player-7"), player_row(3, stable_player_id="player-7")],
        team_ids_by_track_id={7: 1},
    )[0]

    assert event["player_identity"] == {
        "status": "assigned",
        "track_id": 7,
        "stable_player_id": "player-7",
        "team_id": 1,
    }


def test_shot_contract_leaves_ambiguous_players_unassigned():
    event = attach_shot_evidence(
        [{"event_id": 4, "start_frame_index": 1, "end_frame_index": 1, "start_image_xy": [15, 15]}],
        [row(1)],
        expected_frame_indices=[1],
        player_rows=[player_row(1, 7), player_row(1, 8)],
    )[0]

    assert event["player_identity"] == {"status": "unknown", "track_id": None, "stable_player_id": None, "team_id": None}


def test_shot_contract_requires_observed_ball_evidence_before_attribution():
    event = attach_shot_evidence(
        [{"event_id": 5, "start_frame_index": 1, "end_frame_index": 2, "start_image_xy": [15, 15]}],
        [row(1), row(2, "lost")],
        expected_frame_indices=[1, 2],
        player_rows=[player_row(1)],
        team_ids_by_track_id={7: 1},
    )[0]

    assert event["player_identity"] == {"status": "unknown", "track_id": None, "stable_player_id": None, "team_id": None}


def test_shot_contract_keeps_player_when_team_is_missing():
    event = attach_shot_evidence(
        [{"event_id": 6, "start_frame_index": 1, "end_frame_index": 1, "start_image_xy": [15, 15]}],
        [row(1)],
        expected_frame_indices=[1],
        player_rows=[player_row(1, stable_player_id="player-7", status="resolved")],
    )[0]

    assert event["player_identity"] == {
        "status": "assigned",
        "track_id": 7,
        "stable_player_id": "player-7",
        "team_id": None,
    }


def test_shot_contract_rejects_conflicting_stable_identity():
    event = attach_shot_evidence(
        [{"event_id": 7, "start_frame_index": 1, "end_frame_index": 2, "start_image_xy": [15, 15]}],
        [row(1), row(2)],
        expected_frame_indices=[1, 2],
        player_rows=[
            player_row(1, stable_player_id="player-7", status="resolved"),
            player_row(2, stable_player_id="player-8", status="resolved"),
        ],
    )[0]

    assert event["player_identity"] == {"status": "unknown", "track_id": None, "stable_player_id": None, "team_id": None}


def test_shot_contract_rejects_team_conflict_across_raw_tracks_for_one_stable_player():
    event = attach_shot_evidence(
        [{"event_id": 10, "start_frame_index": 1, "end_frame_index": 2, "start_image_xy": [15, 15]}],
        [row(1), row(2)],
        expected_frame_indices=[1, 2],
        player_rows=[
            player_row(1, 7, stable_player_id="player-7", status="resolved", team_id=1),
            player_row(2, 8, stable_player_id="player-7", status="resolved", team_id=2),
        ],
    )[0]

    assert event["player_identity"] == {"status": "unknown", "track_id": None, "stable_player_id": None, "team_id": None}


def test_runner_evidence_join_reads_final_track_and_team_artifacts(tmp_path):
    from pipeline.runner import _attach_ball_evidence

    (tmp_path / "shot_events.jsonl").write_text(
        json.dumps({
            "event_id": 8,
            "start_frame_index": 1,
            "end_frame_index": 1,
            "start_image_xy": [15, 15],
        }) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "ball_tracks.jsonl").write_text(json.dumps(row(1)) + "\n", encoding="utf-8")
    (tmp_path / "tracks.jsonl").write_text(json.dumps(player_row(1, stable_player_id="player-7")) + "\n", encoding="utf-8")
    (tmp_path / "teams.json").write_text(json.dumps({"assignments_by_track_id": {"7": 2}}), encoding="utf-8")

    _attach_ball_evidence(tmp_path, expected_frame_indices=[1])

    evidence = json.loads((tmp_path / "shot_evidence.jsonl").read_text())
    assert evidence["player_identity"]["team_id"] == 2


def test_runner_evidence_uses_projected_shooter_when_ball_control_is_not_authoritative(tmp_path):
    from pipeline.runner import _attach_ball_evidence

    event = {"event_id": 1, "start_frame_index": 1, "end_frame_index": 1, "start_image_xy": [15, 15]}
    (tmp_path / "shot_events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (tmp_path / "ball_tracks.jsonl").write_text(json.dumps(row(1, "lost")) + "\n", encoding="utf-8")
    (tmp_path / "tracks.jsonl").write_text(json.dumps(player_row(1, stable_player_id="player-008")) + "\n", encoding="utf-8")
    (tmp_path / "projected_shots.json").write_text(json.dumps({"shots": [
        {"event_id": 1, "shooter_track_id": 8, "shooter_stable_player_id": "player-008", "shooter_team_id": 2},
    ]}), encoding="utf-8")

    _attach_ball_evidence(tmp_path, expected_frame_indices=[1])

    evidence = json.loads((tmp_path / "shot_evidence.jsonl").read_text())
    assert evidence["ball_evidence"]["status"] == "lost"
    assert evidence["player_identity"] == {
        "status": "inferred",
        "track_id": 8,
        "stable_player_id": "player-008",
        "team_id": 2,
        "attribution_method": "projected_shot_track",
        "authority": "non_authoritative",
    }

    possession = resolve_possession_events(
        [player_row(1, stable_player_id="player-008")], [row(1, "lost")]
    )
    assert possession[0]["status"] == "lost"
    assert possession[0]["stable_player_id"] is None


def test_runner_projected_shots_selector_identity_reaches_final_shot_evidence(tmp_path):
    from pipeline.runner import _attach_ball_evidence, _maybe_write_projected_shots

    event = {
        "event_id": 3,
        "start_frame_index": 100,
        "end_frame_index": 100,
        "start_image_xy": [50.0, 80.0],
    }
    (tmp_path / "shot_events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (tmp_path / "ball_tracks.jsonl").write_text(json.dumps(row(100, "lost")) + "\n", encoding="utf-8")
    (tmp_path / "tracks.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "projected_tracks.jsonl").write_text("".join(json.dumps(projected_row) + "\n" for projected_row in [
        {"frame_index": 100, "track_id": 8, "xyxy": [0, 0, 20, 20], "image_foot_xy": [0, 0], "court_xy_m": [0, 0]},
        {"frame_index": 100, "track_id": 23, "stable_player_id": "player-023", "xyxy": [42, 18, 68, 92], "image_foot_xy": [55, 92], "court_xy_m": [5.5, 9.2]},
    ]), encoding="utf-8")

    projected_file, _ = _maybe_write_projected_shots(tmp_path)
    assert projected_file == "projected_shots.json"
    projected = json.loads((tmp_path / projected_file).read_text(encoding="utf-8"))
    assert projected["shots"][0]["shooter_stable_player_id"] == "player-023"

    _attach_ball_evidence(tmp_path, expected_frame_indices=[100])

    evidence = json.loads((tmp_path / "shot_evidence.jsonl").read_text())
    assert evidence["player_identity"]["stable_player_id"] == "player-023"
    assert evidence["player_identity"]["track_id"] == 23
    assert evidence["ball_evidence"]["status"] == "lost"
    assert resolve_possession_events(
        [player_row(100, 23, stable_player_id="player-023")], [row(100, "lost")]
    )[0]["stable_player_id"] is None


def test_runner_projected_shooter_fallback_keeps_unknown_without_stable_identity(tmp_path):
    from pipeline.runner import _attach_ball_evidence

    event = {"event_id": 2, "start_frame_index": 1, "end_frame_index": 1, "start_image_xy": [15, 15]}
    (tmp_path / "shot_events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (tmp_path / "ball_tracks.jsonl").write_text(json.dumps(row(1, "lost")) + "\n", encoding="utf-8")
    (tmp_path / "tracks.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "projected_shots.json").write_text(json.dumps({"shots": [
        {"event_id": 2, "shooter_track_id": 8, "shooter_stable_player_id": None},
    ]}), encoding="utf-8")

    _attach_ball_evidence(tmp_path, expected_frame_indices=[1])

    evidence = json.loads((tmp_path / "shot_evidence.jsonl").read_text())
    assert evidence["player_identity"] == {
        "status": "unknown", "track_id": None, "stable_player_id": None, "team_id": None,
    }
