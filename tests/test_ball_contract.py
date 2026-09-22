from __future__ import annotations

from pipeline.ball_contract import attach_shot_evidence, validate_ball_track_rows


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
    assert events[0]["player_identity"] == {"status": "unknown", "track_id": None, "team_id": None}


def test_shot_contract_does_not_hide_an_unrepresented_frame():
    event = attach_shot_evidence(
        [{"event_id": 2, "start_frame_index": 1, "end_frame_index": 3}],
        [row(1), row(3)],
    )[0]

    assert event["ball_evidence"]["status"] == "unknown"
    assert event["ball_evidence"]["missing_frame_count"] == 1
