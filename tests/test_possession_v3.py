"""Focused V3 contract, compatibility, and replay tests."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from pipeline.config import PipelineConfig
from pipeline.possession_v3 import (
    BallTransition, CandidateEvidence, EvidenceKind, EventType, PossessionEvent,
    PossessionObservation, PossessionState, Provenance, V3_CONTRACT_VERSION,
    engine_name,
    score_candidates, PossessionResolver,
    normalize_ball_rows, derive_ball_primitives, derive_ball_transitions,
    classify_transition,
)
from pipeline.results import PipelineResults, RunSummary
from pipeline.runner import _possession_result_config


def test_contract_serialization_is_versioned_and_stable():
    transition = BallTransition(
        "t-0004", "player-a", None, ("release", "transit"),
        Provenance(("tracks.jsonl#frame=4",), (4,)),
    )
    payload = asdict(transition)
    assert V3_CONTRACT_VERSION == payload["schema_version"] == 1
    assert payload["state"] == PossessionState.UNKNOWN.value
    assert payload["primitive_types"] == ["release", "transit"]
    assert payload["provenance"]["source_records"] == ["tracks.jsonl#frame=4"]


def test_vocabulary_and_event_contract_are_explicit():
    assert [x.value for x in EvidenceKind] == ["observed", "inferred", "contested", "unknown"]
    event = asdict(PossessionEvent(EventType.TRANSFER, 8, Provenance(frame_indices=[8])))
    assert event["event_type"] == "possession_transfer"
    assert event["provenance"]["frame_indices"] == [8]


def test_event_rejects_reversed_timestamp_range():
    with pytest.raises(ValueError, match="timestamp_end"):
        PossessionEvent(EventType.PASS, 8, Provenance(frame_indices=[8]), timestamp_start=2.0, timestamp_end=1.0)


def _semantic_transition(**overrides):
    values = dict(
        transition_id="semantic-1", source_holder="p1", destination_holder="p2",
        primitive_types=["BALL_RELEASE", "BALL_IN_TRANSIT", "BALL_RECEPTION"],
        provenance=Provenance(source_records=["ball#1"], frame_indices=[1, 2, 3]),
        source_team_id=1, destination_team_id=1, duration_seconds=.2,
        loose_seconds=0.0, proximity_distance=12.0, score=.9, score_margin=.4,
        release_evidence=True, reception_evidence=True, trajectory_evidence=True,
        loss_of_control_evidence=False,
    )
    values.update(overrides)
    return BallTransition(**values)


def test_semantics_clean_pass_emits_transfer_then_pass_with_identity_and_scores():
    events = classify_transition(_semantic_transition())
    assert [event.event_type for event in events] == [EventType.TRANSFER, EventType.PASS]
    assert events[-1].source_holder == "p1" and events[-1].destination_holder == "p2"
    assert events[-1].source_team_id == events[-1].destination_team_id == 1
    assert events[-1].score == .9 and events[-1].score_margin == .4
    assert events[-1].transition_id == "semantic-1"
    assert events[-1].provenance.source_records == ["ball#1"]
    assert events[-1].provenance.frame_indices == [1, 2, 3]


def test_same_team_loose_ball_recovery_is_transfer_not_pass():
    events = classify_transition(_semantic_transition(loose_seconds=.4, primitive_types=["BALL_RELEASE", "BALL_IN_TRANSIT", "LOOSE_BALL", "BALL_RECEPTION"]))
    assert [event.event_type for event in events] == [EventType.TRANSFER]
    assert "loose_ball_recovery" in events[0].reason_codes


def test_direct_steal_requires_rapid_local_cross_team_evidence():
    events = classify_transition(_semantic_transition(source_team_id=1, destination_team_id=2, duration_seconds=.1, proximity_distance=20))
    assert [event.event_type for event in events] == [EventType.TRANSFER, EventType.STEAL]


def test_missing_score_margin_cannot_pass_steal_gate():
    events = classify_transition(_semantic_transition(source_team_id=1, destination_team_id=2, duration_seconds=.1, proximity_distance=20, score_margin=None))
    assert events[-1].event_type is EventType.UNKNOWN
    assert EventType.STEAL not in [event.event_type for event in events]


def test_insufficient_score_margin_blocks_steal():
    events = classify_transition(_semantic_transition(source_team_id=1, destination_team_id=2, duration_seconds=.1, proximity_distance=20, score_margin=.05))
    assert events[-1].event_type is EventType.UNKNOWN


def test_delayed_opponent_recovery_without_confirmed_loss_is_unknown():
    events = classify_transition(_semantic_transition(source_team_id=1, destination_team_id=2, duration_seconds=.8, loose_seconds=.5))
    assert [event.event_type for event in events] == [EventType.TRANSFER, EventType.UNKNOWN]
    assert EventType.STEAL not in [event.event_type for event in events]


def test_confirmed_delayed_opponent_recovery_is_turnover():
    events = classify_transition(_semantic_transition(source_team_id=1, destination_team_id=2, duration_seconds=.8, loose_seconds=.5, loss_of_control_evidence=True))
    assert [event.event_type for event in events] == [EventType.TRANSFER, EventType.TURNOVER]


def test_loss_unknown_and_out_of_bounds_fail_closed():
    loss = classify_transition(_semantic_transition(destination_holder=None, destination_team_id=None, reception_evidence=False, loss_of_control_evidence=True))
    assert loss[0].event_type is EventType.LOSS_OF_CONTROL
    unknown = classify_transition(_semantic_transition(source_holder=None, source_team_id=None, release_evidence=False, trajectory_evidence=False))
    assert unknown[0].event_type is EventType.UNKNOWN
    boundary = classify_transition(_semantic_transition(primitive_types=["BALL_RELEASE", "OUT_OF_BOUNDS"]))
    assert boundary[0].event_type is EventType.UNKNOWN


def test_known_destination_without_destination_team_is_unknown():
    events = classify_transition(_semantic_transition(source_team_id=1, destination_team_id=None))
    assert events[-1].event_type is EventType.UNKNOWN
    assert EventType.LOSS_OF_CONTROL not in [event.event_type for event in events]


def test_engine_is_opt_in_and_legacy_metadata_is_unchanged():
    legacy = PipelineConfig(roboflow_api_key="test-key")
    v3 = PipelineConfig(roboflow_api_key="test-key", possession_engine="v3")
    base = {"source": "sam31_v2_observations", "tracknet_enabled": False}
    assert engine_name(legacy) == "legacy"
    assert legacy.possession_v3_enabled is False
    assert _possession_result_config(legacy, base) == base
    assert _possession_result_config(v3, base) == {
        **base, "possession_engine": "v3", "possession_v3_schema_version": 1,
    }


def test_disabled_results_have_no_v3_pointers():
    assert "possession_v3_file" not in PipelineResults(5, {}, {}, RunSummary(), "cpu").to_dict()


def test_contract_fixture_replay_is_ordered_and_repeatable():
    path = Path(__file__).parent / "fixtures" / "possession_v3" / "contracts.json"
    def replay():
        fixture = json.loads(path.read_text(encoding="utf-8"))
        return [asdict(PossessionObservation(row["frame_index"], row["state"], row["evidence_kind"], provenance=Provenance(frame_indices=[row["frame_index"]]))) for row in fixture["observations"]]
    first, second = replay(), replay()
    assert [row["frame_index"] for row in first] == [0, 4, 8]
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(second, sort_keys=True, separators=(",", ":"))
    assert [row["state"] for row in first] == ["NO_POSSESSION", "OBSERVED", "CONTESTED"]


@pytest.mark.parametrize("factory", [
    lambda: PossessionObservation(-1, PossessionState.UNKNOWN, EvidenceKind.UNKNOWN),
    lambda: PossessionObservation(1, "bad", EvidenceKind.UNKNOWN),
    lambda: PossessionObservation(1, PossessionState.UNKNOWN, EvidenceKind.UNKNOWN, schema_version=0),
    lambda: PossessionObservation(1, PossessionState.UNKNOWN, EvidenceKind.UNKNOWN, schema_version=True),
    lambda: PossessionObservation(1, PossessionState.UNKNOWN, EvidenceKind.UNKNOWN, score=2),
    lambda: BallTransition("x", None, None, [], Provenance(), frame_start=-1),
    lambda: BallTransition("x", None, None, "release", Provenance()),
    lambda: BallTransition("x", None, None, b"release", Provenance()),
    lambda: BallTransition("x", None, None, [], Provenance(), confidence=-.1),
    lambda: PossessionEvent("bad", 1, Provenance()),
])
def test_contracts_reject_invalid_values(factory):
    with pytest.raises(ValueError):
        factory()


def test_contracts_reject_invalid_scores_and_provenance():
    with pytest.raises(ValueError):
        CandidateEvidence(None, None, None, score=float("nan"))
    with pytest.raises(ValueError):
        CandidateEvidence(None, None, None, score=None)
    with pytest.raises(ValueError):
        Provenance(source_records="tracks.jsonl#frame=1")
    with pytest.raises(ValueError):
        Provenance(source_records=[""], frame_indices=[True])


@pytest.mark.parametrize("factory", [
    lambda: CandidateEvidence("", 0, "home", .5),
    lambda: CandidateEvidence("p", 1, "home", .5, {"distance": float("inf")}),
    lambda: CandidateEvidence("p", 1, "home", .5, {}, [""]),
    lambda: PossessionObservation(1, "UNKNOWN", "unknown", profile="", thresholds={"x": "bad"}),
    lambda: PossessionObservation(1, "UNKNOWN", "unknown", team_id="home"),
    lambda: BallTransition("x", "", None, ["release"], Provenance()),
    lambda: BallTransition("x", None, None, ["release"], {}),
    lambda: PossessionEvent(EventType.PASS, 1, Provenance(), transition_id="", reason_codes=[""]),
    lambda: PossessionEvent(EventType.PASS, 1, {}),
])
def test_public_contract_fields_reject_malformed_values(factory):
    with pytest.raises(ValueError):
        factory()


def _player(stable, box, team=1, **extra):
    return {"stable_player_id": stable, "raw_track_id": extra.pop("raw_track_id", 1),
            "team_id": team, "bbox": box, "confidence": extra.pop("confidence", 1.0), **extra}


def test_scoring_prefers_clean_mask_and_falls_back_to_dynamic_bbox_margin():
    players = [_player("mask", (0, 0, 10, 20), clean_mask=True, mask_contains=True),
               _player("edge", (20, 0, 30, 20))]
    result = score_candidates(players, {"x": 25, "y": 20, "confidence": .9}, previous_holder="mask")
    assert result[0].stable_player_id == "mask" and result[0].contributions["mask"] > 0
    assert result[1].contributions["bbox"] > 0  # height-scaled margin admits a dribble edge


def test_scoring_rejects_bad_identity_and_persists_competition_margin():
    result = score_candidates([_player("a", (0, 0, 20, 20)), _player("b", (0, 0, 20, 20)),
                               _player(None, None)], {"x": 10, "y": 10})
    assert result[0].score == result[1].score and result[2].rejected_reasons


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_scoring_fails_closed_for_non_finite_confidence(value):
    result = score_candidates([_player("a", (0, 0, 20, 20), confidence=value)], {"x": 10, "y": 10})
    assert result[0].score == 0.0
    assert "non_finite_confidence" in result[0].rejected_reasons


@pytest.mark.parametrize("confidence", [None, "missing"])
def test_scoring_treats_missing_or_null_player_confidence_as_unavailable(confidence):
    player = _player("a", (0, 0, 20, 20))
    if confidence == "missing":
        player.pop("confidence")
    else:
        player["confidence"] = None
    result = score_candidates([player], {"center_xy": [10, 10], "xyxy": [9, 9, 11, 11]})
    assert result[0].score > 0
    assert result[0].contributions["player_confidence"] == 0
    assert "non_finite_confidence" not in result[0].rejected_reasons


def test_scoring_uses_tracknet_center_and_player_xyxy_shapes():
    result = score_candidates(
        [{"stable_player_id": "a", "raw_track_id": 7, "team_id": 1,
          "xyxy": [0, 0, 20, 20], "confidence": None}],
        {"center_xy": [10, 10], "xyxy": [9, 9, 11, 11]},
    )
    assert result[0].score > 0


def test_scoring_wires_ambiguity_margin_into_candidate_reasons():
    result = score_candidates([_player("a", (0, 0, 20, 20)), _player("b", (0, 0, 20, 20))],
                               {"x": 10, "y": 10}, ambiguity_margin=1.0)
    assert all("ambiguous_margin" in candidate.rejected_reasons for candidate in result)


def test_resolver_uses_stable_identity_and_hysteresis_against_one_frame_switch():
    resolver = PossessionResolver(real_fps=10, hysteresis_margin=.15, persistence_seconds=.2)
    a = CandidateEvidence("stable-a", 1, 1, .8, {"geometry": .8})
    b = CandidateEvidence("stable-b", 2, 2, .95, {"geometry": .95})
    assert resolver.resolve(0, [a]).state is PossessionState.OBSERVED
    assert resolver.resolve(1, [b]).stable_player_id == "stable-a"
    assert resolver.resolve(3, [b]).stable_player_id == "stable-b"


def test_resolver_grace_is_real_fps_bounded_and_retroactively_confirmable():
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    assert resolver.resolve(0, [holder]).state is PossessionState.OBSERVED
    assert resolver.resolve(1, [], ball_lost=True).state is PossessionState.INFERRED
    assert resolver.resolve(2, [holder]).state is PossessionState.OBSERVED
    revised = resolver.retroactive_confirm(1, 2)
    assert revised[0].evidence_kind is EvidenceKind.INFERRED
    assert "retroactive_confirmation" in revised[0].provenance.reason_codes


def test_absent_or_invalid_authoritative_input_does_not_grace_carry():
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    resolver.resolve(0, [holder])
    assert resolver.resolve(1, [], missing=True).state is PossessionState.UNKNOWN
    resolver.resolve(2, [holder])
    assert resolver.resolve(3, [], invalid=True).state is PossessionState.UNKNOWN


def test_zero_score_no_possession_clears_holder_before_later_ball_loss():
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    invalid = CandidateEvidence("a", 1, 1, 0.0, {}, ["invalid_ball_geometry"])
    assert resolver.resolve(0, [holder]).state is PossessionState.OBSERVED
    assert resolver.resolve(1, [invalid]).state is PossessionState.NO_POSSESSION
    assert resolver.resolve(2, [], ball_lost=True).state is PossessionState.NO_POSSESSION


def test_raw_fallback_stays_non_authoritative_and_continues_explicitly():
    candidates = score_candidates([_player(None, (0, 0, 20, 20), raw_track_id=7)], {"x": 10, "y": 10})
    assert candidates[0].stable_player_id is None
    assert candidates[0].fallback_identity == "raw:7"
    assert candidates[0].score > 0
    resolver = PossessionResolver(real_fps=10)
    first = resolver.resolve(0, [CandidateEvidence(None, 7, 1, .9, {"geometry": .9}, fallback_identity="raw:7")])
    second = resolver.resolve(1, [CandidateEvidence(None, 8, 1, .9, {"geometry": .9}, fallback_identity="raw:8")])
    assert first.stable_player_id is None and first.raw_track_id == 7
    assert second.state is PossessionState.TRANSITIONING


def test_raw_only_challenger_cannot_switch_stable_holder():
    resolver = PossessionResolver(real_fps=10, hysteresis_margin=.05, persistence_seconds=0.1)
    stable = CandidateEvidence("stable-a", 1, 1, .8, {"geometry": .8})
    raw = CandidateEvidence(None, 9, 1, .99, {"geometry": .99}, fallback_identity="raw:9")
    assert resolver.resolve(0, [stable]).state is PossessionState.OBSERVED
    assert resolver.resolve(1, [raw]).state is PossessionState.CONTESTED
    assert resolver.resolve(3, [raw]).state is PossessionState.CONTESTED
    assert resolver.history[-1].stable_player_id == "stable-a"


@pytest.mark.parametrize("kwargs, expected", [
    ({"missing": True}, PossessionState.UNKNOWN),
    ({"invalid": True}, PossessionState.UNKNOWN),
    ({"ball_lost": True}, PossessionState.INFERRED),
])
def test_public_state_distinguishes_unknown_from_bounded_ball_loss(kwargs, expected):
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    resolver.resolve(0, [holder])
    assert resolver.resolve(1, [], **kwargs).state is expected


@pytest.mark.parametrize("start, end", [(2, 1), (1, 3), (9, 10)])
def test_retroactive_confirmation_rejects_unordered_sparse_or_unknown_bounds(start, end):
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    resolver.resolve(0, [holder])
    resolver.resolve(1, [], ball_lost=True)
    resolver.resolve(2, [holder])
    with pytest.raises(ValueError):
        resolver.retroactive_confirm(start, end)


def test_retroactive_confirmation_requires_unique_later_observed_holder():
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    rival = CandidateEvidence("b", 2, 2, .9, {"geometry": .9})
    resolver.resolve(0, [holder])
    resolver.resolve(1, [], ball_lost=True)
    resolver.resolve(2, [rival])
    with pytest.raises(ValueError):
        resolver.retroactive_confirm(1, 2)


def _ball(frame, x, y, state="detected"):
    return {"frame_index": frame, "timestamp_seconds": frame / 10, "track_id": 1,
            "track_state": state, "confidence": .9 if state != "lost" else None,
            "center_xy": [x, y], "xyxy": [x - 2, y - 2, x + 2, y + 2]}


def _pos(frame, holder):
    return {"frame_index": frame, "stable_player_id": holder}


def test_ball_interpolation_is_bounded_and_lower_quality_without_detection_state():
    rows = normalize_ball_rows([_ball(0, 0, 0), _ball(2, 20, 0)], real_fps=10, max_gap_frames=1)
    middle = next(row for row in rows if row["frame_index"] == 1)
    assert middle["ball_source"] == "interpolated"
    assert middle["track_state"] != "detected"
    assert middle["quality"] < rows[0]["quality"]
    assert middle["source_frame_indices"] == [0, 2]
    assert len(normalize_ball_rows([_ball(0, 0, 0), _ball(3, 30, 0)], real_fps=10, max_gap_frames=1)) == 2


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), 0, -1])
def test_ball_normalization_rejects_non_finite_or_non_positive_fps(fps):
    with pytest.raises(ValueError):
        normalize_ball_rows([_ball(0, 0, 0)], real_fps=fps, max_gap_frames=1)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
def test_ball_normalization_rejects_non_finite_timestamps(timestamp):
    row = _ball(0, 0, 0)
    row["timestamp_seconds"] = timestamp
    with pytest.raises(ValueError):
        normalize_ball_rows([row], real_fps=10, max_gap_frames=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_ball_normalization_fails_closed_for_invalid_quality(value):
    row = _ball(0, 0, 0)
    row["confidence"] = value
    normalized = normalize_ball_rows([row], real_fps=10, max_gap_frames=1)
    assert normalized[0]["ball_source"] == "lost"


@pytest.mark.parametrize("kwargs", [
    {"release_persistence_frames": 0},
    {"release_persistence_frames": -1},
    {"release_persistence_frames": 1.5},
    {"reception_persistence_frames": 0},
    {"reception_persistence_frames": float("nan")},
    {"reception_persistence_frames": True},
])
def test_primitive_derivation_rejects_invalid_persistence_settings(kwargs):
    with pytest.raises(ValueError):
        derive_ball_primitives([], [], real_fps=10, **kwargs)


def test_lost_ball_with_missing_confidence_fails_closed():
    row = _ball(0, 0, 0, state="lost")
    row["confidence"] = None
    normalized = normalize_ball_rows([row], real_fps=10, max_gap_frames=1)
    assert normalized[0]["ball_source"] == "lost"
    assert normalized[0]["quality"] == 0.0


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -float("inf"), -0.1])
def test_primitive_derivation_rejects_invalid_loose_seconds(seconds):
    with pytest.raises(ValueError):
        derive_ball_primitives([], [], real_fps=10, loose_seconds=seconds)


def test_malformed_coordinates_and_frame_indices_cannot_interpolate_or_emit():
    bad_frame = _ball(-1, 0, 0)
    bad_point = _ball(1, float("nan"), 0)
    rows = normalize_ball_rows([_ball(0, 0, 0), bad_frame, bad_point, _ball(2, 20, 0)], real_fps=10, max_gap_frames=2)
    assert [row["frame_index"] for row in rows] == [0, 1, 2]
    assert rows[1]["ball_source"] == "lost"
    assert derive_ball_primitives(rows, [_pos(0, "a"), _pos(1, None), _pos(2, "b")], real_fps=10) == []


def test_reverse_frame_or_timestamp_order_fails_closed_before_interpolation():
    reverse_frame = [_ball(2, 20, 0), _ball(0, 0, 0)]
    with pytest.raises(ValueError):
        normalize_ball_rows(reverse_frame, real_fps=10, max_gap_frames=2)
    reverse_time = [_ball(0, 0, 0), _ball(1, 10, 0)]
    reverse_time[1]["timestamp_seconds"] = -1
    with pytest.raises(ValueError):
        normalize_ball_rows(reverse_time, real_fps=10, max_gap_frames=2)


@pytest.mark.parametrize("field,value", [("source_records", ["ok", 3]), ("source_records", "raw"),
                                          ("source_frame_indices", [0, -1]), ("source_frame_indices", [True])])
def test_malformed_raw_provenance_fails_closed(field, value):
    left, right = _ball(0, 0, 0), _ball(2, 20, 0)
    left[field] = value
    rows = normalize_ball_rows([left, right], real_fps=10, max_gap_frames=1)
    assert all(row["ball_source"] != "interpolated" for row in rows)
    assert derive_ball_primitives(rows, [_pos(0, "a"), _pos(1, None), _pos(2, "b")], real_fps=10) == []


@pytest.mark.parametrize("field", ["source_records", "source_frame_indices"])
def test_empty_supplied_provenance_fails_closed(field):
    row = _ball(0, 0, 0)
    row[field] = []
    assert normalize_ball_rows([row], real_fps=10, max_gap_frames=1)[0]["ball_source"] == "lost"


@pytest.mark.parametrize("derive", [derive_ball_primitives, derive_ball_transitions])
def test_direct_derivation_rejects_duplicate_frames(derive):
    rows = normalize_ball_rows([_ball(0, 0, 0), _ball(1, 10, 0)], real_fps=10, max_gap_frames=1)
    with pytest.raises(ValueError):
        derive(rows + [dict(rows[0])], [_pos(0, "a"), _pos(1, None)], real_fps=10)


@pytest.mark.parametrize("derive", [derive_ball_primitives, derive_ball_transitions])
def test_direct_derivation_rejects_reverse_order(derive):
    for rows in ([_ball(1, 10, 0), _ball(0, 0, 0)], [_ball(0, 0, 0), _ball(1, 10, 0)], [_ball(0, 0, 0), _ball(1, 10, 0), _ball(2, 20, 0)]):
        if len(rows) == 2:
            rows[1]["timestamp_seconds"] = -1
        else:
            rows[0]["timestamp_seconds"], rows[1]["timestamp_seconds"], rows[2]["timestamp_seconds"] = 2, None, 1
        with pytest.raises(ValueError):
            derive(rows, [], real_fps=10)


def test_ball_normalization_preserves_supplied_provenance():
    left, right = _ball(0, 0, 0), _ball(2, 20, 0)
    left["source_records"] = ["raw/custom#0"]
    left["source_frame_indices"] = [100]
    rows = normalize_ball_rows([left, right], real_fps=10, max_gap_frames=1)
    assert rows[0]["source_records"] == ["raw/custom#0"]
    assert rows[0]["source_frame_indices"] == [100]
    assert rows[1]["source_records"] == ["raw/custom#0", "ball_tracks.jsonl#frame=2"]


def test_release_requires_persistent_separation_and_transit_has_trajectory():
    rows = normalize_ball_rows([_ball(i, i * 10, 0) for i in range(5)], real_fps=10, max_gap_frames=1)
    rows[1]["timestamp_seconds"] = None
    primitives = derive_ball_primitives(rows, [_pos(0, "a"), _pos(1, None), _pos(2, None), _pos(3, "b"), _pos(4, "b")], real_fps=10)
    assert [item["primitive_type"] for item in primitives] == ["BALL_RELEASE", "BALL_IN_TRANSIT", "BALL_RECEPTION"]
    assert primitives[1]["provenance"]["frame_indices"] == [1, 2]
    assert derive_ball_transitions(rows, [_pos(0, "a"), _pos(1, None), _pos(2, None), _pos(3, "b"), _pos(4, "b")], real_fps=10)


def test_ball_transitions_emits_every_qualifying_transition():
    rows = normalize_ball_rows([_ball(i, i * 10, 0) for i in range(10)], real_fps=10, max_gap_frames=1)
    states = [_pos(i, "a" if i == 0 else "b" if i in (3, 4) else "c" if i == 5 else "d" if i in (8, 9) else None)
              for i in range(10)]
    transitions = derive_ball_transitions(rows, states, real_fps=10)
    assert [transition.transition_id for transition in transitions] == ["ball-transition-1", "ball-transition-6"]


def test_event_window_rejects_over_window_but_keeps_in_window_transition():
    rows = normalize_ball_rows([_ball(i, i * 10, 0) for i in range(5)], real_fps=10, max_gap_frames=1)
    states = [_pos(0, "a"), _pos(1, None), _pos(2, None), _pos(3, "b"), _pos(4, "b")]
    assert derive_ball_transitions(rows, states, real_fps=10, event_window_seconds=.2) == []
    assert len(derive_ball_transitions(rows, states, real_fps=10, event_window_seconds=.5)) == 1


def test_near_edge_ball_transition_without_court_evidence_stays_unknown():
    rows = normalize_ball_rows([_ball(0, 1, 40), _ball(1, 2, 40), _ball(2, 3, 40),
                                _ball(3, 4, 40), _ball(4, 5, 40)], real_fps=10, max_gap_frames=1)
    states = [
        {"frame_index": 0, "stable_player_id": "a", "team_id": 1},
        {"frame_index": 1, "stable_player_id": None, "team_id": None},
        {"frame_index": 2, "stable_player_id": None, "team_id": None},
        {"frame_index": 3, "stable_player_id": "b", "team_id": 2},
        {"frame_index": 4, "stable_player_id": "b", "team_id": 2},
    ]

    transitions = derive_ball_transitions(rows, states, real_fps=10)

    assert len(transitions) == 1
    assert all("OUT_OF_BOUNDS" not in transition.primitive_types for transition in transitions)
    assert classify_transition(transitions[0])[-1].event_type is EventType.UNKNOWN


def test_primitives_preserve_available_source_timestamp_ranges():
    rows = normalize_ball_rows([_ball(i, i * 10, 0) for i in range(5)], real_fps=10, max_gap_frames=1)
    primitives = derive_ball_primitives(rows, [_pos(0, "a"), _pos(1, None), _pos(2, None), _pos(3, "b"), _pos(4, "b")], real_fps=10)
    assert all(item["timestamp_start"] is not None and item["timestamp_end"] is not None for item in primitives)
    assert primitives[0]["timestamp_start"] == 0.0
    assert primitives[-1]["timestamp_end"] == 0.4


def test_single_destination_frame_does_not_confirm_reception():
    rows = normalize_ball_rows([_ball(i, i * 10, 0) for i in range(3)], real_fps=10, max_gap_frames=1)
    primitives = derive_ball_primitives(rows, [_pos(0, "a"), _pos(1, None), _pos(2, "b")], real_fps=10)
    assert [item["primitive_type"] for item in primitives] == []


def test_all_qualifying_loose_intervals_are_emitted():
    rows = normalize_ball_rows([_ball(i, 0, 0) for i in range(10)], real_fps=10, max_gap_frames=1)
    states = [_pos(i, None if 0 <= i <= 3 or 6 <= i <= 9 else "a") for i in range(10)]
    loose = derive_ball_primitives(rows, states, real_fps=10, loose_seconds=.3)
    assert [item["frame_index"] for item in loose if item["primitive_type"] == "LOOSE_BALL"] == [3, 9]


def test_single_frame_noise_does_not_emit_primitive():
    rows = normalize_ball_rows([_ball(0, 0, 0), _ball(1, 100, 100), _ball(2, 0, 0)], real_fps=10, max_gap_frames=1)
    assert derive_ball_primitives(rows, [_pos(0, "a"), _pos(1, None), _pos(2, "a")], real_fps=10) == []
