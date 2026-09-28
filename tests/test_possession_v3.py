"""Focused V3 contract, compatibility, and replay tests."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from pipeline.config import PipelineConfig
from pipeline.possession_v3 import (
    BallTransition, CandidateEvidence, EvidenceKind, EventType, PossessionEvent,
    PossessionObservation, PossessionState, Provenance, PossessionContext,
    PossessionAudit, ClearReason, V3_CONTRACT_VERSION,
    engine_name,
    score_candidates, PossessionResolver,
    normalize_ball_rows, derive_ball_primitives, derive_ball_transitions,
    classify_transition,
    debounce_reverted_switches,
)
from pipeline.results import PipelineResults, RunSummary
from pipeline.runner import _build_v3_artifacts, _possession_result_config


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


def test_continuity_context_and_audit_are_additive_and_deterministic():
    context = PossessionContext(
        effective_holder_id="player-006", effective_holder_team=1,
        observed_candidate_id="player-007", challenger_id="player-007",
        challenger_start_frame=20, challenger_confirmation_frame=24,
        grace_start_frame=18, grace_remaining_seconds=.4,
        last_observed_holder_frame=17, last_observed_holder_score=.91,
    )
    audit = PossessionAudit(
        last_holder_id="player-006", last_holder_team=1,
        clear_reason=ClearReason.GRACE_EXPIRED, switch_confirmed=False,
        observation_quality="inferred", observation_source="ball_tracks.jsonl",
        authority="stable", grace_eligible=True, grace_elapsed_seconds=.2,
    )
    observation = PossessionObservation(
        20, PossessionState.INFERRED, EvidenceKind.INFERRED,
        context=context, audit=audit,
    )
    payload = asdict(observation)
    assert payload["context"]["challenger_confirmation_frame"] == 24
    assert payload["context"]["grace_remaining_seconds"] == .4
    assert payload["audit"]["clear_reason"] == "GRACE_EXPIRED"
    assert payload["audit"]["observation_source"] == "ball_tracks.jsonl"
    assert payload["audit"]["authority"] == "stable"
    assert payload["audit"]["grace_eligible"] is True
    assert payload["audit"]["grace_elapsed_seconds"] == .2


@pytest.mark.parametrize("reason", list(ClearReason))
def test_all_clear_reasons_are_serializable(reason):
    payload = asdict(PossessionAudit(clear_reason=reason))
    assert payload["clear_reason"] == reason.value


def test_non_null_audit_payload_round_trips_through_json():
    observation = PossessionObservation(
        20, PossessionState.INFERRED, EvidenceKind.INFERRED,
        audit=PossessionAudit(
            last_holder_id="player-006", last_holder_team=1,
            clear_reason=ClearReason.GRACE_EXPIRED, switch_confirmed=True,
            observation_quality="inferred", observation_source="ball_tracks.jsonl",
            authority="stable", grace_eligible=True, grace_elapsed_seconds=.2,
        ),
    )
    decoded = json.loads(json.dumps(asdict(observation), sort_keys=True))
    audit = decoded["audit"]
    assert audit["clear_reason"] == "GRACE_EXPIRED"
    assert audit["last_holder_id"] == "player-006"
    assert audit["switch_confirmed"] is True
    assert audit["authority"] == "stable"
    assert audit["grace_eligible"] is True
    assert audit["grace_elapsed_seconds"] == .2


def test_old_observation_construction_keeps_deterministic_continuity_defaults():
    payload = asdict(PossessionObservation(1, PossessionState.UNKNOWN, EvidenceKind.UNKNOWN))
    assert payload["context"] == asdict(PossessionContext())
    assert payload["audit"] == asdict(PossessionAudit())


@pytest.mark.parametrize("factory", [
    lambda: PossessionContext(effective_holder_id="p", effective_holder_team=-1),
    lambda: PossessionContext(challenger_start_frame=2, challenger_id=None),
    lambda: PossessionContext(challenger_elapsed_seconds=.1, challenger_id=None),
    lambda: PossessionContext(challenger_confirmation_frame=2, challenger_id=None),
    lambda: PossessionContext(grace_remaining_seconds=-.1),
    lambda: PossessionContext(last_observed_holder_score=1.1),
    lambda: PossessionAudit(clear_reason="NOT_A_REASON"),
    lambda: PossessionAudit(switch_confirmed="yes"),
    lambda: PossessionAudit(observation_quality=""),
    lambda: PossessionAudit(observation_source=""),
    lambda: PossessionAudit(observation_quality=None),
    lambda: PossessionAudit(observation_source=None),
    lambda: PossessionAudit(grace_eligible="yes"),
    lambda: PossessionAudit(grace_elapsed_seconds=-.1),
    lambda: PossessionAudit(grace_elapsed_seconds=float("nan")),
])
def test_continuity_contract_rejects_invalid_state_or_reason(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("authority", [None, "", 123, {"value": "unsupported"}],
                         ids=["none", "empty", "non-string", "unsupported"])
def test_audit_rejects_invalid_authority_values(authority):
    with pytest.raises(ValueError):
        PossessionAudit(authority=authority)


def test_audit_defaults_keep_legacy_observation_construction_serializable():
    audit = asdict(PossessionAudit())
    assert audit["observation_quality"] == "unknown"
    assert audit["observation_source"] == "unknown"
    assert audit["authority"] == "unknown"
    assert audit["grace_eligible"] is False
    assert audit["grace_elapsed_seconds"] == 0.0


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
        "possession_v3_reverted_switch_window_seconds": .5,
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


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_scoring_fails_closed_for_out_of_range_confidence(value):
    result = score_candidates([_player("a", (0, 0, 20, 20), confidence=value)], {"x": 10, "y": 10})
    assert result[0].score == 0.0
    assert "out_of_range_confidence" in result[0].rejected_reasons


@pytest.mark.parametrize("confidence", [None, "missing"])
def test_scoring_treats_missing_or_null_player_confidence_as_unavailable(confidence):
    player = _player("a", (0, 0, 20, 20))
    if confidence == "missing":
        player.pop("confidence")
    else:
        player["confidence"] = None
    result = score_candidates([player], {"center_xy": [10, 10], "xyxy": [9, 9, 11, 11]})
    assert result[0].score > 0.5
    assert result[0].contributions["player_confidence"] == 0
    assert "non_finite_confidence" not in result[0].rejected_reasons


def test_scoring_uses_tracknet_center_and_player_xyxy_shapes():
    result = score_candidates(
        [{"stable_player_id": "a", "raw_track_id": 7, "team_id": 1,
          "xyxy": [0, 0, 20, 20], "confidence": None}],
        {"center_xy": [10, 10], "xyxy": [9, 9, 11, 11]},
    )
    assert result[0].score > 0


@pytest.mark.parametrize("mask_contains", ["yes", {"hit": True}, 1, [True]])
def test_scoring_ignores_malformed_mask_hits(mask_contains):
    player = _player("a", (0, 0, 20, 20), clean_mask=True, mask_contains=mask_contains)
    result = score_candidates([player], {"center_xy": [10, 10], "confidence": .9})
    assert result[0].score < 1.0
    assert result[0].contributions["mask"] == 0


@pytest.mark.parametrize("mask_contains", [lambda _x, _y: "yes", lambda _x, _y: {"hit": True}])
def test_scoring_ignores_malformed_callable_mask_results(mask_contains):
    player = _player("a", (0, 0, 20, 20), clean_mask=True, mask_contains=mask_contains)
    result = score_candidates([player], {"center_xy": [10, 10], "confidence": .9})
    assert result[0].contributions["mask"] == 0


@pytest.mark.parametrize("mask_contains", [lambda _x, _y: True, lambda _x, _y: False])
def test_scoring_accepts_boolean_callable_mask_results(mask_contains):
    player = _player("a", (0, 0, 20, 20), clean_mask=True, mask_contains=mask_contains)
    result = score_candidates([player], {"center_xy": [10, 10], "confidence": .9})
    assert (result[0].contributions["mask"] > 0) is mask_contains(10, 10)


def test_real_shapes_without_masks_remain_unresolved_when_geometry_is_ambiguous():
    players = [
        {"stable_player_id": "a", "raw_track_id": 1, "team_id": 1,
         "xyxy": [0, 0, 20, 20], "confidence": None},
        {"stable_player_id": "b", "raw_track_id": 2, "team_id": 2,
         "xyxy": [0, 0, 20, 20], "confidence": None},
    ]
    candidates = score_candidates(players, {"center_xy": [10, 10], "xyxy": [9, 9, 11, 11]})
    resolver = PossessionResolver(real_fps=4, persistence_seconds=.5)
    assert resolver.resolve(0, candidates).state is PossessionState.CONTESTED
    assert resolver.resolve(1, candidates).state is PossessionState.CONTESTED
    assert resolver.history[-1].stable_player_id is None


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


def test_continuity_separates_effective_observed_and_challenger_for_006_007():
    resolver = PossessionResolver(real_fps=10, persistence_seconds=.3)
    holder = CandidateEvidence("player-006", 6, 1, .9, {"geometry": .9})
    rival = CandidateEvidence("player-007", 7, 1, .95, {"geometry": .95})
    assert resolver.resolve(0, [holder]).state is PossessionState.OBSERVED
    observation = resolver.resolve(1, [rival])
    assert observation.state is PossessionState.TRANSITIONING
    assert observation.context.effective_holder_id == "player-006"
    assert observation.context.observed_candidate_id == "player-007"
    assert observation.context.challenger_id == "player-007"
    assert observation.stable_player_id == "player-006"


def _resolved_row(frame, holder, timestamp, *, state=PossessionState.OBSERVED):
    return PossessionObservation(
        frame, state, EvidenceKind.OBSERVED,
        stable_player_id=holder,
        timestamp_seconds=timestamp,
        context=PossessionContext(effective_holder_id=holder),
    )


@pytest.mark.parametrize("return_time,should_revert", [(0.8, True), (0.81, False)])
def test_offline_reverted_switch_uses_real_timestamp_boundary(return_time, should_revert):
    rows = [_resolved_row(0, "a", 0.0), _resolved_row(3, "b", 0.3), _resolved_row(5, "a", return_time)]
    result = debounce_reverted_switches(rows, real_fps=10, window_seconds=.5)
    assert result[1].context.effective_holder_id == ("a" if should_revert else "b")
    if should_revert:
        assert "TRANSIENT_SWITCH" in result[1].audit.provenance.reason_codes
        assert "reverted_switch" in result[1].audit.provenance.reason_codes
        assert result[1].audit.provenance.frame_indices == [3, 5]
        assert result[1].audit.reverted_switch_duration_seconds == .5


def test_offline_reverted_switch_preserves_raw_candidate_and_unknown_gap():
    rows = [
        _resolved_row(0, "a", 0.0),
        _resolved_row(3, "b", 0.3),
        _resolved_row(4, None, 0.4, state=PossessionState.UNKNOWN),
        _resolved_row(5, "a", 0.5),
    ]
    result = debounce_reverted_switches(rows, real_fps=10, window_seconds=.5)
    assert result[1].stable_player_id == "b"
    assert result[1].context.effective_holder_id == "b"
    assert result[1].audit.provenance.reason_codes == []


def test_reverted_switch_audit_keeps_raw_state_and_artifact_events_use_effective_holder(tmp_path):
    rows = [_resolved_row(0, "a", 0.0), _resolved_row(3, "b", 0.3), _resolved_row(8, "a", 0.8)]
    result = debounce_reverted_switches(rows, real_fps=10, window_seconds=.5)
    assert result[1].state is PossessionState.OBSERVED
    assert result[1].stable_player_id == "b"
    assert result[1].context.effective_holder_id == "a"
    ball_rows = [{"frame_index": frame, "timestamp_seconds": frame / 10, "track_state": "detected",
                  "ball_source": "detected", "quality": 1.0, "center_xy": [frame, 0],
                  "source_records": [f"ball#{frame}"], "source_frame_indices": [frame]}
                 for frame in (0, 3, 8)]
    assert derive_ball_transitions(ball_rows, result, real_fps=10) == []


def test_runner_artifact_boundary_collapses_dense_transient_switch(tmp_path):
    config = PipelineConfig(roboflow_api_key="test-key", possession_engine="v3")
    rows = []
    ball_rows = []
    for frame in range(10):
        ball_x = 5 if frame <= 2 or frame >= 6 else 25
        rows.extend([
            {"frame_index": frame, "stable_player_id": "a", "raw_track_id": 1, "team_id": 1,
             "bbox": [0, 0, 10, 10], "confidence": 1.0},
            {"frame_index": frame, "stable_player_id": "b", "raw_track_id": 2, "team_id": 2,
             "bbox": [20, 0, 30, 10], "confidence": 1.0},
        ])
        ball_rows.append({
            "frame_index": frame, "timestamp_seconds": frame / 10,
            "track_state": "detected", "center_xy": [ball_x, 5], "confidence": 1.0,
            "source_records": [f"ball#{frame}"], "source_frame_indices": [frame],
        })

    manifest = _build_v3_artifacts(tmp_path, rows, ball_rows, fps=10, config=config)
    assert manifest["config"]["reverted_switch_window_seconds"] == .5
    possession_path = tmp_path / manifest["artifacts"]["possession"]["path"]
    event_path = tmp_path / manifest["artifacts"]["events"]["path"]
    possession = [json.loads(line) for line in possession_path.read_text().splitlines()]
    events = [json.loads(line) for line in event_path.read_text().splitlines()]

    transient = [row for row in possession if row["stable_player_id"] == "b"]
    assert transient and all(row["context"]["effective_holder_id"] == "a" for row in transient)
    assert all(row["state"] in {"OBSERVED", "TRANSITIONING"} for row in transient)
    assert all({"TRANSIENT_SWITCH", "reverted_switch"} <= set(row["audit"]["provenance"]["reason_codes"])
               for row in transient)
    assert all({4, 8} <= set(row["audit"]["provenance"]["frame_indices"]) for row in transient)
    assert all(row["audit"]["reverted_switch_duration_seconds"] == .4 for row in transient)
    assert all(event["event_type"] not in {"PASS", "STEAL", "TURNOVER", "LOSS_OF_CONTROL"} for event in events)


def test_runner_lost_tracknet_row_uses_grace_and_allows_subsequent_transition(tmp_path):
    config = PipelineConfig(roboflow_api_key="test-key", possession_engine="v3")
    rows = []
    for frame in range(5):
        rows.extend([
            {"frame_index": frame, "stable_player_id": "a", "raw_track_id": 1, "team_id": 1,
             "bbox": [0, 0, 10, 10], "confidence": 1.0},
            {"frame_index": frame, "stable_player_id": "b", "raw_track_id": 2, "team_id": 2,
             "bbox": [20, 0, 30, 10], "confidence": 1.0},
        ])
    ball_rows = [
        {"frame_index": 0, "timestamp_seconds": 0.0, "track_state": "detected",
         "confidence": 1.0, "center_xy": [5, 5], "source_records": ["ball#0"],
         "source_frame_indices": [0]},
        {"frame_index": 1, "timestamp_seconds": 0.1, "track_state": "lost",
         "confidence": None, "center_xy": None, "source_records": ["ball#1"],
         "source_frame_indices": [1]},
        {"frame_index": 2, "timestamp_seconds": 0.2, "track_state": "detected",
         "confidence": 1.0, "center_xy": [25, 5], "source_records": ["ball#2"],
         "source_frame_indices": [2]},
        {"frame_index": 3, "timestamp_seconds": 0.3, "track_state": "detected",
         "confidence": 1.0, "center_xy": [25, 5], "source_records": ["ball#3"],
         "source_frame_indices": [3]},
        {"frame_index": 4, "timestamp_seconds": 0.4, "track_state": "detected",
         "confidence": 1.0, "center_xy": [25, 5], "source_records": ["ball#4"],
         "source_frame_indices": [4]},
    ]

    manifest = _build_v3_artifacts(tmp_path, rows, ball_rows, fps=10, config=config)
    possession_path = tmp_path / manifest["artifacts"]["possession"]["path"]
    possession = [json.loads(line) for line in possession_path.read_text().splitlines()]

    gap = next(row for row in possession if row["frame_index"] == 1)
    assert gap["state"] == "INFERRED"
    assert gap["context"]["effective_holder_id"] == "a"
    switched = next(row for row in possession if row["frame_index"] == 4)
    assert switched["state"] == "OBSERVED"
    assert switched["context"]["effective_holder_id"] == "b"
    assert switched["audit"]["switch_confirmed"] is True


def test_runner_malformed_ball_geometry_clears_holder_fail_closed(tmp_path):
    config = PipelineConfig(roboflow_api_key="test-key", possession_engine="v3")
    rows = [{"frame_index": frame, "stable_player_id": "a", "raw_track_id": 1, "team_id": 1,
             "bbox": [0, 0, 10, 10], "confidence": 1.0} for frame in range(2)]
    ball_rows = [
        {"frame_index": 0, "timestamp_seconds": 0.0, "track_state": "detected",
         "confidence": 1.0, "center_xy": [5, 5]},
        {"frame_index": 1, "timestamp_seconds": 0.1, "track_state": "detected",
         "confidence": 1.0, "center_xy": [float("nan"), 5]},
    ]

    manifest = _build_v3_artifacts(tmp_path, rows, ball_rows, fps=10, config=config)
    possession_path = tmp_path / manifest["artifacts"]["possession"]["path"]
    possession = [json.loads(line) for line in possession_path.read_text().splitlines()]
    malformed = next(row for row in possession if row["frame_index"] == 1)
    assert malformed["state"] == "UNKNOWN"
    assert malformed["context"]["effective_holder_id"] is None
    assert "invalid_evidence" in malformed["provenance"]["reason_codes"]


def test_exact_006_007_timeline_preserves_then_confirms_switch():
    resolver = PossessionResolver(real_fps=20, persistence_seconds=.15)
    holder = CandidateEvidence("player-006", 6, 1, .9, {"geometry": .9})
    rival = CandidateEvidence("player-007", 7, 1, .95, {"geometry": .95})
    tie = CandidateEvidence("player-008", 8, 2, .94, {"geometry": .94})

    assert resolver.resolve(0, [holder]).state is PossessionState.OBSERVED
    assert resolver.resolve(1, [], ball_lost=True).context.effective_holder_id == "player-006"
    assert resolver.history[-1].state is PossessionState.INFERRED
    unknown = resolver.resolve(2, [])
    assert unknown.state is PossessionState.INFERRED
    assert unknown.context.effective_holder_id == "player-006"
    contested = resolver.resolve(3, [rival, tie])
    assert contested.state is PossessionState.CONTESTED
    assert contested.context.effective_holder_id == "player-006"
    assert contested.context.challenger_id is None

    first_challenge = resolver.resolve(4, [rival])
    assert first_challenge.state is PossessionState.TRANSITIONING
    assert first_challenge.context.challenger_id == "player-007"
    assert first_challenge.context.challenger_start_frame == 4
    cancelled = resolver.resolve(5, [], ball_lost=True)
    assert cancelled.context.effective_holder_id == "player-006"
    assert cancelled.context.challenger_id is None

    restarted = resolver.resolve(6, [rival])
    continued = resolver.resolve(7, [rival])
    assert restarted.context.challenger_start_frame == continued.context.challenger_start_frame == 6
    assert continued.context.challenger_elapsed_seconds == .1
    assert continued.context.effective_holder_id == "player-006"
    confirmed = resolver.resolve(8, [rival])
    assert confirmed.state is PossessionState.OBSERVED
    assert confirmed.context.effective_holder_id == "player-007"
    assert confirmed.context.challenger_id is None
    assert confirmed.audit.switch_confirmed is True


def test_unknown_contested_and_ball_loss_preserve_holder_until_real_fps_grace_expires():
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    resolver.resolve(0, [holder])
    unknown = resolver.resolve(1, [], ball_lost=False)
    contested = resolver.resolve(2, [CandidateEvidence("b", 2, 2, .9, {"geometry": .9}),
                                    CandidateEvidence("c", 3, 2, .9, {"geometry": .9})])
    lost = resolver.resolve(3, [], ball_lost=True)
    expired = resolver.resolve(6, [], ball_lost=True)
    assert unknown.state is PossessionState.INFERRED
    assert contested.state is PossessionState.CONTESTED
    assert lost.state is PossessionState.INFERRED
    assert lost.context.effective_holder_id == "a"
    assert expired.state is PossessionState.LOST_NO_DETECTION
    assert expired.context.effective_holder_id is None
    assert expired.audit.last_holder_id == "a"
    assert expired.audit.clear_reason is ClearReason.LOST_NO_DETECTION


def test_repeated_contested_frames_expire_holder_grace():
    resolver = PossessionResolver(real_fps=10, hysteresis_margin=.1)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    rival = CandidateEvidence("b", 2, 2, .9, {"geometry": .9})
    tie = CandidateEvidence("c", 3, 2, .9, {"geometry": .9})
    resolver.resolve(0, [holder])
    states = [resolver.resolve(frame, [rival, tie]) for frame in range(1, 7)]
    assert [observation.state for observation in states[:-1]] == [PossessionState.CONTESTED] * 5
    expired = states[-1]
    assert expired.state is PossessionState.LOST_NO_DETECTION
    assert expired.context.effective_holder_id is None
    assert expired.audit.last_holder_id == "a"
    assert expired.audit.clear_reason is ClearReason.LOST_NO_DETECTION


def test_challenger_cancels_when_evidence_disappears_and_failed_transition_keeps_holder():
    resolver = PossessionResolver(real_fps=10, persistence_seconds=.4)
    a = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    b = CandidateEvidence("b", 2, 2, .95, {"geometry": .95})
    resolver.resolve(0, [a])
    pending = resolver.resolve(1, [b])
    cancelled = resolver.resolve(2, [])
    assert pending.context.challenger_id == "b"
    assert cancelled.context.challenger_id is None
    assert cancelled.context.effective_holder_id == "a"
    assert resolver.resolve(3, [b]).context.challenger_id == "b"
    assert resolver.resolve(4, [a]).context.challenger_id is None


def test_canceled_challenger_window_is_empty_when_next_evidence_is_not_a_challenger():
    resolver = PossessionResolver(real_fps=10, persistence_seconds=.4)
    holder = CandidateEvidence("holder", 1, 1, .9, {"geometry": .9})
    challenger = CandidateEvidence("challenger", 2, 2, .95, {"geometry": .95})
    weak = CandidateEvidence("weak", 3, 2, .2, {"geometry": .2})

    resolver.resolve(0, [holder])
    resolver.resolve(1, [challenger])
    canceled = resolver.resolve(2, [weak])

    assert canceled.context.challenger_id is None
    assert canceled.context.challenger_start_frame is None
    assert canceled.context.challenger_confirmation_frame is None
    assert canceled.context.challenger_elapsed_seconds == 0.0


def test_observed_requires_keep_threshold_and_confirmed_switch_is_observed():
    resolver = PossessionResolver(real_fps=10, persistence_seconds=.2, keep_threshold=.5)
    current = CandidateEvidence("current", 1, 1, .8, {"geometry": .8})
    weak = CandidateEvidence("weak", 3, 1, .49, {"geometry": .49})
    strong = CandidateEvidence("strong", 2, 1, .8, {"geometry": .8})
    assert resolver.resolve(0, [current]).state is PossessionState.OBSERVED
    weak_observation = resolver.resolve(1, [weak])
    assert weak_observation.state is PossessionState.TRANSITIONING
    assert weak_observation.stable_player_id == "current"
    assert weak_observation.score == .49
    assert resolver.resolve(2, [strong]).state is PossessionState.TRANSITIONING
    switched = resolver.resolve(4, [strong])
    assert switched.state is PossessionState.OBSERVED
    assert switched.context.effective_holder_id == "strong"
    assert switched.score >= .5


def test_delayed_switch_audit_is_visible_on_resolve_with_full_window():
    resolver = PossessionResolver(real_fps=10, persistence_seconds=.2)
    a = CandidateEvidence("player-006", 6, 1, .9, {"geometry": .9})
    b = CandidateEvidence("player-007", 7, 1, .95, {"geometry": .95})
    resolver.resolve(0, [a], timestamp=10.0)
    pending = resolver.resolve(1, [b], timestamp=10.1)
    assert pending.context.effective_holder_id == "player-006"
    resolver.resolve(2, [b], timestamp=10.2)
    switched = resolver.resolve(3, [b], timestamp=10.3)
    assert switched.context.effective_holder_id == "player-007"
    audit = switched.audit
    assert audit.previous_holder_id == "player-006"
    assert audit.new_holder_id == "player-007"
    assert audit.challenger_start_frame == 1
    assert audit.challenger_confirmation_frame == 3
    assert audit.frame_start == 1 and audit.frame_end == 3
    assert audit.timestamp_start == 10.1 and audit.timestamp_end == 10.3
    assert resolver.resolve(4, [], invalid=True).context.effective_holder_id is None


def test_grace_expiry_audit_contains_holder_to_clear_interval():
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("player-006", 6, 1, .9, {"geometry": .9})
    resolver.resolve(0, [holder], timestamp=20.0)
    expired = resolver.resolve(6, [], ball_lost=True, timestamp=20.6)
    assert expired.state is PossessionState.LOST_NO_DETECTION
    assert expired.audit.frame_start == 0
    assert expired.audit.frame_end == 6
    assert expired.audit.timestamp_start == 20.0
    assert expired.audit.timestamp_end == 20.6


@pytest.mark.parametrize("reason", ["bad", ClearReason.LOST_NO_DETECTION])
def test_clear_effective_holder_rejects_unsupported_reasons(reason):
    resolver = PossessionResolver(real_fps=10)
    with pytest.raises(ValueError, match="clear reason"):
        resolver.clear_effective_holder(reason)


def test_confirm_switch_rejects_raw_only_candidate_against_stable_holder():
    resolver = PossessionResolver(real_fps=10)
    stable = CandidateEvidence("player-006", 6, 1, .9, {"geometry": .9})
    raw = CandidateEvidence(None, 7, 1, .95, {"geometry": .95}, fallback_identity="raw:7")
    resolver.resolve(0, [stable])
    with pytest.raises(ValueError, match="stable identity"):
        resolver.confirm_switch(raw, frame_index=1)
    assert resolver.holder == "player-006"


def test_invalid_evidence_cannot_silently_change_holder_or_emit_observed_below_threshold():
    resolver = PossessionResolver(real_fps=10, keep_threshold=.5)
    holder = CandidateEvidence("player-006", 6, 1, .9, {"geometry": .9})
    weak = CandidateEvidence("player-007", 7, 1, .49, {"geometry": .49})
    assert resolver.resolve(0, [holder]).context.effective_holder_id == "player-006"
    observation = resolver.resolve(1, [weak])
    assert observation.state is not PossessionState.OBSERVED
    assert observation.score < resolver.keep_threshold
    assert observation.context.effective_holder_id == "player-006"
    cleared = resolver.resolve(2, [CandidateEvidence("player-007", 7, 1, 0.0, {})])
    assert cleared.context.effective_holder_id is None
    assert cleared.audit.clear_reason is ClearReason.LOOSE_BALL_CONFIRMED


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf"), -.01, 1.01, True, "0.5"])
def test_resolver_rejects_invalid_keep_threshold(threshold):
    with pytest.raises(ValueError, match="keep_threshold"):
        PossessionResolver(real_fps=10, keep_threshold=threshold)


def test_resolver_accepts_keep_threshold_edges():
    assert PossessionResolver(real_fps=10, keep_threshold=0).keep_threshold == 0
    assert PossessionResolver(real_fps=10, keep_threshold=1).keep_threshold == 1


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
def test_resolver_invalid_timestamp_fails_closed_without_negative_elapsed(timestamp):
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    resolver.resolve(0, [holder], timestamp=1.0)
    observation = resolver.resolve(1, [holder], timestamp=timestamp)
    assert observation.state is PossessionState.UNKNOWN
    assert observation.context.effective_holder_id is None
    assert observation.audit.grace_elapsed_seconds == 0


def test_resolver_negative_timestamp_fails_closed_without_audit_construction_error():
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    resolver.resolve(0, [holder], timestamp=1.0)
    observation = resolver.resolve(1, [holder], timestamp=-.5)
    assert observation.state is PossessionState.UNKNOWN
    assert observation.context.effective_holder_id is None
    assert "invalid_timestamp" in observation.provenance.reason_codes
    assert observation.audit.grace_elapsed_seconds == 0


def test_resolver_reverse_timestamp_fails_closed_without_negative_elapsed():
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    resolver.resolve(0, [holder], timestamp=1.0)
    observation = resolver.resolve(1, [holder], timestamp=.5)
    assert observation.state is PossessionState.UNKNOWN
    assert observation.context.effective_holder_id is None
    assert observation.audit.grace_elapsed_seconds == 0


def test_missing_timestamp_preserves_finite_ordering_baseline():
    resolver = PossessionResolver(real_fps=10)
    holder = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    resolver.resolve(0, [holder], timestamp=10.0)
    resolver.resolve(1, [holder])
    observation = resolver.resolve(2, [holder], timestamp=9.0)
    assert observation.state is PossessionState.UNKNOWN
    assert observation.context.effective_holder_id is None
    assert "reverse_timestamp" in observation.provenance.reason_codes


def test_terminal_input_clears_without_grace_and_does_not_retain_challenger():
    resolver = PossessionResolver(real_fps=10)
    a = CandidateEvidence("a", 1, 1, .9, {"geometry": .9})
    b = CandidateEvidence("b", 2, 2, .95, {"geometry": .95})
    resolver.resolve(0, [a])
    resolver.resolve(1, [b])
    cleared = resolver.resolve(2, [], invalid=True)
    assert cleared.state is PossessionState.UNKNOWN
    assert cleared.context.effective_holder_id is None
    assert cleared.context.challenger_id is None
    assert cleared.audit.last_holder_id == "a"


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


def _grace_observation(frame, state, effective, observed, team):
    return PossessionObservation(
        frame, state, EvidenceKind.OBSERVED if state is PossessionState.OBSERVED else EvidenceKind.INFERRED,
        team_id=team,
        context=PossessionContext(effective_holder_id=effective, effective_holder_team=1,
                                  observed_candidate_id=observed),
    )


def test_grace_candidate_context_produces_authoritative_transition():
    rows = normalize_ball_rows([_ball(i, i * 10, 0) for i in range(5)], real_fps=10, max_gap_frames=1)
    states = [
        _grace_observation(0, PossessionState.OBSERVED, "a", "a", 1),
        _grace_observation(1, PossessionState.INFERRED, "a", "b", 1),
        _grace_observation(2, PossessionState.INFERRED, "a", "b", 1),
        _grace_observation(3, PossessionState.OBSERVED, "a", "b", 2),
        _grace_observation(4, PossessionState.OBSERVED, "a", "b", 2),
    ]

    primitives = derive_ball_primitives(rows, states, real_fps=10)
    transitions = derive_ball_transitions(rows, states, real_fps=10)

    assert [item["primitive_type"] for item in primitives] == ["BALL_RELEASE", "BALL_IN_TRANSIT", "BALL_RECEPTION"]
    assert len(transitions) == 1
    assert transitions[0].source_holder == "a"
    assert transitions[0].destination_holder == "b"


def test_persistent_observed_destination_is_consumed_once():
    rows = normalize_ball_rows([_ball(i, i * 10, 0) for i in range(6)], real_fps=10, max_gap_frames=1)
    states = [
        _grace_observation(0, PossessionState.OBSERVED, "a", "a", 1),
        _grace_observation(1, PossessionState.INFERRED, "a", "b", 1),
        _grace_observation(2, PossessionState.INFERRED, "a", "b", 1),
        _grace_observation(3, PossessionState.OBSERVED, "a", "b", 2),
        _grace_observation(4, PossessionState.OBSERVED, "a", "b", 2),
        _grace_observation(5, PossessionState.OBSERVED, "a", "b", 2),
    ]

    primitives = derive_ball_primitives(rows, states, real_fps=10)
    transitions = derive_ball_transitions(rows, states, real_fps=10)
    semantic_events = [event for transition in transitions for event in classify_transition(transition)
                       if event.event_type is not EventType.UNKNOWN]

    assert [item["primitive_type"] for item in primitives].count("BALL_RELEASE") == 1
    assert [item["primitive_type"] for item in primitives].count("BALL_IN_TRANSIT") == 1
    assert [item["primitive_type"] for item in primitives].count("BALL_RECEPTION") == 1
    assert len(transitions) == 1
    assert len(semantic_events) <= 1


def test_reverted_grace_candidate_is_not_a_semantic_transition():
    rows = normalize_ball_rows([_ball(i, i * 10, 0) for i in range(5)], real_fps=10, max_gap_frames=1)
    transient_audit = PossessionAudit(
        observation_quality="inferred", observation_source="resolver", authority="stable",
        grace_eligible=True,
        provenance=Provenance(reason_codes=["TRANSIENT_SWITCH", "reverted_switch"]),
    )
    states = [
        _grace_observation(0, PossessionState.OBSERVED, "a", "a", 1),
        PossessionObservation(1, PossessionState.OBSERVED, EvidenceKind.OBSERVED, team_id=2,
                              context=PossessionContext(effective_holder_id="a", effective_holder_team=1,
                                                        observed_candidate_id="b"), audit=transient_audit),
        PossessionObservation(2, PossessionState.OBSERVED, EvidenceKind.OBSERVED, team_id=2,
                              context=PossessionContext(effective_holder_id="a", effective_holder_team=1,
                                                        observed_candidate_id="b"), audit=transient_audit),
        _grace_observation(3, PossessionState.OBSERVED, "a", "a", 1),
        _grace_observation(4, PossessionState.OBSERVED, "a", "a", 1),
    ]

    assert derive_ball_primitives(rows, states, real_fps=10) == []
    assert derive_ball_transitions(rows, states, real_fps=10) == []
