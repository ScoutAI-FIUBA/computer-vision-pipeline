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
