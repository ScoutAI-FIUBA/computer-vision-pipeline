"""V4 spatial contract and fusion tests."""

from dataclasses import asdict
import json

import pytest

from pipeline.config import PipelineConfig
from pipeline.possession_v3 import (CandidateEvidence, EvidenceKind, PossessionObservation,
                                    PossessionResolver, PossessionState, V4CandidateEvidence,
                                    engine_name, score_candidates_v4)
from pipeline.results import _json_record


def player(name, box, **values):
    return {"stable_player_id": name, "raw_track_id": 1, "team_id": 1,
            "bbox": box, "mask_valid": True,
            "mask_contains": values.pop("mask_contains", True), **values}


def test_valid_mask_bbox_agreement_is_one_spatial_vote_with_diagnostics():
    candidate = score_candidates_v4([player("p1", [0, 0, 20, 20])], {"x": 10, "y": 10})[0]
    assert candidate.spatial_score == candidate.score == 1.0
    assert candidate.mask_score == candidate.bbox_score == 1.0
    assert candidate.mask_bbox_agreement == 1.0
    assert candidate.source == "mask+bbox"
    assert candidate.contributions["spatial_score"] == 1.0


def test_mask_bbox_disagreement_is_reduced_and_marked_ambiguous():
    candidate = score_candidates_v4([player("p1", [0, 0, 20, 20], mask_contains=False)], {"x": 10, "y": 10})[0]
    assert 0 < candidate.spatial_score < 1
    assert candidate.mask_score == 0.0 and candidate.bbox_score == 1.0
    assert candidate.mask_bbox_agreement == 0.0
    assert candidate.source == "mask+bbox-disagreement"
    assert "mask_bbox_disagreement" in candidate.rejected_reasons


def test_missing_mask_does_not_fallback_to_bbox_score():
    candidate = score_candidates_v4([player("p1", [0, 0, 20, 20], mask_valid=False)], {"x": 10, "y": 10})[0]
    assert candidate.mask_score is None
    assert candidate.bbox_score == 1.0
    assert candidate.spatial_score == candidate.score == 0.0
    assert candidate.source == "missing-mask"


def test_overlapping_candidates_are_sorted_and_marked_ambiguous():
    candidates = score_candidates_v4([
        player("b", [0, 0, 20, 20]), player("a", [0, 0, 20, 20])
    ], {"x": 10, "y": 10}, ambiguity_margin=.1)
    assert [candidate.stable_player_id for candidate in candidates] == ["a", "b"]
    assert all("ambiguous_margin" in candidate.rejected_reasons for candidate in candidates)


def test_v4_candidate_serialization_is_deterministic_and_bounded():
    payload = asdict(score_candidates_v4([player("p1", [0, 0, 20, 20])], {"x": 10, "y": 10})[0])
    assert json.dumps(payload, sort_keys=True) == json.dumps(json.loads(json.dumps(payload, sort_keys=True)), sort_keys=True)
    assert 0 <= payload["spatial_score"] <= 1


def test_v4_config_is_opt_in_and_validated(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "key")
    monkeypatch.setenv("POSSESSION_ENGINE", "v4")
    config = PipelineConfig.from_env()
    assert config.possession_engine == "v4"
    assert config.possession_v4_enabled is True
    assert engine_name(config) == "v4"
    assert PipelineConfig(roboflow_api_key="key").possession_engine == "legacy"
    with pytest.raises(ValueError):
        PipelineConfig(roboflow_api_key="key", possession_engine="unsupported")


def test_candidate_contract_rejects_unbounded_v4_diagnostics():
    with pytest.raises(ValueError):
        V4CandidateEvidence("p", 1, 1, .5, spatial_score=1.1)


def test_legacy_candidate_asdict_keeps_exact_v3_shape():
    payload = asdict(CandidateEvidence("p", 1, 1, .5, {"geometry": .5}))
    assert payload == {
        "stable_player_id": "p", "raw_track_id": 1, "team_id": 1,
        "score": .5, "contributions": {"geometry": .5},
        "rejected_reasons": [], "fallback_identity": None,
    }


def test_v4_candidate_asdict_contains_spatial_diagnostics():
    payload = asdict(score_candidates_v4([player("p1", [0, 0, 20, 20])], {"x": 10, "y": 10})[0])
    assert isinstance(score_candidates_v4([player("p1", [0, 0, 20, 20])], {"x": 10, "y": 10})[0], V4CandidateEvidence)
    assert payload["spatial_score"] == 1.0
    assert payload["source"] == "mask+bbox"


@pytest.mark.parametrize("engine", ["legacy", "v3"])
def test_non_v4_serialization_omits_spatial_diagnostics(engine):
    candidate = score_candidates_v4([player("p1", [0, 0, 20, 20])], {"x": 10, "y": 10})[0]
    observation = PossessionObservation(1, PossessionState.CANDIDATE, EvidenceKind.CONTESTED,
                                        candidates=[candidate])
    record = _json_record(observation, {"engine": engine})
    serialized = json.dumps(record)
    assert all(name not in serialized for name in
               ("spatial_score", "mask_score", "bbox_score", "mask_bbox_agreement"))


def test_v4_serialization_keeps_spatial_diagnostics():
    candidate = score_candidates_v4([player("p1", [0, 0, 20, 20])], {"x": 10, "y": 10})[0]
    record = _json_record(PossessionObservation(1, PossessionState.OBSERVED, EvidenceKind.OBSERVED,
                                                candidates=[candidate]), {"engine": "v4"})
    assert record["candidates"][0]["spatial_score"] == 1.0


def test_disagreement_is_contested_even_when_score_exceeds_keep_threshold():
    candidate = score_candidates_v4([player("p1", [0, 0, 20, 20], mask_contains=False)],
                                    {"x": 10, "y": 10}, disagreement_score=.95)[0]
    observation = PossessionResolver(real_fps=30, keep_threshold=.5).resolve(0, [candidate])
    assert observation.state is PossessionState.CONTESTED
    assert observation.context.effective_holder_id is None


def test_v4_ambiguous_margin_cannot_be_promoted_by_legacy_hysteresis():
    candidates = score_candidates_v4([
        player("a", [0, 0, 20, 20]), player("b", [0, 0, 20, 20])
    ], {"x": 10, "y": 10}, ambiguity_margin=.1)
    resolver = PossessionResolver(real_fps=30, hysteresis_margin=0.01, keep_threshold=.5)
    observation = resolver.resolve(0, candidates)
    assert observation.state is PossessionState.CONTESTED
    assert resolver.holder is None


def evidence_player(name, x, *, approach=0.0, continuity=0.0):
    return player(name, [x - 10, 0, x + 10, 40],
                  relative_motion=approach, continuity=continuity)


def test_v4_evidence_exposes_normalized_distance_and_contributions():
    candidate = score_candidates_v4(
        [evidence_player("p1", 10, approach=.8, continuity=1.0)],
        {"x": 10, "y": 20, "confidence": .9},
    )[0]
    assert candidate.normalized_distance == 0.0
    assert candidate.contributions["spatial_fusion"] == 1.0
    assert candidate.holder_control_score > .5
    assert "ball_quality" in candidate.contributions


def test_v4_valid_ball_confidence_contributes_quality():
    candidate = score_candidates_v4([evidence_player("p1", 10, approach=1)],
                                    {"x": 10, "y": 20, "confidence": .8})[0]
    assert candidate.ball_quality == .8
    assert candidate.contributions["ball_quality"] == .8


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), float("-inf")])
def test_v4_malformed_ball_confidence_cannot_score_or_acquire_repeatedly(confidence):
    resolver = PossessionResolver(real_fps=10, v4=True, grace_seconds=.4)
    holder = score_candidates_v4([evidence_player("p1", 10, continuity=1)],
                                 {"x": 10, "y": 20, "confidence": .9})
    resolver.resolve(0, holder)
    malformed_ball = {"x": 10, "y": 20, "confidence": confidence}
    for frame in (1, 2, 3):
        challenger = score_candidates_v4(
            [evidence_player("p2", 10, approach=1, continuity=1)], malformed_ball
        )[0]
        assert challenger.score == 0
        assert challenger.challenger_acquisition_score == 0
        result = resolver.resolve(frame, [challenger])
        assert result.context.effective_holder_id == "p1"
        assert result.state is PossessionState.CONTESTED


def test_v4_resolver_rejects_legacy_candidate_evidence_at_boundary():
    resolver = PossessionResolver(real_fps=10, v4=True)
    legacy = CandidateEvidence("p2", 2, 1, 1.0, {"geometry": 1.0})
    result = resolver.resolve(0, [legacy])
    assert result.state is PossessionState.CONTESTED
    assert resolver.holder is None
    assert "legacy_candidate_without_v4_evidence" in result.provenance.reason_codes

    holder = score_candidates_v4([evidence_player("p1", 10, continuity=1)], {"x": 10, "y": 20})
    resolver.resolve(1, holder)
    result = resolver.resolve(2, [legacy])
    assert result.context.effective_holder_id == "p1"
    assert result.state is PossessionState.CONTESTED


def test_v4_acquisition_requires_approach_not_proximity():
    close = score_candidates_v4([evidence_player("p2", 10)], {"x": 10, "y": 20})[0]
    approaching = score_candidates_v4([evidence_player("p2", 10, approach=1.0)],
                                      {"x": 10, "y": 20})[0]
    assert close.challenger_acquisition_score < .5
    assert approaching.challenger_acquisition_score > close.challenger_acquisition_score


def test_v4_release_persists_and_dribble_return_keeps_holder():
    resolver = PossessionResolver(real_fps=10, v4=True, persistence_seconds=.2,
                                  release_persistence_seconds=.2)
    holder = score_candidates_v4([evidence_player("p1", 10, approach=1, continuity=1)],
                                 {"x": 10, "y": 20})
    assert resolver.resolve(0, holder).state is PossessionState.OBSERVED
    weak = score_candidates_v4([evidence_player("p1", 60)], {"x": 10, "y": 20})
    assert resolver.resolve(1, weak).context.effective_holder_id == "p1"
    returned = score_candidates_v4([evidence_player("p1", 10, approach=1, continuity=1)],
                                   {"x": 10, "y": 20})
    assert resolver.resolve(2, returned).context.effective_holder_id == "p1"


def test_v4_challenger_acquisition_persists_before_switch():
    resolver = PossessionResolver(real_fps=10, v4=True, persistence_seconds=.2,
                                  acquisition_persistence_seconds=.2)
    assert resolver.resolve(0, score_candidates_v4([evidence_player("p1", 10, continuity=1)],
                                                     {"x": 10, "y": 20})).state is PossessionState.OBSERVED
    challenger = score_candidates_v4([evidence_player("p2", 10, approach=1, continuity=1)],
                                     {"x": 10, "y": 20})
    assert resolver.resolve(1, challenger).context.effective_holder_id == "p1"
    assert resolver.resolve(2, challenger).context.effective_holder_id == "p2"


def test_v4_ball_lost_grace_does_not_release_holder():
    resolver = PossessionResolver(real_fps=10, v4=True, grace_seconds=.3)
    candidate = score_candidates_v4([evidence_player("p1", 10, continuity=1)], {"x": 10, "y": 20})
    resolver.resolve(0, candidate)
    lost = resolver.resolve(1, [], ball_lost=True)
    assert lost.state is PossessionState.INFERRED
    assert lost.context.effective_holder_id == "p1"


def test_v4_passing_front_does_not_switch_on_proximity():
    resolver = PossessionResolver(real_fps=10, v4=True, acquisition_persistence_seconds=0)
    holder = score_candidates_v4([evidence_player("p1", 10, continuity=1)], {"x": 10, "y": 20})
    resolver.resolve(0, holder)
    passer = score_candidates_v4([evidence_player("p2", 10)], {"x": 10, "y": 20})
    result = resolver.resolve(1, passer)
    assert result.context.effective_holder_id == "p1"
    assert "acquisition_evidence_required" in result.provenance.reason_codes


def test_v4_unknown_and_malformed_evidence_do_not_release():
    resolver = PossessionResolver(real_fps=10, v4=True)
    holder = score_candidates_v4([evidence_player("p1", 10, continuity=1)], {"x": 10, "y": 20})
    resolver.resolve(0, holder)
    unknown = resolver.resolve(1, [], missing=True)
    assert unknown.state is PossessionState.UNKNOWN
    assert unknown.context.effective_holder_id == "p1"
    malformed = score_candidates_v4([{"stable_player_id": "p2", "team_id": 1, "bbox": object()}],
                                     {"x": 10, "y": 20})[0]
    assert malformed.score == 0 and "invalid_normalized_distance" in malformed.rejected_reasons


def test_v4_malformed_challenger_preserves_holder_as_contested():
    resolver = PossessionResolver(real_fps=10, v4=True, grace_seconds=.3)
    holder = score_candidates_v4([evidence_player("p1", 10, continuity=1)], {"x": 10, "y": 20})
    resolver.resolve(0, holder)
    malformed = score_candidates_v4([{"stable_player_id": "p2", "team_id": 1, "bbox": object()}],
                                    {"x": 10, "y": 20})
    result = resolver.resolve(1, malformed)
    assert result.state is PossessionState.CONTESTED
    assert result.context.effective_holder_id == "p1"


def test_v4_produces_approach_from_ball_and_player_history():
    previous = [evidence_player("p2", 50)]
    current = [evidence_player("p2", 10, continuity=1)]
    candidate = score_candidates_v4(current, {"x": 10, "y": 20, "confidence": 1.0},
                                    previous_ball={"x": 10, "y": 20},
                                    previous_players=previous)[0]
    assert candidate.relative_motion == 1.0
    assert candidate.challenger_acquisition_score > .6
    assert candidate.contributions["relative_motion"] == 1.0


@pytest.mark.parametrize("challenger", [
    {"stable_player_id": "p2", "raw_track_id": 2, "team_id": -1,
     "bbox": [0, 0, 20, 40], "mask_valid": True, "mask_contains": True,
     "relative_motion": 1.0},
    {"stable_player_id": None, "raw_track_id": None, "team_id": 1,
     "bbox": [0, 0, 20, 40], "mask_valid": True, "mask_contains": True,
     "relative_motion": 1.0},
    {"stable_player_id": "p2", "raw_track_id": 2, "team_id": 1,
     "bbox": [0, 0, 20, 40], "mask_valid": True, "mask_contains": True,
     "confidence": float("nan"), "relative_motion": 1.0},
])
def test_v4_invalid_authority_cannot_replace_holder(challenger):
    resolver = PossessionResolver(real_fps=10, v4=True, grace_seconds=.3)
    holder = score_candidates_v4([evidence_player("p1", 10, continuity=1)], {"x": 10, "y": 20})
    resolver.resolve(0, holder)
    candidate = score_candidates_v4([challenger], {"x": 10, "y": 20, "confidence": 1.0})[0]
    assert candidate.score == 0
    assert candidate.challenger_acquisition_score == 0
    result = resolver.resolve(1, [candidate])
    assert result.context.effective_holder_id == "p1"
    assert result.state is PossessionState.CONTESTED


@pytest.mark.parametrize("name,value", [("holder_control_score", -0.1),
                                         ("holder_release_score", 1.1),
                                         ("challenger_acquisition_score", float("nan"))])
def test_v4_evidence_config_scores_are_bounded(name, value):
    with pytest.raises(ValueError):
        PipelineConfig(roboflow_api_key="key", possession_engine="v4", **{name: value})
