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
