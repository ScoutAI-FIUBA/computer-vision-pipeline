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
