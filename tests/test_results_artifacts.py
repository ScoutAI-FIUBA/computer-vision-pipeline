"""Contract tests for V2 tracking artifact packaging."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.results import PipelineResults, RunSummary, package_artifacts, write_v3_artifacts
from pipeline.possession_v3 import (
    BallTransition, EventType, PossessionEvent, PossessionObservation,
    PossessionState, EvidenceKind, Provenance,
)
from pipeline.config import PipelineConfig
from pipeline.runner import _write_v3_artifacts


def _observations() -> list[dict]:
    return [
        {
            "observation_key": "60:3",
            "frame_index": 60,
            "timestamp_seconds": 2.0,
            "raw_track_id": 3,
            "stable_player_id": "player-07",
            "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
            "confidence": 0.94,
            "probability": 0.91,
            "track_state": "tracked",
        }
    ]


def test_package_writes_manifest_tracks_and_lossless_mask(tmp_path: Path):
    mask = np.array([[0, 1], [255, 2]], dtype=np.uint8)

    manifest = package_artifacts(
        tmp_path,
        _observations(),
        {"60:3": mask},
        input_fingerprint="sha256:video",
        video_metadata={"fps": 30.0},
        frame_bounds=[60, 130],
        prompts=["basketball player"],
        config={"retention_frames": 2},
        benchmark={"wall_seconds": 1.25},
        acceptance={"status": "pending"},
        output_prefix="runs/game-03",
    )

    assert manifest["schema"] == "sam31-artifacts"
    assert manifest["input_fingerprint"] == "sha256:video"
    assert manifest["frame_bounds"] == [60, 130]
    assert manifest["artifacts"]["tracks"]["key"] == "runs/game-03/tracks.jsonl"
    assert manifest["artifacts"]["masks"]["count"] == 1
    assert manifest["transfer_errors"] == []
    assert manifest["truncated"] is False

    assert json.loads((tmp_path / "tracks.jsonl").read_text())["observation_key"] == "60:3"
    mask_key = manifest["artifacts"]["masks"]["references"]["60:3"]["key"]
    written = cv2.imread(str(tmp_path / mask_key.removeprefix("runs/game-03/")), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(written, mask)


def test_package_preserves_acceptance_reasons_for_debug_review(tmp_path: Path):
    manifest = package_artifacts(
        tmp_path,
        _observations(),
        {"60:3": np.ones((2, 2), dtype=np.uint8)},
        acceptance={"tracking_identity": "rejected"},
        acceptance_reasons=["missing_stable_player_id:60:3"],
    )

    assert manifest["acceptance_reasons"] == ["missing_stable_player_id:60:3"]


def test_package_adds_optional_clean_mask_without_replacing_raw_mask(tmp_path: Path):
    observations = _observations()
    observations[0]["mask_ref"] = "masks/raw.png"
    manifest = package_artifacts(
        tmp_path,
        observations,
        {"60:3": np.ones((2, 2), dtype=np.uint8)},
        clean_masks={"60:3": np.eye(2, dtype=np.uint8)},
    )

    written = json.loads((tmp_path / "tracks.jsonl").read_text())
    assert written["mask_ref"] == "masks/raw.png"
    assert written["clean_mask_ref"] == "masks_clean/" + manifest["artifacts"]["clean_masks"]["references"]["60:3"]["key"].rsplit("/", 1)[-1]
    clean_key = manifest["artifacts"]["clean_masks"]["references"]["60:3"]["key"]
    assert cv2.imread(str(tmp_path / clean_key), cv2.IMREAD_UNCHANGED) is not None


def test_package_strips_runtime_bbox_coordinate_space_from_tracks(tmp_path: Path):
    observations = [{**_observations()[0], "bbox_coordinate_space": "normalized"}]

    package_artifacts(tmp_path, observations, {})

    written = json.loads((tmp_path / "tracks.jsonl").read_text())
    assert "bbox_coordinate_space" not in written


def test_manifest_integrity_hash_excludes_its_own_hash(tmp_path: Path):
    manifest = package_artifacts(tmp_path, _observations(), {"60:3": np.ones((2, 2), dtype=np.uint8)})
    manifest_ref = manifest["artifacts"]["manifest"]
    expected_payload = json.loads(json.dumps(manifest))
    expected_payload["artifacts"]["manifest"].pop("sha256")
    canonical = json.dumps(expected_payload, sort_keys=True, separators=(",", ":")).encode()

    assert manifest_ref["sha256"] == hashlib.sha256(canonical).hexdigest()


def test_adversarial_observation_keys_get_distinct_mask_artifacts(tmp_path: Path):
    observations = [{**_observations()[0], "observation_key": key} for key in ("a:b", "a-b")]
    manifest = package_artifacts(
        tmp_path,
        observations,
        {"a:b": np.ones((1, 1), dtype=np.uint8), "a-b": np.zeros((1, 1), dtype=np.uint8)},
    )

    refs = manifest["artifacts"]["masks"]["references"]
    assert refs["a:b"]["key"] != refs["a-b"]["key"]
    assert len(list((tmp_path / "masks").glob("*.png"))) == 2


def test_package_records_missing_mask_and_transfer_failure_explicitly(tmp_path: Path):
    manifest = package_artifacts(
        tmp_path,
        _observations(),
        {},
        transfer_errors=["upload timed out"],
        truncated=True,
    )

    assert manifest["truncated"] is True
    assert manifest["complete"] is False
    assert manifest["transfer_errors"] == ["upload timed out"]
    assert manifest["artifacts"]["masks"]["missing"] == ["60:3"]


def test_legacy_results_serialization_is_unchanged_when_no_artifacts_are_set():
    results = PipelineResults(5, {}, {}, RunSummary(), "cpu")

    assert results.to_dict() == {
        "schema_version": 5,
        "video": {},
        "config": {},
        "summary": {
            "frames_processed": 0,
            "segments_processed": 0,
            "detector_calls": 0,
            "track_records": 0,
            "distinct_track_ids": 0,
            "suspected_swaps": 0,
            "wall_seconds": 0.0,
        },
        "device": "cpu",
        "tracks_file": "tracks.jsonl",
        "teams_file": None,
        "numbers_file": None,
        "shot_events_file": None,
        "annotated_video_file": None,
    }


def test_new_result_fields_follow_existing_positional_optional_fields():
    results = PipelineResults(5, {}, {}, RunSummary(), "cpu", "tracks.jsonl", "teams.json", "numbers.json", "shots.json", "balls.json")

    assert results.ball_tracks_file == "balls.json"
    assert results.shot_evidence_file is None


def test_v3_result_pointers_are_additive_when_present():
    payload = PipelineResults(5, {}, {}, RunSummary(), "cpu", possession_v3_file="possession_v3.jsonl", possession_v3_schema_version=1).to_dict()
    assert payload["possession_v3_file"] == "possession_v3.jsonl" and payload["possession_v3_schema_version"] == 1


def test_package_rejects_duplicate_observation_keys(tmp_path: Path):
    observations = _observations() * 2

    with pytest.raises(ValueError, match="duplicate observation_key"):
        package_artifacts(tmp_path, observations, {})


def test_v3_artifacts_are_versioned_hashed_and_deterministic(tmp_path: Path):
    observation = PossessionObservation(2, PossessionState.OBSERVED, EvidenceKind.OBSERVED,
                                        stable_player_id="p1", raw_track_id=7, team_id=0,
                                        score=.9, provenance=Provenance(["tracks#2"], [2], ["candidate"]))
    transition = BallTransition("t1", "p1", "p2", ["BALL_RELEASE", "BALL_RECEPTION"],
                                Provenance(["ball#2"], [2], ["release"]), frame_start=2,
                                frame_end=4, source_team_id=0, destination_team_id=0)
    event = PossessionEvent(EventType.TRANSFER, 4, transition.provenance, transition_id="t1",
                            source_holder="p1", destination_holder="p2", source_team_id=0,
                            destination_team_id=0, reason_codes=["same_team_possession_transfer"])
    first = write_v3_artifacts(tmp_path, [observation], [{"primitive_type": "BALL_RELEASE", "frame_index": 2, "timestamp_seconds": 1.0}], [transition], [event], {"profile": "default"})
    second = write_v3_artifacts(tmp_path / "repeat", [observation], [{"primitive_type": "BALL_RELEASE", "frame_index": 2, "timestamp_seconds": 1.0}], [transition], [event], {"profile": "default"})
    assert first["schema_version"] == 1
    assert first["artifacts"]["possession"]["path"] == "possession_v3.jsonl"
    assert first["artifacts"]["possession"]["sha256"] == second["artifacts"]["possession"]["sha256"]
    assert first["artifacts"]["manifest"]["sha256"] == second["artifacts"]["manifest"]["sha256"]
    assert json.loads((tmp_path / "possession_events_v3.jsonl").read_text())["event_type"] == "possession_transfer"
    possession = json.loads((tmp_path / "possession_v3.jsonl").read_text())
    primitive = json.loads((tmp_path / "possession_primitives_v3.jsonl").read_text())
    transition_row = json.loads((tmp_path / "ball_transitions_v3.jsonl").read_text())
    event_row = json.loads((tmp_path / "possession_events_v3.jsonl").read_text())
    for row in (possession, primitive, transition_row, event_row):
        assert {"frame_start", "frame_end", "timestamp_start", "timestamp_end"} <= row.keys()
    assert (possession["frame_start"], possession["frame_end"]) == (2, 2)
    assert (primitive["timestamp_start"], primitive["timestamp_end"]) == (1.0, 1.0)
    assert (transition_row["frame_start"], transition_row["frame_end"]) == (2, 4)
    assert (event_row["frame_start"], event_row["frame_end"]) == (4, 4)


def test_v3_results_pointers_are_additive_only_when_set():
    payload = PipelineResults(5, {}, {}, RunSummary(), "cpu", possession_v3_file="possession_v3.jsonl",
                              possession_v3_schema_version=1, possession_v3_manifest_file="possession_v3_manifest.json",
                              possession_v3_events_file="possession_events_v3.jsonl").to_dict()
    assert payload["possession_v3_manifest_file"] == "possession_v3_manifest.json"
    assert payload["possession_v3_events_file"] == "possession_events_v3.jsonl"


def test_v3_writer_is_feature_flag_isolated(tmp_path: Path):
    base = PipelineConfig(roboflow_api_key="test-key")
    assert _write_v3_artifacts(tmp_path / "legacy", [], [], fps=30, config=base) == {}
    assert not (tmp_path / "legacy" / "possession_v3.jsonl").exists()
    enabled = PipelineConfig(roboflow_api_key="test-key", possession_engine="v3")
    manifest = _write_v3_artifacts(tmp_path / "v3", [], [], fps=30, config=enabled)
    assert manifest["schema"] == "possession-engine-v3"
    assert (tmp_path / "v3" / "ball_transitions_v3.jsonl").exists()
