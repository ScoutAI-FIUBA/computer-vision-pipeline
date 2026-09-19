"""Contract tests for V2 tracking artifact packaging."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.results import PipelineResults, RunSummary, package_artifacts


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


def test_package_rejects_duplicate_observation_keys(tmp_path: Path):
    observations = _observations() * 2

    with pytest.raises(ValueError, match="duplicate observation_key"):
        package_artifacts(tmp_path, observations, {})
