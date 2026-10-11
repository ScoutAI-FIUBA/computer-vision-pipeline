"""Contract tests for V2 tracking artifact packaging."""

from __future__ import annotations

import json
import hashlib
import io
import os
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.results import (
    PipelineResults, RunSummary, package_artifacts, write_v3_artifacts,
    stage_fingerprint, publish_stage_generation, validate_stage_generation, load_stage_generation,
)
from pipeline.possession_v3 import (
    BallTransition, EventType, PossessionEvent, PossessionObservation,
    PossessionState, EvidenceKind, Provenance, PossessionContext, PossessionAudit,
    ClearReason,
    CandidateEvidence, PossessionResolver,
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


def _valid_tracking_artifacts(root: Path, input_fingerprint: str = "input-a", frame_bounds: list[int] | None = None):
    package = root / "package"
    package_artifacts(package, [], {}, input_fingerprint=input_fingerprint, frame_bounds=frame_bounds or [0, 0])
    return {
        ("tracking-manifest.json" if path.name == "manifest.json" else path.relative_to(package).as_posix()): path
        for path in package.rglob("*") if path.is_file()
    }


def test_stage_fingerprint_is_canonical_and_changes_with_inputs():
    assert stage_fingerprint({"b": 2, "a": 1}) == stage_fingerprint({"a": 1, "b": 2})
    assert stage_fingerprint({"a": 1}) != stage_fingerprint({"a": 2})


def test_stage_generation_is_atomic_and_validated(tmp_path: Path):
    generation = publish_stage_generation(
        tmp_path, stage="tracking", input_fingerprint="input-a", request_fingerprint="request-a",
        artifacts=_valid_tracking_artifacts(tmp_path, frame_bounds=[1, 1]), frame_bounds=[1, 1],
    )
    assert generation["complete"] is True
    assert validate_stage_generation(tmp_path, generation) == generation
    assert json.loads((tmp_path / "current.json").read_text())["generation"] == generation["generation"]


def test_loading_generation_rejects_manifest_from_another_generation(tmp_path: Path):
    first = publish_stage_generation(
        tmp_path, stage="tracking", input_fingerprint="input-a", request_fingerprint="request-a",
        artifacts=_valid_tracking_artifacts(tmp_path, frame_bounds=[1, 1]), frame_bounds=[1, 1],
    )
    second = publish_stage_generation(
        tmp_path, stage="tracking", input_fingerprint="input-b", request_fingerprint="request-b",
        artifacts={"tracks.jsonl": b"tracks-b", "tracking-manifest.json": b"{}"}, frame_bounds=[2, 2],
    )
    manifest_path = tmp_path / "generations" / first["generation"] / "manifest.json"
    manifest_path.write_text(
        (tmp_path / "generations" / second["generation"] / "manifest.json").read_text(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid generation"):
        load_stage_generation(tmp_path, first["generation"])


def test_stage_generation_recomputes_fingerprint_from_published_artifact_bytes(tmp_path: Path):
    generation = publish_stage_generation(
        tmp_path, stage="stats", input_fingerprint="input-a", request_fingerprint="request-a",
        artifacts={"results.json": b"old"}, frame_bounds=[1, 1], tracking_generation="tracking-a",
    )
    manifest_path = tmp_path / "generations" / generation["generation"] / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    artifact = tmp_path / "generations" / generation["generation"] / "results.json"
    artifact.write_bytes(b"new")
    payload["artifacts"]["results.json"]["sha256"] = hashlib.sha256(b"new").hexdigest()
    payload["artifacts"]["results.json"]["size_bytes"] = 3
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in payload.items() if key != "manifest_sha256"},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generation fingerprint"):
        validate_stage_generation(tmp_path, payload)


def test_tracking_generation_rejects_invalid_embedded_package_manifest(tmp_path: Path):
    generation = publish_stage_generation(
        tmp_path, stage="tracking", input_fingerprint="input-a", request_fingerprint="request-a",
        artifacts={"tracks.jsonl": b"tracks", "tracking-manifest.json": b"{}"}, frame_bounds=[1, 1],
    )

    with pytest.raises(ValueError, match="embedded|package|schema"):
        validate_stage_generation(tmp_path, generation)


def test_tracking_generation_rejects_embedded_reference_hash_mismatch(tmp_path: Path):
    package = {
        "schema": "sam31-artifacts", "schema_version": 1, "input_fingerprint": "input-a",
        "frame_bounds": [1, 1], "prompts": ["player"], "config": {}, "benchmark": {},
        "acceptance": {}, "acceptance_reasons": [], "observation_count": 1,
        "mask_encoding": "PNG uint8 lossless", "transfer_errors": [], "truncated": False,
        "complete": True, "artifacts": {
            "manifest": {"key": "manifest.json"},
            "tracks": {"key": "tracks.jsonl", "size_bytes": 6, "sha256": "bad"},
            "masks": {"key": "masks/", "count": 0, "missing": [], "references": {}},
        },
    }
    package["artifacts"]["manifest"]["sha256"] = __import__("pipeline.results", fromlist=["_manifest_hash"])._manifest_hash(package)
    generation = publish_stage_generation(
        tmp_path, stage="tracking", input_fingerprint="input-a", request_fingerprint="request-a",
        artifacts={"tracks.jsonl": b"tracks", "tracking-manifest.json": json.dumps(package).encode()}, frame_bounds=[1, 1],
    )

    with pytest.raises(ValueError, match="embedded|hash"):
        validate_stage_generation(tmp_path, generation)


def test_stage_generation_rejects_hash_changes_and_preserves_previous_pointer(tmp_path: Path):
    previous = publish_stage_generation(
        tmp_path, stage="tracking", input_fingerprint="input-a", request_fingerprint="request-a",
        artifacts=_valid_tracking_artifacts(tmp_path, frame_bounds=[0, 0]), frame_bounds=[0, 0],
    )
    path = tmp_path / "generations" / previous["generation"] / "tracks.jsonl"
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        validate_stage_generation(tmp_path, previous)
    assert json.loads((tmp_path / "current.json").read_text())["generation"] == previous["generation"]


def test_stage_generation_requires_complete_tracking_contract(tmp_path: Path):
    generation = publish_stage_generation(
        tmp_path, stage="tracking", input_fingerprint="input-a", request_fingerprint="request-a",
        artifacts={"tracks.jsonl": b"old", "tracking-manifest.json": b"{}"}, frame_bounds=[1, 1],
    )
    path = tmp_path / "generations" / generation["generation"] / "manifest.json"
    payload = json.loads(path.read_text())
    payload["frame_bounds"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frame_bounds|manifest hash"):
        validate_stage_generation(tmp_path, payload)


def test_stage_generation_pointer_failure_keeps_prior_generation(tmp_path: Path, monkeypatch):
    previous = publish_stage_generation(
        tmp_path, stage="tracking", input_fingerprint="input-a", request_fingerprint="request-a",
        artifacts=_valid_tracking_artifacts(tmp_path, frame_bounds=[0, 0]),
    )
    from pipeline import results as results_module
    original = results_module.os.replace

    def fail_pointer(source, destination):
        if Path(destination).name == "current.json":
            raise OSError("pointer publication failed")
        return original(source, destination)

    monkeypatch.setattr(results_module.os, "replace", fail_pointer)
    with pytest.raises(OSError, match="pointer publication failed"):
        publish_stage_generation(
            tmp_path, stage="tracking", input_fingerprint="input-b", request_fingerprint="request-b",
            artifacts={"tracks.jsonl": b"new", "tracking-manifest.json": b"{}"},
        )
    assert json.loads((tmp_path / "current.json").read_text())["generation"] == previous["generation"]


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


def test_package_writes_hashed_raw_and_clean_mask_bundles(tmp_path: Path):
    manifest = package_artifacts(
        tmp_path,
        _observations(),
        {"60:3": np.ones((2, 2), dtype=np.uint8)},
        clean_masks={"60:3": np.eye(2, dtype=np.uint8)},
    )

    for name, filename in (("masks", "masks.tar"), ("clean_masks", "masks_clean.tar")):
        bundle = manifest["artifacts"][name]["bundle"]
        assert bundle["key"] == filename
        assert bundle["sha256"] == hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        with tarfile.open(tmp_path / filename) as archive:
            assert archive.getnames() == [
                manifest["artifacts"][name]["references"]["60:3"]["key"].rsplit("/", 1)[-1]
            ]


def test_validate_stage_generation_checks_declared_mask_bundle_hash(tmp_path: Path):
    package = tmp_path / "package"
    package_artifacts(package, _observations(), {"60:3": np.ones((2, 2), dtype=np.uint8)},
                      input_fingerprint="input", frame_bounds=[0, 0])
    artifacts = {
        ("tracking-manifest.json" if path.name == "manifest.json" else path.relative_to(package).as_posix()): path
        for path in package.rglob("*") if path.is_file()
    }
    generation = publish_stage_generation(
        tmp_path / "tracking", stage="tracking", input_fingerprint="input", request_fingerprint="request",
        artifacts=artifacts, frame_bounds=[0, 0],
    )
    bundle = tmp_path / "tracking" / "generations" / generation["generation"] / "masks.tar"
    bundle.write_bytes(bundle.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="hash mismatch: masks.tar"):
        validate_stage_generation(tmp_path / "tracking", generation)


@pytest.mark.parametrize("group_name", ["masks", "clean_masks"])
@pytest.mark.parametrize("tamper", ["content", "extra"])
def test_validate_stage_generation_checks_bundle_content_and_exact_members(
    tmp_path: Path, group_name: str, tamper: str,
):
    package = tmp_path / "package"
    package_artifacts(
        package, _observations(), {"60:3": np.ones((2, 2), dtype=np.uint8)},
        {"60:3": np.eye(2, dtype=np.uint8)}, input_fingerprint="input", frame_bounds=[0, 0],
    )
    bundle_path = package / ("masks.tar" if group_name == "masks" else "masks_clean.tar")
    with tarfile.open(bundle_path) as source:
        members = [(member.name, source.extractfile(member).read()) for member in source.getmembers()]
    if tamper == "content":
        encoded = cv2.imencode(".png", np.zeros((2, 2), dtype=np.uint8))[1].tobytes()
        members[0] = (members[0][0], encoded)
    else:
        members.append(("extra.png", members[0][1]))
    with tarfile.open(bundle_path, "w") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    manifest_path = package / "manifest.json"
    package_manifest = json.loads(manifest_path.read_text())
    bundle = package_manifest["artifacts"][group_name]["bundle"]
    bundle_bytes = bundle_path.read_bytes()
    bundle["sha256"] = hashlib.sha256(bundle_bytes).hexdigest()
    bundle["size_bytes"] = len(bundle_bytes)
    package_manifest["artifacts"]["manifest"]["sha256"] = __import__(
        "pipeline.results", fromlist=["_manifest_hash"]
    )._manifest_hash(package_manifest)
    manifest_path.write_text(json.dumps(package_manifest, sort_keys=True), encoding="utf-8")
    artifacts = {
        ("tracking-manifest.json" if path.name == "manifest.json" else path.relative_to(package).as_posix()): path
        for path in package.rglob("*") if path.is_file()
    }
    generation = publish_stage_generation(
        tmp_path / "tracking", stage="tracking", input_fingerprint="input", request_fingerprint="request",
        artifacts=artifacts, frame_bounds=[0, 0],
    )

    error = "bundle (content|member hash)" if tamper == "content" else "bundle members"
    with pytest.raises(ValueError, match=error):
        validate_stage_generation(tmp_path / "tracking", generation)


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
    assert first["artifacts"]["possession"]["path"].startswith("possession_v3/generations/")
    assert first["artifacts"]["possession"]["sha256"] == second["artifacts"]["possession"]["sha256"]
    assert first["artifacts"]["manifest"]["sha256"] == second["artifacts"]["manifest"]["sha256"]
    root = tmp_path / first["artifacts"]["possession"]["path"].split("/", 2)[0]
    possession = json.loads((tmp_path / first["artifacts"]["possession"]["path"]).read_text())
    primitive = json.loads((tmp_path / first["artifacts"]["primitives"]["path"]).read_text())
    transition_row = json.loads((tmp_path / first["artifacts"]["transitions"]["path"]).read_text())
    event_row = json.loads((tmp_path / first["artifacts"]["events"]["path"]).read_text())
    assert json.loads((tmp_path / "possession_v3_manifest.json").read_text())["generation"] == first["generation"]
    for row in (possession, primitive, transition_row, event_row):
        assert {"frame_start", "frame_end", "timestamp_start", "timestamp_end"} <= row.keys()
    assert (possession["frame_start"], possession["frame_end"]) == (2, 2)
    assert (primitive["timestamp_start"], primitive["timestamp_end"]) == (1.0, 1.0)
    assert (transition_row["frame_start"], transition_row["frame_end"]) == (2, 4)
    assert (event_row["frame_start"], event_row["frame_end"]) == (4, 4)


def test_v3_artifact_persists_continuity_roles_and_audit_fields(tmp_path: Path):
    observation = PossessionObservation(
        7, PossessionState.TRANSITIONING, EvidenceKind.CONTESTED,
        context=PossessionContext(
            effective_holder_id="player-006", effective_holder_team=1,
            observed_candidate_id="player-007", challenger_id="player-007",
            grace_remaining_seconds=.4,
        ),
        audit=PossessionAudit(
            last_holder_id="player-006", last_holder_team=1,
            clear_reason=ClearReason.GRACE_EXPIRED,
            frame_start=5, frame_end=7, timestamp_start=.5, timestamp_end=.7,
        ),
    )
    manifest = write_v3_artifacts(tmp_path, [observation], [], [], [])
    row = json.loads((tmp_path / manifest["artifacts"]["possession"]["path"]).read_text())

    assert row["schema_version"] == 1
    assert row["state"] == "TRANSITIONING"
    assert row["context"]["effective_holder_id"] == "player-006"
    assert row["context"]["observed_candidate_id"] == "player-007"
    assert row["context"]["challenger_id"] == "player-007"
    assert row["context"]["grace_remaining_seconds"] == .4
    assert row["audit"]["last_holder_id"] == "player-006"
    assert row["audit"]["clear_reason"] == "GRACE_EXPIRED"
    assert row["audit"]["frame_start"] == 5 and row["audit"]["timestamp_end"] == .7


def test_v3_artifact_serializes_stale_pending_challenger_timeline(tmp_path: Path):
    resolver = PossessionResolver(real_fps=10, persistence_seconds=.4)
    holder = CandidateEvidence("holder", 1, 1, .9, {"geometry": .9})
    challenger = CandidateEvidence("challenger", 2, 2, .95, {"geometry": .95})
    weak = CandidateEvidence("weak", 3, 2, .2, {"geometry": .2})
    resolver.resolve(0, [holder])
    resolver.resolve(1, [challenger])
    assert resolver.pending is not None
    canceled = resolver.resolve(2, [weak])

    # The resolver still has stale pending state, but the observation has no
    # active challenger evidence and must therefore serialize an empty window.
    manifest = write_v3_artifacts(tmp_path, [canceled], [], [], [])
    row = json.loads((tmp_path / manifest["artifacts"]["possession"]["path"]).read_text())

    assert row["context"]["challenger_id"] is None
    assert row["context"]["challenger_start_frame"] is None
    assert row["context"]["challenger_confirmation_frame"] is None
    assert row["context"]["challenger_elapsed_seconds"] == 0.0
    assert (tmp_path / "possession_v3_manifest.json").exists()


def test_real_006_007_continuity_artifact_replays_byte_for_byte(tmp_path: Path):
    config = PipelineConfig(roboflow_api_key="test-key", possession_engine="v3")
    players = [
        {"frame_index": frame, "raw_track_id": 6, "stable_player_id": "player-006",
         "team_id": 1, "xyxy": [0, 0, 20, 20], "confidence": .9}
        for frame in range(3)
    ] + [
        {"frame_index": frame, "raw_track_id": 7, "stable_player_id": "player-007",
         "team_id": 1, "xyxy": [18, 0, 38, 20], "confidence": .9}
        for frame in range(3)
    ]
    balls = [{"frame_index": frame, "timestamp_seconds": frame / 10, "track_state": "detected",
              "confidence": .9, "center_xy": [10, 10]} for frame in range(3)]
    first = _write_v3_artifacts(tmp_path / "first", players, balls, fps=10, config=config)
    first_rows = {
        name: [json.loads(line) for line in (tmp_path / "first" / first["artifacts"][name]["path"]).read_text().splitlines()]
        for name in ("possession", "primitives", "transitions", "events")
    }
    second = write_v3_artifacts(tmp_path / "second", first_rows["possession"], first_rows["primitives"],
                                first_rows["transitions"], first_rows["events"], first["config"])
    second_rows = {
        name: [json.loads(line) for line in (tmp_path / "second" / second["artifacts"][name]["path"]).read_text().splitlines()]
        for name in ("possession", "primitives", "transitions", "events")
    }

    assert first["generation"] == second["generation"]
    assert first_rows["possession"][0]["context"]["effective_holder_id"] == "player-006"
    assert first_rows["possession"][0]["context"]["observed_candidate_id"] == "player-006"
    for name in ("possession", "primitives", "transitions", "events"):
        assert first_rows[name] == second_rows[name]
        assert first["artifacts"][name]["sha256"] == second["artifacts"][name]["sha256"]


def test_v3_results_pointers_are_additive_only_when_set():
    payload = PipelineResults(5, {}, {}, RunSummary(), "cpu", possession_v3_file="possession_v3.jsonl",
                              possession_v3_schema_version=1, possession_v3_manifest_file="possession_v3_manifest.json",
                              possession_v3_events_file="possession_events_v3.jsonl").to_dict()
    assert payload["possession_v3_manifest_file"] == "possession_v3_manifest.json"
    assert payload["possession_v3_events_file"] == "possession_events_v3.jsonl"


def test_runner_v4_artifacts_publish_public_pointer_hash_and_replay(tmp_path: Path):
    config = PipelineConfig(roboflow_api_key="test-key", possession_engine="v4")
    players = [
        {"frame_index": frame, "raw_track_id": 6, "stable_player_id": "player-006",
         "team_id": 1, "xyxy": [0, 0, 20, 40], "confidence": .9,
         "mask_valid": True, "mask_contains": True}
        for frame in range(3)
    ]
    balls = [{"frame_index": frame, "timestamp_seconds": frame / 10,
              "track_state": "detected", "confidence": .9, "center_xy": [10, 20]}
             for frame in range(3)]

    first = _write_v3_artifacts(tmp_path / "first", players, balls, fps=10, config=config)
    second = _write_v3_artifacts(tmp_path / "second", players, balls, fps=10, config=config)
    artifact = first["artifacts"]["v4_diagnostics"]
    artifact_path = tmp_path / "first" / artifact["path"]
    row = json.loads(artifact_path.read_text(encoding="utf-8").splitlines()[0])

    assert first["v4"] == second["v4"]
    assert first["generation"] == second["generation"]
    assert artifact["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert first["artifacts"]["v4_diagnostics"]["sha256"] == second["artifacts"]["v4_diagnostics"]["sha256"]
    assert row["v4"] == {"feature_flag": True, "schema_version": 1}
    assert row["config"]["engine"] == "v4"
    assert row["audit"]["feature_flag"] == "v4"

    results = PipelineResults(5, {}, {}, RunSummary(), "cpu",
                              possession_v3_file=first["artifacts"]["possession"]["path"],
                              possession_v3_schema_version=1,
                              possession_v3_manifest_file="possession_v3_manifest.json",
                              possession_v4_diagnostics_file=artifact["path"],
                              possession_v4_schema_version=1)
    payload = results.to_dict()
    assert payload["possession_v4_diagnostics_file"] == artifact["path"]
    assert payload["possession_v4_schema_version"] == 1


def test_identical_inputs_publish_separate_v4_and_v5_generations(tmp_path: Path):
    players = [{"frame_index": 0, "raw_track_id": 6, "stable_player_id": "player-006",
                "team_id": 1, "xyxy": [0, 0, 20, 40], "confidence": .9,
                "mask_valid": True, "mask_contains": True}]
    balls = [{"frame_index": 0, "timestamp_seconds": 0.0, "track_state": "detected",
              "confidence": .9, "center_xy": [10, 20]}]
    v4 = _write_v3_artifacts(tmp_path, players, balls, fps=10,
                             config=PipelineConfig(roboflow_api_key="key", possession_engine="v4"))
    v5 = _write_v3_artifacts(tmp_path, players, balls, fps=10,
                             config=PipelineConfig(roboflow_api_key="key", possession_engine="v5"))
    assert v4["generation"] != v5["generation"]
    assert v4["artifacts"]["possession"]["path"] != v5["artifacts"]["possession"]["path"]
    assert v4["v4"]["feature_flag"] == "v4"
    assert "v4" not in v5 and "v4_diagnostics" not in v5["artifacts"]


def test_v3_writer_is_feature_flag_isolated(tmp_path: Path):
    base = PipelineConfig(roboflow_api_key="test-key")
    assert _write_v3_artifacts(tmp_path / "legacy", [], [], fps=30, config=base) == {}
    assert not (tmp_path / "legacy" / "possession_v3.jsonl").exists()
    enabled = PipelineConfig(roboflow_api_key="test-key", possession_engine="v3")
    manifest = _write_v3_artifacts(tmp_path / "v3", [], [], fps=30, config=enabled)
    assert manifest["schema"] == "possession-engine-v3"
    assert (tmp_path / "v3" / "possession_v3_manifest.json").exists()


def test_event_window_controls_runner_transition_artifact(tmp_path: Path):
    rows = [{"frame_index": frame, "stable_player_id": holder}
            for frame, holder in ((0, "a"), (1, None), (2, None), (3, "b"), (4, "b"))]
    balls = [{"frame_index": frame, "timestamp_seconds": frame / 10, "track_state": "detected",
              "confidence": .9, "center_xy": [frame * 10, 0], "source_records": [f"ball#{frame}"],
              "source_frame_indices": [frame]}
             for frame in range(5)]
    config = PipelineConfig(roboflow_api_key="test-key", possession_engine="v3",
                            possession_v3_event_window_seconds=.2)
    manifest = _write_v3_artifacts(tmp_path, rows, balls, fps=10, config=config)
    transition_path = tmp_path / manifest["artifacts"]["transitions"]["path"]
    assert transition_path.read_text(encoding="utf-8") == ""


def test_v3_artifact_write_failure_does_not_leave_partial_files(tmp_path: Path, monkeypatch):
    from pipeline import results as results_module

    original = results_module.Path.write_text
    calls = 0

    def fail_after_first(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("diagnostic disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(results_module.Path, "write_text", fail_after_first)
    with pytest.raises(OSError):
        results_module.write_v3_artifacts(tmp_path, [], [], [], [])
    assert not list(tmp_path.glob("*.jsonl"))
    assert not (tmp_path / "possession_v3_manifest.json").exists()


def test_failed_new_generation_preserves_current_pointer(tmp_path: Path, monkeypatch):
    write_v3_artifacts(tmp_path, [], [], [], [], {"profile": "old"})
    previous = (tmp_path / "possession_v3_manifest.json").read_text(encoding="utf-8")
    from pipeline import results as results_module
    original = results_module.os.replace

    def fail_pointer(source, destination):
        if Path(destination).name == "possession_v3_manifest.json":
            raise OSError("pointer publication failed")
        return original(source, destination)

    monkeypatch.setattr(results_module.os, "replace", fail_pointer)
    with pytest.raises(OSError):
        write_v3_artifacts(tmp_path, [], [], [], [], {"profile": "new"})
    assert (tmp_path / "possession_v3_manifest.json").read_text(encoding="utf-8") == previous


def test_build_failure_preserves_last_known_good_pointer(tmp_path: Path, monkeypatch):
    config = PipelineConfig(roboflow_api_key="test-key", possession_engine="v3")
    first = _write_v3_artifacts(tmp_path, [], [], fps=10, config=config)
    pointer = (tmp_path / "possession_v3_manifest.json").read_text(encoding="utf-8")
    generation = first["generation"]
    import pipeline.runner as runner
    monkeypatch.setattr(runner, "normalize_ball_rows", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("build failed")))
    failed = runner._write_v3_artifacts(tmp_path, [], [], fps=10, config=config)
    assert failed["status"] == "failed"
    assert (tmp_path / "possession_v3_manifest.json").read_text(encoding="utf-8") == pointer
    assert (tmp_path / "possession_v3" / "generations" / generation).exists()


def test_failure_report_is_best_effort(tmp_path: Path, monkeypatch):
    import pipeline.runner as runner
    monkeypatch.setattr(runner, "_build_v3_artifacts", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("build failed")))
    monkeypatch.setattr(runner.Path, "write_text", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read-only")))
    result = runner._write_v3_artifacts(tmp_path, [], [], fps=10,
                                        config=PipelineConfig(roboflow_api_key="test-key", possession_engine="v3"))
    assert result["status"] == "failed"


def test_success_removes_stale_failure_report_and_bounds_generations(tmp_path: Path):
    (tmp_path / "possession_v3_failure.json").write_text("stale", encoding="utf-8")
    generation_root = tmp_path / "possession_v3" / "generations"
    generation_root.mkdir(parents=True)
    for name in ("old-a", "old-b", "old-c"):
        (generation_root / name).mkdir()
    _write_v3_artifacts(tmp_path, [], [], fps=10,
                        config=PipelineConfig(roboflow_api_key="test-key", possession_engine="v3"))
    assert not (tmp_path / "possession_v3_failure.json").exists()
    assert len(list(generation_root.iterdir())) <= 2


def test_success_cleans_stale_v3_staging_directory(tmp_path: Path):
    stale = tmp_path / ".possession-v3-stale"
    stale.mkdir()
    old = time.time() - 7200
    os.utime(stale, (old, old))
    _write_v3_artifacts(tmp_path, [], [], fps=10,
                        config=PipelineConfig(roboflow_api_key="test-key", possession_engine="v3"))
    assert not stale.exists()
