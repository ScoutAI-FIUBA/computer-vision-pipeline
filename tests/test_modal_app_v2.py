from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def fake_v2_stats(monkeypatch, request):
    if request.node.name in {
        "test_process_exercises_real_stats_adapter_and_manifest_contract",
        "test_process_uses_v3_image_config_and_returns_v3_artifacts",
        "test_process_end_to_end_number_detector_reaches_real_v2_stats_chain",
    } or request.node.name.startswith("test_process_possession_engine_is_explicit_and_persisted"):
        return
    import modal_app_v2

    def run(_video_path, output_dir, _observations, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text("{}", encoding="utf-8")
        (output_dir / "tracks.jsonl").write_text("", encoding="utf-8")
        return types.SimpleNamespace(
            to_dict=lambda: {
                "tracks_file": "tracks.jsonl",
                "teams_file": None,
                "numbers_file": None,
                "shot_events_file": None,
            }
        )

    monkeypatch.setattr(modal_app_v2, "_run_v2_stats", run)


def test_v2_and_legacy_gpu_capacity_are_explicitly_separate():
    v2_source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    legacy_source = (Path(__file__).parents[1] / "modal_app.py").read_text(encoding="utf-8")

    def decorated_gpu(source: str, class_name: str) -> str:
        tree = ast.parse(source)
        pipeline = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        modal_decorator = next(
            decorator
            for decorator in pipeline.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "app"
            and decorator.func.attr == "cls"
        )
        gpu = next(keyword for keyword in modal_decorator.keywords if keyword.arg == "gpu")
        assert isinstance(gpu.value, ast.Constant)
        assert isinstance(gpu.value.value, str)
        return gpu.value.value

    assert decorated_gpu(v2_source, "PipelineV2") == "A100-80GB"
    assert decorated_gpu(legacy_source, "Pipeline") == "L4"


def test_job_manifest_lifecycle_leases_and_stale_recovery(tmp_path: Path):
    from modal_app_v2 import acquire_job_lease, read_job_manifest, write_job_manifest

    write_job_manifest(tmp_path / "job-1", {"job_id": "job-1", "state": "created"})
    first = acquire_job_lease(tmp_path, "job-1", owner="worker-a", now=100, lease_seconds=30)
    assert first["attempts"]["tracking"] == 1
    with pytest.raises(RuntimeError, match="active lease"):
        acquire_job_lease(tmp_path, "job-1", owner="worker-b", now=110, lease_seconds=30)
    recovered = acquire_job_lease(tmp_path, "job-1", owner="worker-b", now=131, lease_seconds=30)
    assert recovered["attempts"]["tracking"] == 2
    assert read_job_manifest(tmp_path, "job-1")["lease"]["owner"] == "worker-b"


def test_lease_acquisition_is_atomic_and_initializes_running_state(tmp_path: Path):
    from modal_app_v2 import acquire_job_lease, read_job_manifest

    acquired = acquire_job_lease(
        tmp_path, "job-atomic", owner="worker-a", now=100,
        stage="tracking", input_fingerprint="input", request_fingerprint="request",
    )

    assert acquired["state"] == "tracking"
    assert acquired["lease"]["owner"] == "worker-a"
    with pytest.raises(RuntimeError, match="active lease"):
        acquire_job_lease(
            tmp_path, "job-atomic", owner="worker-b", now=101,
            stage="tracking", input_fingerprint="other", request_fingerprint="other",
        )
    assert read_job_manifest(tmp_path, "job-atomic")["input_fingerprint"] == "input"


def _raw_method(cls, name):
    method = cls.__dict__[name]
    return method.__dict__[next(key for key in method.__dict__ if key.startswith("_sync_original"))].raw_f


def test_track_stage_rejects_duplicate_active_worker_before_mutation(tmp_path: Path, monkeypatch):
    import modal_app_v2
    from modal_app_v2 import read_job_manifest, write_job_manifest

    monkeypatch.setattr(modal_app_v2, "V2_JOB_VOLUME_DIR", str(tmp_path))
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    write_job_manifest(tmp_path / "job-1", {
        "job_id": "job-1", "state": "created", "input_fingerprint": "other",
        "request_fingerprint": "other", "lease": {"owner": "worker-a", "expires_at": 9999999999},
    })
    with pytest.raises(RuntimeError, match="active lease"):
        _raw_method(cls, "track_stage")(
            cls(), job_id="job-1", video_bytes=b"video", worker_id="worker-b",
        )
    assert read_job_manifest(tmp_path, "job-1")["lease"]["owner"] == "worker-a"


def test_stage_finalizer_rejects_stale_owner_before_publish_or_manifest_write(tmp_path: Path):
    import modal_app_v2
    from modal_app_v2 import acquire_job_lease, read_job_manifest, write_job_manifest

    job = tmp_path / "job-1"
    acquire_job_lease(tmp_path, "job-1", owner="worker-a", stage="tracking", now=100, lease_seconds=30)
    write_job_manifest(job, {**read_job_manifest(job), "lease": {"owner": "worker-b", "expires_at": 9999999999}})
    published = []

    with pytest.raises(RuntimeError, match="lease ownership"):
        modal_app_v2._finalize_stage(
            tmp_path, "job-1", owner="worker-a",
            publish=lambda: published.append(True),
            manifest={"state": "tracked", "lease": None},
        )

    assert published == []
    assert read_job_manifest(job)["lease"]["owner"] == "worker-b"


def test_stage_owner_requires_unique_identity_and_accepts_explicit_or_modal_id(monkeypatch):
    import modal_app_v2

    for name in ("MODAL_TASK_ID", "MODAL_FUNCTION_CALL_ID", "MODAL_INVOCATION_ID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="unique worker identity"):
        modal_app_v2._stage_owner(None)
    assert modal_app_v2._stage_owner("test-worker") == "test-worker"

    monkeypatch.setenv("MODAL_INVOCATION_ID", "invocation-1")
    assert modal_app_v2._stage_owner(None) == "modal:MODAL_INVOCATION_ID:invocation-1"


def test_tracking_fingerprint_changes_with_effective_environment_configuration(monkeypatch):
    import modal_app_v2
    from pipeline.config import PipelineConfig

    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.delenv("SAM31_V2_SEGMENT_WINDOW", raising=False)
    first = PipelineConfig.from_env()
    first_fingerprint = modal_app_v2._tracking_stage_fingerprint(
        "player", 0, 1, None, None, None, None, None, first,
    )
    monkeypatch.setenv("SAM31_V2_SEGMENT_WINDOW", "64")
    second = PipelineConfig.from_env()
    second_fingerprint = modal_app_v2._tracking_stage_fingerprint(
        "player", 0, 1, None, None, None, None, None, second,
    )
    assert first_fingerprint != second_fingerprint

    first_stats = modal_app_v2._stats_stage_fingerprint("v3", "tracking-1", first)
    monkeypatch.setenv("POSSESSION_V3_GRACE_SECONDS", "0.5")
    third = PipelineConfig.from_env()
    second_stats = modal_app_v2._stats_stage_fingerprint("v3", "tracking-1", third)
    assert first_stats != second_stats


def test_stats_stage_reuses_matching_completed_generation_without_recomputing(tmp_path: Path, monkeypatch):
    import modal_app_v2
    import pipeline.reid
    import pipeline.team_embedder
    from modal_app_v2 import write_job_manifest
    from pipeline.results import package_artifacts, publish_stage_generation

    monkeypatch.setattr(modal_app_v2, "V2_JOB_VOLUME_DIR", str(tmp_path))
    job = tmp_path / "job-1"
    job.mkdir()
    package = job / "package"
    package_artifacts(package, [], {}, input_fingerprint="input", frame_bounds=[0, 0])
    tracking_artifacts = {
        ("tracking-manifest.json" if path.name == "manifest.json" else path.relative_to(package).as_posix()): path
        for path in package.rglob("*") if path.is_file()
    }
    tracking = publish_stage_generation(
        job / "tracking", stage="tracking", input_fingerprint="input", request_fingerprint="request",
        artifacts=tracking_artifacts, frame_bounds=[0, 0],
    )
    stats_request = modal_app_v2._stats_stage_fingerprint(
        "v3", tracking["generation"], modal_app_v2.PipelineConfig.from_env(possession_engine="v3")
    )
    stats = publish_stage_generation(
        job / "stats", stage="stats", input_fingerprint="input", request_fingerprint=stats_request,
        tracking_generation=tracking["generation"], artifacts={"results.json": b"{}"}, frame_bounds=[0, 0],
    )
    write_job_manifest(job, {
        "job_id": "job-1", "state": "complete", "input_fingerprint": "input",
        "request_fingerprint": "request", "tracking_generation": tracking["generation"],
        "stats_generation": stats["generation"],
    })
    monkeypatch.setattr(modal_app_v2, "run_stats_from_v2_generation", lambda *args, **kwargs: pytest.fail("recomputed stats"))
    monkeypatch.setattr(pipeline.reid, "build_embedder", lambda *args, **kwargs: pytest.fail("ReID initialized"))
    monkeypatch.setattr(pipeline.team_embedder, "build_team_embedder", lambda *args, **kwargs: pytest.fail("Team ID initialized"))
    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_model_builder.build_sam3_multiplex_video_predictor = lambda **kwargs: pytest.fail("SAM3 initialized")
    fake_sam3 = types.ModuleType("sam3")
    fake_sam3.model_builder = fake_model_builder
    monkeypatch.setitem(sys.modules, "sam3", fake_sam3)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", fake_model_builder)
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    pipeline = cls()
    load_models = cls.load_models.__dict__[next(key for key in cls.load_models.__dict__ if key.startswith("_sync_original"))].raw_f
    load_models(pipeline)
    assert not hasattr(pipeline, "predictor")
    assert not hasattr(pipeline, "seams")
    result = _raw_method(cls, "stats_stage")(pipeline, job_id="job-1", possession_engine="v3", worker_id="worker-a")
    assert result["generation"] == stats["generation"]


def test_staged_process_orchestrates_tracking_then_stats(tmp_path: Path, monkeypatch):
    import modal_app_v2

    monkeypatch.setattr(modal_app_v2, "V2_JOB_VOLUME_DIR", str(tmp_path))
    calls = []
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    instance = cls()
    monkeypatch.setattr(instance, "track_stage", lambda **_kwargs: pytest.fail("Modal track descriptor called"))
    monkeypatch.setattr(instance, "stats_stage", lambda **_kwargs: pytest.fail("Modal stats descriptor called"))
    monkeypatch.setattr(instance, "_track_stage", lambda **kwargs: calls.append(("track", kwargs)) or {"status": "tracked"})
    monkeypatch.setattr(instance, "_stats_stage", lambda **kwargs: calls.append(("stats", kwargs)) or {"status": "complete"})

    result = _raw_method(cls, "process")(
        instance, video_bytes=b"video", job_id="job-1", staged=True,
        prompt="player", frame_start=0, frame_end=1, possession_engine="v3",
    )

    assert result == {"status": "complete"}
    assert [name for name, _ in calls] == ["track", "stats"]
    assert calls[0][1]["job_id"] == calls[1][1]["job_id"] == "job-1"


def test_staged_process_passes_tracking_payload_without_modal_serialization(tmp_path: Path, monkeypatch):
    import modal_app_v2

    monkeypatch.setattr(modal_app_v2, "V2_JOB_VOLUME_DIR", str(tmp_path))
    payload = {
        "observations": [{"observation_key": "0:1"}],
        "masks_by_observation": {"0:1": object()},
        "clean_masks_by_observation": {},
    }
    captured = {}
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    instance = cls()
    monkeypatch.setattr(instance, "_track_stage", lambda **_kwargs: {"status": "tracked", "tracking_payload": payload})

    def stats(**kwargs):
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(instance, "_stats_stage", stats)
    result = _raw_method(cls, "process")(
        instance, video_bytes=b"video", job_id="job-1", staged=True,
    )

    assert result == {"status": "complete"}
    assert captured["tracking_payload"] is payload


def test_track_stage_publishes_v2_enriched_rows_and_clean_masks(tmp_path: Path, monkeypatch):
    import numpy as np
    import modal_app_v2
    from pipeline.results import load_stage_generation
    from pipeline.runner_v2 import RunnerV2Result

    monkeypatch.setattr(modal_app_v2, "V2_JOB_VOLUME_DIR", str(tmp_path))
    row = {
        "observation_key": "0:3", "frame_index": 0, "raw_track_id": 3,
        "bbox_xyxy": [0, 0, 2, 2], "track_state": "tracked",
    }
    mask = np.ones((2, 2), dtype=np.uint8)
    clean_mask = np.eye(2, dtype=np.uint8)
    enriched = {**row, "stable_player_id": "player-3", "team_id": 1}
    captured = {}

    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *_args: ([row], {"0:3": mask}))
    monkeypatch.setattr(cls := next(value for key, value in modal_app_v2.PipelineV2.__dict__.items()
                                    if key.startswith("_sync_original"))._user_cls,
                        "_ensure_predictor", lambda _self: object())
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {0: object()})
    monkeypatch.setattr(modal_app_v2, "run_v2", lambda observations, **kwargs: (
        captured.update({"rows": observations, **kwargs})
        or RunnerV2Result([enriched], cleaned_masks_by_observation={"0:3": clean_mask})
    ))
    result = _raw_method(cls, "track_stage")(
        cls(), job_id="job-1", video_bytes=b"video", frame_start=0, frame_end=0,
        worker_id="worker-a",
    )
    generation = load_stage_generation(tmp_path / "job-1" / "tracking", result["generation"])
    tracks = [json.loads(line) for line in (tmp_path / "job-1" / "tracking" / "generations" /
                                            result["generation"] / "tracks.jsonl").read_text().splitlines()]

    assert captured["rows"] == [row]
    assert captured["masks_by_observation"]["0:3"] is mask
    assert tracks[0]["stable_player_id"] == "player-3"
    assert tracks[0]["team_id"] == 1
    assert tracks[0]["clean_mask_ref"].startswith("masks_clean/")
    tracking_manifest = json.loads(
        (tmp_path / "job-1" / "tracking" / "generations" / result["generation"] /
         "tracking-manifest.json").read_text()
    )
    assert tracking_manifest["artifacts"]["clean_masks"]["count"] == 1


def test_stats_stage_reads_enriched_staged_tracking_contract(tmp_path: Path, monkeypatch):
    import numpy as np
    import modal_app_v2
    from pipeline.results import package_artifacts, publish_stage_generation
    from modal_app_v2 import write_job_manifest

    monkeypatch.setattr(modal_app_v2, "V2_JOB_VOLUME_DIR", str(tmp_path))
    job = tmp_path / "job-1"
    package = job / "package"
    enriched = [{
        "observation_key": "0:3", "frame_index": 0, "raw_track_id": 3,
        "bbox_xyxy": [0, 0, 2, 2], "stable_player_id": "player-3", "team_id": 1,
        "track_state": "tracked",
    }]
    package_artifacts(package, enriched, {"0:3": np.ones((2, 2), dtype=np.uint8)},
                      input_fingerprint="input", frame_bounds=[0, 0])
    artifacts = {
        ("tracking-manifest.json" if path.name == "manifest.json" else path.relative_to(package).as_posix()): path
        for path in package.rglob("*") if path.is_file()
    }
    tracking = publish_stage_generation(
        job / "tracking", stage="tracking", input_fingerprint="input", request_fingerprint="request",
        artifacts=artifacts, frame_bounds=[0, 0],
    )
    write_job_manifest(job, {
        "job_id": "job-1", "state": "tracked", "input_fingerprint": "input",
        "tracking_generation": tracking["generation"],
    })
    captured = {}

    def fake_stats(video_path, config, tracking_root, output, generation, **kwargs):
        captured["observations"] = json.loads(
            (tracking_root / "generations" / generation["generation"] / "tracks.jsonl").read_text().splitlines()[0]
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text("{}")
        return object()

    monkeypatch.setattr(modal_app_v2, "run_stats_from_v2_generation", fake_stats)
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items()
               if key.startswith("_sync_original"))._user_cls

    _raw_method(cls, "stats_stage")(cls(), job_id="job-1", worker_id="worker-a")

    assert captured["observations"]["stable_player_id"] == "player-3"
    assert captured["observations"]["team_id"] == 1


def test_possession_only_stage_loads_durable_inputs_records_failure_and_retries_idempotently(
    tmp_path: Path, monkeypatch
):
    import numpy as np
    import modal_app_v2
    import pipeline.runner as runner
    from modal_app_v2 import read_job_manifest, write_job_manifest
    from pipeline.results import (
        _artifact_ref, _manifest_hash, load_stage_generation, package_artifacts,
        publish_stage_generation,
    )

    monkeypatch.setattr(modal_app_v2, "V2_JOB_VOLUME_DIR", str(tmp_path))
    job = tmp_path / "job-1"
    package = job / "package"
    observations = [{
        "observation_key": "0:3", "frame_index": 0, "raw_track_id": 3,
        "bbox_xyxy": [0, 0, 2, 2], "stable_player_id": "player-3", "team_id": 1,
        "track_state": "tracked",
    }]
    package_artifacts(
        package, observations, {"0:3": np.ones((2, 2), dtype=np.uint8)},
        input_fingerprint="input", frame_bounds=[0, 0],
        video_metadata={"fps": 30, "width": 2, "height": 2},
    )
    ball_path = package / "ball_tracks.jsonl"
    ball_path.write_text(
        json.dumps({"frame_index": 0, "xyxy": [1, 1, 2, 2], "track_state": "tracked"}) + "\n",
        encoding="utf-8",
    )
    tracking_manifest_path = package / "manifest.json"
    tracking_manifest = json.loads(tracking_manifest_path.read_text(encoding="utf-8"))
    tracking_manifest["artifacts"]["ball_tracks"] = _artifact_ref(ball_path, "ball_tracks.jsonl")
    tracking_manifest["artifacts"]["manifest"]["sha256"] = _manifest_hash(tracking_manifest)
    tracking_manifest_path.write_text(json.dumps(tracking_manifest, sort_keys=True), encoding="utf-8")
    tracking_artifacts = {
        ("tracking-manifest.json" if path.name == "manifest.json" else path.relative_to(package).as_posix()): path
        for path in package.rglob("*") if path.is_file()
    }
    tracking = publish_stage_generation(
        job / "tracking", stage="tracking", input_fingerprint="input", request_fingerprint="tracking-request",
        artifacts=tracking_artifacts, frame_bounds=[0, 0],
    )
    write_job_manifest(job, {
        "job_id": "job-1", "state": "tracked", "input_fingerprint": "input",
        "tracking_generation": tracking["generation"],
    })

    heavy_passes = (
        "_project_v2_rows", "_write_shot_events", "_maybe_write_projected_shots",
        "run_v2_number_evidence", "build_tracknet_detector",
    )
    for name in heavy_passes:
        monkeypatch.setattr(runner, name, lambda *args, _name=name, **kwargs: pytest.fail(_name))
    monkeypatch.setattr(modal_app_v2, "run_stats_from_v2_generation", lambda *args, **kwargs: pytest.fail("stats"))
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *args, **kwargs: pytest.fail("TrackNet/player tracking"))
    monkeypatch.setattr(modal_app_v2, "build_number_predictor", lambda *args, **kwargs: pytest.fail("numbers"))
    monkeypatch.setattr(modal_app_v2, "build_detector", lambda *args, **kwargs: pytest.fail("numbers"))

    original_helper = modal_app_v2.run_possession_only_from_v2_artifacts
    helper_calls = []

    def fail_once(output, loaded_observations, loaded_ball_rows, **kwargs):
        helper_calls.append((loaded_observations, loaded_ball_rows, kwargs))
        raise RuntimeError("possession transient failure")

    monkeypatch.setattr(modal_app_v2, "run_possession_only_from_v2_artifacts", fail_once)
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    with pytest.raises(RuntimeError, match="transient failure"):
        _raw_method(cls, "possession_only_stage")(
            cls(), job_id="job-1", possession_engine="v5", worker_id="worker-a"
        )

    failed = read_job_manifest(job)
    assert failed["attempts"]["possession"] == 1
    assert failed["lease"] is None
    assert failed["errors"][-1]["stage"] == "possession"
    assert helper_calls[0][0] == observations
    assert helper_calls[0][1][0]["frame_index"] == 0

    def succeed(*args, **kwargs):
        helper_calls.append((args[1], args[2], kwargs))
        return original_helper(*args, **kwargs)

    monkeypatch.setattr(modal_app_v2, "run_possession_only_from_v2_artifacts", succeed)
    first = _raw_method(cls, "possession_only_stage")(
        cls(), job_id="job-1", possession_engine="v5", worker_id="worker-a"
    )
    second = _raw_method(cls, "possession_only_stage")(
        cls(), job_id="job-1", possession_engine="v5", worker_id="worker-b"
    )

    assert first == second
    assert len(helper_calls) == 2
    assert read_job_manifest(job)["attempts"]["possession"] == 2
    possession = load_stage_generation(job / "possession", first["generation"])
    assert possession["stage"] == "possession"
    assert possession["tracking_generation"] == tracking["generation"]
    assert possession["complete"] is True
    assert len(list((job / "possession" / "generations").iterdir())) == 1


def test_stale_and_replacement_inputs_are_isolated_and_stats_rejects_tampering(tmp_path: Path, monkeypatch):
    import modal_app_v2
    from modal_app_v2 import _committed_job_input, _persist_job_input, write_job_manifest
    from pipeline.results import package_artifacts, publish_stage_generation

    job = tmp_path / "job-1"
    stale = b"stale-worker-video"
    committed = b"replacement-worker-video"
    stale_fingerprint = "sha256:" + hashlib.sha256(stale).hexdigest()
    committed_fingerprint = "sha256:" + hashlib.sha256(committed).hexdigest()

    stale_path = _persist_job_input(job, stale_fingerprint, stale)
    committed_path = _persist_job_input(job, committed_fingerprint, committed)

    assert stale_path != committed_path
    assert stale_path.read_bytes() == stale
    assert committed_path.read_bytes() == committed

    # A legacy mutable path must not win over the committed immutable input.
    (job / "input.mp4").write_bytes(stale)
    assert _committed_job_input(job, committed_fingerprint) == committed_path

    captured = {}
    package = job / "package"
    package_artifacts(package, [], {}, input_fingerprint=committed_fingerprint, frame_bounds=[0, 0])
    artifacts = {
        ("tracking-manifest.json" if path.name == "manifest.json" else path.relative_to(package).as_posix()): path
        for path in package.rglob("*") if path.is_file()
    }
    tracking = publish_stage_generation(
        job / "tracking", stage="tracking", input_fingerprint=committed_fingerprint,
        request_fingerprint="request", artifacts=artifacts, frame_bounds=[0, 0],
    )
    write_job_manifest(job, {
        "job_id": "job-1", "state": "tracked", "input_fingerprint": committed_fingerprint,
        "tracking_generation": tracking["generation"],
    })
    monkeypatch.setattr(modal_app_v2, "V2_JOB_VOLUME_DIR", str(tmp_path))

    def fake_stats(video_path, *args, **kwargs):
        captured["video_path"] = video_path
        output = args[2]
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text("{}")
        return object()

    monkeypatch.setattr(modal_app_v2, "run_stats_from_v2_generation", fake_stats)
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items()
               if key.startswith("_sync_original"))._user_cls
    _raw_method(cls, "stats_stage")(cls(), job_id="job-1", worker_id="worker-stats")
    assert captured["video_path"] == committed_path

    import shutil
    shutil.rmtree(job / "stats")
    committed_path.write_bytes(b"tampered replacement-worker-video")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _raw_method(cls, "stats_stage")(cls(), job_id="job-1", worker_id="worker-stats-2")


def test_committed_input_does_not_fallback_when_immutable_input_is_missing(tmp_path: Path):
    from modal_app_v2 import _committed_job_input

    legacy = tmp_path / "input.mp4"
    legacy.write_bytes(b"mutable legacy input")
    fingerprint = "sha256:" + ("a" * 64)

    with pytest.raises(ValueError, match="immutable job input is missing"):
        _committed_job_input(tmp_path, fingerprint)
    assert _committed_job_input(tmp_path, "legacy") == legacy


def test_committed_input_rejects_tampered_immutable_input(tmp_path: Path):
    from modal_app_v2 import _committed_job_input, _persist_job_input

    content = b"committed video"
    fingerprint = "sha256:" + hashlib.sha256(content).hexdigest()
    path = _persist_job_input(tmp_path, fingerprint, content)
    path.write_bytes(b"tampered video")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _committed_job_input(tmp_path, fingerprint)


def test_persist_job_input_uses_atomic_rename_for_modal_compatible_publication(
    tmp_path: Path, monkeypatch
):
    import modal_app_v2

    content = b"modal-compatible video"
    fingerprint = "sha256:" + hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(modal_app_v2.os, "link", lambda *_args: pytest.fail("os.link must not be used"))

    path = modal_app_v2._persist_job_input(tmp_path / "job-1", fingerprint, content)

    assert path.read_bytes() == content


def test_persist_job_input_rejects_hash_mismatch_before_publication(tmp_path: Path):
    from modal_app_v2 import _persist_job_input

    fingerprint = "sha256:" + hashlib.sha256(b"expected").hexdigest()
    job = tmp_path / "job-1"

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _persist_job_input(job, fingerprint, b"actual")

    assert not (job / "inputs").exists()


def test_persist_job_input_keeps_existing_matching_destination(tmp_path: Path):
    from modal_app_v2 import _persist_job_input

    content = b"already committed"
    fingerprint = "sha256:" + hashlib.sha256(content).hexdigest()
    job = tmp_path / "job-1"
    first = _persist_job_input(job, fingerprint, content)

    assert _persist_job_input(job, fingerprint, content) == first
    assert list((job / "inputs").iterdir()) == [first]


def test_persist_job_input_cleans_temp_file_when_publication_fails(tmp_path: Path, monkeypatch):
    import modal_app_v2

    content = b"publication failure"
    fingerprint = "sha256:" + hashlib.sha256(content).hexdigest()

    def fail_replace(*_args):
        raise PermissionError("rename failed")

    monkeypatch.setattr(modal_app_v2.os, "replace", fail_replace)
    job = tmp_path / "job-1"

    with pytest.raises(PermissionError, match="rename failed"):
        modal_app_v2._persist_job_input(job, fingerprint, content)

    assert list((job / "inputs").iterdir()) == []


def test_staged_process_records_failure_and_does_not_fallback(tmp_path: Path, monkeypatch):
    import modal_app_v2

    monkeypatch.setattr(modal_app_v2, "V2_JOB_VOLUME_DIR", str(tmp_path))
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    instance = cls()
    monkeypatch.setattr(instance, "_track_stage", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("tracking failed")))
    monkeypatch.setattr(instance, "_stats_stage", lambda **_kwargs: pytest.fail("stats must not run"))

    with pytest.raises(RuntimeError, match="tracking failed"):
        _raw_method(cls, "process")(instance, video_bytes=b"video", job_id="job-1", staged=True)


def test_v2_defaults_to_on_court_prompt():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"process", "main"}:
            if node.name == "process":
                defaults[node.name] = next(
                    default
                    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                    if argument.arg == "prompt"
                )
            else:
                defaults[node.name] = next(
                    default
                    for argument, default in zip(node.args.args, node.args.defaults)
                    if argument.arg == "prompt"
                )

    assert all(
        isinstance(default, ast.Constant) and default.value == "basketball player on the court"
        for default in defaults.values()
    )
    assert set(defaults) == {"process", "main"}


def test_v2_image_isolated_and_uses_only_modal_model_storage():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")

    assert "modal_app.py" not in source
    assert "facebookresearch/sam3" in source
    assert "facebookresearch/sam2" not in source
    assert "R2" not in source
    assert "r2_key" not in source
    assert "output_prefix" not in source
    assert "R2Storage" not in source
    assert '"HF_HOME":' in source
    assert "HF_TOKEN" in source
    assert source.count("modal.Image") == 1


def test_v2_image_mounts_and_enables_tracknet_without_sam2():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")

    assert '"TRACKNET_ENABLED": "1"' in source
    assert '"TRACKNET_CHECKPOINT": TRACKNET_CHECKPOINT' in source
    assert '"BALL_ALL_FRAMES": "1"' in source
    assert '"BBOX_SANITIZER_ENABLED": "1"' in source
    assert '"tracknet-data", environment_name="main", create_if_missing=False' in source
    assert 'volumes={CACHE_DIR: model_cache, TRACKNET_VOLUME_DIR: tracknet_data}' in source


def test_v2_image_enables_v3_without_changing_legacy_config(monkeypatch):
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    env_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "env"
    )
    env = {
        key.value: value.value
        for key, value in zip(env_call.args[0].keys, env_call.args[0].values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    }
    assert env["POSSESSION_ENGINE"] == "v3"

    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("POSSESSION_ENGINE", env["POSSESSION_ENGINE"])
    from pipeline.config import PipelineConfig

    assert PipelineConfig.from_env().possession_v3_enabled is True

    monkeypatch.delenv("POSSESSION_ENGINE")
    legacy = PipelineConfig.from_env()
    assert legacy.possession_engine == "legacy"
    assert legacy.possession_v3_enabled is False


def test_v2_image_enables_hosted_jersey_number_predictor_without_changing_local_default():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    env_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "env"
    )
    env = {
        key.value: value.value
        for key, value in zip(env_call.args[0].keys, env_call.args[0].values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    }
    assert env["NUMBER_PREDICTOR_BACKEND"] == "roboflow-hosted"

    from pipeline.config import PipelineConfig

    assert PipelineConfig.from_env(roboflow_api_key="test-key").number_predictor_backend == "dummy"


def test_v2_runtime_declaration_attaches_roboflow_secret_without_hydrating_modal():
    import modal_app_v2

    service_function = modal_app_v2.PipelineV2._get_class_service_function()
    secret_names = {secret._name for secret in service_function.spec.secrets}

    assert "scoutai-roboflow" in secret_names


def test_v2_image_pins_sam31_runtime_dependencies():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    pip_dependencies = [
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pip_install"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    ]

    assert "setuptools<81" in pip_dependencies
    assert "einops" in pip_dependencies
    assert "pycocotools" in pip_dependencies
    assert "Pillow" in pip_dependencies
    assert "psutil" in pip_dependencies


def test_v2_image_pins_scipy_for_numpy_1_26_compatibility():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    pip_dependencies = [
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pip_install"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    ]

    assert "scipy==1.14.1" in pip_dependencies


def test_v2_image_pins_reviewed_sam31_commit():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")

    assert "2345a4a" in source
    assert "SAM31_REPOSITORY}@{SAM31_COMMIT}" in source
    assert "git+https://github.com/facebookresearch/sam3.git\")" not in source


def test_v2_dependency_smoke_reports_each_import_and_builder():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    smoke = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sam31_dependency_smoke_test"
    )
    source_segment = ast.get_source_segment(source, smoke)
    assert source_segment is not None
    assert "run_sam31_dependency_smoke" in source_segment
    for module_name in (
        "pycocotools.mask",
        "PIL",
        "psutil",
        "einops",
        "scipy",
        "cv2",
        "sam3.model_builder",
    ):
        assert module_name in source
    assert "callable" in source
    assert "raise RuntimeError" in source


def test_v2_dependency_smoke_does_not_construct_model():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    smoke = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sam31_dependency_smoke_test"
    )

    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_sam3_multiplex_video_predictor"
        for node in ast.walk(smoke)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PipelineV2"
        for node in ast.walk(smoke)
    )


def test_v2_import_check_attempts_every_module_and_reports_failures_independently():
    from modal_app_v2 import SAM31_SMOKE_MODULES, check_sam31_imports

    attempted = []
    failed_modules = {"PIL", "scipy"}

    def fake_import_module(module_name):
        attempted.append(module_name)
        if module_name in failed_modules:
            raise ImportError(f"blocked {module_name}")
        if module_name == "sam3.model_builder":
            return types.SimpleNamespace(build_sam3_multiplex_video_predictor=lambda: None)
        return types.SimpleNamespace()

    report = check_sam31_imports(import_module=fake_import_module)

    assert attempted == list(SAM31_SMOKE_MODULES)
    assert report["imported"] == [module for module in SAM31_SMOKE_MODULES if module not in failed_modules]
    assert set(report["failures"]) == failed_modules
    assert all(module in report["failures"] for module in failed_modules)
    assert all("blocked" in report["failures"][module] for module in failed_modules)


def test_v2_import_check_reports_non_callable_builder_without_skipping_modules():
    from modal_app_v2 import SAM31_SMOKE_MODULES, check_sam31_imports

    attempted = []

    def fake_import_module(module_name):
        attempted.append(module_name)
        return types.SimpleNamespace()

    report = check_sam31_imports(import_module=fake_import_module)

    assert attempted == list(SAM31_SMOKE_MODULES)
    assert report["imported"] == [module for module in SAM31_SMOKE_MODULES if module != "sam3.model_builder"]
    assert set(report["failures"]) == {"sam3.model_builder"}
    assert "not callable" in report["failures"]["sam3.model_builder"]


def test_v2_smoke_wrapper_raises_concise_failure_for_controlled_checker():
    from modal_app_v2 import run_sam31_dependency_smoke

    calls = []

    def fake_checker():
        calls.append(True)
        return {"ok": False, "imported": [], "failures": {"PIL": "blocked", "scipy": "blocked"}}

    with pytest.raises(
        RuntimeError,
        match=r"^SAM3\.1 dependency smoke failed: PIL, scipy$",
    ):
        run_sam31_dependency_smoke(import_checker=fake_checker)

    assert calls == [True]


def test_v2_smoke_wrapper_returns_successful_checker_report():
    from modal_app_v2 import run_sam31_dependency_smoke

    report = {"ok": True, "imported": ["PIL"], "failures": {}}

    assert run_sam31_dependency_smoke(import_checker=lambda: report) is report


def test_v2_baseline_predictor_disables_optional_flashattention3():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    predictor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_sam3_multiplex_video_predictor"
    ]

    assert len(predictor_calls) == 1
    assert [
        keyword.value.value for keyword in predictor_calls[0].keywords if keyword.arg == "use_fa3"
    ] == [False]


def test_v2_predictor_initialization_disables_optional_flashattention3(monkeypatch):
    import modal_app_v2

    calls = []
    fake_predictor = object()

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return fake_predictor

    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_model_builder.build_sam3_multiplex_video_predictor = fake_builder
    fake_sam3 = types.ModuleType("sam3")
    fake_sam3.model_builder = fake_model_builder
    monkeypatch.setitem(sys.modules, "sam3", fake_sam3)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", fake_model_builder)
    monkeypatch.setattr(modal_app_v2, "model_cache", types.SimpleNamespace(commit=lambda: None))

    pipeline_cls = next(
        value
        for key, value in modal_app_v2.PipelineV2.__dict__.items()
        if key.startswith("_sync_original")
    )._user_cls

    pipeline = pipeline_cls()
    assert pipeline._ensure_predictor() is fake_predictor
    assert calls == [
        {"max_num_objects": 16, "multiplex_count": 16, "use_fa3": False}
    ]
    assert pipeline._ensure_predictor() is fake_predictor
    assert calls == [
        {"max_num_objects": 16, "multiplex_count": 16, "use_fa3": False}
    ]


def test_v2_tracking_models_are_lazy_and_cached(monkeypatch):
    import modal_app_v2
    import pipeline.reid
    import pipeline.team_embedder
    from pipeline.runner_v2 import RuntimeIdentityAuthority

    predictor_calls = []
    cache_commits = []
    reid_calls = []
    team_calls = []
    fake_predictor = object()
    fake_reid = object()
    fake_team = object()

    def fake_sam_builder(**kwargs):
        predictor_calls.append(kwargs)
        return fake_predictor

    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_model_builder.build_sam3_multiplex_video_predictor = fake_sam_builder
    fake_sam3 = types.ModuleType("sam3")
    fake_sam3.model_builder = fake_model_builder
    monkeypatch.setitem(sys.modules, "sam3", fake_sam3)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", fake_model_builder)
    monkeypatch.setattr(
        modal_app_v2,
        "model_cache",
        types.SimpleNamespace(commit=lambda: cache_commits.append(True)),
    )
    monkeypatch.setenv("REID_BACKEND", "fake-reid")
    monkeypatch.setenv("TEAM_EMBEDDER_BACKEND", "fake-team")
    monkeypatch.setenv("TEAM_EMBEDDER_MODEL_ID", "fake-team-model")
    monkeypatch.setattr(
        pipeline.reid,
        "build_embedder",
        lambda backend, weights: reid_calls.append((backend, weights)) or fake_reid,
    )
    monkeypatch.setattr(
        pipeline.team_embedder,
        "build_team_embedder",
        lambda backend, model_id: team_calls.append((backend, model_id)) or fake_team,
    )

    pipeline_cls = next(
        value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original")
    )._user_cls
    load_models = pipeline_cls.load_models.__dict__[
        next(key for key in pipeline_cls.load_models.__dict__ if key.startswith("_sync_original"))
    ].raw_f

    pipeline = pipeline_cls()
    load_models(pipeline)

    assert pipeline._models_loaded is True
    assert not hasattr(pipeline, "predictor")
    assert not hasattr(pipeline, "seams")
    assert predictor_calls == []
    pipeline._ensure_predictor()
    pipeline._ensure_seams()
    assert pipeline.predictor is fake_predictor
    assert predictor_calls == [{"max_num_objects": 16, "multiplex_count": 16, "use_fa3": False}]
    assert reid_calls == [("fake-reid", "/cache/reid/osnet_x1_0_msmt17.pt")]
    assert team_calls == [("fake-team", "fake-team-model")]
    assert pipeline.seams.identity is not None
    assert pipeline.seams.reid is fake_reid
    assert pipeline.seams.team_embedder is fake_team
    assert isinstance(pipeline.seams.identity_authority, RuntimeIdentityAuthority)
    assert len(cache_commits) == 2


def test_entrypoint_accepts_nonempty_video_and_validates_public_values():
    from modal_app_v2 import validate_entrypoint_args

    validate_entrypoint_args(video_bytes=b"video", prompt="player", frame_start=0, frame_end=1)

    with pytest.raises(ValueError, match="non-empty"):
        validate_entrypoint_args(video_bytes=b"", prompt="player", frame_start=0, frame_end=1)
    with pytest.raises(ValueError, match="prompt"):
        validate_entrypoint_args(video_bytes=b"video", prompt=" ", frame_start=0, frame_end=1)
    with pytest.raises(ValueError, match="frame"):
        validate_entrypoint_args(video_bytes=b"video", prompt="player", frame_start=2, frame_end=1)


@pytest.mark.parametrize(
    ("video_bytes", "prompt", "frame_start", "frame_end"),
    [
        ("video", "player", 0, 1),
        (b"video", "player", -1, 1),
        (b"video", "player", True, 1),
        (b"video", "player", 0, 1.0),
    ],
)
def test_entrypoint_rejects_wrong_public_types(video_bytes, prompt, frame_start, frame_end):
    from modal_app_v2 import validate_entrypoint_args

    with pytest.raises(ValueError):
        validate_entrypoint_args(
            video_bytes=video_bytes,
            prompt=prompt,
            frame_start=frame_start,
            frame_end=frame_end,
        )


def test_direct_artifact_mapping_returns_relative_bytes(tmp_path: Path):
    from modal_app_v2 import collect_artifacts

    (tmp_path / "manifest.json").write_text('{"complete": false}', encoding="utf-8")
    (tmp_path / "masks").mkdir()
    (tmp_path / "masks" / "1.png").write_bytes(b"png")

    artifacts = collect_artifacts(tmp_path)

    assert artifacts == {
        "manifest.json": b'{"complete": false}',
        "masks/1.png": b"png",
    }
    assert all(not Path(name).is_absolute() for name in artifacts)


def test_direct_artifact_mapping_rejects_configured_size_limit(tmp_path: Path):
    from modal_app_v2 import collect_artifacts

    (tmp_path / "tracks.jsonl").write_bytes(b"1234")

    with pytest.raises(ValueError, match="exceeds configured artifact return limit"):
        collect_artifacts(tmp_path, max_bytes=3)


def test_direct_artifact_mapping_reads_size_limit_from_environment(tmp_path: Path, monkeypatch):
    from modal_app_v2 import collect_artifacts

    (tmp_path / "tracks.jsonl").write_bytes(b"1234")
    monkeypatch.setenv("SAM31_MAX_ARTIFACT_BYTES", "3")

    with pytest.raises(ValueError, match="exceeds configured artifact return limit"):
        collect_artifacts(tmp_path)


def test_direct_artifact_mapping_rejects_symlinked_file_outside_output_root(tmp_path: Path):
    from modal_app_v2 import collect_artifacts

    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    outside = tmp_path / "secret.bin"
    outside.write_bytes(b"secret")
    (output_dir / "tracks.jsonl").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink|outside"):
        collect_artifacts(output_dir)


def test_direct_artifact_mapping_rejects_symlinked_directory(tmp_path: Path):
    from modal_app_v2 import collect_artifacts

    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "mask.png").write_bytes(b"secret")
    (output_dir / "masks").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        collect_artifacts(output_dir)


def test_local_output_writer_recreates_nested_artifact_paths(tmp_path: Path):
    from modal_app_v2 import write_artifacts

    write_artifacts(
        {"manifest.json": b"{}", "masks/1.png": b"png"},
        tmp_path / "output",
    )

    assert (tmp_path / "output" / "manifest.json").read_bytes() == b"{}"
    assert (tmp_path / "output" / "masks" / "1.png").read_bytes() == b"png"


@pytest.mark.parametrize("relative_path", ["", ".", "/tmp/out", "../escape", "masks/../../escape", "bad\\path"])
def test_local_output_writer_rejects_unsafe_artifact_paths(tmp_path: Path, relative_path: str):
    from modal_app_v2 import write_artifacts

    with pytest.raises(ValueError, match="artifact path"):
        write_artifacts({relative_path: b"nope"}, tmp_path / "output")

    assert not (tmp_path / "escape").exists()


def test_local_entrypoint_writes_remote_artifact_bytes(tmp_path: Path, monkeypatch, capsys):
    import modal_app_v2

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")

    class FakeProcess:
        def remote(self, **kwargs):
            assert kwargs["video_bytes"] == b"video"
            assert kwargs["return_artifacts"] is True
            return {"acceptance": {"tracking_identity": "unvalidated"}, "artifacts": {"manifest.json": b"{}"}}

    class FakePipeline:
        process = FakeProcess()

    monkeypatch.setattr(modal_app_v2, "PipelineV2", FakePipeline)
    modal_app_v2.main(video_path=str(video), output_dir=str(tmp_path / "output"))

    assert (tmp_path / "output" / "manifest.json").read_bytes() == b"{}"
    assert "tracking_identity" in capsys.readouterr().out


@pytest.mark.parametrize("selected", ["v4", "v5", "v3"])
def test_local_entrypoint_forwards_selected_possession_engine(
    tmp_path: Path, monkeypatch, capsys, selected: str
):
    import modal_app_v2

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    forwarded = {}

    class FakeProcess:
        def remote(self, **kwargs):
            forwarded.update(kwargs)
            return {"acceptance": {}, "artifacts": {}}

    class FakePipeline:
        process = FakeProcess()

    monkeypatch.setattr(modal_app_v2, "PipelineV2", FakePipeline)
    modal_app_v2.main(
        video_path=str(video), output_dir=str(tmp_path / "output"), possession_engine=selected
    )

    assert forwarded["possession_engine"] == selected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"staged": True, "stats_only": True, "job_id": "job"}, "mutually exclusive"),
        ({"tracking_only": True, "staged": True, "job_id": "job", "video_path": "input.mp4"}, "mutually exclusive"),
        ({"tracking_only": True, "stats_only": True, "job_id": "job", "video_path": "input.mp4"}, "mutually exclusive"),
        ({"tracking_only": True}, "job-id is required"),
        ({"tracking_only": True, "job_id": "job"}, "video-path is required"),
        ({"staged": True}, "job-id is required"),
        ({"stats_only": True}, "job-id is required"),
        ({"job_id": "job"}, "requires --staged, --stats-only, --tracking-only, or --possession-only"),
        ({"stats_only": True, "job_id": "job", "video_path": "input.mp4"}, "cannot be used"),
        ({"staged": True, "job_id": "job"}, "video-path is required"),
    ],
)
def test_local_entrypoint_validates_restartable_cli_modes(kwargs, message):
    from modal_app_v2 import _validate_local_entrypoint_mode

    with pytest.raises(ValueError, match=message):
        _validate_local_entrypoint_mode(
            video_path=kwargs.get("video_path", ""),
            job_id=kwargs.get("job_id", ""),
            staged=kwargs.get("staged", False),
            stats_only=kwargs.get("stats_only", False),
            tracking_only=kwargs.get("tracking_only", False),
        )


def test_local_entrypoint_routes_staged_process_to_deployed_method(tmp_path, monkeypatch):
    import modal_app_v2

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    forwarded = {}

    class FakeProcess:
        def remote(self, **kwargs):
            forwarded.update(kwargs)
            return {"status": "complete"}

    class FakePipeline:
        process = FakeProcess()

    monkeypatch.setattr(modal_app_v2, "PipelineV2", FakePipeline)
    modal_app_v2.main(video_path=str(video), staged=True, job_id="job-1")

    assert forwarded["staged"] is True
    assert forwarded["job_id"] == "job-1"


@pytest.mark.parametrize("selected", ["v3", "v5"])
def test_local_entrypoint_routes_stats_only_to_deployed_stats_method(monkeypatch, capsys, selected):
    import modal_app_v2

    forwarded = {}

    class FakeStats:
        def remote(self, **kwargs):
            forwarded.update(kwargs)
            return {"status": "complete", "generation": "stats-1"}

    class FakePipeline:
        stats_stage = FakeStats()

    monkeypatch.setattr(modal_app_v2, "PipelineV2", FakePipeline)
    modal_app_v2.main(stats_only=True, job_id="job-1", possession_engine=selected)

    assert forwarded == {"job_id": "job-1", "possession_engine": selected}
    assert "stats-1" in capsys.readouterr().out


def test_local_entrypoint_routes_possession_only_without_video(monkeypatch, capsys):
    import modal_app_v2

    forwarded = {}

    class FakePossession:
        def remote(self, **kwargs):
            forwarded.update(kwargs)
            return {"status": "complete", "generation": "possession-1"}

    class FakePipeline:
        possession_only_stage = FakePossession()

    monkeypatch.setattr(modal_app_v2, "PipelineV2", FakePipeline)
    modal_app_v2.main(possession_only=True, job_id="job-1", possession_engine="v5")

    assert forwarded == {"job_id": "job-1", "possession_engine": "v5"}
    assert "possession-1" in capsys.readouterr().out


def test_possession_only_rejects_video_and_other_modes():
    from modal_app_v2 import _validate_local_entrypoint_mode

    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_local_entrypoint_mode(
            video_path="input.mp4", job_id="job-1", staged=True,
            stats_only=False, tracking_only=False, possession_only=True,
        )
    with pytest.raises(ValueError, match="cannot be used"):
        _validate_local_entrypoint_mode(
            video_path="input.mp4", job_id="job-1", staged=False,
            stats_only=False, tracking_only=False, possession_only=True,
        )


def test_stats_fingerprint_is_engine_specific():
    from modal_app_v2 import _stats_stage_fingerprint
    from pipeline.config import PipelineConfig

    v4 = PipelineConfig(roboflow_api_key="key", possession_engine="v4")
    v5 = PipelineConfig(roboflow_api_key="key", possession_engine="v5")
    assert _stats_stage_fingerprint("v4", "tracking-1", v4) != _stats_stage_fingerprint("v5", "tracking-1", v5)


def test_local_entrypoint_routes_tracking_only_to_deployed_track_method(tmp_path, monkeypatch, capsys):
    import modal_app_v2

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    forwarded = {}

    class FakeTrack:
        def remote(self, **kwargs):
            forwarded.update(kwargs)
            return {"status": "tracked", "generation": "tracking-1", "fingerprint": "sha256:test"}

    class FakePipeline:
        track_stage = FakeTrack()

    monkeypatch.setattr(modal_app_v2, "PipelineV2", FakePipeline)
    modal_app_v2.main(video_path=str(video), tracking_only=True, job_id="job-1")

    assert forwarded == {
        "job_id": "job-1",
        "video_bytes": b"video",
        "prompt": "basketball player on the court",
        "frame_start": 60,
        "frame_end": 130,
        "segmented": None,
        "segment_window": None,
        "segment_overlap": None,
        "segment_seam_min_score": None,
        "segment_seam_ambiguity_margin": None,
    }
    assert "tracking-1" in capsys.readouterr().out


def test_v2_process_returns_artifacts_and_never_remote_debug_video():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")

    assert 'payload["artifacts"]' in source
    assert "return_artifacts" in source
    assert "debug.mp4" not in source
    assert "upload" not in source.lower()


def test_failed_or_incomplete_local_manifest_is_returned_without_claiming_complete(tmp_path: Path):
    from modal_app_v2 import collect_artifacts

    manifest = {"complete": False, "truncated": True, "transfer_errors": ["artifact too large"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    returned = json.loads(collect_artifacts(tmp_path)["manifest.json"])

    assert returned == manifest


def test_legacy_modal_file_is_not_imported_by_v2():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    assert "import modal_app" not in source


@pytest.mark.parametrize("accepts_offload", [True, False])
def test_start_session_filters_rebuilt_init_state_kwargs_and_returns_session_id(
    accepts_offload: bool, tmp_path: Path
):
    from modal_app_v2 import _start_session

    class FakeModel:
        if accepts_offload:

            def init_state(self, resource_path, offload_state_to_cpu, async_loading_frames):
                return resource_path, offload_state_to_cpu, async_loading_frames
        else:

            def init_state(self, resource_path, async_loading_frames):
                return resource_path, async_loading_frames

    class FakePredictor:
        def __init__(self):
            self.model = FakeModel()
            self.requests = []
            self.session_id = None

        def handle_request(self, request):
            self.requests.append(request)
            if request["type"] == "start_session":
                assert "offload_state_to_cpu" in request
                assert "offload_video_to_cpu" in request
                self.model.init_state(**{key: value for key, value in request.items() if key != "type"})
                self.session_id = "compatible-session"
                return {"session_id": self.session_id}
            return None

        def handle_stream_request(self, request):
            assert request["session_id"] == self.session_id
            return iter(())

    predictor = FakePredictor()
    start = _start_session(predictor, tmp_path / "clip.mp4")

    assert start["session_id"] == "compatible-session"
    start_request = predictor.requests[0]
    assert start_request["resource_path"] == str(tmp_path / "clip.mp4")
    assert start_request["async_loading_frames"] is False
    assert start_request["offload_state_to_cpu"] is True
    assert start_request["offload_video_to_cpu"] is True


def test_start_session_propagates_offload_video_to_cpu_when_supported(tmp_path: Path):
    from modal_app_v2 import _start_session

    class FakeModel:
        def init_state(self, resource_path, offload_video_to_cpu, async_loading_frames):
            return resource_path, offload_video_to_cpu, async_loading_frames

    class FakePredictor:
        def __init__(self):
            self.model = FakeModel()
            self.init_result = None

        def handle_request(self, request):
            self.init_result = self.model.init_state(
                **{key: value for key, value in request.items() if key != "type"}
            )
            return {"session_id": "video-offload"}

    predictor = FakePredictor()

    assert _start_session(predictor, tmp_path / "clip.mp4")["session_id"] == "video-offload"
    assert predictor.init_result == (str(tmp_path / "clip.mp4"), True, False)


def test_start_session_filters_each_rebuilt_kwarg_against_intermediate_signature(tmp_path: Path):
    from modal_app_v2 import _start_session

    class FakeModel:
        def init_state(self, resource_path, offload_state_to_cpu):
            return resource_path, offload_state_to_cpu

    class FakePredictor:
        def __init__(self):
            self.model = FakeModel()
            self.init_result = None

        def handle_request(self, request):
            self.init_result = self.model.init_state(
                **{key: value for key, value in request.items() if key != "type"}
            )
            return {"session_id": "intermediate-signature"}

    predictor = FakePredictor()

    start = _start_session(predictor, tmp_path / "clip.mp4")

    assert start["session_id"] == "intermediate-signature"
    assert predictor.init_result == (str(tmp_path / "clip.mp4"), True)


def test_start_session_restores_unsupported_init_state_after_success(tmp_path: Path):
    from modal_app_v2 import _start_session

    class FakeModel:
        def init_state(self, resource_path, async_loading_frames):
            return resource_path, async_loading_frames

    class FakePredictor:
        def __init__(self):
            self.model = FakeModel()

        def handle_request(self, request):
            self.model.init_state(**{key: value for key, value in request.items() if key != "type"})
            return {"session_id": "restored-success"}

    predictor = FakePredictor()
    original = predictor.model.init_state

    assert _start_session(predictor, tmp_path / "clip.mp4")["session_id"] == "restored-success"
    assert predictor.model.init_state.__func__ is FakeModel.init_state
    assert predictor.model.init_state.__self__ is original.__self__


def test_start_session_restores_unsupported_init_state_after_failure(tmp_path: Path):
    from modal_app_v2 import _start_session

    expected = RuntimeError("session failed")

    class FakeModel:
        def init_state(self, resource_path, async_loading_frames):
            raise expected

    class FakePredictor:
        def __init__(self):
            self.model = FakeModel()

        def handle_request(self, request):
            self.model.init_state(**{key: value for key, value in request.items() if key != "type"})

    predictor = FakePredictor()

    with pytest.raises(RuntimeError) as error:
        _start_session(predictor, tmp_path / "clip.mp4")

    assert error.value is expected
    assert predictor.model.init_state.__func__ is FakeModel.init_state


def test_track_video_propagates_session_id_and_closes_in_lifecycle_order(tmp_path: Path):
    from modal_app_v2 import _track_video

    class FakeModel:
        def init_state(self, resource_path, offload_state_to_cpu, async_loading_frames):
            return resource_path, offload_state_to_cpu, async_loading_frames

    class FakePredictor:
        def __init__(self):
            self.model = FakeModel()
            self.calls = []

        def handle_request(self, request):
            self.calls.append(request)
            if request["type"] == "start_session":
                self.model.init_state(**{key: value for key, value in request.items() if key != "type"})
                return {"session_id": "session-17"}
            return None

        def handle_stream_request(self, request):
            self.calls.append(request)
            return iter(())

    predictor = FakePredictor()

    assert _track_video(predictor, tmp_path / "clip.mp4", "person", 4, 8) == ([], {})
    assert [call["type"] for call in predictor.calls] == [
        "start_session",
        "add_prompt",
        "propagate_in_video",
        "close_session",
    ]
    assert predictor.calls[1]["session_id"] == "session-17"
    assert predictor.calls[2]["session_id"] == "session-17"
    assert predictor.calls[3]["session_id"] == "session-17"


def test_track_video_normalizes_official_nested_output_envelope(tmp_path: Path):
    import numpy as np

    from modal_app_v2 import _track_video
    from pipeline.runner_v2 import MaskAppearanceAdapter

    class FakeModel:
        def init_state(self, **_kwargs):
            return None

    class FakePredictor:
        model = FakeModel()

        def handle_request(self, request):
            if request["type"] == "start_session":
                return {"session_id": "session-17"}
            return None

        def handle_stream_request(self, _request):
            return iter(
                [
                    {
                        "frame_index": 42,
                        "outputs": {
                            "out_obj_ids": [7],
                            "out_boxes_xywh": [[0.1, 0.2, 0.4, 0.6]],
                            "out_probs": [0.8],
                        },
                    }
                ]
            )

    rows, masks = _track_video(FakePredictor(), tmp_path / "clip.mp4", "person", 42, 42)

    assert rows[0]["frame_index"] == 42
    assert rows[0]["raw_track_id"] == 7
    assert rows[0]["bbox_xyxy"] == [0.1, 0.2, 0.5, 0.8]
    assert rows[0]["bbox_coordinate_space"] == "normalized"
    assert rows[0]["probability"] == 0.8
    assert masks == {}

    frame = np.ones((40, 50, 3), dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=bool)
    mask[8:32, 5:25] = True
    crop = MaskAppearanceAdapter().crop(
        frame, rows[0]["bbox_xyxy"], mask, coordinate_space=rows[0]["bbox_coordinate_space"]
    )
    assert crop is not None
    assert crop.bbox_xyxy == (5, 8, 25, 32)


def test_track_video_normalizes_empty_nested_outputs(tmp_path: Path):
    from modal_app_v2 import _track_video

    class FakeModel:
        def init_state(self, **_kwargs):
            return None

    class FakePredictor:
        model = FakeModel()

        def handle_request(self, request):
            if request["type"] == "start_session":
                return {"session_id": "session-17"}
            return None

        def handle_stream_request(self, _request):
            return iter([{"frame_index": 42, "outputs": {}}])

    assert _track_video(FakePredictor(), tmp_path / "clip.mp4", "person", 42, 42) == ([], {})


def test_track_video_rejects_malformed_non_empty_nested_outputs(tmp_path: Path):
    from modal_app_v2 import _track_video

    class FakeModel:
        def init_state(self, **_kwargs):
            return None

    class FakePredictor:
        model = FakeModel()

        def handle_request(self, request):
            if request["type"] == "start_session":
                return {"session_id": "session-17"}
            return None

        def handle_stream_request(self, _request):
            return iter(
                [
                    {
                        "frame_index": 42,
                        "outputs": {
                            "out_obj_ids": [7, 8],
                            "out_boxes_xywh": [[1, 2, 3, 4]],
                        },
                    }
                ]
            )

    with pytest.raises(RuntimeError, match="object ids and boxes"):
        _track_video(FakePredictor(), tmp_path / "clip.mp4", "person", 42, 42)


def test_session_objects_keeps_direct_output_compatibility():
    from modal_app_v2 import _session_objects

    assert _session_objects(
        {"out_obj_ids": [7], "out_boxes_xywh": [[1, 2, 3, 4]], "out_probs": [0.8]}
    ) == [
        {
            "object_id": 7,
            "bbox": [1, 2, 3, 4],
            "bbox_format": "xywh",
            "probability": 0.8,
            "mask": None,
        }
    ]


def test_track_video_propagates_genuine_start_exception_unchanged(tmp_path: Path):
    from modal_app_v2 import _track_video

    expected = RuntimeError("init failed")

    class FakeModel:
        def init_state(self, resource_path, offload_state_to_cpu, async_loading_frames):
            raise expected

    class FakePredictor:
        def __init__(self):
            self.model = FakeModel()

        def handle_request(self, request):
            return self.model.init_state(
                **{key: value for key, value in request.items() if key != "type"}
            )

    with pytest.raises(RuntimeError) as error:
        _track_video(FakePredictor(), tmp_path / "clip.mp4", "person", 4, 8)

    assert error.value is expected


def test_process_returns_complete_raw_artifacts_from_fake_session(monkeypatch):
    import cv2
    import numpy as np
    import modal_app_v2
    from pipeline.runner_v2 import RunnerV2, RunnerV2Seams, RuntimeIdentityAuthority

    class FakePredictor:
        class Model:
            def init_state(self, **_kwargs):
                return None

        model = Model()

        def handle_request(self, request):
            if request["type"] == "start_session":
                return {"session_id": "fake-session"}
            return None

        def handle_stream_request(self, _request):
            yield {
                "frame_index": 60,
                "out_obj_ids": [3],
                "out_boxes_xywh": [[1, 2, 10, 20]],
                "out_probs": [0.9],
                "out_binary_masks": [np.array([[0, 255]], dtype=np.uint8)],
            }

    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {"fps": 30.0})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(
        modal_app_v2,
        "run_v2",
        lambda rows, **kwargs: RunnerV2(RunnerV2Seams(identity=lambda items: {item["observation_key"]: "player-3" for item in items})).run(rows, **kwargs),
    )
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: FakePredictor())

    payload = process(cls(),
        video_bytes=b"fake-video", frame_start=60, frame_end=60, return_artifacts=True
    )

    artifacts = payload["artifacts"]
    assert payload["acceptance"] == {"tracking_identity": "unvalidated", "statistics": "deferred"}
    assert {"tracks.jsonl", "manifest.json"} <= set(artifacts)
    mask = np.array([[0, 255]], dtype=np.uint8)
    mask_path = next(name for name in artifacts if name.startswith("masks/"))
    manifest = json.loads(artifacts["manifest.json"])
    assert manifest["complete"] is True
    assert manifest["observation_count"] == 1
    assert manifest["config"] == {
        "prompt": "basketball player on the court",
        "frame_end": 60,
        "frame_start": 60,
    }
    assert "wall_seconds" in manifest["benchmark"]
    observation = json.loads(artifacts["tracks.jsonl"].decode())
    assert observation == {
        "bbox_xyxy": [1.0, 2.0, 11.0, 22.0],
        "confidence": None,
        "frame_index": 60,
        "mask_ref": mask_path,
        "observation_key": "60:3",
        "probability": 0.9,
        "raw_track_id": 3,
        "stable_player_id": "player-3",
        "timestamp_seconds": 0.0,
        "track_state": "tracked",
    }
    decoded_mask = cv2.imdecode(np.frombuffer(artifacts[mask_path], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(decoded_mask, mask)


def test_process_persists_clean_masks_without_replacing_raw_masks(monkeypatch):
    import cv2
    import numpy as np
    import modal_app_v2
    from pipeline.runner_v2 import run_v2

    rows = [
        {"observation_key": f"{frame}:3", "frame_index": frame, "raw_track_id": 3,
         "bbox_xyxy": list(box), "track_state": "tracked"}
        for frame, box in enumerate(((10, 10, 20, 30), (100, 100, 180, 190), (12, 10, 22, 30)))
    ]
    masks = {}
    for row in rows:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 11:21] = 255
        mask[35:38, 35:38] = 255
        masks[row["observation_key"]] = mask

    monkeypatch.setenv("BBOX_SANITIZER_ENABLED", "1")
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *_args: (rows, masks))
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(modal_app_v2, "run_v2", run_v2)
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())

    payload = process(cls(), video_bytes=b"fake-video", frame_start=0, frame_end=2, return_artifacts=True)
    artifacts = payload["artifacts"]
    tracks = [json.loads(line) for line in artifacts["tracks.jsonl"].decode().splitlines()]
    manifest = json.loads(artifacts["manifest.json"])

    middle = tracks[1]
    assert middle["mask_ref"].startswith("masks/")
    assert middle["clean_mask_ref"].startswith("masks_clean/")
    assert middle["mask_ref"] != middle["clean_mask_ref"]
    assert middle["raw_bbox"] == [100, 100, 180, 190]
    assert middle["bbox_xyxy"] == [11.0, 10.0, 21.0, 30.0]
    assert any(name.startswith("masks_clean/") for name in artifacts)
    assert manifest["artifacts"]["clean_masks"]["count"] == 3
    assert manifest["config"]["bbox_sanitizer"]["mask_cleanup_count"] == 3
    raw = cv2.imdecode(np.frombuffer(artifacts[middle["mask_ref"]], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    clean = cv2.imdecode(np.frombuffer(artifacts[middle["clean_mask_ref"]], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert int(raw.sum()) > int(clean.sum())


def test_process_exercises_real_stats_adapter_and_manifest_contract(monkeypatch):
    import numpy as np
    import modal_app_v2
    from pipeline.config import PipelineConfig
    import pipeline.runner as legacy_runner

    config = PipelineConfig(
        roboflow_api_key="test-key",
        reid_backend="none",
        team_embedder_backend="none",
        court_model_id=None,
        number_predictor_backend="dummy",
        prefetch_detections=False,
        max_frames=1,
        team_debug_enabled=False,
    )
    probe_calls = []
    info = types.SimpleNamespace(
        fps=30.0,
        frame_count=1,
        width=4,
        height=4,
        duration_seconds=1 / 30,
    )
    segment = types.SimpleNamespace(
        frames=[np.zeros((4, 4, 3), dtype=np.uint8)],
        frame_indices=[0],
        is_last=True,
    )

    class FakeDetector:
        def __init__(self):
            self.call_count = 0

        def detect(self, _frame):
            self.call_count += 1
            return {"predictions": []}

    class FakeTracker:
        device = "cpu"

        def propagate_segment(self, _frames, _seed_rows):
            return {}

    monkeypatch.setattr(PipelineConfig, "from_env", classmethod(lambda cls: config))
    monkeypatch.setattr(legacy_runner, "probe", lambda path: probe_calls.append(path) or info)
    monkeypatch.setattr(legacy_runner, "iter_segments", lambda *_args, **_kwargs: iter((segment,)))
    monkeypatch.setattr(legacy_runner, "build_detector", lambda *_args, **_kwargs: FakeDetector())
    monkeypatch.setattr(legacy_runner, "Sam2SegmentTracker", lambda _model_id: FakeTracker())
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *_args: ([], {}))
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(modal_app_v2, "build_detector", lambda _config: (_ for _ in ()).throw(
        AssertionError("number detector must not initialize without stable V2 players")
    ))
    monkeypatch.setattr(modal_app_v2, "run_v2", lambda rows, **_kwargs: types.SimpleNamespace(
        observations=rows, acceptance={}, acceptance_reasons=[]
    ))
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f

    payload = process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60, return_artifacts=True)

    assert [path.name for path in probe_calls] == ["input.mp4"]
    artifacts = payload["artifacts"]
    manifest = json.loads(artifacts["manifest.json"])
    assert manifest["artifacts"]["legacy_stats"]["results"]["key"] == "stats/results.json"
    assert manifest["artifacts"]["legacy_stats"]["tracks"]["key"] == "stats/tracks.jsonl"
    assert manifest["artifacts"]["legacy_stats"]["possession_events"]["key"] == "stats/possession_events.jsonl"
    assert manifest["artifacts"]["legacy_stats"]["possession_transitions"]["key"] == "stats/possession_transitions.jsonl"
    assert manifest["artifacts"]["legacy_stats"]["shot_evidence"]["key"] == "stats/shot_evidence.jsonl"
    assert {"stats/results.json", "stats/tracks.jsonl", "stats/teams.json", "stats/shot_events.jsonl", "stats/shot_evidence.jsonl", "stats/possession_events.jsonl", "stats/possession_transitions.jsonl"} <= set(artifacts)
    results = json.loads(artifacts["stats/results.json"])
    assert results["schema_version"] == 5
    assert results["tracks_file"] == "tracks.jsonl"
    assert results["teams_file"] == "teams.json"
    # This fixture has no V2 player rows, so the adapter must preserve the
    # no-evidence/null contract. Configured rows are covered by the real adapter
    # integration test and produce numbers artifacts.
    assert results["numbers_file"] is None
    assert "numbers" not in manifest["artifacts"]["legacy_stats"]
    assert results["shot_events_file"] == "shot_events.jsonl"
    assert results["possession_events_file"] == "possession_events.jsonl"
    assert results["possession_transitions_file"] == "possession_transitions.jsonl"
    assert {"stats/results.json", "stats/tracks.jsonl", "stats/teams.json", "stats/shot_events.jsonl", "stats/possession_events.jsonl"} <= set(artifacts)
    for name, reference in manifest["artifacts"]["legacy_stats"].items():
        data = artifacts[reference["key"]]
        assert reference["size_bytes"] == len(data)
        assert reference["sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["legacy_stats_results"] == "stats/results.json"


def test_process_disabled_number_evidence_skips_predictor_and_detector(monkeypatch):
    import modal_app_v2

    rows = [{"observation_key": "60:3", "frame_index": 60, "raw_track_id": 3, "stable_player_id": "stable-3",
             "bbox_xyxy": [1, 2, 11, 22], "track_state": "tracked"}]
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("NUMBER_EVIDENCE_ENABLED", "0")
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *_args: (rows, {}))
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(modal_app_v2, "run_v2", lambda observations, **_kwargs: types.SimpleNamespace(
        observations=observations, acceptance={}, acceptance_reasons=[]
    ))
    monkeypatch.setattr(modal_app_v2, "build_number_predictor", lambda _config: (_ for _ in ()).throw(
        AssertionError("disabled number predictor must not initialize")
    ))
    monkeypatch.setattr(modal_app_v2, "build_detector", lambda _config: (_ for _ in ()).throw(
        AssertionError("disabled number detector must not initialize")
    ))
    seen = {}

    def disabled_stats(_video, output_dir, _observations, **kwargs):
        seen.update(kwargs)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in {
            "results.json": json.dumps({"numbers_file": None}),
            "tracks.jsonl": "", "teams.json": "{}", "shot_events.jsonl": "",
        }.items():
            (output_dir / name).write_text(content, encoding="utf-8")
        return types.SimpleNamespace(to_dict=lambda: {
            "tracks_file": "tracks.jsonl", "teams_file": "teams.json",
            "numbers_file": None, "shot_events_file": "shot_events.jsonl",
        })

    monkeypatch.setattr(modal_app_v2, "_run_v2_stats", disabled_stats)
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items()
               if key.startswith("_sync_original"))._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__
                                       if key.startswith("_sync_original"))].raw_f

    payload = process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60, return_artifacts=True)
    assert seen["number_predictor"] is None
    assert seen["number_detector"] is None
    results = json.loads(payload["artifacts"]["stats/results.json"])
    assert results["numbers_file"] is None
    assert not any(name.startswith("stats/number") for name in payload["artifacts"])


def test_process_populated_v2_number_boundary_reuses_predictor_and_hashes_artifacts(monkeypatch):
    import modal_app_v2

    rows = [{
        "observation_key": "60:3", "frame_index": 60, "raw_track_id": 3,
        "stable_player_id": "player-3", "bbox_xyxy": [1, 2, 11, 22],
        "track_state": "tracked",
    }]
    predictor = object()
    predictor_configs = []
    seen = []
    detector = object()

    monkeypatch.setenv("NUMBER_PREDICTOR_BACKEND", "roboflow-hosted")
    monkeypatch.setenv("NUMBER_PREDICTOR_ROBOFLOW_MODEL_ID", "test-jersey-model/9")
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")

    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *_args: (rows, {}))
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(modal_app_v2, "run_v2", lambda observations, **_kwargs: types.SimpleNamespace(
        observations=observations, acceptance={}, acceptance_reasons=[],
    ))
    def fake_build_number_predictor(config):
        predictor_configs.append(config)
        return predictor

    monkeypatch.setattr(modal_app_v2, "build_number_predictor", fake_build_number_predictor)
    monkeypatch.setattr(modal_app_v2, "build_detector", lambda _config: detector)

    def fake_stats(_video, output_dir, _observations, **kwargs):
        seen.append(kwargs["number_predictor"])
        seen.append(kwargs["number_detector"])
        output_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "results.json": json.dumps({"numbers_file": "numbers.json"}),
            "tracks.jsonl": "{}\n", "teams.json": "{}", "numbers.json": json.dumps({"players": []}),
            "number_predictions.json": json.dumps({"predictions": []}),
            "number_prediction_cases.jsonl": "", "number_manifest.json": json.dumps({"artifacts": {}}),
            "shot_events.jsonl": "",
        }
        for name, content in files.items():
            (output_dir / name).write_text(content, encoding="utf-8")
        return types.SimpleNamespace(to_dict=lambda: {
            "tracks_file": "tracks.jsonl", "teams_file": "teams.json", "numbers_file": "numbers.json",
            "shot_events_file": "shot_events.jsonl",
        })

    monkeypatch.setattr(modal_app_v2, "_run_v2_stats", fake_stats)
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())

    payload = process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60, return_artifacts=True)
    artifacts = payload["artifacts"]
    assert len(predictor_configs) == 1
    assert predictor_configs[0].number_predictor_backend == "roboflow-hosted"
    assert predictor_configs[0].number_predictor_roboflow_model_id == "test-jersey-model/9"
    assert predictor_configs[0].roboflow_api_key == "test-key"
    assert seen == [predictor, detector]
    manifest = json.loads(artifacts["manifest.json"])
    stats_refs = manifest["artifacts"]["legacy_stats"]
    for name in ("numbers", "number_predictions", "number_prediction_cases", "number_manifest"):
        reference = stats_refs[name]
        data = artifacts[reference["key"]]
        assert reference["size_bytes"] == len(data)
        assert reference["sha256"] == hashlib.sha256(data).hexdigest()
    results = json.loads(artifacts["stats/results.json"])
    assert results["numbers_file"] == "numbers.json"


def test_process_end_to_end_number_detector_reaches_real_v2_stats_chain(monkeypatch):
    import numpy as np
    import modal_app_v2
    import pipeline.runner as runner
    from pipeline.config import PipelineConfig
    from pipeline.number_predictor import NumberPrediction

    rows = [{
        "observation_key": "60:7", "frame_index": 60, "raw_track_id": 7,
        "stable_player_id": "stable-7", "bbox_xyxy": [5, 5, 25, 35],
        "track_state": "tracked", "confidence": 0.95,
    }]
    info = types.SimpleNamespace(fps=30.0, frame_count=1, width=60, height=40, duration_seconds=1 / 30)

    class Detector:
        def __init__(self): self.frames = []
        def detect(self, frame):
            self.frames.append(int(frame[0, 0, 0]))
            return {"predictions": [{
                "class": "number", "x": 15, "y": 15, "width": 10, "height": 8,
                "confidence": 0.91,
            }]}

    class Predictor:
        name = "fake"
        def __init__(self): self.samples = []
        def predict(self, sample):
            self.samples.append(sample)
            return NumberPrediction("23", 0.99)

    detector, predictor = Detector(), Predictor()
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setattr(PipelineConfig, "from_env", classmethod(lambda cls: PipelineConfig(
        roboflow_api_key="test-key", frame_stride=1, team_embedder_backend="none",
        number_predictor_backend="dummy", possession_engine="legacy",
    )))
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *_args: (rows, {}))
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "run_v2", lambda observations, **_kwargs: types.SimpleNamespace(
        observations=observations, acceptance={}, acceptance_reasons=[], runtime_diagnostics={},
    ))
    monkeypatch.setattr(modal_app_v2, "build_detector", lambda _config: detector)
    monkeypatch.setattr(modal_app_v2, "build_number_predictor", lambda _config: predictor)
    monkeypatch.setattr(runner, "probe", lambda _path: info)

    class Capture:
        def __init__(self, _path): self.frame = None
        def set(self, _property, frame): self.frame = frame
        def read(self): return True, np.full((40, 60, 3), self.frame or 0, dtype=np.uint8)
        def release(self): pass

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        VideoCapture=Capture, CAP_PROP_POS_FRAMES=1,
    ))
    monkeypatch.setattr(runner, "_write_shot_events", lambda _video, path, **_kwargs: path.write_text(""))
    monkeypatch.setattr(runner, "_attach_ball_evidence", lambda output_dir, **_kwargs: (
        (output_dir / "shot_evidence.jsonl").write_text("")
    ))

    def write_possession(output_dir, *_args, **_kwargs):
        events, transitions = output_dir / "possession_events.jsonl", output_dir / "possession_transitions.jsonl"
        events.write_text("")
        transitions.write_text("")
        return events, transitions

    monkeypatch.setattr(runner, "_write_possession_artifacts", write_possession)
    monkeypatch.setattr(runner, "_write_v3_artifacts", lambda *_args, **_kwargs: {})

    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items()
               if key.startswith("_sync_original"))._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__
                                       if key.startswith("_sync_original"))].raw_f

    payload = process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60, return_artifacts=True)
    artifacts = payload["artifacts"]
    assert detector.frames == [60]
    assert [sample.frame_index for sample in predictor.samples] == [60]
    assert predictor.samples[0].number_xyxy == (10.0, 11.0, 20.0, 19.0)

    prediction = json.loads(artifacts["stats/number_predictions.json"])["predictions"][0]
    assert prediction["stable_player_id"] == "stable-7"
    assert prediction["evidence"]["source"] == "legacy_basketball_object_detector"
    assert prediction["evidence"]["association_method"] == "object_detector_number_box"
    number_manifest = json.loads(artifacts["stats/number_manifest.json"])
    assert number_manifest["evidence_sources"] == ["legacy_basketball_object_detector"]
    numbers = json.loads(artifacts["stats/numbers.json"])
    assert numbers["players"][0]["stable_player_id"] == "stable-7"

    manifest = json.loads(artifacts["manifest.json"])
    stats = manifest["artifacts"]["legacy_stats"]
    assert stats["numbers"]["key"] == "stats/numbers.json"
    assert stats["number_predictions"]["key"] == "stats/number_predictions.json"
    assert stats["number_manifest"]["key"] == "stats/number_manifest.json"
    results = json.loads(artifacts["stats/results.json"])
    assert results["numbers_file"] == "numbers.json"


def test_process_uses_v3_image_config_and_returns_v3_artifacts(monkeypatch):
    import modal_app_v2
    from pipeline.config import PipelineConfig

    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("POSSESSION_ENGINE", "v3")
    configs = []
    original_from_env = PipelineConfig.from_env

    def from_env(cls):
        config = original_from_env()
        configs.append(config)
        return config

    monkeypatch.setattr(PipelineConfig, "from_env", classmethod(from_env))
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *_args: ([], {}))
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(
        modal_app_v2,
        "run_v2",
        lambda rows, **_kwargs: types.SimpleNamespace(
            observations=rows, acceptance={}, acceptance_reasons=[]
        ),
    )

    def fake_stats(_video_path, output_dir, _observations, *, config, **_kwargs):
        assert config.possession_v3_enabled is True
        output_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "results.json": "{}",
            "tracks.jsonl": "",
            "teams.json": "{}",
            "shot_events.jsonl": "",
        }
        for name, contents in files.items():
            (output_dir / name).write_text(contents, encoding="utf-8")
        generation = output_dir / "possession_v3" / "generations" / "fake-generation"
        generation.mkdir(parents=True)
        (generation / "possession_v3.jsonl").write_text("{}\n", encoding="utf-8")
        manifest = {
            "schema": "possession-engine-v3",
            "artifacts": {
                "manifest": {"path": "possession_v3_manifest.json"},
                "possession": {
                    "path": "possession_v3/generations/fake-generation/possession_v3.jsonl"
                },
            },
        }
        (output_dir / "possession_v3_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return types.SimpleNamespace(
            to_dict=lambda: {
                "tracks_file": "tracks.jsonl",
                "teams_file": "teams.json",
                "numbers_file": None,
                "shot_events_file": "shot_events.jsonl",
            }
        )

    monkeypatch.setattr(modal_app_v2, "_run_v2_stats", fake_stats)
    cls = next(
        value for key, value in modal_app_v2.PipelineV2.__dict__.items()
        if key.startswith("_sync_original")
    )._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[
        next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))
    ].raw_f

    payload = process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60,
                      return_artifacts=True)

    assert configs and all(config.possession_v3_enabled for config in configs)
    artifacts = payload["artifacts"]
    assert "stats/possession_v3_manifest.json" in artifacts
    manifest = json.loads(artifacts["stats/possession_v3_manifest.json"])
    pointer = manifest["artifacts"]["possession"]["path"]
    assert f"stats/{pointer}" in artifacts


@pytest.mark.parametrize(
    ("requested", "configured", "expected"),
    [("v4", "v3", "v4"), (None, "v3", "v3")],
)
def test_process_possession_engine_is_explicit_and_persisted(
    monkeypatch, requested, configured, expected
):
    import modal_app_v2

    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("POSSESSION_ENGINE", configured)
    monkeypatch.setenv("NUMBER_EVIDENCE_ENABLED", "0")
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *_args: ([], {}))
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(
        modal_app_v2,
        "run_v2",
        lambda rows, **_kwargs: types.SimpleNamespace(
            observations=rows, acceptance={}, acceptance_reasons=[], runtime_diagnostics={}
        ),
    )
    captured = []

    def fake_stats(_video_path, output_dir, _observations, *, config, **_kwargs):
        captured.append(config.possession_engine)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(
            json.dumps({"config": {"possession_engine": config.possession_engine}}),
            encoding="utf-8",
        )
        for name in ("tracks.jsonl", "teams.json", "shot_events.jsonl"):
            (output_dir / name).write_text("{}\n" if name.endswith("jsonl") else "{}", encoding="utf-8")
        generation = output_dir / "possession_v3" / "generations" / "engine-check"
        generation.mkdir(parents=True)
        (generation / "possession_v3.jsonl").write_text("{}\n", encoding="utf-8")
        v4_diagnostics = output_dir / "possession_v4_diagnostics.jsonl"
        if config.possession_engine == "v4":
            v4_diagnostics.write_text('{"v4": {"feature_flag": true}}\n', encoding="utf-8")
        manifest = {
            "schema": "possession-engine-v3",
            "engine": config.possession_engine,
            "v4": {"enabled": config.possession_engine == "v4"},
            "artifacts": {
                "manifest": {"path": "possession_v3_manifest.json"},
                "possession": {
                    "path": "possession_v3/generations/engine-check/possession_v3.jsonl"
                },
                **(
                    {"v4_diagnostics": {"path": "possession_v4_diagnostics.jsonl"}}
                    if config.possession_engine == "v4"
                    else {}
                ),
            },
        }
        (output_dir / "possession_v3_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return types.SimpleNamespace(
            to_dict=lambda: {
                "tracks_file": "tracks.jsonl",
                "teams_file": "teams.json",
                "numbers_file": None,
                "shot_events_file": "shot_events.jsonl",
                "possession_v4_diagnostics_file": (
                    "possession_v4_diagnostics.jsonl"
                    if config.possession_engine == "v4"
                    else None
                ),
            }
        )

    monkeypatch.setattr(modal_app_v2, "_run_v2_stats", fake_stats)
    cls = next(
        value for key, value in modal_app_v2.PipelineV2.__dict__.items()
        if key.startswith("_sync_original")
    )._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[
        next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))
    ].raw_f

    payload = process(
        cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60,
        possession_engine=requested, return_artifacts=True,
    )

    assert captured == [expected]
    root_manifest = json.loads(payload["artifacts"]["manifest.json"])
    assert root_manifest["config"]["possession_engine"] == expected
    stats = json.loads(payload["artifacts"]["stats/results.json"])
    assert stats["config"]["possession_engine"] == expected
    v3_manifest = json.loads(payload["artifacts"]["stats/possession_v3_manifest.json"])
    assert v3_manifest["engine"] == expected
    assert v3_manifest["v4"]["enabled"] is (expected == "v4")
    stats_refs = root_manifest["artifacts"]["legacy_stats"]
    if expected == "v4":
        reference = stats_refs["possession_v4_diagnostics"]
        assert reference["key"] == "stats/possession_v4_diagnostics.jsonl"
        data = payload["artifacts"][reference["key"]]
        assert reference["size_bytes"] == len(data)
        assert reference["sha256"] == hashlib.sha256(data).hexdigest()
    else:
        assert "possession_v4_diagnostics" not in stats_refs


def test_process_rejects_legacy_possession_engine(monkeypatch):
    import modal_app_v2

    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    cls = next(
        value for key, value in modal_app_v2.PipelineV2.__dict__.items()
        if key.startswith("_sync_original")
    )._user_cls
    process = cls.process.__dict__[
        next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))
    ].raw_f

    with pytest.raises(ValueError, match="must be 'v3', 'v4', or 'v5'"):
        process(cls(), video_bytes=b"fake-video", possession_engine="legacy")


def test_process_court_projection_manifest_boundary(monkeypatch):
    import hashlib
    import modal_app_v2
    import numpy as np

    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("COURT_MODEL_ID", "court/1")
    monkeypatch.setenv("NUMBER_EVIDENCE_ENABLED", "0")
    row = {
        "observation_key": "60:7",
        "frame_index": 60,
        "raw_track_id": 7,
        "stable_player_id": "player-007",
        "bbox_xyxy": [10, 10, 30, 50],
        "team_id": 1,
        "track_state": "tracked",
    }
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *_args: ([dict(row)], {row["observation_key"]: np.ones((2, 2), dtype=np.uint8)}))
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(
        modal_app_v2, "run_v2",
        lambda rows, **_kwargs: types.SimpleNamespace(observations=rows, acceptance={}, acceptance_reasons=[]),
    )

    def fake_stats(_video_path, output_dir, _observations, *, config, **_kwargs):
        assert config.court_model_id == "court/1"
        output_dir.mkdir(parents=True, exist_ok=True)
        contents = {
                "results.json": json.dumps({
                    "tracks_file": "tracks.jsonl", "teams_file": "teams.json",
                    "numbers_file": None,
                    "shot_events_file": "shot_events.jsonl", "projected_tracks_file": "projected_tracks.jsonl",
                    "projected_shots_file": "projected_shots.json", "projected_shots_image_file": "projected_shots.png",
                    "shot_evidence_file": None, "possession_events_file": None,
                    "possession_transitions_file": None, "ball_tracks_file": None,
                }),
            "tracks.jsonl": "{}\n", "teams.json": "{}", "shot_events.jsonl": "",
            "projected_tracks.jsonl": '{"frame_index":60}\n',
            "projected_shots.json": '{"shots":[]}\n', "projected_shots.png": b"png-bytes",
        }
        for name, data in contents.items():
            path = output_dir / name
            path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return types.SimpleNamespace(to_dict=lambda: json.loads(contents["results.json"]))

    monkeypatch.setattr(modal_app_v2, "_run_v2_stats", fake_stats)
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f

    payload = process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60, return_artifacts=True)
    artifacts = payload["artifacts"]
    manifest = json.loads(artifacts["manifest.json"])
    stats_refs = manifest["artifacts"]["legacy_stats"]
    expected = {
        "projected_tracks": "stats/projected_tracks.jsonl",
        "projected_shots": "stats/projected_shots.json",
        "projected_shots_image": "stats/projected_shots.png",
    }
    results = json.loads(artifacts["stats/results.json"])
    for name, relative_path in expected.items():
        reference = stats_refs[name]
        assert reference["key"] == relative_path
        assert not relative_path.startswith("/")
        data = artifacts[relative_path]
        assert reference["size_bytes"] == len(data)
        assert reference["sha256"] == hashlib.sha256(data).hexdigest()
        assert results[f"{name}_file"] == relative_path.removeprefix("stats/")


def test_process_passes_tracked_frames_and_masks_to_v2_runner(monkeypatch):
    import numpy as np
    import modal_app_v2
    from pipeline.runner_v2 import RunnerV2Result

    captured = {}
    frame = np.full((4, 5, 3), 7, dtype=np.uint8)
    mask = np.ones((4, 5), dtype=np.uint8)

    class FakePredictor:
        class Model:
            def init_state(self, **_kwargs):
                return None

        model = Model()

        def handle_request(self, request):
            return {"session_id": "fake-session"} if request["type"] == "start_session" else None

        def handle_stream_request(self, _request):
            yield {
                "frame_index": 60,
                "out_obj_ids": [3],
                "out_boxes_xywh": [[1, 0, 4, 4]],
                "out_binary_masks": [mask],
            }

    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {60: frame})
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "run_v2", lambda rows, **kwargs: captured.update(kwargs) or RunnerV2Result(rows))
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: FakePredictor())

    process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60)

    assert captured["frames_by_index"] == {60: frame}
    assert np.array_equal(captured["masks_by_observation"]["60:3"], mask)


def test_process_segmented_boundary_passes_rows_masks_and_artifacts(monkeypatch):
    import numpy as np
    import modal_app_v2
    from pipeline.runner_v2 import RunnerV2Result

    calls = []
    rows = [
        {
            "observation_key": "0:7",
            "frame_index": 0,
            "raw_track_id": 7,
            "bbox_xyxy": [0, 0, 1, 1],
            "track_state": "tracked",
        }
    ]
    masks = {"0:7": np.ones((2, 2), dtype=np.uint8)}
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(modal_app_v2, "_track_segmented_video", lambda *args, **kwargs: calls.append((args, kwargs)) or (rows, masks))
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *args, **kwargs: pytest.fail("full-video path used"))
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {0: frame})
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    captured = {}
    monkeypatch.setattr(
        modal_app_v2,
        "run_v2",
        lambda observations, **kwargs: captured.update({"rows": observations, **kwargs}) or RunnerV2Result(observations),
    )
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f

    payload = process(cls(), video_bytes=b"fake-video", frame_start=0, frame_end=0,
                      segmented=True, segment_window=4, segment_overlap=1, return_artifacts=True)

    assert calls and {key: calls[0][1][key] for key in (
        "window", "overlap", "seam_min_score", "seam_ambiguity_margin"
    )} == {
        "window": 4,
        "overlap": 1,
        "seam_min_score": 0.5,
        "seam_ambiguity_margin": 0.05,
    }
    assert calls[0][1]["seam_diagnostics"] == []
    assert captured["rows"] == rows
    assert np.array_equal(captured["masks_by_observation"]["0:7"], masks["0:7"])
    assert "0:7" in json.loads(payload["artifacts"]["tracks.jsonl"].decode())["observation_key"]
    assert any(name.startswith("masks/") for name in payload["artifacts"])
    manifest = json.loads(payload["artifacts"]["manifest.json"])
    assert manifest["config"]["sam31_segmented"] is True


def test_process_segmented_serializes_seam_diagnostics_without_mask_mismatch(monkeypatch):
    import numpy as np
    import modal_app_v2
    from pipeline.runner_v2 import RunnerV2Result

    rows = [{
        "observation_key": "0:7", "frame_index": 0, "raw_track_id": 7,
        "bbox_xyxy": [0, 0, 1, 1], "track_state": "tracked",
    }]
    masks = {"0:7": np.ones((2, 2), dtype=np.uint8)}
    diagnostics = [
        {"event": "matched", "segment_index": 1, "previous_raw_track_id": 7,
         "current_raw_track_id": 1000007, "score": 0.91},
        {"event": "ambiguous", "segment_index": 2},
        {"event": "ownership", "frame_index": 2, "owner_segment_index": 1},
        {"event": "owned_observation_missing_mask", "frame_index": 3},
    ]

    def segmented(*_args, seam_diagnostics, **_kwargs):
        seam_diagnostics.extend(diagnostics)
        return rows, masks

    monkeypatch.setattr(modal_app_v2, "_track_segmented_video", segmented)
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *_args, **_kwargs: pytest.fail("full-video path used"))
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {0: np.zeros((2, 2, 3), dtype=np.uint8)})
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "run_v2", lambda observations, **_kwargs: RunnerV2Result(observations))
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f

    payload = process(
        cls(), video_bytes=b"fake-video", frame_start=0, frame_end=0,
        segmented=True, return_artifacts=True, return_diagnostics=True,
    )

    assert payload["debug_diagnostics"]["seam_diagnostics"] == diagnostics
    manifest = json.loads(payload["artifacts"]["manifest.json"])
    assert manifest["config"]["sam31_seam_diagnostics"] == diagnostics
    track = json.loads(payload["artifacts"]["tracks.jsonl"].decode())
    assert track["mask_ref"] in payload["artifacts"]


def test_segmented_conflicted_team_is_scoped_in_artifacts_and_manifest(monkeypatch):
    import numpy as np
    import modal_app_v2
    from pipeline.runner_v2 import RunnerV2, RunnerV2Seams, RuntimeIdentityAuthority
    from pipeline.team_embedder import TeamFeatures

    rows = [
        {"observation_key": f"{frame}:{raw}", "frame_index": frame,
         "raw_track_id": raw, "bbox_xyxy": [0, 0, 2, 2], "track_state": "tracked"}
        for frame, raw in enumerate(range(1, 8), start=60)
    ]
    stable = {
        "60:1": "player-1", "61:2": "player-1",
        "62:3": "player-2", "63:4": "player-2",
        "64:5": "player-3",
        "65:6": "player-4", "66:7": "player-4",
    }

    class ReID:
        def embed(self, _frame, boxes):
            return [[3.0, 4.0] for _ in boxes]

    class TeamEmbedder:
        name = "fake-team"

        def embed(self, _frame, samples):
            return [TeamFeatures(color_signature=[float(sample.track_id)], embedding=None) for sample in samples]

    class Clusterer:
        name = "segmented-conflict-clusterer"

        def fit_predict(self, _embeddings, _team_count):
            return [0, 0, 0, 1, 1, 1, 0]

    monkeypatch.setattr(modal_app_v2, "_track_segmented_video", lambda *args, **kwargs: (
        rows, {row["observation_key"]: np.ones((2, 2), dtype=bool) for row in rows}
    ))
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {
        frame: np.ones((2, 2, 3), dtype=np.uint8) for frame in range(60, 67)
    })
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(
        modal_app_v2, "run_v2",
        lambda observations, **kwargs: RunnerV2(RunnerV2Seams(
            identity=lambda items: {item["observation_key"]: stable[item["observation_key"]] for item in items},
            reid=ReID(), team_embedder=TeamEmbedder(), team_clusterer=Clusterer(),
            identity_authority=RuntimeIdentityAuthority(),
        )).run(observations, **kwargs),
    )

    def stats(_video_path, output_dir, observations, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text("{}", encoding="utf-8")
        (output_dir / "tracks.jsonl").write_text("", encoding="utf-8")
        (output_dir / "teams.json").write_text(json.dumps({
            "assignments_by_track_id": {str(row["raw_track_id"]): row["team_id"]
                                         for row in observations if row.get("team_id") is not None},
            "unassigned_track_ids": [row["raw_track_id"] for row in observations
                                     if row.get("team_id") is None],
            "source": "sam31_v2_observations",
        }), encoding="utf-8")
        return types.SimpleNamespace(to_dict=lambda: {
            "tracks_file": "tracks.jsonl", "teams_file": "teams.json",
            "numbers_file": None, "shot_events_file": None,
        })

    monkeypatch.setattr(modal_app_v2, "_run_v2_stats", stats)
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items()
               if key.startswith("_sync_original"))._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__
                                       if key.startswith("_sync_original"))].raw_f

    payload = process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=66,
                      segmented=True, return_artifacts=True)
    manifest = json.loads(payload["artifacts"]["manifest.json"])
    tracks = [json.loads(line) for line in payload["artifacts"]["tracks.jsonl"].decode().splitlines()]
    teams = json.loads(payload["artifacts"]["stats/teams.json"])

    assert payload["acceptance"]["tracking_identity"] == "accepted"
    assert "conflicting_team_assignment:stable_player_id=player-2" in manifest["acceptance_reasons"]
    assert "conflicting_team_assignment:stable_player_id=player-4" in manifest["acceptance_reasons"]
    assert {row["stable_player_id"] for row in tracks if row.get("team_id") is not None} == {"player-1", "player-3"}
    assert {row["stable_player_id"] for row in tracks if row.get("team_id") is None} == {"player-2", "player-4"}
    assert all(row["team_assignment"]["status"] == "contested"
               for row in tracks if row.get("team_id") is None)
    assert teams["assignments_by_track_id"] == {"1": 0, "2": 0, "5": 1}
    assert teams["unassigned_track_ids"] == [3, 4, 6, 7]


@pytest.mark.parametrize("segmented", [False, None])
def test_process_segmented_false_or_unset_rolls_back_to_full_video(monkeypatch, segmented):
    import modal_app_v2
    from pipeline.runner_v2 import RunnerV2Result

    calls = []
    monkeypatch.delenv("SAM31_V2_SEGMENTED", raising=False)
    monkeypatch.setattr(modal_app_v2, "_track_segmented_video", lambda *args, **kwargs: pytest.fail("segmented path used"))
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *args, **kwargs: calls.append((args, kwargs)) or ([], {}))
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "run_v2", lambda observations, **_kwargs: RunnerV2Result(observations))
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f

    process(cls(), video_bytes=b"fake-video", frame_start=0, frame_end=0, segmented=segmented)

    assert calls and calls[0][1] == {}


def test_process_exposes_runtime_diagnostics_only_through_explicit_debug_response(
    monkeypatch,
):
    import modal_app_v2
    from pipeline.runner_v2 import RunnerV2Result

    diagnostics = {
        "debug_runtime_only": True,
        "team_vote_diagnostics": {
            "player-3": {"cluster_votes": {"0": 1, "1": 2}, "total_samples": 3}
        },
    }
    rows = [{
        "observation_key": "60:3",
        "frame_index": 60,
        "raw_track_id": 3,
        "bbox_xyxy": [1, 2, 11, 22],
        "track_state": "tracked",
    }]
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *args: (rows, {}))
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(
        modal_app_v2,
        "run_v2",
        lambda observations, **_kwargs: RunnerV2Result(
            observations, runtime_diagnostics=diagnostics
        ),
    )
    cls = next(
        value for key, value in modal_app_v2.PipelineV2.__dict__.items()
        if key.startswith("_sync_original")
    )._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f

    normal = process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60, return_artifacts=True)
    debug = process(
        cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60, return_diagnostics=True
    )

    assert "debug_diagnostics" not in normal
    assert "runtime_diagnostics" not in json.loads(normal["artifacts"]["tracks.jsonl"].decode())
    assert "runtime_diagnostics" not in json.loads(normal["artifacts"]["manifest.json"].decode())
    assert debug == {"acceptance": {}, "debug_diagnostics": diagnostics}


def test_conflicting_team_response_keeps_rejected_policy_with_debug_opt_in(monkeypatch):
    import modal_app_v2
    import numpy as np
    from pipeline.runner_v2 import RunnerV2, RunnerV2Seams
    from pipeline.team_embedder import TeamFeatures

    rows = [
        {
            "observation_key": "60:3",
            "frame_index": 60,
            "raw_track_id": 3,
            "bbox_xyxy": [1, 2, 11, 22],
            "track_state": "tracked",
        },
        {
            "observation_key": "61:4",
            "frame_index": 61,
            "raw_track_id": 4,
            "bbox_xyxy": [1, 2, 11, 22],
            "track_state": "tracked",
        },
    ]

    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, _frame, samples):
            return [TeamFeatures(color_signature=[1.0], embedding=None) for _ in samples]

    class ConflictingClusterer:
        name = "conflicting-clusterer"

        def fit_predict(self, embeddings, team_count):
            return [0, 1]

    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *args: (rows, {
        "60:3": np.ones((24, 12), dtype=bool),
        "61:4": np.ones((24, 12), dtype=bool),
    }))
    monkeypatch.setattr(
        modal_app_v2,
        "_load_video_frames",
        lambda *_args: {
            60: np.ones((24, 12, 3), dtype=np.uint8),
            61: np.ones((24, 12, 3), dtype=np.uint8),
        },
    )
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(
        modal_app_v2,
        "run_v2",
        lambda observations, **kwargs: RunnerV2(
            RunnerV2Seams(
                identity=lambda items: {item["observation_key"]: "player-7" for item in items},
                team_embedder=FakeTeamEmbedder(),
                team_clusterer=ConflictingClusterer(),
            )
        ).run(observations, **kwargs),
    )
    cls = next(
        value for key, value in modal_app_v2.PipelineV2.__dict__.items()
        if key.startswith("_sync_original")
    )._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f

    normal = process(
        cls(), video_bytes=b"same-input", frame_start=60, frame_end=61, return_diagnostics=False
    )
    debug = process(
        cls(), video_bytes=b"same-input", frame_start=60, frame_end=61, return_diagnostics=True
    )

    assert normal == {
        "acceptance": {"tracking_identity": "unvalidated", "statistics": "deferred"}
    }
    assert debug["acceptance"] == normal["acceptance"]
    assert debug.keys() == {"acceptance", "debug_diagnostics"}
    assert debug["debug_diagnostics"]["team_vote_diagnostics"] == {
        "player-7": {"cluster_votes": {"0": 1, "1": 1}, "total_samples": 2}
    }


def test_process_passes_runtime_seams_to_v2_runner(monkeypatch):
    import modal_app_v2
    from pipeline.runner_v2 import RunnerV2Result, RunnerV2Seams

    captured = {}
    seams = RunnerV2Seams(identity=lambda rows: {row["observation_key"]: "player-3" for row in rows})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "run_v2", lambda rows, **kwargs: captured.update(kwargs) or RunnerV2Result(rows))
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: type("Predictor", (), {
        "model": type("Model", (), {"init_state": lambda _self, **_kwargs: None})(),
        "handle_request": lambda _self, request: {"session_id": "fake"} if request["type"] == "start_session" else None,
        "handle_stream_request": lambda _self, _request: iter(()),
    })())
    instance = cls()
    instance.seams = seams
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f

    process(instance, video_bytes=b"fake-video", frame_start=60, frame_end=60)

    assert captured["seams"] is seams


def test_process_rejects_missing_identity_and_defers_statistics(monkeypatch):
    import modal_app_v2
    from pipeline.runner_v2 import RunnerV2, RunnerV2Seams

    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(
        modal_app_v2,
        "run_v2",
        lambda rows, **kwargs: RunnerV2(RunnerV2Seams(identity=lambda _items: {})).run(rows, **kwargs),
    )
    monkeypatch.setattr(
        cls,
        "_ensure_predictor",
        lambda _self: type("FakePredictor", (), {
            "model": type("FakeModel", (), {"init_state": lambda _self, **_kwargs: None})(),
            "handle_request": lambda _self, request: {"session_id": "fake"}
            if request["type"] == "start_session" else None,
            "handle_stream_request": lambda _self, _request: iter([{
                "frame_index": 60,
                "out_obj_ids": [3],
                "out_boxes_xywh": [[1, 2, 10, 20]],
                "out_probs": [0.9],
            }]),
        })(),
    )
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f

    payload = process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60, return_artifacts=True)

    assert payload["acceptance"] == {"tracking_identity": "rejected", "statistics": "deferred"}
    manifest = json.loads(payload["artifacts"]["manifest.json"])
    assert manifest["acceptance"] == payload["acceptance"]
    observation = json.loads(payload["artifacts"]["tracks.jsonl"].decode())
    assert observation["raw_track_id"] == 3
    assert "stable_player_id" not in observation
    assert "reid_embedding" not in observation
    assert "team_id" not in observation


@pytest.mark.parametrize(
    "acceptance",
    [
        {
            "tracking_identity": "accepted",
            "statistics": "deferred",
            "authority": "runtime_identity",
        },
        {"tracking_identity": "rejected", "statistics": "deferred"},
        {"tracking_identity": "unvalidated", "statistics": "deferred"},
    ],
)
def test_process_preserves_explicit_runner_acceptance_states(monkeypatch, acceptance):
    import modal_app_v2
    from pipeline.runner_v2 import RunnerV2Result

    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *args: ([], {}))
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    monkeypatch.setattr(
        modal_app_v2,
        "run_v2",
        lambda rows, **_kwargs: RunnerV2Result(rows, acceptance=acceptance),
    )
    cls = next(
        value for key, value in modal_app_v2.PipelineV2.__dict__.items()
        if key.startswith("_sync_original")
    )._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f

    payload = process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60)

    assert payload == {"acceptance": acceptance}


def test_process_size_limit_fails_before_returning_partial_artifacts(monkeypatch):
    import modal_app_v2

    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *args: ([], {}))
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_args: {})
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())

    with pytest.raises(ValueError, match="exceeds configured artifact return limit"):
        process(cls(),
            video_bytes=b"fake-video", frame_start=0, frame_end=0,
            return_artifacts=True, max_artifact_bytes=1,
        )


def test_process_flushes_top_level_failure_log(monkeypatch, capsys):
    import modal_app_v2

    monkeypatch.setattr(
        modal_app_v2,
        "_track_video",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("propagation failed")),
    )
    cls = next(
        value for key, value in modal_app_v2.PipelineV2.__dict__.items()
        if key.startswith("_sync_original")
    )._user_cls
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f

    with pytest.raises(RuntimeError, match="propagation failed"):
        process(cls(), video_bytes=b"fake-video", frame_start=0, frame_end=0)

    assert "[v2] error=RuntimeError: propagation failed" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("ids", "boxes", "message"),
    [([1, 2], [[0, 0, 1, 1]], "object ids and boxes"), ([1], [[0, 0, 1, 1]], "probabilities")],
)
def test_fake_sam_cardinality_errors_are_explicit(ids, boxes, message):
    from modal_app_v2 import _session_objects

    output = {"out_obj_ids": ids, "out_boxes_xywh": boxes}
    if message == "probabilities":
        output["out_probs"] = [0.5, 0.6]
    with pytest.raises(RuntimeError, match=message):
        _session_objects(output)
