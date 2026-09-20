from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest


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

    assert decorated_gpu(v2_source, "PipelineV2") == "A100-40GB"
    assert decorated_gpu(legacy_source, "Pipeline") == "L4"


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
    assert calls == [{"use_fa3": False}]
    assert pipeline._ensure_predictor() is fake_predictor
    assert calls == [{"use_fa3": False}]


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
    from pipeline.runner_v2 import RunnerV2, RunnerV2Seams

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
    monkeypatch.setattr(
        modal_app_v2,
        "run_v2",
        lambda rows: RunnerV2(RunnerV2Seams(identity=lambda items: {item["observation_key"]: "player-3" for item in items})).run(rows),
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
    assert manifest["config"] == {"prompt": "basketball player", "frame_end": 60, "frame_start": 60}
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


def test_process_rejects_missing_identity_and_defers_statistics(monkeypatch):
    import modal_app_v2
    from pipeline.runner_v2 import RunnerV2, RunnerV2Seams

    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    monkeypatch.setattr(
        modal_app_v2,
        "run_v2",
        lambda rows: RunnerV2(RunnerV2Seams(identity=lambda _items: {})).run(rows),
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


def test_process_size_limit_fails_before_returning_partial_artifacts(monkeypatch):
    import modal_app_v2

    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *args: ([], {}))
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _path: {})
    cls = next(value for key, value in modal_app_v2.PipelineV2.__dict__.items() if key.startswith("_sync_original"))._user_cls
    process = cls.process.__dict__[next(key for key in cls.process.__dict__ if key.startswith("_sync_original"))].raw_f
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())

    with pytest.raises(ValueError, match="exceeds configured artifact return limit"):
        process(cls(),
            video_bytes=b"fake-video", frame_start=0, frame_end=0,
            return_artifacts=True, max_artifact_bytes=1,
        )


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
