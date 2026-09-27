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
    }:
        return
    import modal_app_v2

    def run(_video_path, output_dir, _observations):
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


def test_v2_load_models_warms_cold_container_and_assigns_runtime_seams(monkeypatch):
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
    assert results["numbers_file"] is None
    assert results["shot_events_file"] == "shot_events.jsonl"
    assert results["possession_events_file"] == "possession_events.jsonl"
    assert results["possession_transitions_file"] == "possession_transitions.jsonl"
    assert {"stats/results.json", "stats/tracks.jsonl", "stats/teams.json", "stats/shot_events.jsonl", "stats/possession_events.jsonl"} <= set(artifacts)
    for name, reference in manifest["artifacts"]["legacy_stats"].items():
        data = artifacts[reference["key"]]
        assert reference["size_bytes"] == len(data)
        assert reference["sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["legacy_stats_results"] == "stats/results.json"


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

    def fake_stats(_video_path, config, output_dir, _observations):
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

    monkeypatch.setattr(modal_app_v2, "run_stats_from_v2_observations", fake_stats)
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

    assert calls and calls[0][1] == {"window": 4, "overlap": 1}
    assert captured["rows"] == rows
    assert np.array_equal(captured["masks_by_observation"]["0:7"], masks["0:7"])
    assert "0:7" in json.loads(payload["artifacts"]["tracks.jsonl"].decode())["observation_key"]
    assert any(name.startswith("masks/") for name in payload["artifacts"])
    manifest = json.loads(payload["artifacts"]["manifest.json"])
    assert manifest["config"]["sam31_segmented"] is True


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

    def stats(_video_path, output_dir, observations):
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
