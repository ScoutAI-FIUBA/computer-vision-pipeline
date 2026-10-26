from pathlib import Path
import sys
import json

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ball.modal_pipeline_experiment import (
    clip_identifier,
    comparison_frame_records,
    hosted_client_or_raise,
    load_labels,
    ordered_frame_paths,
    padded_window,
    select_frame_indices,
    validate_frame_labels,
    validate_output_video,
)
from scripts.ball import modal_pipeline_experiment as experiment


def _clip(root: Path, split: str, match: str, count: int = 3) -> Path:
    import cv2

    clip_root = root / split / match
    frame_dir = clip_root / "frame" / "clip_001"
    frame_dir.mkdir(parents=True)
    for index in range(count):
        assert cv2.imwrite(str(frame_dir / f"{index}.png"), np.zeros((16, 24, 3), dtype=np.uint8))
    (clip_root / "csv").mkdir()
    rows = ["Frame,Visibility,X,Y"] + [f"{index},0,0,0" for index in range(count)]
    (clip_root / "csv" / "clip_001_ball.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return clip_root


class _FakeDetector:
    context_size = 2
    backend_name = "fake"

    def predict(self, _frames, frame_indices):
        return type("Prediction", (), {"heatmap": np.zeros((8, 12), dtype=np.float32), "frame_indices": tuple(frame_indices)})()


def test_padded_window_emits_edge_padded_causal_sequences():
    assert padded_window([0, 1, 2, 3], 0, 4) == (0, 0, 0, 0)
    assert padded_window([0, 1, 2, 3], 2, 4) == (0, 0, 1, 2)
    assert padded_window([0, 1, 2, 3], 3, 4) == (0, 1, 2, 3)


def test_ordered_frame_paths_reject_gaps(tmp_path: Path):
    (tmp_path / "0.png").touch()
    (tmp_path / "2.png").touch()
    with pytest.raises(ValueError, match="contiguous"):
        ordered_frame_paths(tmp_path)


def test_select_frame_indices_is_deterministic():
    assert select_frame_indices(10, 3) == [0, 3, 6]
    assert select_frame_indices(3, 20) == [0, 1, 2]


def test_clip_outputs_use_requested_split_and_match():
    assert clip_identifier(Path("processed/basketball-v1/test/match20")) == "test_match20"
    assert clip_identifier(Path("processed/basketball-v1/train/match01")) == "train_match01"


def test_comparison_frame_records_persist_exact_paths_and_labels():
    paths = [Path("frames/0.png"), Path("frames/1.png")]
    labels = {0: (1, 10.0, 20.0), 1: (0, 0.0, 0.0)}
    assert comparison_frame_records(paths, labels, [1]) == [
        {"frame_index": 1, "path": "frames/1.png", "label": [0, 0.0, 0.0]}
    ]


def test_hosted_comparison_fails_for_missing_credentials_or_dependency():
    with pytest.raises(RuntimeError, match="missing ROBOFLOW_API_KEY"):
        hosted_client_or_raise(lambda: "", lambda _key: object())
    with pytest.raises(RuntimeError, match="missing dependency"):
        hosted_client_or_raise(
            lambda: "secret",
            lambda _key: (_ for _ in ()).throw(ModuleNotFoundError("inference_sdk")),
        )


def test_load_labels_rejects_duplicate_frame_rows(tmp_path: Path):
    labels = tmp_path / "duplicate.csv"
    labels.write_text(
        "Frame,Visibility,X,Y\n0,0,0,0\n0,1,10,20\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate label for frame 0"):
        load_labels(labels)


def test_invalid_output_video_fails_clearly(tmp_path: Path):
    with pytest.raises(RuntimeError, match="empty or missing"):
        validate_output_video(tmp_path / "missing.mp4", 1)
    invalid = tmp_path / "invalid.mp4"
    invalid.write_text("not a video", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot be opened|incomplete"):
        validate_output_video(invalid, 1)


def test_cli_tracknet_only_writes_unique_videos_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    clips = [_clip(tmp_path, "test", "match20"), _clip(tmp_path, "train", "match01")]
    monkeypatch.setattr(experiment, "inspect_tracknet_checkpoint", lambda _path: {"exists": True, "seq_len": 2, "bg_mode": "concat"})
    monkeypatch.setattr(experiment, "build_detector", lambda *args, **kwargs: _FakeDetector())
    output_dir = tmp_path / "outputs"

    experiment.main([*(str(clip) for clip in clips), "--output-dir", str(output_dir), "--tracknet-only"])

    videos = sorted(output_dir.glob("*.mp4"))
    reports = sorted(output_dir.glob("*.json"))
    assert [path.name for path in videos] == [
        "test_match20_tracknet_vs_detector.mp4",
        "train_match01_tracknet_vs_detector.mp4",
    ]
    assert [json.loads(path.read_text())["clip_id"] for path in reports] == ["test_match20", "train_match01"]


@pytest.mark.parametrize("failure", ["credential", "dependency"])
def test_cli_full_comparison_fails_nonzero_without_hosted_prerequisite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
):
    clips = [_clip(tmp_path, "test", "match20"), _clip(tmp_path, "train", "match01")]
    monkeypatch.setattr(experiment, "inspect_tracknet_checkpoint", lambda _path: {"exists": True, "seq_len": 2, "bg_mode": "concat"})
    monkeypatch.setattr(experiment, "build_detector", lambda *args, **kwargs: _FakeDetector())
    from scripts.shared import env_config

    monkeypatch.setattr(env_config, "get_roboflow_api_key", lambda: "" if failure == "credential" else "secret")
    if failure == "dependency":
        monkeypatch.setattr(experiment, "roboflow_client", lambda _key: (_ for _ in ()).throw(ModuleNotFoundError("inference_sdk")))

    with pytest.raises(SystemExit, match="Hosted detector comparison unavailable") as error:
        experiment.main([*(str(clip) for clip in clips), "--output-dir", str(tmp_path / "outputs")])
    assert error.value.code not in (None, 0)


def test_cli_json_persists_exact_comparison_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    clips = [_clip(tmp_path, "test", "match20", 4), _clip(tmp_path, "train", "match01", 4)]
    monkeypatch.setattr(experiment, "inspect_tracknet_checkpoint", lambda _path: {"exists": True, "seq_len": 2, "bg_mode": "concat"})
    monkeypatch.setattr(experiment, "build_detector", lambda *args, **kwargs: _FakeDetector())
    from scripts.shared import env_config

    monkeypatch.setattr(env_config, "get_roboflow_api_key", lambda: "secret")
    monkeypatch.setattr(experiment, "roboflow_client", lambda _key: object())
    monkeypatch.setattr(experiment, "infer_frame", lambda *_args, **_kwargs: {"predictions": []})
    output_dir = tmp_path / "outputs"

    experiment.main([*(str(clip) for clip in clips), "--output-dir", str(output_dir), "--detector-frame-cap", "2"])

    report = json.loads((output_dir / "test_match20_tracknet_vs_detector.json").read_text())
    assert report["comparison_frames"] == [
        {"frame_index": 0, "path": str(clips[0] / "frame" / "clip_001" / "0.png"), "label": [0, 0.0, 0.0]},
        {"frame_index": 2, "path": str(clips[0] / "frame" / "clip_001" / "2.png"), "label": [0, 0.0, 0.0]},
    ]


def test_frame_labels_must_match_exact_ordered_frames(tmp_path: Path):
    paths = [tmp_path / "0.png", tmp_path / "1.png"]
    with pytest.raises(ValueError, match="missing=.*extra="):
        validate_frame_labels(paths, {0: (0, 0.0, 0.0), 2: (0, 0.0, 0.0)})


def test_modal_tracknet_configuration_is_explicit_without_importing_modal():
    source = (Path(__file__).resolve().parents[1] / "modal_app.py").read_text(encoding="utf-8")
    image_source = source.split("image = (", 1)[1].split("\n)\n\n# Secrets", 1)[0]
    image_lines = image_source.splitlines()
    last_local_line = max(
        index for index, line in enumerate(image_lines) if ".add_local_" in line
    )
    assert ".env(" in "\n".join(image_lines[:last_local_line])
    assert not any(line.lstrip().startswith(".") for line in image_lines[last_local_line + 1 :])
    assert '"tracknet-data", environment_name="main", create_if_missing=False' in source
    assert 'TRACKNET_VOLUME_DIR = "/tracknet-data"' in source
    assert "basketball-v1-20ep-r3/checkpoints/TrackNet_best.pt" in source
    assert '.add_local_file("scripts/ball/tracknet.py"' in source
    assert '.add_local_file("scripts/ball/candidates.py"' in source
    assert "pipeline == MAIN_PIPELINE and config.tracknet_enabled" in source


def _patch_modal_model_dependencies(monkeypatch: pytest.MonkeyPatch):
    import modal_app

    class FakeTracker:
        device = "cpu"

    class FakeEmbedder:
        name = "none"
        device = "cpu"

    monkeypatch.setattr("pipeline.tracking.Sam2SegmentTracker", lambda _model_id: FakeTracker())
    monkeypatch.setattr("pipeline.reid.build_embedder", lambda *_args, **_kwargs: FakeEmbedder())
    monkeypatch.setattr(modal_app.model_cache, "commit", lambda: None)
    return modal_app


def test_modal_loader_builds_tracknet_from_configured_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    modal_app = _patch_modal_model_dependencies(monkeypatch)
    checkpoint = tmp_path / "TrackNet_best.pt"
    checkpoint.write_bytes(b"checkpoint")
    detector = object()
    calls = []
    monkeypatch.setenv("TRACKNET_ENABLED", "1")
    monkeypatch.setenv("TRACKNET_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr("pipeline.ball.build_tracknet_detector", lambda path: calls.append(path) or detector)

    assert modal_app._load_tracknet_detector() is detector
    assert calls == [checkpoint]


def test_modal_cold_start_defers_tracknet_loading():
    source = (Path(__file__).resolve().parents[1] / "modal_app.py").read_text(encoding="utf-8")
    load_model = source.split("def load_model", 1)[1].split("def _ensure_tracknet_detector", 1)[0]

    assert "_load_tracknet_detector" not in load_model
    assert "self.tracknet_detector = None" in load_model
    assert "pipeline == MAIN_PIPELINE and config.tracknet_enabled" in source


def test_modal_loader_fails_clearly_for_missing_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    modal_app = _patch_modal_model_dependencies(monkeypatch)
    monkeypatch.setenv("TRACKNET_ENABLED", "1")
    monkeypatch.setenv("TRACKNET_CHECKPOINT", str(tmp_path / "missing.pt"))

    with pytest.raises(RuntimeError, match="checkpoint is missing"):
        modal_app._load_tracknet_detector()


def test_modal_disabled_loader_keeps_hosted_compatibility(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    modal_app = _patch_modal_model_dependencies(monkeypatch)
    monkeypatch.setenv("TRACKNET_ENABLED", "0")
    monkeypatch.setenv("TRACKNET_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setattr(
        "pipeline.ball.build_tracknet_detector",
        lambda _path: pytest.fail("disabled Modal path must not build TrackNet"),
    )

    assert modal_app._load_tracknet_detector() is None


def test_modal_process_passes_preloaded_detector_to_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import modal_app

    detector = object()
    captured = {}

    def fake_run_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("pipeline.runner.run_pipeline", fake_run_pipeline)
    modal_app._run_selected_pipeline(
        modal_app.MAIN_PIPELINE,
        video_path=tmp_path / "input.mp4",
        config=object(),
        output_dir=tmp_path / "out",
        tracker=object(),
        embedder=object(),
        tracknet_detector=detector,
    )

    assert captured["tracknet_detector"] is detector


def test_parse_args_requires_exactly_one_clip_source(tmp_path: Path):
    clips = [str(_clip(tmp_path, "test", "match20")), str(_clip(tmp_path, "train", "match01"))]
    output = ["--output-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        experiment.parse_args(output)
    with pytest.raises(SystemExit):
        experiment.parse_args([*clips, "--processed-root", str(tmp_path), *output])
    assert len(experiment.parse_args([*clips, *output]).clip_roots) == 2
