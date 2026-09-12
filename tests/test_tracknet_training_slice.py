from __future__ import annotations

import csv
import importlib
import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

torch_module = types.ModuleType("torch")
utils_module = types.ModuleType("torch.utils")
data_module = types.ModuleType("torch.utils.data")
nn_module = types.ModuleType("torch.nn")


class _Dataset:
    pass


class _IterableDataset:
    pass


class _Module:
    pass


data_module.Dataset = _Dataset
data_module.IterableDataset = _IterableDataset
data_module.DataLoader = object
nn_module.Module = _Module
utils_module.data = data_module
torch_module.utils = utils_module
torch_module.nn = nn_module
tensorboard_module = types.ModuleType("torch.utils.tensorboard")
tensorboard_module.SummaryWriter = object
sys.modules.setdefault("torch", torch_module)
sys.modules.setdefault("torch.nn", nn_module)
sys.modules.setdefault("torch.utils", utils_module)
sys.modules.setdefault("torch.utils.data", data_module)
sys.modules.setdefault("torch.utils.tensorboard", tensorboard_module)

test_module = types.ModuleType("test")
test_module.eval_tracknet = lambda *args, **kwargs: (0.0, {})
test_module.eval_inpaintnet = lambda *args, **kwargs: (0.0, {})
sys.modules.setdefault("test", test_module)

general_module = types.ModuleType("utils.general")
general_module.ResumeArgumentParser = object
general_module.get_model = lambda *args, **kwargs: None
general_module.get_rally_dirs = lambda root_dir, split: sorted(
    str(path.relative_to(root_dir))
    for match_dir in (Path(root_dir) / split).iterdir()
    if match_dir.is_dir()
    for path in (match_dir / "frame").iterdir()
    if path.is_dir()
)
general_module.get_match_median = lambda match_dir: np.zeros((32, 64, 3), dtype=np.uint8)
general_module.HEIGHT = 288
general_module.WIDTH = 512
general_module.SIGMA = 2.5
general_module.IMG_FORMAT = "png"
general_module.to_img_format = lambda value, **kwargs: value
metric_module = types.ModuleType("utils.metric")
metric_module.WBCELoss = lambda *args, **kwargs: None
visualize_module = types.ModuleType("utils.visualize")
visualize_module.plot_heatmap_pred_sample = lambda *args, **kwargs: None
visualize_module.plot_traj_pred_sample = lambda *args, **kwargs: None
visualize_module.write_to_tb = lambda *args, **kwargs: None
sys.modules.setdefault("utils.general", general_module)
sys.modules.setdefault("utils.metric", metric_module)
sys.modules.setdefault("utils.visualize", visualize_module)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TrackNetV3"))

from basketball_data import write_json_manifest
from dataset import Shuttlecock_Trajectory_Dataset
from finetune import (
    TrainingStateError,
    compute_validation_metrics,
    load_finetune_state,
    save_training_config,
    select_best_checkpoint,
)

train_module = importlib.import_module("train")


def _write_video(path: Path, *, width: int = 64, height: int = 32, frames: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    for _ in range(frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


def _write_labels(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Frame", "Visibility", "X", "Y"])
        writer.writeheader()
        writer.writerows(
            [
                {"Frame": 0, "Visibility": 1, "X": 10, "Y": 8},
                {"Frame": 1, "Visibility": 1, "X": 12, "Y": 10},
                {"Frame": 2, "Visibility": 0, "X": 0, "Y": 0},
            ]
        )


def _write_adapted_clip(root: Path, split: str, match_name: str, clip: str) -> None:
    _write_video(root / split / match_name / "video" / f"{clip}.mp4")
    _write_labels(root / split / match_name / "csv" / f"{clip}_ball.csv")
    frame_dir = root / split / match_name / "frame" / clip
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in range(3):
        cv2.imwrite(str(frame_dir / f"{frame_index}.png"), np.zeros((32, 64, 3), dtype=np.uint8))
    np.savez(frame_dir / "median.npz", median=np.zeros((32, 64, 3), dtype=np.uint8))
    np.savez(root / split / match_name / "median.npz", median=np.zeros((32, 64, 3), dtype=np.uint8))


def _write_manifest(path: Path, fingerprint: str) -> None:
    write_json_manifest(
        {
            "fingerprint": fingerprint,
            "generated_splits": ["train", "val", "test"],
            "requested_splits": ["train", "validation", "test"],
            "preprocess_params": {"seq_len": 2, "bg_mode": "", "height": 32, "width": 64},
            "split_fingerprints": {"val": fingerprint},
            "splits": {},
        },
        path,
    )


def test_dataset_uses_manifest_aware_cache_root_and_validation_alias(tmp_path: Path) -> None:
    adapted_root = tmp_path / "adapted-data"
    cache_root = tmp_path / "run-cache"
    _write_adapted_clip(adapted_root, "val", "match02", "clip_002")
    _write_manifest(adapted_root / "manifest.json", "fingerprint-a")

    dataset = Shuttlecock_Trajectory_Dataset(
        root_dir=adapted_root,
        cache_root=cache_root,
        split="validation",
        seq_len=2,
        sliding_step=2,
        data_mode="heatmap",
        HEIGHT=32,
        WIDTH=64,
    )

    assert dataset.split == "val"
    assert len(dataset) == 1
    assert sorted(path.name for path in cache_root.glob("*.npz")) == [
        "data_fingerprint-a_l2_s2_heatmap_val.npz",
        "img_config_fingerprint-a_32x64_val.npz",
    ]
    assert not list(adapted_root.glob("*.npz"))


def test_dataset_cache_file_changes_when_manifest_fingerprint_changes(tmp_path: Path) -> None:
    adapted_root = tmp_path / "adapted-data"
    cache_root = tmp_path / "run-cache"
    _write_adapted_clip(adapted_root, "val", "match02", "clip_002")
    manifest_path = adapted_root / "manifest.json"

    _write_manifest(manifest_path, "fingerprint-a")
    Shuttlecock_Trajectory_Dataset(
        root_dir=adapted_root,
        cache_root=cache_root,
        split="val",
        seq_len=2,
        sliding_step=2,
        data_mode="heatmap",
        HEIGHT=32,
        WIDTH=64,
    )

    _write_manifest(manifest_path, "fingerprint-b")
    Shuttlecock_Trajectory_Dataset(
        root_dir=adapted_root,
        cache_root=cache_root,
        split="val",
        seq_len=2,
        sliding_step=2,
        data_mode="heatmap",
        HEIGHT=32,
        WIDTH=64,
    )

    assert (cache_root / "data_fingerprint-a_l2_s2_heatmap_val.npz").exists()
    assert (cache_root / "data_fingerprint-b_l2_s2_heatmap_val.npz").exists()


def test_dataset_test_split_reads_adapted_csv_when_corrected_csv_is_missing(tmp_path: Path) -> None:
    adapted_root = tmp_path / "adapted-data"
    cache_root = tmp_path / "run-cache"
    _write_adapted_clip(adapted_root, "test", "match03", "clip_003")
    _write_manifest(adapted_root / "manifest.json", "fingerprint-test")

    dataset = Shuttlecock_Trajectory_Dataset(
        root_dir=adapted_root,
        cache_root=cache_root,
        split="test",
        seq_len=2,
        sliding_step=2,
        data_mode="heatmap",
        HEIGHT=32,
        WIDTH=64,
    )

    _, frames, heatmaps, _, visibility = dataset[0]

    assert len(dataset) == 1
    assert frames.shape == (6, 32, 64)
    assert heatmaps.shape == (2, 32, 64)
    assert visibility.tolist() == [1.0, 1.0]


def test_load_finetune_state_separates_pretrained_weights_from_resume_state(tmp_path: Path) -> None:
    pretrained_path = tmp_path / "pretrained.pt"
    resume_path = tmp_path / "resume.pt"

    pretrained_path.write_text(
        json.dumps({"model": {"weights": [1, 2, 3]}, "model_name": "TrackNet", "seq_len": 2, "bg_mode": ""}),
        encoding="utf-8",
    )
    resume_path.write_text(
        json.dumps(
            {
                "epoch": 4,
                "best_metric": {"p_at_20": 0.8, "detection_rate": 0.85},
                "model": {"weights": [4, 5, 6]},
                "optimizer": {"lr": 0.1},
                "scheduler": {"last_epoch": 4},
                "param_dict": {"model_name": "TrackNet", "seq_len": 2, "bg_mode": "", "dataset_root": "adapted", "save_dir": "run"},
                "adapted_data_manifest": "manifest-a",
            }
        ),
        encoding="utf-8",
    )

    pretrained_state_result = load_finetune_state(
        model_name="TrackNet",
        seq_len=2,
        bg_mode="",
        dataset_root="adapted",
        save_dir="run",
        adapted_data_manifest="manifest-a",
        pretrained_tracknet=pretrained_path,
        resume_checkpoint=None,
        device="cpu",
    )

    resume_state_result = load_finetune_state(
        model_name="TrackNet",
        seq_len=2,
        bg_mode="",
        dataset_root="adapted",
        save_dir="run",
        adapted_data_manifest="manifest-a",
        pretrained_tracknet=None,
        resume_checkpoint=resume_path,
        device="cpu",
    )

    assert pretrained_state_result["start_epoch"] == 0
    assert pretrained_state_result["best_metric"] is None
    assert pretrained_state_result["model_state_dict"]["weights"] == [1, 2, 3]
    assert resume_state_result["start_epoch"] == 5
    assert resume_state_result["best_metric"]["p_at_20"] == pytest.approx(0.8)


def test_load_finetune_state_rejects_incompatible_resume_checkpoint(tmp_path: Path) -> None:
    resume_path = tmp_path / "resume.pt"
    resume_path.write_text(
        json.dumps(
            {
                "epoch": 1,
                "model": {"weights": [1]},
                "optimizer": {"lr": 0.1},
                "scheduler": None,
                "param_dict": {"model_name": "TrackNet", "seq_len": 4, "bg_mode": "", "dataset_root": "adapted", "save_dir": "run"},
                "adapted_data_manifest": "manifest-old",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainingStateError, match="incompatible"):
        load_finetune_state(
            model_name="TrackNet",
            seq_len=2,
            bg_mode="",
            dataset_root="adapted",
            save_dir="run",
            adapted_data_manifest="manifest-new",
            pretrained_tracknet=None,
            resume_checkpoint=resume_path,
            device="cpu",
        )


def test_load_finetune_state_uses_weights_only_for_pretrained_torch_checkpoints(tmp_path: Path) -> None:
    pretrained_path = tmp_path / "pretrained.pt"
    pretrained_path.write_bytes(b"not-json")
    recorded: dict[str, object] = {}

    def fake_load(path: Path, *, map_location: str, weights_only: bool) -> dict:
        recorded["path"] = path
        recorded["map_location"] = map_location
        recorded["weights_only"] = weights_only
        return {"model": {"weights": [9]}, "model_name": "TrackNet", "seq_len": 2, "bg_mode": ""}

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(sys.modules["torch"], "load", fake_load, raising=False)
    try:
        state = load_finetune_state(
            model_name="TrackNet",
            seq_len=2,
            bg_mode="",
            dataset_root="adapted",
            save_dir="run",
            adapted_data_manifest="manifest-a",
            pretrained_tracknet=pretrained_path,
            resume_checkpoint=None,
            device="cpu",
        )
    finally:
        monkeypatch.undo()

    assert recorded["path"] == pretrained_path
    assert recorded["map_location"] == "cpu"
    assert recorded["weights_only"] is True
    assert state["model_state_dict"]["weights"] == [9]


def test_load_finetune_state_rejects_untrusted_pretrained_fallback_without_weights_only(tmp_path: Path) -> None:
    pretrained_path = tmp_path / "pretrained.pt"
    pretrained_path.write_bytes(b"not-json")

    def fake_load(path: Path, *, map_location: str) -> dict:
        return {"model": {"weights": [9]}, "model_name": "TrackNet", "seq_len": 2, "bg_mode": ""}

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(sys.modules["torch"], "load", fake_load, raising=False)
    try:
        with pytest.raises(TrainingStateError, match="weights_only"):
            load_finetune_state(
                model_name="TrackNet",
                seq_len=2,
                bg_mode="",
                dataset_root="adapted",
                save_dir="run",
                adapted_data_manifest="manifest-a",
                pretrained_tracknet=pretrained_path,
                resume_checkpoint=None,
                device="cpu",
            )
    finally:
        monkeypatch.undo()


def test_load_finetune_state_allows_trusted_resume_fallback_only_inside_save_dir(tmp_path: Path) -> None:
    trusted_run_dir = tmp_path / "run"
    trusted_run_dir.mkdir()
    trusted_resume_path = trusted_run_dir / "resume.pt"
    trusted_resume_path.write_bytes(b"not-json")
    calls: list[dict[str, object]] = []

    def fake_load(path: Path, *, map_location: str) -> dict:
        calls.append({"path": path, "map_location": map_location})
        return {
            "epoch": 2,
            "best_metric": {"p_at_20": 0.82, "detection_rate": 0.88},
            "model": {"weights": [4, 5, 6]},
            "optimizer": {"lr": 0.1},
            "scheduler": {"last_epoch": 2},
            "param_dict": {"model_name": "TrackNet", "seq_len": 2, "bg_mode": "", "dataset_root": "adapted", "save_dir": str(trusted_run_dir)},
            "adapted_data_manifest": "manifest-a",
        }

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(sys.modules["torch"], "load", fake_load, raising=False)
    try:
        state = load_finetune_state(
            model_name="TrackNet",
            seq_len=2,
            bg_mode="",
            dataset_root="adapted",
            save_dir=str(trusted_run_dir),
            adapted_data_manifest="manifest-a",
            pretrained_tracknet=None,
            resume_checkpoint=trusted_resume_path,
            device="cpu",
        )
    finally:
        monkeypatch.undo()

    assert calls == [{"path": trusted_resume_path, "map_location": "cpu"}]
    assert state["mode"] == "resume"
    assert state["start_epoch"] == 3


def test_load_finetune_state_rejects_untrusted_resume_fallback_outside_save_dir(tmp_path: Path) -> None:
    resume_path = tmp_path / "resume.pt"
    resume_path.write_bytes(b"not-json")

    def fake_load(path: Path, *, map_location: str) -> dict:
        return {}

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(sys.modules["torch"], "load", fake_load, raising=False)
    try:
        with pytest.raises(TrainingStateError, match="trusted"):
            load_finetune_state(
                model_name="TrackNet",
                seq_len=2,
                bg_mode="",
                dataset_root="adapted",
                save_dir=str(tmp_path / "run"),
                adapted_data_manifest="manifest-a",
                pretrained_tracknet=None,
                resume_checkpoint=resume_path,
                device="cpu",
            )
    finally:
        monkeypatch.undo()


def test_pandas_compat_prefers_installed_pandas_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pandas = types.SimpleNamespace(source="installed")
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)
    sys.modules.pop("pandas_compat", None)

    pandas_compat = importlib.import_module("pandas_compat")

    assert pandas_compat.pd is fake_pandas


def test_tracknet_directory_no_longer_exposes_local_pandas_module() -> None:
    tracknet_root = Path(__file__).resolve().parents[1] / "TrackNetV3"

    assert importlib.machinery.PathFinder.find_spec("pandas", [str(tracknet_root)]) is None


def test_parse_compat_prefers_installed_parse_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_parse = types.SimpleNamespace(parse=lambda pattern, value: ("installed", pattern, value))
    monkeypatch.setitem(sys.modules, "parse", fake_parse)
    sys.modules.pop("parse_compat", None)

    parse_compat = importlib.import_module("parse_compat")

    assert parse_compat.parse is fake_parse


def test_tracknet_directory_no_longer_exposes_local_parse_module() -> None:
    tracknet_root = Path(__file__).resolve().parents[1] / "TrackNetV3"

    assert importlib.machinery.PathFinder.find_spec("parse", [str(tracknet_root)]) is None


def test_parse_compat_fallback_accepts_absolute_tracknet_rally_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "parse", raising=False)
    sys.modules.pop("parse_compat", None)

    parse_compat = importlib.import_module("parse_compat")

    match_dir, rally_id = parse_compat.parse.parse(
        "{}/frame/{}",
        "/tmp/adapted-data/val/match14/frame/clip_014",
    )

    assert match_dir == "/tmp/adapted-data/val/match14"
    assert rally_id == "clip_014"


def test_parse_compat_fallback_keeps_simple_segment_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "parse", raising=False)
    sys.modules.pop("parse_compat", None)

    parse_compat = importlib.import_module("parse_compat")

    match_id, = parse_compat.parse.parse("match{}", "match14")

    assert match_id == "14"


def test_select_best_checkpoint_prefers_p_at_20_when_guard_passes() -> None:
    baseline = {"detection_rate": 0.80}
    best = select_best_checkpoint(
        baseline,
        [
            {"epoch": 1, "p_at_20": 0.81, "detection_rate": 0.78, "median_error": 11.0},
            {"epoch": 2, "p_at_20": 0.85, "detection_rate": 0.79, "median_error": 14.0},
        ],
    )

    assert best["epoch"] == 2


def test_select_best_checkpoint_rejects_detection_rate_regression_below_guard() -> None:
    baseline = {"detection_rate": 0.80}

    with pytest.raises(TrainingStateError, match="guard"):
        select_best_checkpoint(
            baseline,
            [{"epoch": 3, "p_at_20": 0.90, "detection_rate": 0.77, "median_error": 9.0}],
        )


def test_save_training_config_writes_reproducible_json(tmp_path: Path) -> None:
    output_path = tmp_path / "training-config.json"

    save_training_config(
        output_path,
        {
            "dataset_root": "adapted-data",
            "pretrained_tracknet": "weights/base.pt",
            "resume_checkpoint": None,
            "selection_metric": "p_at_20",
            "early_stopping_patience": 5,
        },
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["dataset_root"] == "adapted-data"
    assert saved["selection_metric"] == "p_at_20"
    assert saved["early_stopping_patience"] == 5


def test_training_parser_and_validation_patience_use_p_at_20_guard() -> None:
    args = train_module.build_parser().parse_args(["--early_stopping_patience", "5"])
    assert args.early_stopping_patience == 5
    assert train_module.accepted_validation_improvement(
        {"p_at_20": 0.81, "detection_rate": 0.79},
        {"p_at_20": 0.80},
        baseline_detection_rate=0.80,
    )
    assert not train_module.accepted_validation_improvement(
        {"p_at_20": 0.99, "detection_rate": 0.77},
        {"p_at_20": 0.80},
        baseline_detection_rate=0.80,
    )
    assert train_module.update_early_stopping(
        1,
        metrics={"p_at_20": 0.80, "detection_rate": 0.80},
        best_metric={"p_at_20": 0.80},
        baseline_detection_rate=0.80,
        patience=2,
    ) == (False, 2, True)


def test_training_history_serializes_selection_metadata() -> None:
    history = train_module.append_history(
        [], epoch=0, train_loss=1.0, val_loss=0.5,
        metrics={"p_at_20": 0.8},
        metadata={"selection_metric": "p_at_20", "early_stopping_patience": 5},
    )
    assert history[0]["metadata"] == {"selection_metric": "p_at_20", "early_stopping_patience": 5}


def test_compute_validation_metrics_and_dataset_loader_smoke(tmp_path: Path) -> None:
    adapted_root = tmp_path / "adapted-data"
    cache_root = tmp_path / "run-cache"
    _write_adapted_clip(adapted_root, "train", "match01", "clip_001")
    _write_manifest(adapted_root / "manifest.json", "fingerprint-train")

    dataset = Shuttlecock_Trajectory_Dataset(
        root_dir=adapted_root,
        cache_root=cache_root,
        split="train",
        seq_len=2,
        sliding_step=1,
        data_mode="heatmap",
        HEIGHT=32,
        WIDTH=64,
    )
    _, frames, heatmaps, _, visibility = dataset[0]
    metrics = compute_validation_metrics(
        [
            {"visibility": 1, "predicted_visibility": 1, "distance": 12.0},
            {"visibility": 1, "predicted_visibility": 1, "distance": 24.0},
            {"visibility": 0, "predicted_visibility": 1, "distance": None},
        ]
    )

    assert frames.shape == (6, 32, 64)
    assert heatmaps.shape == (2, 32, 64)
    assert visibility.tolist() == [1.0, 1.0]
    assert metrics["p_at_20"] == pytest.approx(0.5)
    assert metrics["detection_rate"] == pytest.approx(1.0)


def test_persist_epoch_validation_artifacts_saves_checkpoint_before_report(tmp_path: Path) -> None:
    recorded: dict[str, object] = {}

    def fake_save_checkpoint(path: str, **kwargs: object) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("checkpoint", encoding="utf-8")
        recorded["checkpoint_path"] = path

    def fake_write_validation_artifacts(reports_root: str, **kwargs: object) -> Path:
        checkpoint_path = Path(str(kwargs["checkpoint"]))
        recorded["checkpoint_exists_during_report"] = checkpoint_path.exists()
        report_path = Path(reports_root) / str(kwargs["output_name"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"checkpoint": str(checkpoint_path)}), encoding="utf-8")
        return report_path

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(train_module, "_save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(train_module, "write_validation_artifacts", fake_write_validation_artifacts)
    try:
        artifact = train_module.persist_epoch_validation_artifacts(
            save_dir=str(tmp_path / "checkpoints"),
            reports_dir=str(tmp_path / "reports"),
            model_name="TrackNet",
            epoch=0,
            dataset_root="adapted-data",
            evaluation_mode="weight",
            summary={"TP": 5.0, "FP1": 1.0, "FN": 2.0, "median_error": 11.0},
            selection_metrics={"p_at_20": 0.625, "detection_rate": 0.75, "median_error": 11.0},
            best_metric=None,
            model=object(),
            optimizer=object(),
            scheduler=None,
            param_dict={},
            adapted_data_manifest="manifest-a",
            history=[],
        )
    finally:
        monkeypatch.undo()

    assert Path(str(recorded["checkpoint_path"])).exists()
    assert recorded["checkpoint_exists_during_report"] is True
    assert Path(artifact["report"]).exists()
    assert artifact["checkpoint"].endswith("TrackNet_epoch_001.pt")


def test_retain_top_epoch_artifacts_keeps_top_three_and_preserves_cur_and_best(tmp_path: Path) -> None:
    checkpoints_root = tmp_path / "checkpoints"
    reports_root = tmp_path / "reports"
    current_checkpoint = checkpoints_root / "TrackNet_cur.pt"
    best_checkpoint = checkpoints_root / "TrackNet_best.pt"
    current_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    current_checkpoint.write_text("cur", encoding="utf-8")
    best_checkpoint.write_text("best", encoding="utf-8")

    candidates = []
    for epoch, p_at_20 in ((1, 0.70), (2, 0.86), (3, 0.82), (4, 0.90)):
        checkpoint_path = checkpoints_root / f"TrackNet_epoch_{epoch:03d}.pt"
        report_path = reports_root / f"epoch-{epoch:03d}-validation.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(f"epoch-{epoch}", encoding="utf-8")
        report_path.write_text(json.dumps({"checkpoint": str(checkpoint_path)}), encoding="utf-8")
        candidates.append(
            {
                "epoch": epoch - 1,
                "checkpoint": str(checkpoint_path),
                "report": str(report_path),
                "metrics": {"p_at_20": p_at_20, "detection_rate": 0.80, "median_error": float(10 - epoch)},
            }
        )

    kept = train_module.retain_top_epoch_artifacts(candidates, keep=3)

    kept_names = sorted(Path(item["checkpoint"]).name for item in kept)
    assert kept_names == ["TrackNet_epoch_002.pt", "TrackNet_epoch_003.pt", "TrackNet_epoch_004.pt"]
    assert not (checkpoints_root / "TrackNet_epoch_001.pt").exists()
    assert (reports_root / "epoch-001-validation.json").exists()
    assert current_checkpoint.exists()
    assert best_checkpoint.exists()


def test_epoch_reports_history_and_presence_matrix_survive_checkpoint_pruning(tmp_path: Path) -> None:
    reports_root = tmp_path / 'reports'
    checkpoints_root = tmp_path / 'checkpoints'
    candidates = []
    for epoch, p_at_20 in enumerate((0.70, 0.86, 0.82, 0.90), start=1):
        checkpoint = checkpoints_root / f'TrackNet_epoch_{epoch:03d}.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(str(epoch), encoding='utf-8')
        report = reports_root / 'epochs' / f'epoch_{epoch:03d}.json'
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({'checkpoint': str(checkpoint)}), encoding='utf-8')
        candidates.append({'epoch': epoch - 1, 'checkpoint': str(checkpoint), 'report': str(report), 'metrics': {'p_at_20': p_at_20, 'detection_rate': 0.8, 'median_error': 10 - epoch}})

    train_module.retain_top_epoch_artifacts(candidates, keep=3)

    assert sorted(path.name for path in (reports_root / 'epochs').glob('epoch_*.json')) == [f'epoch_{epoch:03d}.json' for epoch in range(1, 5)]
    assert sorted(path.name for path in checkpoints_root.glob('epoch_top_*.pt')) == ['epoch_top_001.pt', 'epoch_top_002.pt', 'epoch_top_003.pt']


def test_presence_confusion_matrix_is_not_localization_confusion() -> None:
    assert train_module.presence_confusion_matrix({'TP': 7, 'FP1': 2, 'FN': 3, 'FP2': 4, 'TN': 9}) == {'TP': 9.0, 'FN': 3.0, 'FP': 4.0, 'TN': 9.0}


def test_training_plots_include_all_epoch_reports_and_skip_without_matplotlib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reports_root = tmp_path / 'reports'
    epochs_root = reports_root / 'epochs'
    epochs_root.mkdir(parents=True)
    for epoch in range(1, 4):
        (epochs_root / f'epoch_{epoch:03d}.json').write_text(json.dumps({'metrics': {'p_at_20': epoch / 10, 'detection_rate': 0.8}}), encoding='utf-8')
    (epochs_root / 'epoch_001_confusion_matrix.json').write_text(json.dumps({'TP': 1, 'TN': 2}), encoding='utf-8')
    history = [{'epoch': epoch - 1, 'train_loss': 1 / epoch, 'val_loss': 2 / epoch, 'metrics': {'p_at_20': epoch / 10, 'detection_rate': 0.8}} for epoch in range(1, 4)]
    train_module.write_metrics_history_csv(reports_root, history)
    outputs = train_module._plot_training_reports(reports_root, history)
    if outputs:
        assert (reports_root / 'plots' / 'metrics_evolution.png').exists()
        assert (reports_root / 'plots' / 'checkpoint_comparison.png').exists()
        assert len(list(epochs_root.glob('*_confusion_matrix.png'))) == 3
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', None)
    assert train_module._plot_training_reports(reports_root, history) == []


def test_load_epoch_artifacts_prunes_reports_with_missing_checkpoints(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    checkpoints_root = tmp_path / "checkpoints"
    reports_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    existing_checkpoint = checkpoints_root / "TrackNet_epoch_002.pt"
    existing_checkpoint.write_text("epoch-2", encoding="utf-8")
    existing_report = reports_root / "epoch-002-validation.json"
    existing_report.write_text(
        json.dumps(
            {
                "checkpoint": str(existing_checkpoint),
                "metrics": {"p_at_20": 0.8, "detection_rate": 0.8, "median_error": 9.0},
            }
        ),
        encoding="utf-8",
    )

    orphan_report = reports_root / "epoch-001-validation.json"
    orphan_report.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoints_root / "TrackNet_epoch_001.pt"),
                "metrics": {"p_at_20": 0.7, "detection_rate": 0.79, "median_error": 10.0},
            }
        ),
        encoding="utf-8",
    )

    artifacts = train_module.load_epoch_artifacts(reports_root)

    assert [Path(item["checkpoint"]).name for item in artifacts] == ["TrackNet_epoch_002.pt"]
    assert orphan_report.exists() is True
    assert existing_report.exists() is True


def test_select_best_epoch_artifact_uses_baseline_validation_report_policy(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    checkpoints_root = tmp_path / "checkpoints"
    reports_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    baseline_report = reports_root / "baseline-validation-report.json"
    baseline_report.write_text(
        json.dumps(
            {
                "checkpoint": "baseline.pt",
                "split": "validation",
                "dataset_root": "/cache/run/adapted-data",
                "metrics": {"detection_rate": 0.90, "p_at_20": 0.70, "median_error": 11.0},
            }
        ),
        encoding="utf-8",
    )

    for epoch, p_at_20, detection_rate in ((1, 0.95, 0.87), (2, 0.91, 0.89)):
        checkpoint_path = checkpoints_root / f"TrackNet_epoch_{epoch:03d}.pt"
        report_path = reports_root / f"epoch-{epoch:03d}-validation.json"
        checkpoint_path.write_text(f"epoch-{epoch}", encoding="utf-8")
        report_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "split": "validation",
                    "dataset_root": "/cache/run/adapted-data",
                    "metrics": {"p_at_20": p_at_20, "detection_rate": detection_rate, "median_error": float(10 - epoch)},
                }
            ),
            encoding="utf-8",
        )

    selected = train_module.select_best_epoch_artifact(
        reports_dir=reports_root,
        baseline_validation_report=baseline_report,
    )

    assert Path(selected["checkpoint"]).name == "TrackNet_epoch_002.pt"
    assert selected["metrics"]["detection_rate"] == pytest.approx(0.89)


def test_retain_top_epoch_artifacts_preserves_guard_selected_checkpoint(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    checkpoints_root = tmp_path / "checkpoints"
    reports_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    baseline_report = reports_root / "baseline-validation-report.json"
    baseline_report.write_text(
        json.dumps(
            {
                "checkpoint": "baseline.pt",
                "split": "validation",
                "dataset_root": "/cache/run/adapted-data",
                "metrics": {"detection_rate": 0.90, "p_at_20": 0.70, "median_error": 11.0},
            }
        ),
        encoding="utf-8",
    )

    candidates = []
    for epoch, p_at_20, detection_rate in ((1, 0.97, 0.85), (2, 0.96, 0.86), (3, 0.95, 0.87), (4, 0.80, 0.89)):
        checkpoint_path = checkpoints_root / f"TrackNet_epoch_{epoch:03d}.pt"
        report_path = reports_root / f"epoch-{epoch:03d}-validation.json"
        checkpoint_path.write_text(f"epoch-{epoch}", encoding="utf-8")
        report_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "split": "validation",
                    "dataset_root": "/cache/run/adapted-data",
                    "metrics": {"p_at_20": p_at_20, "detection_rate": detection_rate, "median_error": float(10 - epoch)},
                }
            ),
            encoding="utf-8",
        )
        candidates.append(
            {
                "epoch": epoch - 1,
                "checkpoint": str(checkpoint_path),
                "report": str(report_path),
                "metrics": {"p_at_20": p_at_20, "detection_rate": detection_rate, "median_error": float(10 - epoch)},
            }
        )

    selected = train_module.select_best_epoch_artifact(
        reports_dir=reports_root,
        baseline_validation_report=baseline_report,
    )

    kept = train_module.retain_top_epoch_artifacts(
        candidates,
        keep=3,
        preserve={(selected["checkpoint"], selected["report"])},
    )

    assert Path(selected["checkpoint"]).name == "TrackNet_epoch_004.pt"
    assert Path(selected["checkpoint"]).exists() is True
    assert Path(selected["report"]).exists() is True
    assert any(item["checkpoint"] == selected["checkpoint"] for item in kept)


def test_load_epoch_artifacts_rejects_checkpoint_paths_outside_active_run_roots(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    checkpoints_root = tmp_path / "checkpoints"
    outside_checkpoint = tmp_path / "outside" / "TrackNet_epoch_001.pt"
    reports_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    outside_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    outside_checkpoint.write_text("outside", encoding="utf-8")

    unsafe_report = reports_root / "epoch-001-validation.json"
    unsafe_report.write_text(
        json.dumps(
            {
                "checkpoint": str(outside_checkpoint),
                "metrics": {"p_at_20": 0.9, "detection_rate": 0.9, "median_error": 8.0},
            }
        ),
        encoding="utf-8",
    )

    artifacts = train_module.load_epoch_artifacts(reports_root)

    assert artifacts == []
    assert unsafe_report.exists() is True
    assert outside_checkpoint.exists() is True


def test_retain_top_epoch_artifacts_never_unlinks_checkpoint_outside_run_root(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    checkpoints_root = tmp_path / "checkpoints"
    outside_checkpoint = tmp_path / "outside" / "TrackNet_epoch_001.pt"
    safe_checkpoint = checkpoints_root / "TrackNet_epoch_002.pt"
    unsafe_report = reports_root / "epoch-001-validation.json"
    safe_report = reports_root / "epoch-002-validation.json"

    outside_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    safe_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    unsafe_report.parent.mkdir(parents=True, exist_ok=True)
    outside_checkpoint.write_text("outside", encoding="utf-8")
    safe_checkpoint.write_text("safe", encoding="utf-8")
    unsafe_report.write_text("unsafe", encoding="utf-8")
    safe_report.write_text("safe", encoding="utf-8")

    kept = train_module.retain_top_epoch_artifacts(
        [
            {
                "epoch": 0,
                "checkpoint": str(outside_checkpoint),
                "report": str(unsafe_report),
                "metrics": {"p_at_20": 0.1, "detection_rate": 0.1, "median_error": 99.0},
            },
            {
                "epoch": 1,
                "checkpoint": str(safe_checkpoint),
                "report": str(safe_report),
                "metrics": {"p_at_20": 0.9, "detection_rate": 0.9, "median_error": 8.0},
            },
        ],
        keep=1,
    )

    assert [Path(item["checkpoint"]).name for item in kept] == ["TrackNet_epoch_002.pt"]
    assert outside_checkpoint.exists() is True
    assert unsafe_report.exists() is True
    assert safe_checkpoint.exists() is True


def test_epoch_artifact_discovery_ignores_stale_future_and_mixed_reports(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    checkpoints_root = tmp_path / "checkpoints"
    epochs_root = reports_root / "epochs"
    epochs_root.mkdir(parents=True)
    checkpoints_root.mkdir(parents=True)
    checkpoint = checkpoints_root / "TrackNet_epoch_001.pt"
    checkpoint.write_text("one", encoding="utf-8")
    (epochs_root / "epoch_001.json").write_text(json.dumps({"checkpoint": str(checkpoint), "metrics": {"p_at_20": 0.5}}), encoding="utf-8")
    (epochs_root / "epoch_002.json").write_text("{broken", encoding="utf-8")
    (epochs_root / "epoch_001_confusion_matrix.json").write_text(json.dumps({"TP": 1}), encoding="utf-8")
    (epochs_root / "notes.json").write_text(json.dumps({"checkpoint": str(checkpoint)}), encoding="utf-8")

    artifacts = train_module.load_epoch_artifacts(reports_root, max_epoch=0)

    assert [item["epoch"] for item in artifacts] == [0]
    assert (epochs_root / "epoch_002.json").exists()
    assert (epochs_root / "epoch_001_confusion_matrix.json").exists()


def test_fresh_run_reset_is_limited_to_training_outputs(tmp_path: Path) -> None:
    save_dir = tmp_path / "checkpoints"
    reports_root = tmp_path / "reports"
    save_dir.mkdir()
    reports_root.mkdir()
    (save_dir / "TrackNet_cur.pt").write_text("stale", encoding="utf-8")
    (save_dir / "TrackNet_epoch_001.pt").write_text("stale", encoding="utf-8")
    (reports_root / "epoch-001-validation.json").write_text("{}", encoding="utf-8")
    (reports_root / "baseline-validation-report.json").write_text("baseline", encoding="utf-8")
    adapted_root = tmp_path / "adapted-data"
    adapted_root.mkdir()
    (adapted_root / "manifest.json").write_text("manifest", encoding="utf-8")

    train_module._reset_training_outputs(save_dir, reports_root)

    assert not (save_dir / "TrackNet_cur.pt").exists()
    assert not (save_dir / "TrackNet_epoch_001.pt").exists()
    assert not (reports_root / "epoch-001-validation.json").exists()
    assert (reports_root / "baseline-validation-report.json").read_text() == "baseline"
    assert (adapted_root / "manifest.json").read_text() == "manifest"


def test_canonical_retention_is_deterministic_and_leaves_no_duplicate_top_files(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    checkpoints_root = tmp_path / "checkpoints"
    candidates = []
    for epoch in range(1, 5):
        checkpoint = checkpoints_root / f"TrackNet_epoch_{epoch:03d}.pt"
        report = reports_root / "epochs" / f"epoch_{epoch:03d}.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(str(epoch), encoding="utf-8")
        report.write_text(json.dumps({"checkpoint": str(checkpoint)}), encoding="utf-8")
        candidates.append({"epoch": epoch - 1, "checkpoint": str(checkpoint), "report": str(report), "metrics": {"p_at_20": 0.8, "detection_rate": 0.9, "median_error": 10.0}})
    (checkpoints_root / "TrackNet_best.pt").write_text("best", encoding="utf-8")
    (checkpoints_root / "TrackNet_cur.pt").write_text("cur", encoding="utf-8")

    kept = train_module.retain_top_epoch_artifacts(candidates, keep=3)

    assert [Path(item["checkpoint"]).name for item in kept] == ["epoch_top_001.pt", "epoch_top_002.pt", "epoch_top_003.pt"]
    assert [json.loads(Path(item["report"]).read_text())["checkpoint"] for item in kept] == [item["checkpoint"] for item in kept]
    assert not list(checkpoints_root.glob(".epoch_top_*.tmp"))
    assert (checkpoints_root / "TrackNet_best.pt").read_text() == "best"
    assert (checkpoints_root / "TrackNet_cur.pt").read_text() == "cur"


def test_atomic_checkpoint_write_preserves_existing_file_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "TrackNet_cur.pt"
    target.write_text("old", encoding="utf-8")

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(train_module.torch, "save", fail_save, raising=False)
    with pytest.raises(OSError, match="disk full"):
        train_module._save_checkpoint(target, epoch=0, best_metric=None, model=types.SimpleNamespace(state_dict=lambda: {}), optimizer=types.SimpleNamespace(state_dict=lambda: {}), scheduler=None, param_dict={}, adapted_data_manifest=None, history=[])

    assert target.read_text() == "old"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("patience", "metrics", "expected"),
    [
        (0, [{"p_at_20": 0.0, "detection_rate": 0.8}], (True, 0, False)),
        (1, [{"p_at_20": 0.0, "detection_rate": 0.8}, {"p_at_20": 0.0, "detection_rate": 0.8}], (False, 1, True)),
        (3, [{"p_at_20": 0.0, "detection_rate": 0.8}], (True, 0, False)),
    ],
)
def test_early_stopping_patience_counts_only_after_initial_best(patience, metrics, expected) -> None:
    best = None
    count = 0
    for metric in metrics:
        accepted, count, stopped = train_module.update_early_stopping(count, metrics=metric, best_metric=best, baseline_detection_rate=0.8, patience=patience)
        if accepted:
            best = metric
    assert (accepted, count, stopped) == expected
