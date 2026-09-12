from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_tracknet_test_module():
    project_root = Path(__file__).resolve().parents[1]
    tracknet_root = project_root / "TrackNetV3"
    sys.path.insert(0, str(tracknet_root))

    torch_module = types.ModuleType("torch")
    nn_module = types.ModuleType("torch.nn")
    utils_module = types.ModuleType("torch.utils")
    data_module = types.ModuleType("torch.utils.data")
    nn_module.Module = object
    data_module.Dataset = object
    data_module.IterableDataset = object
    data_module.DataLoader = object
    utils_module.data = data_module
    torch_module.nn = nn_module
    torch_module.utils = utils_module
    sys.modules.setdefault("torch", torch_module)
    sys.modules.setdefault("torch.nn", nn_module)
    sys.modules.setdefault("torch.utils", utils_module)
    sys.modules.setdefault("torch.utils.data", data_module)

    coco_module = types.ModuleType("pycocotools.coco")
    cocoeval_module = types.ModuleType("pycocotools.cocoeval")
    coco_module.COCO = object
    cocoeval_module.COCOeval = object
    sys.modules.setdefault("pycocotools", types.ModuleType("pycocotools"))
    sys.modules.setdefault("pycocotools.coco", coco_module)
    sys.modules.setdefault("pycocotools.cocoeval", cocoeval_module)

    general_module = types.ModuleType("utils.general")
    general_module.HEIGHT = 288
    general_module.WIDTH = 512
    general_module.SIGMA = 2.5
    general_module.IMG_FORMAT = "png"
    general_module.get_rally_dirs = lambda *args, **kwargs: []
    general_module.get_match_median = lambda *args, **kwargs: None
    sys.modules.setdefault("utils.general", general_module)
    sys.modules.setdefault("utils.metric", types.ModuleType("utils.metric"))

    spec = importlib.util.spec_from_file_location("tracknet_test_module", tracknet_root / "test.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_parser_exposes_dataset_root_flag() -> None:
    module = _load_tracknet_test_module()

    parser = module.build_parser()
    args = parser.parse_args(["--dataset_root", "/tmp/adapted-data"])

    assert args.dataset_root == "/tmp/adapted-data"


def test_build_rally_dataset_forwards_dataset_root_to_dataset_ctor() -> None:
    module = _load_tracknet_test_module()
    captured = {}

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        return kwargs

    module.Shuttlecock_Trajectory_Dataset = fake_dataset

    dataset = module.build_rally_dataset(
        {"dataset_root": "/tmp/adapted-data", "cache_root": "/tmp/run/checkpoints/cache"},
        rally_dir="/tmp/adapted-data/val/match14/frame/clip_014",
        seq_len=8,
        sliding_step=1,
        data_mode="heatmap",
        bg_mode="concat",
    )

    assert dataset["root_dir"] == "/tmp/adapted-data"
    assert captured["root_dir"] == "/tmp/adapted-data"
    assert captured["cache_root"] == "/tmp/run/checkpoints/cache"
    assert captured["rally_dir"] == "/tmp/adapted-data/val/match14/frame/clip_014"


def test_build_rally_dataset_uses_default_root_when_flag_is_omitted() -> None:
    module = _load_tracknet_test_module()
    captured = {}

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        return kwargs

    module.Shuttlecock_Trajectory_Dataset = fake_dataset

    parser = module.build_parser()
    args = parser.parse_args([])
    dataset = module.build_rally_dataset(
        vars(args),
        rally_dir="/tmp/data/test/match01/frame/clip_001",
        seq_len=8,
        sliding_step=8,
        data_mode="coordinate",
        padding=True,
    )

    assert args.dataset_root == module.data_dir
    assert dataset["root_dir"] == module.data_dir
    assert captured["root_dir"] == module.data_dir
    assert captured["cache_root"] is None
    assert captured["padding"] is True


def test_build_parser_exposes_cache_root_flag() -> None:
    module = _load_tracknet_test_module()

    args = module.build_parser().parse_args(["--cache_root", "/tmp/run/checkpoints/cache"])

    assert args.cache_root == "/tmp/run/checkpoints/cache"


def test_get_test_res_treats_missing_drop_frame_json_as_no_drop(tmp_path) -> None:
    module = _load_tracknet_test_module()
    module.data_dir = str(tmp_path)
    module.get_metric = lambda *args: (0.0, 0.0, 0.0, 0.0, 0.0)

    pred_dict = {
        "match1_rally1": {
            "Type": [
                module.pred_types_map["TP"],
                module.pred_types_map["FP1"],
                module.pred_types_map["FN"],
            ]
        }
    }

    res = module.get_test_res(pred_dict, drop=True)

    assert res["TP"] == 1
    assert res["FP1"] == 1
    assert res["FN"] == 1
    assert res["TN"] == 0
    assert res["FP2"] == 0


def test_get_test_res_preserves_drop_frame_behavior_when_file_exists(tmp_path) -> None:
    module = _load_tracknet_test_module()
    module.data_dir = str(tmp_path)
    module.get_metric = lambda *args: (0.0, 0.0, 0.0, 0.0, 0.0)
    (tmp_path / "drop_frame.json").write_text(
        '{"start": {"match1_rally1": 1}, "end": {"match1_rally1": 3}}',
        encoding="utf-8",
    )

    pred_dict = {
        "match1_rally1": {
            "Type": [
                module.pred_types_map["TP"],
                module.pred_types_map["FP1"],
                module.pred_types_map["FN"],
                module.pred_types_map["TN"],
            ]
        }
    }

    res = module.get_test_res(pred_dict, drop=True)

    assert res["TP"] == 0
    assert res["FP1"] == 1
    assert res["FN"] == 1
    assert res["TN"] == 0
    assert res["FP2"] == 0
