from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import io
import pickle
from pathlib import Path
import zipfile
from typing import Any, Protocol, Sequence

import numpy as np


FrameSize = tuple[int, int]
OnnxInputLayout = str
ColorOrder = str
NormalizationMode = str
CheckpointInspection = dict[str, Any]


DEFAULT_TRACKNET_PT = Path(__file__).with_name("TrackNet_best.pt")


@dataclass(slots=True)
class BallHeatmapPrediction:
    heatmap: np.ndarray
    input_size: FrameSize
    source_frame_size: FrameSize
    frame_indices: tuple[int, ...]
    backend_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BallTemporalDetector(Protocol):
    context_size: int
    backend_name: str

    def predict(self, frames: Sequence[np.ndarray], frame_indices: Sequence[int]) -> BallHeatmapPrediction:
        ...


@dataclass(slots=True)
class BufferedFrame:
    index: int
    frame: np.ndarray


class TemporalFrameBuffer:
    def __init__(self, window_size: int) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self._items: deque[BufferedFrame] = deque(maxlen=window_size)

    def push(self, frame_index: int, frame: np.ndarray) -> None:
        self._items.append(BufferedFrame(index=frame_index, frame=frame.copy()))

    @property
    def ready(self) -> bool:
        return len(self._items) == self.window_size

    def frames(self) -> list[np.ndarray]:
        return [item.frame for item in self._items]

    def frame_indices(self) -> tuple[int, ...]:
        return tuple(item.index for item in self._items)

    def latest(self) -> np.ndarray:
        if not self._items:
            raise RuntimeError("Temporal frame buffer is empty")
        return self._items[-1].frame


def padded_window(frame_indices: Sequence[int], target_position: int, size: int) -> tuple[int, ...]:
    """Return a causal window with repeated edge frames at sequence bounds."""
    if not frame_indices or size < 1:
        raise ValueError("frame indices and window size must be positive")
    last_position = len(frame_indices) - 1
    return tuple(
        frame_indices[max(0, min(last_position, target_position - size + 1 + offset))]
        for offset in range(size)
    )


def preprocess_frame(
    frame: np.ndarray,
    input_size: FrameSize,
    *,
    color_order: ColorOrder,
    normalization: NormalizationMode,
) -> np.ndarray:
    import cv2

    width, height = input_size
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    if color_order == "rgb":
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    elif color_order == "bgr":
        image = resized
    else:
        raise ValueError(f"Unsupported color order: {color_order}")

    image = image.astype(np.float32) / 255.0
    if normalization == "zero_one":
        return image
    if normalization == "neg_one_one":
        return image * 2.0 - 1.0
    raise ValueError(f"Unsupported normalization mode: {normalization}")


def stack_frames_for_model(
    frames: Sequence[np.ndarray],
    input_size: FrameSize,
    *,
    color_order: ColorOrder,
    normalization: NormalizationMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not frames:
        raise ValueError("At least one frame is required")
    processed = np.stack(
        [
            preprocess_frame(
                frame,
                input_size,
                color_order=color_order,
                normalization=normalization,
            )
            for frame in frames
        ],
        axis=0,
    )
    nchw = np.transpose(processed, (0, 3, 1, 2))
    ncthw = np.expand_dims(nchw, axis=0)
    stacked_4d = np.expand_dims(np.concatenate(nchw, axis=0), axis=0)
    return processed, ncthw, stacked_4d


def default_tracknet_checkpoint_path() -> Path | None:
    return DEFAULT_TRACKNET_PT if DEFAULT_TRACKNET_PT.exists() else None


class _CheckpointStorageType:
    def __init__(self, name: str) -> None:
        self.__name__ = name


class _CheckpointMetadataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict

            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            def rebuild_tensor_v2(storage: Any, storage_offset: int, size: Sequence[int], stride: Sequence[int], requires_grad: bool, backward_hooks: Any) -> dict[str, Any]:
                return {
                    "storage": storage,
                    "storage_offset": storage_offset,
                    "size": tuple(size),
                    "stride": tuple(stride),
                    "requires_grad": requires_grad,
                }

            return rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return _CheckpointStorageType(name)
        raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        return {"persistent_id": pid}


def inspect_tracknet_checkpoint(checkpoint_path: str | Path) -> CheckpointInspection:
    checkpoint = Path(checkpoint_path).expanduser()
    info: CheckpointInspection = {
        "path": str(checkpoint),
        "name": checkpoint.name,
        "suffix": checkpoint.suffix.lower(),
        "exists": checkpoint.exists(),
    }
    if not checkpoint.exists():
        return info

    if checkpoint.suffix.lower() == ".onnx":
        info.update({"format": "onnx"})
        return info

    if checkpoint.suffix.lower() != ".pt":
        info.update({"format": "unknown"})
        return info

    info["format"] = "pytorch"
    info["zip_serialized"] = zipfile.is_zipfile(checkpoint)
    if not info["zip_serialized"]:
        info["inspection_error"] = "Only zip-serialized torch checkpoints can be inspected without torch installed."
        return info

    with zipfile.ZipFile(checkpoint) as archive:
        data_pickle_name = next((name for name in archive.namelist() if name.endswith("data.pkl")), None)
        if data_pickle_name is None:
            info["inspection_error"] = "Could not find data.pkl inside the torch checkpoint archive."
            return info
        payload = archive.read(data_pickle_name)

    try:
        parsed = _CheckpointMetadataUnpickler(io.BytesIO(payload)).load()
    except Exception as error:  # pragma: no cover - defensive path for unknown checkpoints
        info["inspection_error"] = f"Failed to inspect torch checkpoint metadata: {error}"
        return info

    param_dict = parsed.get("param_dict") if isinstance(parsed, dict) else None
    model_state = parsed.get("model") if isinstance(parsed, dict) else None
    first_weight_shape = None
    model_key_count = 0
    if hasattr(model_state, "items"):
        model_items = list(model_state.items())
        model_key_count = len(model_items)
        for key, value in model_items:
            if key.endswith(".weight") and isinstance(value, dict) and "size" in value:
                first_weight_shape = tuple(value["size"])
                break
        info["state_dict_preview"] = [key for key, _ in model_items[:12]]
    info["top_level_keys"] = sorted(parsed.keys()) if isinstance(parsed, dict) else []
    info["param_dict"] = param_dict if isinstance(param_dict, dict) else None
    info["model_key_count"] = model_key_count
    info["first_weight_shape"] = first_weight_shape
    if isinstance(param_dict, dict):
        info["seq_len"] = param_dict.get("seq_len")
        info["bg_mode"] = param_dict.get("bg_mode", "")
        info["model_name"] = param_dict.get("model_name")
    if first_weight_shape is not None:
        info["input_channels"] = int(first_weight_shape[1])
    return info


def _infer_tracknet_input_channels(seq_len: int, bg_mode: str) -> int:
    if bg_mode == "":
        return seq_len * 3
    if bg_mode == "subtract":
        return seq_len
    if bg_mode == "subtract_concat":
        return seq_len * 4
    if bg_mode == "concat":
        return seq_len * 3 + 3
    raise RuntimeError(
        f"Unsupported TrackNet background mode '{bg_mode}'. "
        "This experimental scaffold only understands '', subtract, subtract_concat, and concat."
    )


def _build_tracknet_channels(frames: Sequence[np.ndarray], input_size: FrameSize, bg_mode: str) -> np.ndarray:
    import cv2

    if not frames:
        raise ValueError("At least one frame is required")
    rgb_frames = []
    for frame in frames:
        resized = cv2.resize(frame, input_size, interpolation=cv2.INTER_AREA)
        rgb_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32))
    stacked = np.stack(rgb_frames, axis=0)
    median = np.median(stacked, axis=0)

    if bg_mode == "":
        channels = [np.moveaxis(frame / 255.0, -1, 0) for frame in stacked]
    elif bg_mode == "subtract":
        channels = [np.sum(np.abs(frame - median), axis=2, dtype=np.float32, keepdims=True) / 255.0 for frame in stacked]
        channels = [np.moveaxis(frame, -1, 0) for frame in channels]
    elif bg_mode == "subtract_concat":
        channels = []
        for frame in stacked:
            diff = np.sum(np.abs(frame - median), axis=2, dtype=np.float32, keepdims=True) / 255.0
            channels.append(np.concatenate((np.moveaxis(frame / 255.0, -1, 0), np.moveaxis(diff, -1, 0)), axis=0))
    elif bg_mode == "concat":
        channels = [np.moveaxis(median / 255.0, -1, 0)]
        channels.extend(np.moveaxis(frame / 255.0, -1, 0) for frame in stacked)
    else:
        raise RuntimeError(f"Unsupported TrackNet background mode '{bg_mode}'")

    return np.expand_dims(np.concatenate(channels, axis=0).astype(np.float32), axis=0)


def _normalize_heatmap_output(output: Any, *, expected_seq_len: int) -> tuple[np.ndarray, dict[str, Any]]:
    heatmap = np.asarray(output)
    metadata: dict[str, Any] = {"raw_output_shape": list(heatmap.shape)}
    if heatmap.ndim != 4:
        raise RuntimeError(
            f"Expected TrackNet output to have rank 4 [N, C, H, W], got shape {tuple(heatmap.shape)}"
        )
    if heatmap.shape[0] != 1:
        raise RuntimeError(
            f"Expected a single-batch TrackNet output, got batch size {heatmap.shape[0]}"
        )
    channel_count = int(heatmap.shape[1])
    metadata["output_channels"] = channel_count
    if channel_count == 1:
        metadata["selected_frame_channel"] = 0
        return heatmap[0, 0].astype(np.float32), metadata
    if channel_count != expected_seq_len:
        raise RuntimeError(
            f"TrackNet checkpoint produced {channel_count} output channels, but the checkpoint metadata says seq_len={expected_seq_len}. "
            "The local experimental adapter cannot safely decide how to map those channels to a single frame heatmap."
        )
    metadata["selected_frame_channel"] = channel_count - 1
    metadata["selection_note"] = "Using the latest frame heatmap from the TrackNet output sequence."
    return heatmap[0, channel_count - 1].astype(np.float32), metadata


class StubTrackNetDetector:
    def __init__(self, *, context_size: int, input_size: FrameSize) -> None:
        self.context_size = context_size
        self.input_size = input_size
        self.backend_name = "stub"

    def predict(self, frames: Sequence[np.ndarray], frame_indices: Sequence[int]) -> BallHeatmapPrediction:
        if len(frames) != self.context_size:
            raise ValueError(f"Expected {self.context_size} frames, got {len(frames)}")
        source_height, source_width = frames[-1].shape[:2]
        heatmap = np.zeros((self.input_size[1], self.input_size[0]), dtype=np.float32)
        return BallHeatmapPrediction(
            heatmap=heatmap,
            input_size=self.input_size,
            source_frame_size=(source_width, source_height),
            frame_indices=tuple(frame_indices),
            backend_name=self.backend_name,
            metadata={"stub": True, "note": "No checkpoint was provided; emitted an empty heatmap."},
        )


class OnnxTrackNetDetector:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        context_size: int,
        input_size: FrameSize,
        input_layout: OnnxInputLayout = "auto",
        color_order: ColorOrder = "rgb",
        normalization: NormalizationMode = "zero_one",
    ) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "ONNX backend requested but onnxruntime is not installed. "
                "Install it first or rerun with --allow-stub."
            ) from error

        self.context_size = context_size
        self.input_size = input_size
        self.backend_name = "onnx"
        self.checkpoint_path = checkpoint_path
        self.input_layout = input_layout
        self.color_order = color_order
        self.normalization = normalization
        self._session = ort.InferenceSession(str(checkpoint_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def predict(self, frames: Sequence[np.ndarray], frame_indices: Sequence[int]) -> BallHeatmapPrediction:
        if len(frames) != self.context_size:
            raise ValueError(f"Expected {self.context_size} frames, got {len(frames)}")

        _, ncthw, stacked_4d = stack_frames_for_model(
            frames,
            self.input_size,
            color_order=self.color_order,
            normalization=self.normalization,
        )
        input_meta = self._session.get_inputs()[0]
        rank = len(input_meta.shape)
        selected_layout = self.input_layout
        if selected_layout == "auto":
            if rank == 5:
                selected_layout = "temporal_5d"
            elif rank == 4:
                selected_layout = "stacked_4d"
            else:
                raise RuntimeError(f"Unsupported ONNX input rank for experimental scaffold: {rank}")

        if selected_layout == "temporal_5d":
            model_input = ncthw
        elif selected_layout == "stacked_4d":
            model_input = stacked_4d
        else:
            raise RuntimeError(f"Unsupported ONNX input layout for experimental scaffold: {selected_layout}")

        output = self._session.run([self._output_name], {self._input_name: model_input})[0]
        heatmap = np.asarray(output)
        while heatmap.ndim > 2:
            heatmap = heatmap[0]
        source_height, source_width = frames[-1].shape[:2]
        return BallHeatmapPrediction(
            heatmap=heatmap.astype(np.float32),
            input_size=self.input_size,
            source_frame_size=(source_width, source_height),
            frame_indices=tuple(frame_indices),
            backend_name=self.backend_name,
            metadata={
                "checkpoint_path": str(self.checkpoint_path),
                "input_layout": selected_layout,
                "color_order": self.color_order,
                "normalization": self.normalization,
            },
        )


class PytorchTrackNetDetector:
    def __init__(self, checkpoint_path: Path, *, context_size: int, input_size: FrameSize) -> None:
        try:
            import torch
            import torch.nn as nn
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "PyTorch backend requested but torch is not installed. "
                "Install torch first, or rerun with --inspect-model to inspect the checkpoint without inference."
            ) from error

        inspection = inspect_tracknet_checkpoint(checkpoint_path)
        param_dict = inspection.get("param_dict") or {}
        seq_len = int(param_dict.get("seq_len") or 0)
        bg_mode = str(param_dict.get("bg_mode", ""))
        model_name = param_dict.get("model_name")
        input_channels = inspection.get("input_channels")

        if model_name not in {None, "TrackNet"}:
            raise RuntimeError(
                f"Unsupported PyTorch checkpoint model_name={model_name!r}. "
                "This experimental scaffold only implements the official TrackNet tracking module path."
            )
        if seq_len < 1:
            raise RuntimeError(
                "Could not resolve seq_len from the .pt checkpoint. "
                "The local experimental adapter needs checkpoint metadata to safely instantiate TrackNet."
            )
        expected_input_channels = _infer_tracknet_input_channels(seq_len, bg_mode)
        if input_channels is not None and int(input_channels) != expected_input_channels:
            raise RuntimeError(
                "The .pt checkpoint input channels do not match the local TrackNet assumptions. "
                f"Expected {expected_input_channels} channels for seq_len={seq_len}, bg_mode={bg_mode!r}, "
                f"but the checkpoint's first conv expects {input_channels}."
            )

        class Conv2DBlock(nn.Module):
            def __init__(self, in_dim: int, out_dim: int) -> None:
                super().__init__()
                self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
                self.bn = nn.BatchNorm2d(out_dim)
                self.relu = nn.ReLU()

            def forward(self, x: Any) -> Any:
                return self.relu(self.bn(self.conv(x)))

        class Double2DConv(nn.Module):
            def __init__(self, in_dim: int, out_dim: int) -> None:
                super().__init__()
                self.conv_1 = Conv2DBlock(in_dim, out_dim)
                self.conv_2 = Conv2DBlock(out_dim, out_dim)

            def forward(self, x: Any) -> Any:
                return self.conv_2(self.conv_1(x))

        class Triple2DConv(nn.Module):
            def __init__(self, in_dim: int, out_dim: int) -> None:
                super().__init__()
                self.conv_1 = Conv2DBlock(in_dim, out_dim)
                self.conv_2 = Conv2DBlock(out_dim, out_dim)
                self.conv_3 = Conv2DBlock(out_dim, out_dim)

            def forward(self, x: Any) -> Any:
                return self.conv_3(self.conv_2(self.conv_1(x)))

        class TrackNet(nn.Module):
            def __init__(self, in_dim: int, out_dim: int) -> None:
                super().__init__()
                self.down_block_1 = Double2DConv(in_dim, 64)
                self.down_block_2 = Double2DConv(64, 128)
                self.down_block_3 = Triple2DConv(128, 256)
                self.bottleneck = Triple2DConv(256, 512)
                self.up_block_1 = Triple2DConv(768, 256)
                self.up_block_2 = Double2DConv(384, 128)
                self.up_block_3 = Double2DConv(192, 64)
                self.predictor = nn.Conv2d(64, out_dim, kernel_size=1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x: Any) -> Any:
                x1 = self.down_block_1(x)
                x2 = self.down_block_2(nn.functional.max_pool2d(x1, kernel_size=2, stride=2))
                x3 = self.down_block_3(nn.functional.max_pool2d(x2, kernel_size=2, stride=2))
                x = self.bottleneck(nn.functional.max_pool2d(x3, kernel_size=2, stride=2))
                x = torch.cat([nn.functional.interpolate(x, scale_factor=2, mode="nearest"), x3], dim=1)
                x = self.up_block_1(x)
                x = torch.cat([nn.functional.interpolate(x, scale_factor=2, mode="nearest"), x2], dim=1)
                x = self.up_block_2(x)
                x = torch.cat([nn.functional.interpolate(x, scale_factor=2, mode="nearest"), x1], dim=1)
                x = self.up_block_3(x)
                return self.sigmoid(self.predictor(x))

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        state_dict = None
        if isinstance(checkpoint_data, dict):
            if isinstance(checkpoint_data.get("model"), dict):
                state_dict = checkpoint_data["model"]
            elif isinstance(checkpoint_data.get("state_dict"), dict):
                state_dict = checkpoint_data["state_dict"]
        if state_dict is None and hasattr(checkpoint_data, "state_dict"):
            state_dict = checkpoint_data.state_dict()
        if state_dict is None:
            raise RuntimeError(
                "Unsupported PyTorch checkpoint structure. Expected a state_dict-like object under 'model' or 'state_dict', "
                "or a serialized module exposing state_dict()."
            )

        model = TrackNet(expected_input_channels, seq_len)
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as error:
            raise RuntimeError(
                "The .pt checkpoint could be inspected, but the local lightweight TrackNet adapter could not load the weights. "
                f"Details: {error}"
            ) from error

        model.eval()
        self._torch = torch
        self._model = model
        self.context_size = seq_len
        self.input_size = input_size
        self.backend_name = "pytorch"
        self.checkpoint_path = checkpoint_path
        self.bg_mode = bg_mode
        self.checkpoint_info = inspection
        self.requested_context_size = context_size

    def predict(self, frames: Sequence[np.ndarray], frame_indices: Sequence[int]) -> BallHeatmapPrediction:
        if len(frames) != self.context_size:
            raise ValueError(f"Expected {self.context_size} frames, got {len(frames)}")

        model_input = _build_tracknet_channels(frames, self.input_size, self.bg_mode)
        with self._torch.no_grad():
            output = self._model(self._torch.from_numpy(model_input))
        heatmap, output_metadata = _normalize_heatmap_output(output.detach().cpu().numpy(), expected_seq_len=self.context_size)
        source_height, source_width = frames[-1].shape[:2]
        return BallHeatmapPrediction(
            heatmap=heatmap,
            input_size=self.input_size,
            source_frame_size=(source_width, source_height),
            frame_indices=tuple(frame_indices),
            backend_name=self.backend_name,
            metadata={
                "checkpoint_path": str(self.checkpoint_path),
                "bg_mode": self.bg_mode,
                "checkpoint_seq_len": self.context_size,
                "requested_context_size": self.requested_context_size,
                **output_metadata,
            },
        )


def build_detector(
    checkpoint_path: str | Path | None,
    *,
    backend: str,
    allow_stub: bool,
    context_size: int,
    input_size: FrameSize,
    onnx_input_layout: OnnxInputLayout = "auto",
    onnx_color_order: ColorOrder = "rgb",
    onnx_normalization: NormalizationMode = "zero_one",
) -> BallTemporalDetector:
    checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path else default_tracknet_checkpoint_path()
    if checkpoint is None:
        if allow_stub:
            return StubTrackNetDetector(context_size=context_size, input_size=input_size)
        raise RuntimeError(
            "No TrackNet checkpoint was provided. Pass --model PATH to try a real backend, "
            "or pass --allow-stub to exercise the scaffold with empty heatmaps."
        )
    if not checkpoint.exists():
        raise RuntimeError(f"TrackNet checkpoint not found: {checkpoint}")

    selected_backend = backend.lower()
    if selected_backend == "auto":
        if checkpoint.suffix.lower() == ".onnx":
            selected_backend = "onnx"
        elif checkpoint.suffix.lower() == ".pt":
            selected_backend = "pytorch"
        else:
            selected_backend = "unknown"
    if selected_backend == "pytorch":
        return PytorchTrackNetDetector(
            checkpoint,
            context_size=context_size,
            input_size=input_size,
        )
    if selected_backend == "onnx":
        return OnnxTrackNetDetector(
            checkpoint,
            context_size=context_size,
            input_size=input_size,
            input_layout=onnx_input_layout,
            color_order=onnx_color_order,
            normalization=onnx_normalization,
        )
    raise RuntimeError(
        f"Unsupported backend '{backend}' for checkpoint {checkpoint.name}. "
        "This scaffold currently supports stub mode, best-effort ONNX, and a local TrackNet-style PyTorch adapter for .pt checkpoints."
    )
