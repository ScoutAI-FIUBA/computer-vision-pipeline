from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ball import tracknet


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class _FakeTorch:
    def __init__(self, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)

    def device(self, name: str) -> str:
        return name


def test_select_torch_device_uses_cuda_when_available() -> None:
    assert tracknet._select_torch_device(_FakeTorch(cuda_available=True)) == "cuda"


def test_select_torch_device_falls_back_to_cpu() -> None:
    assert tracknet._select_torch_device(_FakeTorch(cuda_available=False)) == "cpu"


class _FakeTensor:
    def __init__(self, value: np.ndarray, calls: list[tuple[str, object]], *, requires_float: bool = False) -> None:
        self.value = value
        self.calls = calls
        self.requires_float = requires_float

    def to(self, device: object) -> _FakeTensor:
        self.calls.append(("tensor.to", device))
        return self

    def detach(self) -> _FakeTensor:
        return self

    def float(self) -> _FakeTensor:
        self.calls.append(("tensor.float", None))
        self.requires_float = False
        return self

    def cpu(self) -> _FakeTensor:
        self.calls.append(("tensor.cpu", None))
        return self

    def numpy(self) -> np.ndarray:
        if self.requires_float:
            raise TypeError("Got unsupported ScalarType BFloat16")
        return self.value


class _InferenceMode:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    def __enter__(self) -> None:
        self.calls.append(("inference_mode.enter", None))

    def __exit__(self, *_args: object) -> None:
        self.calls.append(("inference_mode.exit", None))


class _InferenceTorch:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    def inference_mode(self) -> _InferenceMode:
        return _InferenceMode(self.calls)

    def from_numpy(self, value: np.ndarray) -> _FakeTensor:
        self.calls.append(("from_numpy", value))
        return _FakeTensor(np.zeros((1, 1, 2, 2), dtype=np.float32), self.calls)


class _FakeModel:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    def __call__(self, value: _FakeTensor) -> _FakeTensor:
        self.calls.append(("model", value))
        return value


def test_predict_moves_input_and_uses_inference_mode(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    detector = object.__new__(tracknet.PytorchTrackNetDetector)
    detector._torch = _InferenceTorch(calls)
    detector._model = _FakeModel(calls)
    detector._device = "cuda"
    detector.context_size = 1
    detector.input_size = (2, 2)
    detector.bg_mode = ""
    detector.backend_name = "pytorch"
    detector.checkpoint_path = "model.pt"
    detector.requested_context_size = 1

    monkeypatch.setattr(tracknet, "_build_tracknet_channels", lambda *args: np.zeros((1, 3, 2, 2), dtype=np.float32))
    prediction = detector.predict([np.zeros((2, 2, 3), dtype=np.uint8)], [7])

    assert prediction.heatmap.shape == (2, 2)
    assert [name for name, _ in calls] == [
        "inference_mode.enter",
        "from_numpy",
        "tensor.to",
        "model",
        "inference_mode.exit",
        "tensor.float",
        "tensor.cpu",
    ]
    assert calls[2] == ("tensor.to", "cuda")


class _BFloat16Model(_FakeModel):
    def __call__(self, value: _FakeTensor) -> _FakeTensor:
        self.calls.append(("model", value))
        return _FakeTensor(value.value, self.calls, requires_float=True)


def test_predict_casts_bfloat16_output_before_numpy(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    detector = object.__new__(tracknet.PytorchTrackNetDetector)
    detector._torch = _InferenceTorch(calls)
    detector._model = _BFloat16Model(calls)
    detector._device = "cpu"
    detector.context_size = 1
    detector.input_size = (2, 2)
    detector.bg_mode = ""
    detector.backend_name = "pytorch"
    detector.checkpoint_path = "model.pt"
    detector.requested_context_size = 1

    monkeypatch.setattr(tracknet, "_build_tracknet_channels", lambda *args: np.zeros((1, 3, 2, 2), dtype=np.float32))
    prediction = detector.predict([np.zeros((2, 2, 3), dtype=np.uint8)], [7])

    assert prediction.heatmap.dtype == np.float32
    assert [name for name, _ in calls][-2:] == ["tensor.float", "tensor.cpu"]
