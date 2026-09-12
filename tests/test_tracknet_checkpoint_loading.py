from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest


def test_checkpoint_loader_uses_weights_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from TrackNetV3 import checkpoint_loading

    recorded: dict[str, object] = {}

    def fake_load(path: Path, *, map_location: str, weights_only: bool) -> dict:
        recorded.update(path=path, map_location=map_location, weights_only=weights_only)
        return {"model": {}}

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=fake_load))
    result = checkpoint_loading.load_checkpoint(tmp_path / "model.pt", map_location="cuda")

    assert result == {"model": {}}
    assert recorded == {"path": tmp_path / "model.pt", "map_location": "cuda", "weights_only": True}


def test_checkpoint_loader_rejects_unsafe_legacy_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from TrackNetV3 import checkpoint_loading

    def legacy_load(path: Path, *, map_location: str) -> dict:
        return {"model": {}}

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=legacy_load))

    with pytest.raises(checkpoint_loading.UnsafeCheckpointError, match="trusted root"):
        checkpoint_loading.load_checkpoint(tmp_path / "model.pt")


def test_runtime_scripts_delegate_checkpoint_loading() -> None:
    root = Path(__file__).resolve().parents[1] / "TrackNetV3"
    for name in ("test.py", "predict.py", "generate_mask_data.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "torch.load(" not in source
        assert "load_checkpoint(" in source
