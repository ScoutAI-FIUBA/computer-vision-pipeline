"""Re-ID backend tests.

Cover the parts that do not need a GPU or model download: occlusion accounting,
backend selection, and the fallback path. The osnet backend itself is exercised
on Modal.

    python -m pytest tests/test_reid_backends.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.reid import (  # noqa: E402
    ColourEmbedder,
    NullEmbedder,
    OsnetEmbedder,
    build_embedder,
    occlusion_fraction,
)


def test_unoccluded_box_reports_zero():
    assert occlusion_fraction([100, 100, 200, 300], []) == pytest.approx(0.0)


def test_fully_covered_box_reports_one():
    assert occlusion_fraction([100, 100, 200, 300], [[0, 0, 1280, 720]]) == pytest.approx(1.0)


def test_half_covered_box_reports_half():
    # Cover the top half of a 100x200 box.
    assert occlusion_fraction([100, 100, 200, 300], [[100, 100, 200, 200]]) == pytest.approx(0.5)


def test_overlapping_occluders_are_not_double_counted():
    """Two occluders covering the same strip must not sum past reality."""

    fraction = occlusion_fraction(
        [100, 100, 200, 300], [[100, 100, 200, 200], [100, 150, 200, 200]]
    )
    assert fraction == pytest.approx(0.5)


def test_degenerate_box_is_treated_as_fully_occluded():
    assert occlusion_fraction([100, 100, 100, 100], []) == pytest.approx(1.0)


def test_null_embedder_returns_one_none_per_box():
    embedder = NullEmbedder()
    assert embedder.embed(None, [[0, 0, 10, 10], [5, 5, 20, 20]]) == [None, None]


def test_colour_embedder_returns_one_entry_per_box():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 200)
    signatures = ColourEmbedder().embed(frame, [[100, 50, 200, 250], [10, 10, 15, 15]])
    assert len(signatures) == 2
    assert signatures[0] is not None
    assert signatures[1] is None, "a tiny box cannot be described"


def test_embed_handles_no_boxes():
    assert ColourEmbedder().embed(np.zeros((10, 10, 3), dtype=np.uint8), []) == []


def test_build_embedder_selects_backends():
    assert isinstance(build_embedder("none"), NullEmbedder)
    assert isinstance(build_embedder("colour"), ColourEmbedder)


def test_build_embedder_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown reid_backend"):
        build_embedder("magic")


def test_build_embedder_falls_back_when_boxmot_is_absent(monkeypatch):
    """A laptop without boxmot should still run, just with weaker features."""

    import builtins

    real_import = builtins.__import__

    def fail_on_boxmot(name, *args, **kwargs):
        if name.startswith("boxmot"):
            raise ImportError("no boxmot here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_on_boxmot)
    assert isinstance(build_embedder("osnet"), ColourEmbedder)


def test_osnet_embed_converts_bfloat16_features_before_numpy():
    torch = pytest.importorskip("torch")
    embedder = object.__new__(OsnetEmbedder)
    embedder._model = lambda frame, boxes: torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)

    assert embedder.embed(np.zeros((2, 2, 3), dtype=np.uint8), [[0, 0, 2, 2]]) == [
        pytest.approx([0.6, 0.8])
    ]


def test_osnet_embed_disables_cuda_autocast_at_boxmot_boundary(monkeypatch):
    torch = pytest.importorskip("torch")
    embedder = object.__new__(OsnetEmbedder)
    seen = []

    def model(frame, boxes):
        seen.append(torch.is_autocast_enabled("cuda"))
        return torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)

    embedder._model = model
    embedder.embed(np.zeros((2, 2, 3), dtype=np.uint8), [[0, 0, 2, 2]])

    assert seen == [False]
