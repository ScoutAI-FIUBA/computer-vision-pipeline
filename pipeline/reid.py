"""Person re-identification embeddings.

Two backends, same interface:

  OsnetEmbedder    osnet_x1_0_msmt17 via boxmot — a network trained to tell
                   *people* apart. Needs a GPU to be fast. Default.
  ColourEmbedder   torso colour histogram, CPU-only, no model download.

Why the osnet default: colour histograms were measured on real footage and could
not separate two players in the same kit. Correct matches scored anywhere from
0.63 to 0.95 self-similarity, and the swap detector built on them produced false
positives on a player known to be tracked correctly. Learned re-ID features are
what the problem actually needs; the colour backend is kept as a fallback for
running without a GPU or without boxmot installed.

Both backends return unit-length vectors, so `appearance.similarity` (a dot
product) is cosine similarity either way, and everything downstream — matching,
the swap detector, the tests — is backend-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pipeline import appearance
from pipeline.appearance import ReidConfig

# Weights live on the Modal Volume so they survive container restarts. boxmot
# downloads them here on first use.
DEFAULT_OSNET_WEIGHTS = "osnet_x1_0_msmt17.pt"

# Fraction of a player's box that other players may cover before we decline to
# describe them. An embedding of a mostly-hidden player describes whoever is in
# front, which would poison that track's stored identity. More permissive than
# the colour backend's torso rule because osnet tolerates partial occlusion.
MAX_OCCLUSION_FRACTION = 0.6


class Embedder(Protocol):
    """Turns player boxes in a frame into comparable identity vectors."""

    def embed(self, frame, boxes: list[list[float]]) -> list[list[float] | None]:
        """One vector per box, or None where the box cannot be described."""
        ...


def occlusion_fraction(box: list[float], others: list[list[float]]) -> float:
    """How much of `box` is covered by `others`, accounting for overlap between them."""

    target = tuple(int(value) for value in box)
    total = appearance.area(target)
    if total <= 0:
        return 1.0

    regions = [target]
    for other in others:
        overlap = appearance.intersect(target, tuple(int(v) for v in other))
        if overlap is None:
            continue
        regions = [piece for region in regions for piece in appearance.subtract(region, overlap)]
        if not regions:
            return 1.0

    return 1.0 - (sum(appearance.area(region) for region in regions) / total)


class OsnetEmbedder:
    """Learned person re-ID via boxmot's osnet backbone.

    Loading is separate from embedding so the model is paid for once per
    container (in Modal's `@modal.enter()`), not once per job.
    """

    def __init__(self, weights: str = DEFAULT_OSNET_WEIGHTS, device: str | None = None) -> None:
        import torch
        from boxmot.reid.core.reid import ReID

        # boxmot uses Ultralytics-style device strings: a GPU is "0", not "cuda".
        # Passing "cuda" raises ValueError.
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")
        self._model = ReID(weights=weights, device=self.device, half=False)
        self.name = f"osnet:{Path(weights).name}"

    def embed(self, frame, boxes: list[list[float]]) -> list[list[float] | None]:
        import numpy as np

        if not boxes:
            return []

        # Describe only players who are mostly visible; the rest stay None and
        # fall back to geometry during matching.
        keep = [
            index
            for index, box in enumerate(boxes)
            if occlusion_fraction(box, [b for position, b in enumerate(boxes) if position != index])
            <= MAX_OCCLUSION_FRACTION
        ]
        results: list[list[float] | None] = [None] * len(boxes)
        if not keep:
            return results

        # NOTE: the argument order is (frame, boxes). Reversing it does NOT
        # raise — it silently returns one row per image pixel-row instead of one
        # per box. Verified against boxmot 22.0.0.
        features = self._model(frame, np.asarray([boxes[index] for index in keep], dtype=float))

        features = np.asarray(features, dtype=np.float32)
        norms = np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)
        for position, index in enumerate(keep):
            results[index] = (features[position] / norms[position]).tolist()
        return results


class ColourEmbedder:
    """Torso colour histograms. CPU-only fallback; weak on same-team players."""

    def __init__(self, config: ReidConfig | None = None) -> None:
        self._config = config or ReidConfig()
        self.device = "cpu"
        self.name = "colour-histogram"

    def embed(self, frame, boxes: list[list[float]]) -> list[list[float] | None]:
        return [
            appearance.extract(
                frame,
                box,
                [other for position, other in enumerate(boxes) if position != index],
                self._config,
            )
            for index, box in enumerate(boxes)
        ]


class NullEmbedder:
    """Disables appearance entirely; matching falls back to pure geometry."""

    device = "none"
    name = "none"

    def embed(self, frame, boxes: list[list[float]]) -> list[list[float] | None]:
        return [None] * len(boxes)


def build_embedder(backend: str, weights: str = DEFAULT_OSNET_WEIGHTS) -> Embedder:
    """Construct the requested backend, falling back to colour if osnet is absent."""

    if backend == "none":
        return NullEmbedder()
    if backend == "colour":
        return ColourEmbedder()
    if backend != "osnet":
        raise ValueError(f"Unknown reid_backend: {backend!r} (osnet, colour, none)")

    try:
        return OsnetEmbedder(weights=weights)
    except ImportError:
        # Running somewhere without boxmot or torch, e.g. a laptop. Degrade
        # loudly-but-usably rather than refusing to run at all.
        return ColourEmbedder()
