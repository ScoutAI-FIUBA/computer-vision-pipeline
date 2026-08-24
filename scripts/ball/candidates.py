from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np


@dataclass(slots=True)
class BallCandidate:
    x: float
    y: float
    score: float
    heatmap_x: int
    heatmap_y: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "x": self.x,
            "y": self.y,
            "score": self.score,
            "heatmap_x": self.heatmap_x,
            "heatmap_y": self.heatmap_y,
        }


def _as_heatmap_array(heatmap: np.ndarray) -> np.ndarray:
    arr = np.asarray(heatmap, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _local_peak_mask(heatmap: np.ndarray) -> np.ndarray:
    import cv2

    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(heatmap, kernel)
    return heatmap >= dilated


def extract_candidates(
    heatmap: np.ndarray,
    *,
    original_frame_size: tuple[int, int],
    min_score: float,
    min_distance: float,
    max_candidates: int,
) -> list[BallCandidate]:
    if max_candidates < 1:
        return []

    arr = _as_heatmap_array(heatmap)
    peak_mask = _local_peak_mask(arr)
    ys, xs = np.where(peak_mask & (arr >= float(min_score)))
    if len(xs) == 0:
        return []

    frame_width, frame_height = original_frame_size
    heatmap_height, heatmap_width = arr.shape
    ranked = sorted(
        (
            BallCandidate(
                x=((x + 0.5) * frame_width) / heatmap_width,
                y=((y + 0.5) * frame_height) / heatmap_height,
                score=float(arr[y, x]),
                heatmap_x=int(x),
                heatmap_y=int(y),
            )
            for y, x in zip(ys, xs)
        ),
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    kept: list[BallCandidate] = []
    for candidate in ranked:
        if all(hypot(candidate.x - other.x, candidate.y - other.y) >= min_distance for other in kept):
            kept.append(candidate)
        if len(kept) >= max_candidates:
            break
    return kept
