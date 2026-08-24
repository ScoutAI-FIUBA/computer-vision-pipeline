from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.ball.candidates import BallCandidate


def _normalized_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    arr = np.asarray(heatmap, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        arr = np.zeros((size[1], size[0]), dtype=np.float32)
    resized = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)
    low = float(resized.min())
    high = float(resized.max())
    if high > low:
        resized = (resized - low) / (high - low)
    else:
        resized = np.zeros_like(resized)
    return (resized * 255.0).astype(np.uint8)


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, *, alpha: float = 0.35) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    color = cv2.applyColorMap(_normalized_heatmap(heatmap, (width, height)), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0.0)


def annotate_candidates(frame: np.ndarray, candidates: Sequence[BallCandidate], *, frame_index: int | None = None) -> np.ndarray:
    import cv2

    annotated = frame.copy()
    for rank, candidate in enumerate(candidates, start=1):
        center = (int(round(candidate.x)), int(round(candidate.y)))
        cv2.circle(annotated, center, 10, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"#{rank} {candidate.score:.2f}",
            (center[0] + 8, max(20, center[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if frame_index is not None:
        cv2.putText(annotated, f"frame={frame_index}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def save_debug_frame(path: Path, frame: np.ndarray, heatmap: np.ndarray, candidates: Sequence[BallCandidate], *, frame_index: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    annotated = annotate_candidates(overlay_heatmap(frame, heatmap), candidates, frame_index=frame_index)
    if not cv2.imwrite(str(path), annotated):
        raise RuntimeError(f"Failed to write debug frame: {path}")


def append_candidates_jsonl(
    path: Path,
    *,
    frame_index: int,
    candidates: Sequence[BallCandidate],
    backend_name: str,
    frame_indices: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "frame_index": frame_index,
        "buffer_frame_indices": list(frame_indices),
        "backend": backend_name,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
