#!/usr/bin/env python3
"""Render SAM3.1 raw tracking artifacts over the original local video."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def load_observations(tracks_jsonl: Path) -> dict[int, list[dict[str, Any]]]:
    """Group JSONL observations by source frame index."""
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with tracks_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                by_frame[int(row["frame_index"])].append(row)
    return dict(by_frame)


def color_for_track(raw_track_id: int) -> tuple[int, int, int]:
    """Return a stable, visible BGR color for a raw SAM track ID."""
    digest = hashlib.sha256(str(int(raw_track_id)).encode()).digest()
    return tuple(80 + byte % 176 for byte in digest[:3])


def _mask_path(mask_ref: Any, masks_dir: Path) -> Path | None:
    if not isinstance(mask_ref, str) or not mask_ref:
        return None
    root = masks_dir.resolve()
    references = [Path(mask_ref)]
    if references[0].parts[:1] == ("masks",):
        references.append(Path(*references[0].parts[1:]))
    for reference in references:
        candidate = (root / reference).resolve()
        if candidate != root and root in candidate.parents:
            return candidate
    return None


def _load_mask(row: dict[str, Any], masks_dir: Path, shape: tuple[int, int]) -> np.ndarray | None:
    path = _mask_path(row.get("mask_ref"), masks_dir)
    if path is None or not path.is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2 or mask.shape != shape:
        return None
    return mask > 0


def validate_artifacts(tracks_jsonl: Path, masks_dir: Path, *, allow_incomplete: bool = False) -> None:
    """Reject unavailable or explicitly incomplete returned packages."""
    if not tracks_jsonl.is_file():
        raise ValueError(f"required artifact is missing: {tracks_jsonl}")
    if not masks_dir.is_dir():
        raise ValueError(f"required artifact directory is missing: {masks_dir}")

    manifest_path = tracks_jsonl.parent / "manifest.json"
    if not manifest_path.is_file():
        if allow_incomplete:
            return
        raise ValueError(f"artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read artifact manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("invalid artifact manifest structure: expected an object")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("invalid artifact manifest structure: artifacts must be an object")
    if not isinstance(artifacts.get("tracks"), dict) or not isinstance(artifacts.get("masks"), dict):
        raise ValueError("invalid artifact manifest structure: tracks and masks are required")

    reasons: list[str] = []
    if manifest.get("complete") is False:
        reasons.append("incomplete")
    if manifest.get("truncated") is True:
        reasons.append("truncated")
    if manifest.get("transfer_errors"):
        reasons.append("transfer errors")
    missing = artifacts["masks"].get("missing", [])
    if not isinstance(missing, list):
        raise ValueError("invalid artifact manifest structure: masks.missing must be a list")
    if missing:
        reasons.append("missing required artifacts")
    if reasons and not allow_incomplete:
        detail = ", ".join(reasons)
        raise ValueError(f"artifact package is {detail}; pass --allow-incomplete to override")


def render_frame(frame: np.ndarray, rows: Iterable[dict[str, Any]], masks_dir: Path) -> np.ndarray:
    """Draw masks, boxes, and raw IDs without mutating the source frame."""
    rendered = frame.copy()
    for row in rows:
        try:
            raw_id = int(row["raw_track_id"])
            box = [int(round(float(value))) for value in row["bbox_xyxy"]]
            if len(box) != 4:
                continue
        except (KeyError, TypeError, ValueError):
            continue

        color = color_for_track(raw_id)
        mask = _load_mask(row, masks_dir, rendered.shape[:2])
        if mask is not None:
            overlay = rendered.copy()
            overlay[mask] = color
            rendered = cv2.addWeighted(overlay, 0.35, rendered, 0.65, 0)

        x1, y1, x2, y2 = box
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            rendered,
            f"RAW: {raw_id}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return rendered


def render_debug_video(
    video_path: Path,
    tracks_jsonl: Path,
    masks_dir: Path,
    output_path: Path,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    allow_incomplete: bool = False,
) -> dict[str, int | None]:
    """Render the selected inclusive source-frame range to an MP4."""
    if start_frame is not None and start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame is not None and end_frame < 0:
        raise ValueError("end_frame must be non-negative")
    if start_frame is not None and end_frame is not None and end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")

    validate_artifacts(tracks_jsonl, masks_dir, allow_incomplete=allow_incomplete)
    observations = load_observations(tracks_jsonl)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    first_written: int | None = None
    last_written: int | None = None
    frames_written = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if start_frame is not None and frame_index < start_frame:
                frame_index += 1
                continue
            if end_frame is not None and frame_index > end_frame:
                break
            writer.write(render_frame(frame, observations.get(frame_index, ()), masks_dir))
            first_written = frame_index if first_written is None else first_written
            last_written = frame_index
            frames_written += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    return {"frames_written": frames_written, "first_frame": first_written, "last_frame": last_written}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SAM3.1 raw tracks and masks locally.")
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--tracks-jsonl", type=Path)
    parser.add_argument("--masks-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path, help="Directory containing tracks.jsonl and masks/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Render despite an incomplete/truncated manifest (unsafe; default rejects it)",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    tracks_jsonl = args.tracks_jsonl or (artifact_dir / "tracks.jsonl" if artifact_dir else None)
    masks_dir = args.masks_dir or (artifact_dir / "masks" if artifact_dir else None)
    if tracks_jsonl is None or masks_dir is None:
        parser.error("provide --artifact-dir or both --tracks-jsonl and --masks-dir")
    stats = render_debug_video(
        args.video_path,
        tracks_jsonl,
        masks_dir,
        args.output,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        allow_incomplete=args.allow_incomplete,
    )
    print(f"Wrote {stats['frames_written']} frames to {args.output}")


if __name__ == "__main__":
    main()
