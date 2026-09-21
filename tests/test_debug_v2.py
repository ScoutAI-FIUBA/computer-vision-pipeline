from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.tracking.debug_v2 import render_debug_video, render_frame


def _write_video(path: Path, count: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for index in range(count):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()


def _write_artifacts(root: Path) -> None:
    (root / "masks").mkdir(parents=True)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:12, 5:13] = 255
    cv2.imwrite(str(root / "masks" / "one.png"), mask)
    rows = [
        {
            "frame_index": 1,
            "raw_track_id": 7,
            "bbox_xyxy": [5, 4, 13, 12],
            "mask_ref": "masks/one.png",
        },
        {"frame_index": 2, "raw_track_id": 8, "bbox_xyxy": [15, 5, 22, 14], "mask_ref": "masks/missing.png"},
    ]
    (root / "tracks.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_render_frame_draws_mask_bbox_and_raw_label(tmp_path: Path, monkeypatch):
    root = tmp_path / "artifacts"
    _write_artifacts(root)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    labels: list[str] = []
    original_put_text = cv2.putText

    def capture_label(image, text, *args, **kwargs):
        labels.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_label)
    rendered = render_frame(frame, [{
        "frame_index": 1,
        "raw_track_id": 7,
        "bbox_xyxy": [5, 4, 13, 12],
        "mask_ref": "masks/one.png",
    }], root)

    assert rendered[7, 7].any()
    assert rendered[4, 5].any()
    assert labels == ["RAW: 7"]
    assert rendered.shape == frame.shape


def test_missing_masks_do_not_drop_detection_or_frames(tmp_path: Path):
    video = tmp_path / "input.mp4"
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "debug.mp4"
    _write_video(video)
    _write_artifacts(artifacts)

    stats = render_debug_video(video, artifacts / "tracks.jsonl", artifacts, output)

    assert stats["frames_written"] == 4
    capture = cv2.VideoCapture(str(output))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
        capture.set(cv2.CAP_PROP_POS_FRAMES, 2)
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok
    assert frame[5, 15].any()


def test_frame_bounds_render_only_inclusive_source_range(tmp_path: Path):
    video = tmp_path / "input.mp4"
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "debug.mp4"
    _write_video(video)
    _write_artifacts(artifacts)

    stats = render_debug_video(video, artifacts / "tracks.jsonl", artifacts, output, start_frame=1, end_frame=2)

    assert stats["frames_written"] == 2
    assert stats["first_frame"] == 1
    assert stats["last_frame"] == 2
