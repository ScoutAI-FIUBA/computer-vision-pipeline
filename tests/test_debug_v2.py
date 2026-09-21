from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.tracking.debug_v2 import render_debug_video, render_frame


def _write_video(path: Path, count: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    colors = [(20, 30, 200), (30, 200, 40), (200, 40, 30), (180, 160, 40)]
    for index in range(count):
        writer.write(np.full((24, 32, 3), colors[index % len(colors)], dtype=np.uint8))
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
            "bbox_xyxy": [5, 18, 13, 23],
            "mask_ref": "masks/one.png",
        },
        {"frame_index": 2, "raw_track_id": 8, "bbox_xyxy": [15, 5, 22, 14], "mask_ref": "masks/missing.png"},
    ]
    (root / "tracks.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (root / "manifest.json").write_text(json.dumps({
        "complete": True,
        "truncated": False,
        "transfer_errors": [],
        "artifacts": {"tracks": {"key": "tracks.jsonl"}, "masks": {"key": "masks/", "missing": []}},
    }))


def test_render_frame_draws_mask_bbox_and_raw_label(tmp_path: Path):
    root = tmp_path / "artifacts"
    _write_artifacts(root)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    rendered = render_frame(frame, [{
        "frame_index": 1,
        "raw_track_id": 7,
        "bbox_xyxy": [5, 18, 13, 23],
        "mask_ref": "masks/one.png",
    }], root)

    assert rendered[7, 7].any()
    assert rendered[18, 5].any()
    assert rendered[12:15, 5:30].any(axis=2).any()
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
    assert not np.array_equal(frame[0, 0], np.array([30, 200, 40], dtype=np.uint8))


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
    capture = cv2.VideoCapture(str(output))
    try:
        ok, first = capture.read()
        assert ok
        ok, second = capture.read()
        assert ok
    finally:
        capture.release()
    assert np.linalg.norm(first[0, 0].astype(float) - np.array([30, 200, 40])) < 25
    assert np.linalg.norm(second[0, 0].astype(float) - np.array([200, 40, 30])) < 25


def test_incomplete_manifest_is_rejected_by_default(tmp_path: Path):
    video = tmp_path / "input.mp4"
    artifacts = tmp_path / "artifacts"
    _write_video(video)
    _write_artifacts(artifacts)
    manifest = json.loads((artifacts / "manifest.json").read_text())
    manifest["complete"] = False
    manifest["truncated"] = True
    manifest["transfer_errors"] = ["missing artifact"]
    (artifacts / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="incomplete|truncated|transfer"):
        render_debug_video(video, artifacts / "tracks.jsonl", artifacts, tmp_path / "debug.mp4")


def test_incomplete_manifest_can_be_rendered_explicitly(tmp_path: Path):
    video = tmp_path / "input.mp4"
    artifacts = tmp_path / "artifacts"
    _write_video(video)
    _write_artifacts(artifacts)
    manifest = json.loads((artifacts / "manifest.json").read_text())
    manifest["complete"] = False
    (artifacts / "manifest.json").write_text(json.dumps(manifest))

    stats = render_debug_video(
        video, artifacts / "tracks.jsonl", artifacts, tmp_path / "debug.mp4", allow_incomplete=True
    )

    assert stats["frames_written"] == 4


def test_manifest_missing_required_artifacts_is_rejected(tmp_path: Path):
    video = tmp_path / "input.mp4"
    artifacts = tmp_path / "artifacts"
    _write_video(video)
    _write_artifacts(artifacts)
    manifest = json.loads((artifacts / "manifest.json").read_text())
    manifest["artifacts"]["masks"]["missing"] = ["1:7"]
    (artifacts / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing required artifacts"):
        render_debug_video(video, artifacts / "tracks.jsonl", artifacts, tmp_path / "debug.mp4")
