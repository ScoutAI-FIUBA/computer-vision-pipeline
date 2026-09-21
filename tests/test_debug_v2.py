from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.tracking import debug_v2
from scripts.tracking.debug_v2 import render_debug_video, render_frame, validate_artifacts


def _write_video(path: Path, count: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    colors = [(20, 30, 200), (30, 200, 40), (200, 40, 30), (180, 160, 40)]
    for index in range(count):
        writer.write(np.full((24, 32, 3), colors[index % len(colors)], dtype=np.uint8))
    writer.release()


def _write_artifacts(root: Path) -> None:
    (root / "masks").mkdir(parents=True, exist_ok=True)
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


def test_manifest_is_required_unless_incomplete_override_is_explicit(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_artifacts(artifacts)
    (artifacts / "manifest.json").unlink()

    with pytest.raises(ValueError, match="manifest"):
        validate_artifacts(artifacts / "tracks.jsonl", artifacts)
    validate_artifacts(artifacts / "tracks.jsonl", artifacts, allow_incomplete=True)


def test_missing_required_files_have_controlled_errors(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_artifacts(artifacts)

    (artifacts / "tracks.jsonl").unlink()
    with pytest.raises(ValueError, match="tracks.jsonl"):
        validate_artifacts(artifacts / "tracks.jsonl", artifacts)

    _write_artifacts(artifacts)
    for path in (artifacts / "masks").iterdir():
        path.unlink()
    (artifacts / "masks").rmdir()
    with pytest.raises(ValueError, match="directory"):
        validate_artifacts(artifacts / "tracks.jsonl", artifacts / "masks")


def test_malformed_manifest_has_a_controlled_error(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_artifacts(artifacts)
    (artifacts / "manifest.json").write_text("[]")

    with pytest.raises(ValueError, match="manifest structure"):
        validate_artifacts(artifacts / "tracks.jsonl", artifacts)


def test_valid_empty_package_renders_empty_frames(tmp_path: Path):
    video = tmp_path / "input.mp4"
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "debug.mp4"
    _write_video(video, count=2)
    (artifacts / "masks").mkdir(parents=True)
    (artifacts / "tracks.jsonl").write_text("")
    (artifacts / "manifest.json").write_text(json.dumps({
        "complete": True,
        "truncated": False,
        "transfer_errors": [],
        "artifacts": {"tracks": {"key": "tracks.jsonl"}, "masks": {"key": "masks/", "missing": []}},
    }))

    validate_artifacts(artifacts / "tracks.jsonl", artifacts)
    stats = render_debug_video(video, artifacts / "tracks.jsonl", artifacts, output)

    assert stats["frames_written"] == 2


@pytest.mark.parametrize("field", ["complete", "truncated"])
@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_manifest_status_fields_require_real_booleans(tmp_path: Path, field, value):
    artifacts = tmp_path / "artifacts"
    _write_artifacts(artifacts)
    manifest = json.loads((artifacts / "manifest.json").read_text())
    manifest[field] = value
    (artifacts / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=f"{field} must be a boolean"):
        validate_artifacts(artifacts / "tracks.jsonl", artifacts)


def test_malformed_nested_manifest_fields_have_controlled_errors(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_artifacts(artifacts)
    manifest = json.loads((artifacts / "manifest.json").read_text())
    manifest["transfer_errors"] = "timed out"
    (artifacts / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="transfer_errors must be a list"):
        validate_artifacts(artifacts / "tracks.jsonl", artifacts)


def test_cli_passes_allow_incomplete_to_render_flow(tmp_path: Path, monkeypatch):
    captured: dict = {}

    def fake_render(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"frames_written": 0}

    monkeypatch.setattr(debug_v2, "render_debug_video", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debug_v2.py",
            "--video-path",
            str(tmp_path / "input.mp4"),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--output",
            str(tmp_path / "debug.mp4"),
            "--allow-incomplete",
        ],
    )

    debug_v2.main()

    assert captured["kwargs"]["allow_incomplete"] is True
