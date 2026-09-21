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


def test_render_frame_scales_normalized_bbox_before_drawing_mask_and_label(tmp_path: Path):
    root = tmp_path / "artifacts"
    _write_artifacts(root)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    rendered = render_frame(frame, [{
        "frame_index": 1,
        "raw_track_id": 7,
        "bbox_xyxy": [0.25, 0.25, 0.5, 0.75],
        "mask_ref": "masks/one.png",
    }], root)

    color = np.array(debug_v2.color_for_track(7), dtype=np.uint8)
    assert np.array_equal(rendered[6, 8], color)
    assert rendered[7, 7].any()


def test_render_frame_scales_all_bbox_corners_and_full_raw_label(tmp_path: Path):
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    root = tmp_path / "artifacts"
    root.mkdir()
    color = debug_v2.color_for_track(7)
    row = {"raw_track_id": 7, "bbox_xyxy": [0.2, 0.25, 0.6, 0.75]}

    rendered = render_frame(frame, [row], root)

    expected = frame.copy()
    cv2.rectangle(expected, (20, 20), (60, 60), color, 1)
    cv2.putText(expected, "RAW: 7", (20, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    assert np.array_equal(rendered[5:21, 20:70], expected[5:21, 20:70])
    for x, y in ((20, 20), (60, 20), (20, 60), (60, 60)):
        assert np.array_equal(rendered[y, x], color)


@pytest.mark.parametrize(("index", "value"), [(0, float("inf")), (1, float("nan")), (2, float("inf")), (3, float("nan"))])
def test_render_frame_skips_non_finite_bbox_coordinates(tmp_path: Path, index: int, value: float):
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    root = tmp_path / "artifacts"
    root.mkdir()
    bbox = [0.25, 0.25, 0.5, 0.75]
    bbox[index] = value

    rendered = render_frame(frame, [{"raw_track_id": 7, "bbox_xyxy": bbox}], root)

    assert np.array_equal(rendered, frame)


@pytest.mark.parametrize("index", range(4))
def test_render_frame_skips_extreme_finite_bbox_coordinates(tmp_path: Path, index: int):
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    root = tmp_path / "artifacts"
    root.mkdir()
    bbox = [0.25, 0.25, 0.5, 0.75]
    bbox[index] = 1e308

    rendered = render_frame(frame, [{"raw_track_id": 7, "bbox_xyxy": bbox}], root)

    assert np.array_equal(rendered, frame)


@pytest.mark.parametrize("index", range(4))
def test_render_frame_skips_huge_integer_bbox_coordinates(tmp_path: Path, index: int):
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    root = tmp_path / "artifacts"
    root.mkdir()
    bbox = [0, 0, 10, 10]
    bbox[index] = 10**1000

    rendered = render_frame(frame, [{"raw_track_id": 7, "bbox_xyxy": bbox}], root)

    assert np.array_equal(rendered, frame)


def test_render_frame_treats_exact_normalized_boundaries_as_normalized(tmp_path: Path):
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    root = tmp_path / "artifacts"
    root.mkdir()
    color = debug_v2.color_for_track(7)

    rendered = render_frame(frame, [{"raw_track_id": 7, "bbox_xyxy": [0, 0, 1, 1]}], root)

    expected = frame.copy()
    cv2.rectangle(expected, (0, 0), (60, 40), color, 1)
    cv2.putText(expected, "RAW: 7", (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    assert np.array_equal(rendered, expected)


def test_render_frame_treats_mixed_coordinates_as_pixel_style(tmp_path: Path):
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    root = tmp_path / "artifacts"
    root.mkdir()
    color = debug_v2.color_for_track(7)

    rendered = render_frame(frame, [{"raw_track_id": 7, "bbox_xyxy": [5, 6, 20, 18]}], root)

    expected = frame.copy()
    cv2.rectangle(expected, (5, 6), (20, 18), color, 1)
    cv2.putText(expected, "RAW: 7", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    assert np.array_equal(rendered, expected)


def test_missing_masks_do_not_drop_detection_or_frames(tmp_path: Path):
    video = tmp_path / "input.mp4"
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "debug.mp4"
    _write_video(video)
    _write_artifacts(artifacts)

    frame_for_missing_mask = np.zeros((24, 80, 3), dtype=np.uint8)
    rendered = render_frame(frame_for_missing_mask, [{
        "raw_track_id": 8,
        "bbox_xyxy": [15, 5, 22, 14],
        "mask_ref": "masks/missing.png",
    }], artifacts)
    expected = frame_for_missing_mask.copy()
    color = debug_v2.color_for_track(8)
    cv2.rectangle(expected, (15, 5), (22, 14), color, 1)
    cv2.putText(expected, "RAW: 8", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    assert np.array_equal(rendered[0:15, 15:75], expected[0:15, 15:75])

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
