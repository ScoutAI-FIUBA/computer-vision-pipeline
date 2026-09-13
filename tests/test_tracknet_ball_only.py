from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ball import tracknet_ball_only as script


def test_edge_padded_windows_and_lost_record():
    assert script.padded_window([10, 12, 14], 0, 4) == (10, 10, 10, 10)
    assert script.padded_window([10, 12, 14], 1, 4) == (10, 10, 10, 12)
    record = script.frame_record(0, 10, 30, (10, 10), [], 24, 16)
    assert record["track_state"] == "lost"
    assert record["selected_center_xy"] is None


def test_candidate_coordinate_mapping_and_box():
    candidate = script.BallCandidate(3.5, 4.5, 0.9, 1, 1)
    record = script.frame_record(2, 8, 30, (4, 8), [candidate], 20, 16)
    assert record["source_frame_index"] == 8
    assert record["selected_center_xy"] == [3.5, 4.5]
    assert record["selected_box_xyxy"] == [0.0, 0.0, 11.5, 12.5]


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (24, 16))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((16, 24, 3), dtype=np.uint8))
    writer.release()


def test_cli_writes_ball_only_json_and_video_without_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video = tmp_path / "clip.mp4"
    model = tmp_path / "model.pt"
    _video(video)
    model.write_bytes(b"fake")

    class FakeDetector:
        context_size = 2
        backend_name = "fake"

        def predict(self, _frames, frame_indices):
            assert len(frame_indices) == 2
            return type("Prediction", (), {"heatmap": np.zeros((8, 12), dtype=np.float32)})()

    monkeypatch.setattr(script, "inspect_tracknet_checkpoint", lambda _path: {"seq_len": 2})
    monkeypatch.setattr(script, "build_detector", lambda *args, **kwargs: FakeDetector())
    output = tmp_path / "out"
    script.main([str(video), "--model", str(model), "--output-dir", str(output)])

    rows = [json.loads(line) for line in (output / "ball_tracks.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["track_state"] == "lost"
    assert rows[0]["tracknet_window_source_frames"] == [0, 0]
    assert "player" not in rows[0]
    assert (output / "ball_only.mp4").stat().st_size > 0


def test_module_has_no_forbidden_detector_dependencies():
    source = Path(script.__file__).read_text(encoding="utf-8")
    assert "roboflow" not in source.lower()
    assert "sam2" not in source.lower()
    assert "modal" not in source.lower()
    assert "import torch" not in source
