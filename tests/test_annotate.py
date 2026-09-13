from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.annotate import draw_tracks, team_color, track_label, track_label_id  # noqa: E402


def test_team_color_uses_distinct_palette_entries():
    assert team_color(0) != team_color(1)


def test_track_label_id_uses_only_confirmed_number():
    assert track_label_id({"track_id": 7, "confirmed_number": "23"}) == "23"
    assert track_label_id({"track_id": 7}) is None
    assert track_label_id({"track_id": 7, "confirmed_number": "unknown"}) is None


def test_track_label_omits_internal_track_id_when_number_is_missing():
    assert track_label({"track_id": 7, "confirmed_number": "23", "class_name": "player"}) == "#23 player"
    assert track_label({"track_id": 7, "class_name": "player"}) == "player"


def test_draw_tracks_uses_team_colors_when_team_ids_exist():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    annotated = draw_tracks(
        frame,
        [
            {
                "xyxy": [10, 10, 40, 60],
                "confidence": 0.9,
                "track_id": 1,
                "team_id": 0,
                "class_name": "player",
            },
            {
                "xyxy": [70, 10, 100, 60],
                "confidence": 0.8,
                "track_id": 2,
                "team_id": 1,
                "class_name": "player",
            },
        ],
    )

    assert tuple(int(value) for value in annotated[10, 10]) == team_color(0)
    assert tuple(int(value) for value in annotated[10, 70]) == team_color(1)


def test_draw_tracks_skips_lost_rows_without_xyxy():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    annotated = draw_tracks(
        frame,
        [
            {
                "xyxy": [10, 10, 40, 60],
                "confidence": 0.9,
                "track_id": 1,
                "team_id": 0,
                "class_name": "player",
            },
            {
                "xyxy": None,
                "confidence": None,
                "track_id": 2,
                "team_id": 1,
                "class_name": "ball",
                "track_state": "lost",
            },
        ],
    )

    assert tuple(int(value) for value in annotated[10, 10]) == team_color(0)


def test_draw_tracks_supervision_path_skips_lost_rows(monkeypatch):
    captured = {}

    class FakeDetections:
        def __init__(self, *, xyxy, **_kwargs):
            self.xyxy = xyxy

        def __len__(self):
            return len(self.xyxy)

    class FakeBoxAnnotator:
        def __init__(self, **_kwargs):
            pass

        def annotate(self, *, scene, detections):
            captured["detections"] = detections
            return scene

    class FakeLabelAnnotator(FakeBoxAnnotator):
        def annotate(self, *, scene, detections, labels):
            captured["labels"] = labels
            return scene

    monkeypatch.setitem(
        sys.modules,
        "supervision",
        SimpleNamespace(
            Detections=FakeDetections,
            ColorLookup=SimpleNamespace(TRACK=object()),
            BoxAnnotator=FakeBoxAnnotator,
            LabelAnnotator=FakeLabelAnnotator,
        ),
    )

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    annotated = draw_tracks(
        frame,
        [
            {
                "xyxy": [10, 10, 40, 60],
                "confidence": 0.9,
                "track_id": 1,
                "class_name": "player",
                "track_state": "detected",
            },
            {
                "xyxy": None,
                "confidence": None,
                "track_id": 2,
                "class_name": "ball",
                "track_state": "lost",
            },
        ],
    )

    assert annotated is frame
    assert captured["detections"].xyxy.tolist() == [[10.0, 10.0, 40.0, 60.0]]
    assert captured["labels"] == ["player 0.90"]
