from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.annotate import draw_tracks, team_color  # noqa: E402


def test_team_color_uses_distinct_palette_entries():
    assert team_color(0) != team_color(1)


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
