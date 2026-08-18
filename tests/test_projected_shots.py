"""Regression coverage for projected shot export."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.projected_shots import build_projected_shots, write_projected_shots_image  # noqa: E402


def test_build_projected_shots_matches_shot_anchor_to_projected_player_bbox():
    artifact = build_projected_shots(
        [
            {
                "event_id": 1,
                "shot_type": "jump_shot",
                "outcome": "made",
                "start_frame_index": 100,
                "start_timestamp_seconds": 4.0,
                "start_image_xy": [50.0, 80.0],
            }
        ],
        [
            {
                "frame_index": 100,
                "track_id": 8,
                "class_name": "player",
                "track_state": "propagated",
                "xyxy": [0, 0, 20, 20],
                "image_foot_xy": [0, 0],
                "court_xy_m": [0.0, 0.0],
            },
            {
                "frame_index": 100,
                "track_id": 23,
                "class_name": "player",
                "track_state": "detected",
                "xyxy": [42, 18, 68, 92],
                "image_foot_xy": [55, 92],
                "court_xy_m": [5.5, 9.2],
            },
            {
                "frame_index": 100,
                "track_id": 30,
                "class_name": "player",
                "track_state": "detected",
                "xyxy": [95, 0, 105, 20],
                "image_foot_xy": [100, 0],
                "court_xy_m": [10.0, 0.0],
            },
            {
                "frame_index": 100,
                "track_id": 31,
                "class_name": "player",
                "track_state": "detected",
                "xyxy": [0, 95, 10, 105],
                "image_foot_xy": [0, 100],
                "court_xy_m": [0.0, 10.0],
            },
            {
                "frame_index": 100,
                "track_id": 32,
                "class_name": "player",
                "track_state": "detected",
                "xyxy": [95, 95, 105, 105],
                "image_foot_xy": [100, 100],
                "court_xy_m": [10.0, 10.0],
            },
        ],
        team_ids_by_track_id={23: 1},
        confirmed_numbers_by_track_id={23: "11"},
    )

    shot = artifact["shots"][0]
    assert artifact["heuristics"]["frame_matching"] == "nearest projected frame by minimum temporal distance"
    assert shot["matched_projected_frame_index"] == 100
    assert shot["matched_projected_frame_gap"] == 0
    assert shot["shooter_track_id"] == 23
    assert shot["shooter_team_id"] == 1
    assert shot["shooter_confirmed_number"] == "11"
    assert shot["shot_origin_court_xy_m"] == [5.5, 9.2]
    assert shot["shot_origin_projection_method"] == "shooter_foot"
    assert shot["shooter_attribution_method"] == "nearest_player_bbox_to_shot_anchor"
    assert [row["track_id"] for row in shot["projected_player_snapshot"]] == [8, 23, 30, 31, 32]


def test_build_projected_shots_matches_nearest_frame_even_outside_max_gap():
    artifact = build_projected_shots(
        [
            {
                "event_id": 7,
                "shot_type": "jump_shot",
                "outcome": "missed",
                "start_frame_index": 12,
                "start_timestamp_seconds": 0.48,
                "start_image_xy": [16.0, 24.0],
            }
        ],
        [
            {
                "frame_index": 0,
                "track_id": 3,
                "class_name": "player",
                "track_state": "detected",
                "xyxy": [0, 0, 20, 30],
                "image_foot_xy": [10, 30],
                "court_xy_m": [1.0, 2.0],
            },
            {
                "frame_index": 40,
                "track_id": 4,
                "class_name": "player",
                "track_state": "detected",
                "xyxy": [8, 10, 24, 34],
                "image_foot_xy": [16, 34],
                "court_xy_m": [3.0, 4.0],
            },
        ],
        max_frame_gap=6,
    )

    shot = artifact["shots"][0]
    assert shot["matched_projected_frame_index"] == 0
    assert shot["matched_projected_frame_gap"] == 12
    assert shot["shooter_track_id"] == 3
    assert shot["shot_origin_court_xy_m"] == [1.0, 2.0]
    assert shot["shot_origin_projection_method"] == "shooter_foot"


def test_write_projected_shots_image_writes_png_with_plotted_marks(tmp_path: Path):
    artifact = {
        "shots": [
            {"event_id": 1, "outcome": "made", "shot_origin_court_xy_m": [7.5, 3.25]},
            {"event_id": 2, "outcome": "missed", "shot_origin_court_xy_m": [20.0, 10.0]},
        ]
    }

    output_path = tmp_path / "projected_shots.png"
    plotted = write_projected_shots_image(artifact, output_path, pixels_per_meter=20, point_radius=5)

    image = cv2.imread(str(output_path))
    assert plotted == 2
    assert output_path.exists()
    assert image is not None
    assert image.shape[:2] == (300, 560)
