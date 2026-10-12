"""Regression coverage for projected shot export."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.projected_shots import build_projected_shots, write_projected_shots_image  # noqa: E402


def test_projected_shot_artifacts_are_removed_without_projected_rows(tmp_path: Path):
    from pipeline.runner import _maybe_write_projected_shots

    for projected_tracks in (None, ""):
        if projected_tracks is not None:
            (tmp_path / "projected_tracks.jsonl").write_text(projected_tracks, encoding="utf-8")
        for filename in ("projected_shots.json", "projected_shots.png"):
            (tmp_path / filename).write_text("stale", encoding="utf-8")

        assert _maybe_write_projected_shots(tmp_path) == (None, None)
        assert not (tmp_path / "projected_shots.json").exists()
        assert not (tmp_path / "projected_shots.png").exists()
        (tmp_path / "projected_tracks.jsonl").unlink(missing_ok=True)


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
                "stable_player_id": "player-23",
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
    assert shot["shooter_stable_player_id"] == "player-23"
    assert shot["shooter_team_id"] == 1
    assert shot["shooter_confirmed_number"] == "11"
    assert shot["shot_origin_court_xy_m"] == [5.5, 9.2]
    assert shot["shot_origin_projection_method"] == "shooter_foot"
    assert shot["shooter_attribution_method"] == "nearest_player_bbox_to_shot_anchor"
    assert artifact["heuristics"]["team_number_enrichment"].startswith("compatibility-only")
    assert [row["track_id"] for row in shot["projected_player_snapshot"]] == [8, 23, 30, 31, 32]
    assert shot["projected_player_snapshot"][1]["stable_player_id"] == "player-23"


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
    assert shot["shooter_track_id"] is None
    assert shot["shooter_stable_player_id"] is None
    assert shot["shooter_team_id"] is None
    assert shot["shooter_confirmed_number"] is None
    assert shot["shot_origin_court_xy_m"] is None
    assert shot["shot_origin_projection_method"] is None
    assert shot["shooter_attribution_method"] == "no_stable_player_id"
    assert shot["projected_player_snapshot"][0]["track_id"] == 3


def test_build_projected_shots_does_not_attribute_raw_only_row():
    artifact = build_projected_shots(
        [{"event_id": 8, "start_frame_index": 1, "start_image_xy": [10.0, 10.0]}],
        [{
            "frame_index": 1,
            "track_id": 9,
            "xyxy": [0, 0, 20, 20],
            "image_foot_xy": [10, 20],
            "court_xy_m": [2.0, 3.0],
        }],
        team_ids_by_track_id={9: 2},
        confirmed_numbers_by_track_id={9: "7"},
    )

    shot = artifact["shots"][0]
    assert shot["shooter_stable_player_id"] is None
    assert shot["shooter_track_id"] is None
    assert shot["shooter_team_id"] is None
    assert shot["shooter_confirmed_number"] is None
    assert shot["shooter_attribution_method"] == "no_stable_player_id"
    assert shot["projected_player_snapshot"] == [{
        "track_id": 9,
        "stable_player_id": None,
        "team_id": 2,
        "confirmed_number": "7",
        "track_state": None,
        "image_foot_xy": [10.0, 20.0],
        "court_xy_m": [2.0, 3.0],
        "xyxy": [0.0, 0.0, 20.0, 20.0],
    }]


def test_build_projected_shots_copies_selected_track_stable_player_id_for_overlapping_players():
    artifact = build_projected_shots(
        [{"event_id": 9, "start_frame_index": 1, "start_image_xy": [10.0, 10.0]}],
        [
            {
                "frame_index": 1,
                "track_id": 3,
                "stable_player_id": "player-007",
                "xyxy": [0, 0, 20, 20],
                "image_foot_xy": [10, 20],
                "court_xy_m": [2.0, 3.0],
            },
            {
                "frame_index": 1,
                "track_id": 5,
                "stable_player_id": "player-008",
                "xyxy": [5, 5, 25, 25],
                "image_foot_xy": [15, 25],
                "court_xy_m": [4.0, 5.0],
            },
            {
                "frame_index": 1,
                "track_id": 9,
                "xyxy": [0, 0, 20, 20],
                "image_foot_xy": [10, 20],
                "court_xy_m": [6.0, 7.0],
            },
        ],
    )

    shot = artifact["shots"][0]
    assert shot["shooter_track_id"] == 3
    assert shot["shooter_stable_player_id"] == "player-007"
    assert shot["shot_origin_court_xy_m"] == [2.0, 3.0]
    assert shot["shot_origin_projection_method"] == "shooter_foot"
    assert shot["shooter_attribution_method"] == "nearest_player_bbox_to_shot_anchor"
    assert [(row["track_id"], row["stable_player_id"]) for row in shot["projected_player_snapshot"]] == [
        (3, "player-007"),
        (5, "player-008"),
        (9, None),
    ]


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
