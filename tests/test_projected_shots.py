"""Regression coverage for projected shot export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import pytest

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


def _player(track_id: int, stable_id: str, xyxy: list[float], court_xy: list[float], frame_index: int) -> dict:
    return {
        "frame_index": frame_index,
        "track_id": track_id,
        "stable_player_id": stable_id,
        "class_name": "player",
        "track_state": "tracked",
        "xyxy": xyxy,
        "image_foot_xy": [(xyxy[0] + xyxy[2]) / 2.0, xyxy[3]],
        "court_xy_m": court_xy,
    }


def _made_event() -> dict:
    return {
        "event_id": 1,
        "shot_type": "jump_shot",
        "outcome": "made",
        "start_frame_index": 20,
        "start_timestamp_seconds": 0.3,
        "start_image_xy": [20.0, 520.0],  # an edge false positive near the cut-off player
        "ball_in_basket_frame_index": 20,
    }


def _pred(class_id: int, class_name: str, xyxy: list[float], confidence: float = 0.8) -> dict:
    x1, y1, x2, y2 = xyxy
    return {
        "class_id": class_id,
        "class": class_name,
        "confidence": confidence,
        "x": (x1 + x2) / 2.0,
        "y": (y1 + y2) / 2.0,
        "width": x2 - x1,
        "height": y2 - y1,
    }


def _ball_pred(x: float, y: float) -> dict:
    return _pred(0, "ball", [x - 10, y - 10, x + 10, y + 10], 0.9)


def _raw_rows(by_frame: dict[int, list[dict]]) -> list[dict]:
    return [
        {"frame_index": frame, "timestamp_seconds": frame / 60.0, "predictions": predictions}
        for frame, predictions in sorted(by_frame.items())
    ]


def _vote_scene() -> tuple[list[dict], dict[int, list[dict]]]:
    # The shooter (player-shooter) and a teammate (player-mate) both hold a shooting pose. The ball is
    # at the shooter's hands; it is far above the teammate's box.
    frames = (10, 12, 14, 16, 18, 20)
    tracks = []
    for frame_index in frames:
        tracks += [
            _player(1, "player-edge", [0, 430, 45, 620], [18.7, 11.1], frame_index),
            _player(2, "player-shooter", [500, 280, 600, 460], [24.9, 8.3], frame_index),
            _player(3, "player-mate", [200, 300, 300, 480], [20.0, 6.0], frame_index),
            _player(4, "player-crowd", [540, 100, 640, 260], [26.0, -2.7], frame_index),
        ]
    raw = {
        frame_index: [
            _pred(7, "player-shot-block", [500, 280, 600, 460]),
            _pred(7, "player-shot-block", [200, 300, 300, 480], 0.79),
            _pred(7, "player-shot-block", [540, 100, 640, 260]),
            _ball_pred(550, 270),
        ]
        for frame_index in frames
    }
    # Ball for the teammate's pose is 0.43 box-heights above his box; the crowd box is off court.
    raw[10] = raw[10][:3] + [_ball_pred(250, 230)]
    return tracks, raw


def test_made_shot_is_attributed_by_shooting_pose_boxes_with_the_ball_at_the_hands():
    tracks, raw = _vote_scene()

    shot = build_projected_shots([_made_event()], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]

    assert shot["shooter_stable_player_id"] == "player-shooter"
    assert shot["shooter_attribution_method"] == "action_box_vote"
    assert shot["shot_origin_court_xy_m"] == [24.9, 8.3]


def test_pose_box_with_the_ball_far_above_the_box_does_not_vote():
    tracks, raw = _vote_scene()
    # Remove the shooter's own boxes: only the teammate (ball 0.43h above his box) and the crowd remain.
    raw = {
        frame: [p for p in preds if not (p["class_id"] == 7 and p["x"] == 550.0)] for frame, preds in raw.items()
    }
    for frame in raw:
        raw[frame] = [p for p in raw[frame] if p["class_id"] != 0] + [_ball_pred(250, 230)]

    shot = build_projected_shots([_made_event()], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]

    assert shot["shooter_stable_player_id"] is None
    assert shot["shooter_attribution_method"] == "ambiguous"
    assert shot["shot_origin_projection_method"] == "shot_anchor_homography_from_projected_players"


def test_close_vote_is_ambiguous_instead_of_a_guess():
    tracks, raw = _vote_scene()
    raw = {
        frame: [
            _pred(7, "player-shot-block", [500, 280, 600, 460]),
            _pred(7, "player-shot-block", [200, 300, 300, 480]),
            _ball_pred(550, 270),
            _ball_pred(250, 320),
        ]
        for frame in raw
    }

    shot = build_projected_shots([_made_event()], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]

    assert shot["shooter_stable_player_id"] is None
    assert shot["shooter_attribution_method"] == "ambiguous"


def test_made_shot_without_raw_predictions_keeps_the_anchor_rule():
    tracks, _ = _vote_scene()

    shot = build_projected_shots([_made_event()], tracks)["shots"][0]

    assert shot["shooter_stable_player_id"] == "player-edge"
    assert shot["shooter_attribution_method"] == "nearest_player_bbox_to_shot_anchor"


def test_shot_without_a_basket_frame_keeps_the_anchor_rule_even_with_raw_predictions():
    tracks, raw = _vote_scene()
    event = {k: v for k, v in _made_event().items() if k != "ball_in_basket_frame_index"}

    shot = build_projected_shots([event], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]

    assert shot["shooter_attribution_method"] == "nearest_player_bbox_to_shot_anchor"


def _shooter_only_scene(frames: list[int]) -> tuple[list[dict], dict[int, list[dict]]]:
    tracks = [_player(2, "player-shooter", [500, 280, 600, 460], [24.9, 8.3], frame) for frame in frames]
    raw = {
        frame: [_pred(7, "player-shot-block", [500, 280, 600, 460]), _ball_pred(550, 270)] for frame in frames
    }
    return tracks, raw


def _basket_event(basket_frame: int) -> dict:
    return {**_made_event(), "start_frame_index": basket_frame, "ball_in_basket_frame_index": basket_frame}


def test_made_shot_start_moves_to_the_first_shooting_pose_frame():
    tracks, raw = _vote_scene()
    raw[10].append(_ball_pred(550, 270))  # the scene's frame 10 only has the ball at the teammate

    shot = build_projected_shots([_made_event()], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]

    assert shot["start_frame_index"] == 10
    assert shot["start_timestamp_seconds"] == 10 / 60.0
    assert shot["start_shot_xyxy"] == [500, 280, 600, 460]
    assert shot["start_image_xy"] == [550, 370]
    assert shot["detector_start_frame_index"] == 20
    assert shot["detector_start_timestamp_seconds"] == 0.3
    assert shot["matched_projected_frame_index"] == 10
    assert shot["matched_projected_frame_gap"] == 0
    assert shot["start_frame_source"] == "action_box_release"


def test_release_is_the_start_of_the_last_run_so_a_putback_beats_an_earlier_miss():
    tracks, raw = _shooter_only_scene([0, 2, 30, 32, 34])

    shot = build_projected_shots([_basket_event(40)], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]

    assert shot["start_frame_index"] == 30
    assert shot["matched_projected_frame_index"] == 30


def test_a_step_of_exactly_the_max_gap_does_not_split_the_run():
    # 0.25s at 60fps is 15 frames.
    tracks, raw = _shooter_only_scene([0, 15, 30])
    shot = build_projected_shots([_basket_event(40)], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]
    assert shot["start_frame_index"] == 0

    tracks, raw = _shooter_only_scene([0, 15, 31])
    shot = build_projected_shots([_basket_event(40)], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]
    assert shot["start_frame_index"] == 31


def test_start_is_unchanged_when_the_vote_is_ambiguous_or_cannot_run():
    tracks, raw = _vote_scene()
    close = {
        frame: [
            _pred(7, "player-shot-block", [500, 280, 600, 460]),
            _pred(7, "player-shot-block", [200, 300, 300, 480]),
            _ball_pred(550, 270),
            _ball_pred(250, 320),
        ]
        for frame in raw
    }
    no_basket = {k: v for k, v in _made_event().items() if k != "ball_in_basket_frame_index"}
    cases = [
        build_projected_shots([_made_event()], tracks, raw_prediction_rows=_raw_rows(close))["shots"][0],
        build_projected_shots([_made_event()], tracks)["shots"][0],
        build_projected_shots([no_basket], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0],
    ]

    for shot in cases:
        assert shot["start_frame_index"] == 20
        assert shot["start_timestamp_seconds"] == 0.3
        assert shot["start_image_xy"] == [20.0, 520.0]
        assert shot["start_frame_source"] == "shot_detector"


def _close_vote_raw(frames) -> dict[int, list[dict]]:
    return {
        frame: [
            _pred(7, "player-shot-block", [500, 280, 600, 460]),
            _pred(7, "player-shot-block", [200, 300, 300, 480]),
            _ball_pred(550, 270),
            _ball_pred(250, 320),
        ]
        for frame in frames
    }


def _affine_tracks(tracks: list[dict]) -> list[dict]:
    # court = foot / 20 makes the image->court homography well defined for any four players.
    return [{**row, "court_xy_m": [row["image_foot_xy"][0] / 20.0, row["image_foot_xy"][1] / 20.0]} for row in tracks]


def test_ambiguous_made_shot_keeps_an_origin_projected_from_the_detector_box_foot():
    tracks, raw = _vote_scene()
    event = {**_made_event(), "start_shot_xyxy": [300.0, 300.0, 400.0, 500.0]}

    shot = build_projected_shots(
        [event], _affine_tracks(tracks), raw_prediction_rows=_raw_rows(_close_vote_raw(raw))
    )["shots"][0]

    assert shot["shooter_stable_player_id"] is None
    assert shot["shooter_attribution_method"] == "ambiguous"
    assert shot["start_frame_source"] == "shot_detector"
    assert shot["shot_origin_court_xy_m"] == pytest.approx([17.5, 25.0], abs=1e-3)
    assert shot["shot_origin_projection_method"] == "shot_anchor_homography_from_projected_players"


def test_ambiguous_made_shot_without_four_projected_players_has_no_origin():
    tracks, raw = _vote_scene()
    tracks = _affine_tracks([row for row in tracks if row["track_id"] in (2, 3)])
    event = {**_made_event(), "start_shot_xyxy": [300.0, 300.0, 400.0, 500.0]}

    shot = build_projected_shots([event], tracks, raw_prediction_rows=_raw_rows(_close_vote_raw(raw)))["shots"][0]

    assert shot["shooter_attribution_method"] == "ambiguous"
    assert shot["shot_origin_court_xy_m"] is None
    assert shot["shot_origin_projection_method"] is None


def test_trajectory_made_shot_is_attributed_with_the_rim_pass_as_anchor():
    tracks, raw = _vote_scene()
    event = {k: v for k, v in _made_event().items() if k != "ball_in_basket_frame_index"}
    event["rim_pass_frame_index"] = 20

    shot = build_projected_shots([event], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]

    assert shot["shooter_stable_player_id"] == "player-shooter"
    assert shot["shooter_attribution_method"] == "action_box_vote"
    assert shot["start_frame_source"] == "action_box_release"


def test_basket_frame_wins_over_rim_pass_as_vote_anchor():
    tracks, raw = _vote_scene()
    event = {**_made_event(), "rim_pass_frame_index": 5000}

    shot = build_projected_shots([event], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]

    assert shot["shooter_attribution_method"] == "action_box_vote"


def test_unconfirmed_basket_event_takes_the_non_made_path_without_a_vote():
    tracks, raw = _vote_scene()
    event = {**_made_event(), "outcome": "unknown", "resolution_reason": "ball_in_basket_unconfirmed"}

    shot = build_projected_shots([event], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]

    assert shot["outcome"] == "unknown"
    assert shot["shooter_attribution_method"] != "action_box_vote"
    assert shot["start_frame_source"] != "action_box_release"


def _number_scene() -> tuple[list[dict], list[dict]]:
    event = [{"event_id": 9, "start_frame_index": 1, "start_image_xy": [10.0, 10.0]}]
    tracks = [
        _player(3, "player-007", [0, 0, 20, 20], [2.0, 3.0], 1),
        _player(5, "player-008", [100, 100, 120, 120], [4.0, 5.0], 1),
    ]
    return event, tracks


def test_shooter_and_snapshot_numbers_resolve_by_stable_player_id(tmp_path: Path):
    from pipeline.projected_shots import _load_confirmed_numbers_by_stable_player_id

    numbers_path = tmp_path / "numbers.json"
    numbers_path.write_text(
        json.dumps(
            {
                "players": [
                    {"stable_player_id": "player-007", "number": "7", "status": "confirmed"},
                    {"stable_player_id": "player-008", "number": "8", "status": "ambiguous"},
                    {"stable_player_id": "player-009", "number": "9", "status": "unknown"},
                    {"stable_player_id": "player-010", "number": "", "status": "confirmed"},
                ]
            }
        ),
        encoding="utf-8",
    )
    by_stable = _load_confirmed_numbers_by_stable_player_id(numbers_path)
    assert by_stable == {"player-007": "7"}

    event, tracks = _number_scene()
    shot = build_projected_shots(event, tracks, confirmed_numbers_by_stable_player_id=by_stable)["shots"][0]

    assert shot["shooter_confirmed_number"] == "7"
    assert [row["confirmed_number"] for row in shot["projected_player_snapshot"]] == ["7", None]


def test_numbers_fall_back_to_track_mapping_when_no_stable_entry():
    event, tracks = _number_scene()
    shot = build_projected_shots(
        event,
        tracks,
        confirmed_numbers_by_track_id={3: "30", 5: "50"},
        confirmed_numbers_by_stable_player_id={"player-008": "8"},
    )["shots"][0]

    assert shot["shooter_confirmed_number"] == "30"
    assert [row["confirmed_number"] for row in shot["projected_player_snapshot"]] == ["30", "8"]


def test_stable_number_wins_over_track_mapping():
    event, tracks = _number_scene()
    shot = build_projected_shots(
        event,
        tracks,
        confirmed_numbers_by_track_id={3: "30"},
        confirmed_numbers_by_stable_player_id={"player-007": "7"},
    )["shots"][0]

    assert shot["shooter_confirmed_number"] == "7"


def test_numbers_are_none_without_any_mapping():
    event, tracks = _number_scene()
    shot = build_projected_shots(event, tracks)["shots"][0]

    assert shot["shooter_confirmed_number"] is None
    assert [row["confirmed_number"] for row in shot["projected_player_snapshot"]] == [None, None]


def _with_team(tracks: list[dict], teams_by_track: dict[int, int | None]) -> list[dict]:
    return [{**row, "team_id": teams_by_track[row["track_id"]]} if row["track_id"] in teams_by_track else row for row in tracks]


def test_shooter_candidates_are_the_top_three_of_the_vote_ranking():
    tracks, _ = _vote_scene()
    tracks = _with_team([*tracks, *(_player(5, "player-extra", [300, 100, 380, 260], [22.0, 4.0], f) for f in (10, 12, 14, 16, 18, 20))], {2: 0, 3: 1, 1: 1})
    boxes = (
        ([500, 280, 600, 460], 0.9, (550, 270)),
        ([200, 300, 300, 480], 0.6, (250, 320)),
        ([0, 430, 45, 620], 0.4, (20, 480)),
        ([300, 100, 380, 260], 0.2, (340, 150)),
    )
    scene = {
        frame: [_pred(7, "player-shot-block", box, conf) for box, conf, _ in boxes] + [_ball_pred(*ball) for _, _, ball in boxes]
        for frame in (10, 12, 14, 16, 18, 20)
    }

    shot = build_projected_shots([_made_event()], tracks, raw_prediction_rows=_raw_rows(scene))["shots"][0]

    candidates = shot["shooter_candidates"]
    assert [c["stable_player_id"] for c in candidates] == ["player-shooter", "player-mate", "player-edge"]
    assert [c["team_id"] for c in candidates] == [0, 1, 1]
    assert candidates[0]["score"] > candidates[1]["score"] > candidates[2]["score"]
    assert set(candidates[0]) == {"stable_player_id", "team_id", "score"}


def test_shooter_candidates_are_returned_for_an_ambiguous_vote_and_empty_without_a_vote():
    tracks, raw = _vote_scene()
    ambiguous = build_projected_shots([_made_event()], tracks, raw_prediction_rows=_raw_rows(_close_vote_raw(raw)))["shots"][0]
    assert ambiguous["shooter_attribution_method"] == "ambiguous"
    assert {c["stable_player_id"] for c in ambiguous["shooter_candidates"]} == {"player-shooter", "player-mate"}

    no_raw = build_projected_shots([_made_event()], tracks)["shots"][0]
    missed = build_projected_shots([{**_made_event(), "outcome": "missed", "ball_in_basket_frame_index": None}], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]
    no_ball = build_projected_shots([_made_event()], tracks, raw_prediction_rows=_raw_rows({f: [] for f in raw}))["shots"][0]
    assert no_raw["shooter_candidates"] == []
    assert missed["shooter_candidates"] == []
    assert no_ball["shooter_candidates"] == []


def test_shooter_team_falls_back_to_the_player_majority_team():
    tracks, raw = _vote_scene()
    # The shooter's rows carry team 1 at most frames; the track-level map knows nothing.
    tracks = [{**row, "team_id": 1} if row["track_id"] == 2 else row for row in tracks]
    tracks[5] = {**tracks[5], "team_id": None}

    shot = build_projected_shots([_made_event()], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]

    assert shot["shooter_stable_player_id"] == "player-shooter"
    assert shot["shooter_team_id"] == 1


def test_track_level_team_wins_over_the_player_majority_and_zero_is_a_team():
    tracks, raw = _vote_scene()
    tracks = _with_team(tracks, {2: 1})

    shot = build_projected_shots(
        [_made_event()], tracks, team_ids_by_track_id={2: 0}, raw_prediction_rows=_raw_rows(raw)
    )["shots"][0]

    assert shot["shooter_team_id"] == 0


def test_shooter_team_stays_null_without_a_strict_majority():
    tracks, raw = _vote_scene()
    shooter_rows = [row for row in tracks if row["track_id"] == 2]
    teams = {id(row): (0 if index % 2 else 1) for index, row in enumerate(shooter_rows)}
    tracks = [{**row, "team_id": teams[id(row)]} if id(row) in teams else row for row in tracks]

    shot = build_projected_shots([_made_event()], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]

    assert shot["shooter_stable_player_id"] == "player-shooter"
    assert shot["shooter_team_id"] is None
    assert shot["needs_review"] is True


def test_ambiguous_vote_between_teammates_keeps_the_team_but_not_the_player():
    tracks, raw = _vote_scene()
    tracks = _with_team(tracks, {2: 1, 3: 1})
    event = {**_made_event(), "start_shot_xyxy": [300.0, 300.0, 400.0, 500.0]}

    shot = build_projected_shots([event], tracks, raw_prediction_rows=_raw_rows(_close_vote_raw(raw)))["shots"][0]

    assert shot["shooter_attribution_method"] == "ambiguous_same_team"
    assert shot["shooter_stable_player_id"] is None
    assert shot["shooter_track_id"] is None
    assert shot["shooter_team_id"] == 1
    assert shot["start_frame_source"] == "shot_detector"
    assert shot["shot_origin_projection_method"] == "shot_anchor_homography_from_projected_players"
    assert shot["needs_review"] is True


def test_ambiguous_vote_between_different_teams_or_unknown_teams_stays_ambiguous():
    tracks, raw = _vote_scene()
    for teams in ({2: 0, 3: 1}, {2: 0, 3: None}, {}):
        shot = build_projected_shots(
            [_made_event()], _with_team(tracks, teams), raw_prediction_rows=_raw_rows(_close_vote_raw(raw))
        )["shots"][0]
        assert shot["shooter_attribution_method"] == "ambiguous"
        assert shot["shooter_team_id"] is None


def _review_shot(*, attribution_clear: bool, team: int | None, outcome: str = "made") -> dict:
    tracks, raw = _vote_scene()
    tracks = _with_team(tracks, {2: team, 3: team})
    scene = raw if attribution_clear else _close_vote_raw(raw)
    event = {**_made_event(), "outcome": outcome, "outcome_source": "ball_in_basket"}
    return build_projected_shots([event], tracks, raw_prediction_rows=_raw_rows(scene))["shots"][0]


def test_needs_review_truth_table():
    assert _review_shot(attribution_clear=True, team=1)["needs_review"] is False
    assert _review_shot(attribution_clear=True, team=None)["needs_review"] is True
    assert _review_shot(attribution_clear=False, team=1)["needs_review"] is True
    assert _review_shot(attribution_clear=True, team=1, outcome="unknown")["needs_review"] is True
    # Missed shots never run the vote, so the anchor rule decides and the shot stays under review.
    assert _review_shot(attribution_clear=True, team=1, outcome="missed")["needs_review"] is True


def test_needs_review_for_the_nearest_bbox_anchor_rule():
    tracks, _ = _vote_scene()
    tracks = _with_team(tracks, {1: 0, 2: 0, 3: 0, 4: 0})

    shot = build_projected_shots([_made_event()], tracks)["shots"][0]

    assert shot["shooter_attribution_method"] == "nearest_player_bbox_to_shot_anchor"
    assert shot["shooter_stable_player_id"] is not None
    assert shot["shooter_team_id"] == 0
    assert shot["needs_review"] is True


def test_release_timestamp_only_for_action_box_release():
    tracks, raw = _vote_scene()
    released = build_projected_shots([_made_event()], tracks, raw_prediction_rows=_raw_rows(raw))["shots"][0]
    assert released["start_frame_source"] == "action_box_release"
    assert released["release_timestamp_seconds"] == released["start_timestamp_seconds"]
    assert released["release_timestamp_seconds"] is not None

    detector = build_projected_shots([_made_event()], tracks)["shots"][0]
    assert detector["release_timestamp_seconds"] is None


def test_resolved_timestamp_and_source_fields_are_copied_or_null():
    tracks, _ = _vote_scene()
    event = {**_made_event(), "resolved_timestamp_seconds": 0.5, "outcome_source": "ball_in_basket", "resolution_reason": "ball_in_basket"}

    shot = build_projected_shots([event], tracks)["shots"][0]
    bare = build_projected_shots([_made_event()], tracks)["shots"][0]

    assert (shot["resolved_timestamp_seconds"], shot["outcome_source"], shot["resolution_reason"]) == (0.5, "ball_in_basket", "ball_in_basket")
    assert (bare["resolved_timestamp_seconds"], bare["outcome_source"], bare["resolution_reason"]) == (None, None, None)


@pytest.mark.parametrize(
    "outcome, source, reason, expected",
    [
        ("made", "ball_in_basket", "ball_in_basket", "detected"),
        ("made", "rim_trajectory", "rim_trajectory", "inferred"),
        ("missed", "shot_detector", "shot_timeout", "timeout"),
        ("missed", "shot_detector", "new_shot_started", "timeout"),
        ("unknown", "shot_detector", "end_of_clip", "unresolved"),
        ("unknown", "ball_in_basket", "ball_in_basket_unconfirmed", "unresolved"),
        ("unknown", "shot_detector", "shot_timeout", "unresolved"),
    ],
)
def test_outcome_confidence_mapping(outcome, source, reason, expected):
    tracks, _ = _vote_scene()
    event = {**_made_event(), "outcome": outcome, "outcome_source": source, "resolution_reason": reason}

    shot = build_projected_shots([event], tracks)["shots"][0]

    assert shot["outcome_confidence"] == expected
    assert shot["needs_review"] is True or expected != "unresolved"
