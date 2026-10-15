"""Rim-trajectory outcome refinement on synthetic rim and ball fixtures.

    python -m pytest tests/test_shot_outcome.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import runner  # noqa: E402
from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.shot_outcome import refine_shot_events, refine_shot_events_file  # noqa: E402

RIM_X, RIM_Y, RIM_W = 500.0, 200.0, 40.0


def rim_box(confidence: float = 0.9, x: float = RIM_X) -> dict:
    return {"class_id": 9, "class": "rim", "confidence": confidence, "x": x, "y": RIM_Y, "width": RIM_W, "height": 25}


def raw_rows(rim_frames, *, last_frame: int = 200) -> list[dict]:
    return [
        {
            "frame_index": frame,
            "timestamp_seconds": frame / 30.0,
            "predictions": [rim_box()] if frame in rim_frames else [],
        }
        for frame in range(0, last_frame + 1)
    ]


def tracknet_row(frame: int, u: float, v: float) -> dict:
    return {
        "frame_index": frame,
        "timestamp_seconds": frame / 30.0,
        "track_state": "detected",
        "center_xy": [RIM_X + u * RIM_W, RIM_Y + v * RIM_W],
    }


def event(**overrides) -> dict:
    base = {
        "event_id": 1,
        "shot_type": "jump_shot",
        "outcome": "missed",
        "start_frame_index": 20,
        "end_frame_index": 40,
        "resolved_frame_index": 190,
        "ball_in_basket_frame_index": None,
        "resolution_reason": "shot_timeout",
    }
    return {**base, **overrides}


# Above the rim, then in the zone, then out below and centred.
MADE_PATH = [(60, 0.1, -1.2), (66, 0.0, -0.7), (72, 0.0, -0.1), (76, 0.0, 0.2), (80, 0.05, 0.6), (84, 0.0, 1.2)]


def balls(path) -> list[dict]:
    return [tracknet_row(frame, u, v) for frame, u, v in path]


def refine_one(path, *, rim_frames=range(0, 201), **event_overrides) -> dict:
    return refine_shot_events([event(**event_overrides)], raw_rows(rim_frames), balls(path))[0]


def test_ball_passing_down_through_the_rim_zone_upgrades_a_missed_event():
    refined = refine_one(MADE_PATH)

    assert refined["outcome"] == "made"
    assert refined["resolution_reason"] == "rim_trajectory"
    assert refined["outcome_source"] == "rim_trajectory"
    assert refined["rim_pass_frame_index"] == 72
    assert refined["rim_exit_frame_index"] == 80
    assert refined["ball_in_basket_frame_index"] is None


def test_unknown_event_is_upgraded_too():
    refined = refine_one(MADE_PATH, outcome="unknown", resolution_reason="end_of_clip")

    assert refined["outcome"] == "made"
    assert refined["resolution_reason"] == "rim_trajectory"


def test_detector_ball_boxes_count_as_ball_points_only_when_confident():
    def ball_box(frame, u, v, confidence):
        return {"class_id": 0, "class": "ball", "confidence": confidence, "x": RIM_X + u * RIM_W, "y": RIM_Y + v * RIM_W, "width": 10, "height": 10}

    rows = raw_rows(range(0, 201))
    for frame, u, v in MADE_PATH:
        rows[frame]["predictions"].append(ball_box(frame, u, v, 0.8))

    assert refine_shot_events([event()], rows, [])[0]["outcome"] == "made"

    for frame, u, v in MADE_PATH:
        rows[frame]["predictions"][-1]["confidence"] = 0.5
    assert refine_shot_events([event()], rows, [])[0]["outcome"] == "missed"


def test_lost_tracknet_rows_are_ignored():
    rows = [{**row, "track_state": "lost"} for row in balls(MADE_PATH)]

    refined = refine_shot_events([event()], raw_rows(range(0, 201)), rows)[0]

    assert refined["outcome"] == "missed"


def test_ball_leaving_the_zone_to_the_side_keeps_the_outcome():
    path = [(60, 0.1, -1.2), (66, 0.0, -0.7), (72, 0.0, -0.1), (76, 0.9, 0.0), (80, 1.4, 0.8)]

    refined = refine_one(path)

    assert refined["outcome"] == "missed"
    assert refined["outcome_source"] == "shot_detector"
    assert refined["rim_pass_frame_index"] is None


def test_ball_leaving_the_zone_upwards_keeps_the_outcome():
    path = [(60, 0.1, -1.2), (66, 0.0, -0.7), (72, 0.0, -0.1), (76, 0.0, -0.8), (80, 0.0, -1.5)]

    assert refine_one(path)["outcome"] == "missed"


def test_missing_exit_keeps_the_outcome():
    path = MADE_PATH[:4]

    assert refine_one(path)["outcome"] == "missed"


def test_exit_too_long_after_the_zone_keeps_the_outcome():
    path = MADE_PATH[:4] + [(76 + 46, 0.0, 1.0)]

    assert refine_one(path)["outcome"] == "missed"


def test_zone_without_a_ball_above_the_rim_keeps_the_outcome():
    path = [(72, 0.0, -0.3), (76, 0.0, 0.2), (80, 0.05, 0.6)]

    assert refine_one(path)["outcome"] == "missed"


def test_zone_too_long_after_the_ball_above_the_rim_keeps_the_outcome():
    path = [(30, 0.1, -1.2), (72, 0.0, -0.1), (76, 0.0, 0.2), (80, 0.05, 0.6)]

    assert refine_one(path)["outcome"] == "missed"


def test_a_second_ball_far_from_the_rim_does_not_count_as_a_side_exit():
    path = MADE_PATH + [(73, 7.0, -2.3)]

    assert refine_one(path)["outcome"] == "made"


def test_ball_before_the_event_window_or_long_after_it_is_ignored():
    early = [(frame - 40, u, v) for frame, u, v in MADE_PATH]
    late = [(frame + 150, u, v) for frame, u, v in MADE_PATH]

    assert refine_one(early, rim_frames=range(0, 301), start_frame_index=30, end_frame_index=40)["outcome"] == "missed"
    assert refine_one(late, rim_frames=range(0, 301))["outcome"] == "missed"


def test_pass_in_the_next_shot_does_not_upgrade_an_event_closed_by_new_shot_started():
    refined = refine_one(MADE_PATH, resolution_reason="new_shot_started", resolved_frame_index=50)

    assert refined["outcome"] == "missed"


def test_rim_box_within_thirty_frames_is_used_when_the_frame_has_none():
    assert refine_one(MADE_PATH, rim_frames={50})["outcome"] == "made"
    assert refine_one(MADE_PATH, rim_frames={20})["outcome"] == "missed"


def test_highest_confidence_rim_in_a_frame_wins():
    rows = raw_rows(range(0, 201))
    for row in rows:
        row["predictions"].append(rim_box(confidence=0.2, x=RIM_X + 300))

    assert refine_shot_events([event()], rows, balls(MADE_PATH))[0]["outcome"] == "made"


def test_no_rim_means_no_rule():
    assert refine_one(MADE_PATH, rim_frames=[])["outcome"] == "missed"


def basket_event(**overrides) -> dict:
    return event(outcome="made", resolution_reason="ball_in_basket", ball_in_basket_frame_index=100, **overrides)


def refine_basket(ball_path, *, rim_frames=range(0, 201), raw=None, **event_overrides) -> dict:
    rows = raw if raw is not None else raw_rows(rim_frames)
    return refine_shot_events([basket_event(**event_overrides)], rows, balls(ball_path))[0]


def test_ball_in_basket_events_with_a_ball_near_the_rim_stay_made():
    refined = refine_basket([(90, 0.3, -1.0), (96, 0.0, -0.2)])

    assert refined["outcome"] == "made"
    assert refined["resolution_reason"] == "ball_in_basket"
    assert refined["outcome_source"] == "ball_in_basket"
    assert refined["ball_in_basket_frame_index"] == 100
    assert refined["rim_pass_frame_index"] is None
    assert refined["ball_near_rim_evidence"] is True


def test_basket_detection_with_the_ball_far_from_the_rim_becomes_unknown():
    original = basket_event()

    refined = refine_basket([(frame, 8.0, 0.0) for frame in range(40, 111, 5)])

    assert refined["outcome"] == "unknown"
    assert refined["resolution_reason"] == "ball_in_basket_unconfirmed"
    assert refined["outcome_source"] == "ball_in_basket"
    assert refined["ball_near_rim_evidence"] is False
    for key in ("event_id", "shot_type", "start_frame_index", "end_frame_index", "resolved_frame_index", "ball_in_basket_frame_index"):
        assert refined[key] == original[key]


def test_basket_detection_without_any_ball_becomes_unknown():
    assert refine_basket([])["outcome"] == "unknown"


def test_basket_detection_without_rim_boxes_is_left_alone():
    refined = refine_basket([(frame, 8.0, 0.0) for frame in range(40, 111, 5)], rim_frames=[])

    assert refined["outcome"] == "made"
    assert refined["resolution_reason"] == "ball_in_basket"
    assert "ball_near_rim_evidence" not in refined


def test_rim_box_only_far_outside_the_window_is_not_a_rim_for_the_window():
    refined = refine_basket([(frame, 8.0, 0.0) for frame in range(40, 111, 5)], rim_frames=[0, 1, 2])

    assert refined["outcome"] == "made"


def test_tracknet_only_and_detector_only_balls_both_give_evidence():
    near = (95, 0.5, 0.0)
    assert refine_basket([near])["ball_near_rim_evidence"] is True

    ball_box = {"class_id": 0, "class": "ball", "confidence": 0.8, "x": RIM_X + 0.5 * RIM_W, "y": RIM_Y, "width": 10, "height": 10}
    rows = raw_rows(range(0, 201))
    rows[95]["predictions"].append(ball_box)
    detector_only = refine_shot_events([basket_event()], rows, [])[0]
    assert detector_only["outcome"] == "made"
    assert detector_only["ball_near_rim_evidence"] is True


def test_weak_detector_ball_and_lost_tracknet_rows_give_no_evidence():
    ball_box = {"class_id": 0, "class": "ball", "confidence": 0.4, "x": RIM_X, "y": RIM_Y, "width": 10, "height": 10}
    rows = raw_rows(range(0, 201))
    rows[95]["predictions"].append(ball_box)
    lost = {**tracknet_row(96, 0.0, 0.0), "track_state": "lost"}

    assert refine_shot_events([basket_event()], rows, [lost])[0]["outcome"] == "unknown"


def test_ball_distance_threshold_boundary():
    # Euclidean in rim widths: (1.2, 1.6) is exactly 2.0 away.
    assert refine_basket([(95, 1.2, 1.6)])["outcome"] == "made"
    assert refine_basket([(95, 1.2, 1.7)])["outcome"] == "unknown"


def test_frame_window_boundary():
    near = (0.0, 0.0)
    assert refine_basket([(40, *near)])["outcome"] == "made"
    assert refine_basket([(39, *near)])["outcome"] == "unknown"
    assert refine_basket([(110, *near)])["outcome"] == "made"
    assert refine_basket([(111, *near)])["outcome"] == "unknown"


def test_trajectory_upgraded_events_are_not_second_guessed():
    refined = refine_one(MADE_PATH)

    assert refined["resolution_reason"] == "rim_trajectory"
    assert "ball_near_rim_evidence" not in refined


def test_missed_and_unknown_events_are_never_touched_by_the_basket_check():
    for outcome in ("missed", "unknown"):
        refined = refine_one([], outcome=outcome, resolution_reason="end_of_clip", ball_in_basket_frame_index=100)
        assert refined["outcome"] == outcome
        assert "ball_near_rim_evidence" not in refined


def test_input_events_are_not_mutated():
    original = event()

    refine_shot_events([original], raw_rows(range(0, 201)), balls(MADE_PATH))

    assert original == event()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_stats_dir(output_dir: Path) -> None:
    write_jsonl(output_dir / runner.SHOT_EVENTS_FILENAME, [event()])
    write_jsonl(output_dir / runner.SHOT_RAW_PREDICTIONS_FILENAME, raw_rows(range(0, 201)))
    write_jsonl(output_dir / runner.BALL_TRACKS_FILENAME, balls(MADE_PATH))


def test_flag_defaults_off_and_reads_the_environment(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "dummy")
    monkeypatch.delenv("SHOT_RIM_TRAJECTORY_OUTCOME", raising=False)
    assert PipelineConfig(roboflow_api_key="dummy").shot_rim_trajectory_outcome is False
    assert PipelineConfig.from_env().shot_rim_trajectory_outcome is False

    monkeypatch.setenv("SHOT_RIM_TRAJECTORY_OUTCOME", "1")
    assert PipelineConfig.from_env().shot_rim_trajectory_outcome is True


def test_flag_off_leaves_the_written_files_byte_identical(tmp_path):
    write_stats_dir(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    runner._refine_shot_outcomes(tmp_path, PipelineConfig(roboflow_api_key="dummy"))

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_flag_on_rewrites_shot_events_and_leaves_no_temporary_file(tmp_path):
    write_stats_dir(tmp_path)

    runner._refine_shot_outcomes(tmp_path, PipelineConfig(roboflow_api_key="dummy", shot_rim_trajectory_outcome=True))

    (rewritten,) = [json.loads(line) for line in (tmp_path / runner.SHOT_EVENTS_FILENAME).read_text().splitlines()]
    assert rewritten["outcome"] == "made"
    assert rewritten["rim_pass_frame_index"] == 72
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [runner.SHOT_EVENTS_FILENAME, runner.SHOT_RAW_PREDICTIONS_FILENAME, runner.BALL_TRACKS_FILENAME]
    )


def test_rewrite_is_atomic_and_a_failed_replace_keeps_the_original(tmp_path, monkeypatch):
    write_stats_dir(tmp_path)
    events_path = tmp_path / runner.SHOT_EVENTS_FILENAME
    original = events_path.read_bytes()
    replaced = []
    real_replace = os.replace

    def spy_replace(source, destination):
        replaced.append((Path(source).name, Path(destination).name))
        assert events_path.read_bytes() == original  # not touched before the replace
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", spy_replace)
    refine_shot_events_file(
        events_path, tmp_path / runner.SHOT_RAW_PREDICTIONS_FILENAME, tmp_path / runner.BALL_TRACKS_FILENAME
    )
    assert replaced == [(runner.SHOT_EVENTS_FILENAME + ".tmp", runner.SHOT_EVENTS_FILENAME)]

    write_stats_dir(tmp_path)

    def failing_replace(source, destination):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        refine_shot_events_file(
            events_path, tmp_path / runner.SHOT_RAW_PREDICTIONS_FILENAME, tmp_path / runner.BALL_TRACKS_FILENAME
        )
    assert events_path.read_bytes() == original
