"""Appearance-based re-identification tests.

The headline test is `test_appearance_prevents_swap_when_geometry_is_ambiguous`,
which reproduces the failure seen on real footage: a shooter and the player
blocking her cross, and afterwards they exchange track IDs. Geometry alone
cannot tell them apart; torso colour can.

No GPU or network needed.

    python -m pytest tests/test_appearance_matching.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import appearance  # noqa: E402
from pipeline.appearance import ReidConfig  # noqa: E402
from pipeline.tracking import assign_track_ids  # noqa: E402


def unit(values: list[float]) -> list[float]:
    vector = np.asarray(values, dtype=np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


RED = unit([1.0, 0.0, 0.0, 0.0])
BLUE = unit([0.0, 0.0, 1.0, 0.0])


def row(xyxy, track_id=None, signature=None, state="detected"):
    return {
        "xyxy": list(xyxy),
        "confidence": 0.9,
        "class_name": "player",
        "track_id": track_id,
        "track_state": state,
        "appearance": signature,
    }


# --- geometry helpers ---------------------------------------------------


def test_subtract_removes_the_covered_strip():
    pieces = appearance.subtract((0, 0, 100, 100), (0, 0, 100, 40))
    assert pieces == [(0, 40, 100, 100)]


def test_subtract_returns_source_when_disjoint():
    assert appearance.subtract((0, 0, 10, 10), (50, 50, 60, 60)) == [(0, 0, 10, 10)]


def test_torso_box_skips_head_and_legs():
    torso = appearance.torso_box((720, 1280, 3), (100, 100, 200, 300), ReidConfig())
    # y_start 0.15 and y_end 0.68 of a 200px-tall box.
    assert torso == (105, 130, 195, 236)


# --- occlusion refusal --------------------------------------------------


def test_visible_torso_refuses_when_heavily_covered():
    """The crossing case: refusing is correct, guessing would be confidently wrong."""

    covered = appearance.visible_torso(
        (720, 1280, 3), (100, 100, 200, 300), [(0, 0, 1280, 720)], ReidConfig()
    )
    assert covered is None


def test_visible_torso_allows_a_small_occlusion():
    region = appearance.visible_torso(
        (720, 1280, 3), (100, 100, 200, 300), [(100, 100, 200, 140)], ReidConfig()
    )
    assert region is not None


def test_extract_returns_none_for_tiny_boxes():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert appearance.extract(frame, (10, 10, 20, 25), [], ReidConfig()) is None


def test_extract_produces_a_normalised_signature():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 200)  # solid red in BGR
    signature = appearance.extract(frame, (100, 100, 200, 300), [], ReidConfig())
    assert signature is not None
    assert np.linalg.norm(np.asarray(signature)) == pytest.approx(1.0, abs=1e-5)


def test_different_colours_are_less_similar_than_identical_ones():
    frame_red = np.zeros((400, 400, 3), dtype=np.uint8)
    frame_red[:, :] = (0, 0, 200)
    frame_blue = np.zeros((400, 400, 3), dtype=np.uint8)
    frame_blue[:, :] = (200, 0, 0)

    box = (100, 50, 200, 250)
    red = appearance.extract(frame_red, box, [], ReidConfig())
    blue = appearance.extract(frame_blue, box, [], ReidConfig())

    assert appearance.similarity(red, red) == pytest.approx(1.0, abs=1e-5)
    assert appearance.similarity(red, blue) < 0.5


def test_similarity_is_none_when_a_signature_is_missing():
    assert appearance.similarity(None, RED) is None
    assert appearance.similarity(RED, None) is None


def test_blend_keeps_result_normalised():
    merged = appearance.blend(RED, BLUE, momentum=0.5)
    assert np.linalg.norm(np.asarray(merged)) == pytest.approx(1.0, abs=1e-5)


def test_blend_without_history_returns_the_observation():
    assert appearance.blend(None, RED) == RED


# --- the regression -----------------------------------------------------


def crossing_scenario():
    """Two players mid-crossing, each nearer the OTHER's previous position.

    Geometry alone therefore favours the swap; colour favours the truth.
    """

    active = [
        row([100, 100, 160, 250], track_id=1, state="propagated"),
        row([140, 100, 200, 250], track_id=2, state="propagated"),
    ]
    # Track 1 (red) has moved right, track 2 (blue) has moved left: they passed.
    detections = [
        row([145, 100, 205, 250], signature=RED),
        row([95, 100, 155, 250], signature=BLUE),
    ]
    signatures = {1: RED, 2: BLUE}
    return detections, active, signatures


def test_geometry_alone_swaps_the_two_players():
    """Establishes that the scenario really is ambiguous without colour."""

    detections, active, _ = crossing_scenario()
    assigned, _ = assign_track_ids(detections, active, 3, 0.15, appearance_weight=0.0)
    # The red player (first detection) wrongly inherits track 2.
    assert assigned[0]["track_id"] == 2
    assert assigned[1]["track_id"] == 1


def test_appearance_prevents_swap_when_geometry_is_ambiguous():
    """The fix: colour keeps each player on their own track through a crossing."""

    detections, active, signatures = crossing_scenario()
    assigned, next_id = assign_track_ids(
        detections,
        active,
        3,
        0.15,
        appearance_by_track_id=signatures,
        appearance_weight=0.5,
    )

    assert assigned[0]["track_id"] == 1, "red detection must stay on the red track"
    assert assigned[1]["track_id"] == 2, "blue detection must stay on the blue track"
    assert next_id == 3, "no new IDs should be minted"


def test_missing_signature_falls_back_to_geometry_without_crashing():
    detections, active, signatures = crossing_scenario()
    detections[0]["appearance"] = None
    assigned, _ = assign_track_ids(
        detections, active, 3, 0.15, appearance_by_track_id=signatures, appearance_weight=0.5
    )
    assert sorted(r["track_id"] for r in assigned) == [1, 2]


def test_appearance_cannot_match_across_the_court():
    """Colour is a tie-breaker only: the IoU gate still bounds every match."""

    active = [row([100, 100, 160, 250], track_id=1, state="propagated")]
    detections = [row([900, 400, 960, 550], signature=RED)]
    assigned, next_id = assign_track_ids(
        detections, active, 5, 0.15, appearance_by_track_id={1: RED}, appearance_weight=0.5
    )
    assert assigned[0]["track_id"] == 5, "identical colour must not defeat the IoU gate"
    assert next_id == 6


# --- swap detection -----------------------------------------------------


def test_find_better_match_flags_a_swapped_detection():
    """A red detection sitting on the blue track is the signal we want."""

    result = appearance.find_better_match(RED, assigned_track_id=2, signatures={1: RED, 2: BLUE})
    assert result is not None
    better_id, own, other = result
    assert better_id == 1
    assert other > own


def test_find_better_match_stays_quiet_when_assignment_is_right():
    assert appearance.find_better_match(RED, 1, {1: RED, 2: BLUE}) is None


def test_find_better_match_needs_a_stored_signature():
    """A brand-new track has no history, so nothing can be concluded."""

    assert appearance.find_better_match(RED, 99, {1: RED}) is None


def test_find_better_match_ignores_differences_within_the_margin():
    """Near-identical kit must not produce a stream of false alarms."""

    almost = unit([1.0, 0.02, 0.0, 0.0])
    assert appearance.find_better_match(RED, 1, {1: RED, 2: almost}, margin=0.03) is None


def test_find_better_match_reports_the_best_alternative():
    green = unit([0.0, 1.0, 0.0, 0.0])
    result = appearance.find_better_match(BLUE, 1, {1: RED, 2: green, 3: BLUE})
    assert result is not None
    assert result[0] == 3


def test_optimal_assignment_beats_greedy_ordering():
    """Greedy lets an early detection steal a track a later one needed more."""

    active = [
        row([100, 100, 200, 300], track_id=1, state="propagated"),
        row([210, 100, 310, 300], track_id=2, state="propagated"),
    ]
    detections = [
        row([160, 100, 260, 300]),  # overlaps both, slightly favours track 2
        row([100, 100, 200, 300]),  # exact match for track 1
    ]
    assigned, _ = assign_track_ids(detections, active, 3, 0.15)
    assert assigned[1]["track_id"] == 1, "the exact match must win its track"
    assert assigned[0]["track_id"] == 2
