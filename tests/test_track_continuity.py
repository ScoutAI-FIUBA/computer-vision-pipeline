"""Track identity tests.

These cover the two functions that decide whether a player keeps their ID:
`assign_track_ids` (matching fresh detections to live tracks) and
`carry_unmatched_tracks` (surviving a detector miss).

The second exists because of a measured failure: on real footage the detector
alternated between finding 9 and 10 players, and every recovery minted a new ID.
No GPU or network needed.

    python -m pytest tests/test_track_continuity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.tracking import (  # noqa: E402
    assign_track_ids,
    carry_unmatched_tracks,
    xyxy_iou,
)


def row(xyxy, track_id=None, state="detected", confidence=0.9):
    return {
        "xyxy": list(xyxy),
        "confidence": confidence,
        "class_name": "player",
        "track_id": track_id,
        "track_state": state,
    }


# --- IoU ----------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    assert xyxy_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert xyxy_iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0


def test_iou_touching_edges_is_zero():
    assert xyxy_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0


def test_iou_half_overlap():
    # 10x10 boxes sharing a 5x10 strip: 50 / (100 + 100 - 50)
    assert xyxy_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


# --- assign_track_ids ---------------------------------------------------


def test_overlapping_detection_keeps_existing_id():
    active = [row([100, 100, 150, 200], track_id=7, state="propagated")]
    assigned, next_id = assign_track_ids([row([102, 103, 152, 202])], active, 8, 0.15)
    assert assigned[0]["track_id"] == 7
    assert next_id == 8, "reusing an ID must not consume a new one"


def test_distant_detection_gets_new_id():
    active = [row([100, 100, 150, 200], track_id=7, state="propagated")]
    assigned, next_id = assign_track_ids([row([900, 900, 950, 990])], active, 8, 0.15)
    assert assigned[0]["track_id"] == 8
    assert next_id == 9


def test_each_active_track_claimed_at_most_once():
    """Two detections over one track must not both inherit its ID."""

    active = [row([100, 100, 150, 200], track_id=7, state="propagated")]
    assigned, _ = assign_track_ids(
        [row([100, 100, 150, 200]), row([104, 104, 154, 204])], active, 8, 0.15
    )
    assert sorted(r["track_id"] for r in assigned) == [7, 8]


def test_no_active_tracks_numbers_sequentially():
    assigned, next_id = assign_track_ids(
        [row([0, 0, 10, 10]), row([20, 20, 30, 30])], [], 1, 0.15
    )
    assert [r["track_id"] for r in assigned] == [1, 2]
    assert next_id == 3


# --- carry_unmatched_tracks --------------------------------------------


def test_missed_track_is_carried_and_keeps_its_id():
    """The regression this function exists for: one missed detection."""

    active = [
        row([100, 100, 150, 200], track_id=1, state="propagated"),
        row([300, 100, 350, 200], track_id=2, state="propagated"),
    ]
    seeds = [row([102, 102, 152, 202], track_id=1)]  # detector found only track 1

    missed: dict[int, int] = {}
    result = carry_unmatched_tracks(seeds, active, missed, max_missed=2)

    assert sorted(r["track_id"] for r in result) == [1, 2]
    assert missed == {2: 1}
    carried = next(r for r in result if r["track_id"] == 2)
    assert carried["track_state"] == "coasted"


def test_track_retired_after_max_missed():
    active = [row([300, 100, 350, 200], track_id=2, state="propagated")]
    missed: dict[int, int] = {}

    for expected in (1, 2):
        result = carry_unmatched_tracks([], active, missed, max_missed=2)
        assert [r["track_id"] for r in result] == [2]
        assert missed == {2: expected}

    # Third consecutive miss exceeds the budget: the track is dropped.
    result = carry_unmatched_tracks([], active, missed, max_missed=2)
    assert result == []
    assert missed == {}


def test_recovered_track_resets_its_miss_counter():
    active = [row([300, 100, 350, 200], track_id=2, state="propagated")]
    missed: dict[int, int] = {}

    carry_unmatched_tracks([], active, missed, max_missed=2)
    assert missed == {2: 1}

    # Detector finds it again.
    carry_unmatched_tracks([row([301, 101, 351, 201], track_id=2)], active, missed, max_missed=2)
    assert missed == {}, "a recovered track must get its full grace period back"


def test_max_missed_zero_restores_original_behaviour():
    """The old script deleted any track the detector did not confirm."""

    active = [row([300, 100, 350, 200], track_id=2, state="propagated")]
    missed: dict[int, int] = {}
    result = carry_unmatched_tracks([], active, missed, max_missed=0)
    assert result == []
    assert missed == {}


def test_carried_row_does_not_mutate_the_active_row():
    active_row = row([300, 100, 350, 200], track_id=2, state="propagated")
    result = carry_unmatched_tracks([], [active_row], {}, max_missed=2)
    assert result[0]["track_state"] == "coasted"
    assert active_row["track_state"] == "propagated"


def test_alternating_detector_keeps_one_id_over_many_calls():
    """End-to-end regression: 9/10/9/10 detections must not mint new IDs.

    This is the exact pattern measured on real footage that produced 32 track
    IDs for 10 players.
    """

    boxes = [[i * 100, 100, i * 100 + 50, 200] for i in range(10)]
    active = [row(b, track_id=i + 1, state="propagated") for i, b in enumerate(boxes)]
    missed: dict[int, int] = {}
    next_id = 11

    for call in range(12):
        # Every other call the detector misses the last player.
        visible = boxes if call % 2 == 0 else boxes[:-1]
        seeds, next_id = assign_track_ids([row(b) for b in visible], active, next_id, 0.15)
        seeds = carry_unmatched_tracks(seeds, active, missed, max_missed=2)
        active = [row(r["xyxy"], track_id=r["track_id"], state="propagated") for r in seeds]

    assert next_id == 11, "no new IDs should have been issued"
    assert sorted(r["track_id"] for r in active) == list(range(1, 11))
