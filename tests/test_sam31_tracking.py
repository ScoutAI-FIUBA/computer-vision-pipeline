"""Contract tests for the isolated SAM 3.1 tracking adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.sam31_tracking import (  # noqa: E402
    Sam31MultiplexTracker,
    normalize_multiplex_output,
    observation_key,
)


def test_normalization_preserves_raw_id_and_uses_separate_stable_id():
    rows = normalize_multiplex_output(
        frame_index=60,
        timestamp_seconds=2.0,
        objects=[
            {
                "object_id": 3,
                "bbox": [10, 20, 30, 40],
                "confidence": 0.94,
                "probability": 0.91,
            }
        ],
        stable_player_ids={3: "player-07"},
    )

    assert rows == [
        {
            "observation_key": "60:3",
            "frame_index": 60,
            "timestamp_seconds": 2.0,
            "raw_track_id": 3,
            "stable_player_id": "player-07",
            "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
            "confidence": 0.94,
            "probability": 0.91,
            "track_state": "tracked",
        }
    ]


def test_normalization_converts_xywh_and_leaves_unassigned_identity_null():
    rows = normalize_multiplex_output(
        frame_index=61,
        timestamp_seconds=2.033,
        objects=[{"object_id": 4, "bbox": [10, 20, 30, 40], "bbox_format": "xywh"}],
    )

    assert rows[0]["observation_key"] == observation_key(61, 4)
    assert rows[0]["bbox_xyxy"] == [10.0, 20.0, 40.0, 60.0]
    assert rows[0]["stable_player_id"] is None
    assert rows[0]["probability"] is None


def test_normalization_keeps_multiple_raw_tracks_separate():
    rows = normalize_multiplex_output(
        frame_index=62,
        timestamp_seconds=2.066,
        objects=[
            {"object_id": 3, "bbox": [0, 0, 10, 10]},
            {"object_id": 8, "bbox": [20, 0, 10, 10]},
        ],
        stable_player_ids={3: "player-07"},
    )

    assert [(row["raw_track_id"], row["stable_player_id"]) for row in rows] == [
        (3, "player-07"),
        (8, None),
    ]


def test_tracker_calls_prompt_and_propagate_and_closes_session():
    class FakeSession:
        def __init__(self):
            self.calls = []

        def start(self, video):
            self.calls.append(("start", video))

        def add_prompt(self, prompt):
            self.calls.append(("prompt", prompt))

        def propagate(self, frame_index):
            self.calls.append(("propagate", frame_index))
            return [{"object_id": 3, "bbox": [1, 2, 3, 4]}]

        def close(self):
            self.calls.append(("close",))

    session = FakeSession()
    tracker = Sam31MultiplexTracker(session)
    tracker.start("clip.mp4")
    tracker.add_prompt("person")
    rows = tracker.propagate(60, 2.0)
    tracker.close()

    assert rows[0]["raw_track_id"] == 3
    assert [call[0] for call in session.calls] == ["start", "prompt", "propagate", "close"]


def test_tracker_terminates_after_bounded_loss_and_does_not_reassign():
    class FakeSession:
        def start(self, _video):
            pass

        def propagate(self, frame_index):
            return [{"object_id": 3, "bbox": [1, 2, 3, 4]}] if frame_index == 10 else []

        def close(self):
            pass

    tracker = Sam31MultiplexTracker(FakeSession(), retention_frames=2)
    tracker.start("clip.mp4")
    assert tracker.propagate(10, 0.0)[0]["track_state"] == "tracked"
    assert tracker.propagate(11, 0.1)[0]["track_state"] == "coasted"
    assert tracker.propagate(12, 0.2)[0]["track_state"] == "coasted"
    assert tracker.propagate(13, 0.3) == []
    assert tracker.propagate(14, 0.4) == []


def test_tracker_requires_start_before_model_calls():
    tracker = Sam31MultiplexTracker(object())

    with pytest.raises(RuntimeError, match="must be started"):
        tracker.propagate(1, 0.0)
