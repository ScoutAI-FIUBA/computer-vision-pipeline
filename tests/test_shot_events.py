"""Minimal make/miss tests on the active pipeline path.

    python -m pytest tests/test_shot_events.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.shot_events import (  # noqa: E402
    flush_open_shot_event,
    shot_frame_signals,
    shot_signal_trace_record,
    update_shot_event_candidate,
)
from pipeline.video import Segment, VideoInfo  # noqa: E402


def frame(*class_ids: int) -> dict:
    return {"predictions": [{"class_id": class_id} for class_id in class_ids]}


def detection_frame(*predictions: dict) -> dict:
    return {"predictions": list(predictions)}


def test_shot_frame_signals_match_verified_notebook_class_ids():
    signals = shot_frame_signals(frame(1, 5, 6))

    assert signals == {
        "has_jump_shot": True,
        "has_layup_dunk": True,
        "has_ball_in_basket": True,
        "shot_type_signal": "layup_dunk",
        "shot_signal_confidence": None,
        "shot_signal_xyxy": None,
        "shot_signal_center_xy": None,
    }


def test_shot_frame_signals_capture_best_shot_bbox_anchor():
    signals = shot_frame_signals(
        {
            "predictions": [
                {"class_id": 5, "class": "jump_shot", "confidence": 0.6, "x": 20, "y": 30, "width": 10, "height": 20},
                {"class_id": 5, "class": "jump_shot", "confidence": 0.9, "x": 40, "y": 50, "width": 12, "height": 18},
            ]
        }
    )

    assert signals["shot_type_signal"] == "jump_shot"
    assert signals["shot_signal_confidence"] == pytest.approx(0.9)
    assert signals["shot_signal_center_xy"] == pytest.approx((40.0, 50.0))
    assert signals["shot_signal_xyxy"] == pytest.approx((34.0, 41.0, 46.0, 59.0))


def test_jump_shot_becomes_made_when_ball_in_basket_arrives():
    candidate = None
    next_event_id = 1

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=100,
        timestamp_seconds=4.0,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None
    assert candidate is not None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=130,
        timestamp_seconds=5.2,
        signals=shot_frame_signals(frame(1)),
    )

    assert candidate is not None
    assert next_event_id == 2
    assert resolved is not None
    assert resolved["shot_type"] == "jump_shot"
    assert resolved["outcome"] == "made"
    assert resolved["start_frame_index"] == 100
    assert resolved["ball_in_basket_frame_index"] == 130


def test_repeated_shot_detections_within_one_second_stay_same_event():
    candidate = None
    next_event_id = 1

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=100,
        timestamp_seconds=4.0,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None
    assert candidate is not None

    original_event_id = candidate.event_id

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=120,
        timestamp_seconds=4.9,
        signals=shot_frame_signals(frame(5)),
    )

    assert resolved is None
    assert candidate is not None
    assert candidate.event_id == original_event_id
    assert candidate.start_frame_index == 100
    assert candidate.latest_shot_frame_index == 120
    assert next_event_id == 2


def test_made_shot_reanchors_start_to_latest_shot_signal_within_same_event_window():
    candidate = None
    next_event_id = 1

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=0,
        timestamp_seconds=0.0,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=8,
        timestamp_seconds=0.2667,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=12,
        timestamp_seconds=0.4,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=18,
        timestamp_seconds=0.6,
        signals=shot_frame_signals(frame(1)),
    )

    assert resolved is not None
    assert resolved["outcome"] == "made"
    assert resolved["start_frame_index"] == 12
    assert resolved["start_timestamp_seconds"] == pytest.approx(0.4)
    assert resolved["ball_in_basket_frame_index"] == 18
    assert next_event_id == 2


def test_repeated_shot_detections_without_reset_do_not_start_new_event_after_one_second():
    candidate = None
    next_event_id = 1

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=100,
        timestamp_seconds=4.0,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None
    assert candidate is not None

    original_event_id = candidate.event_id

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=145,
        timestamp_seconds=5.5,
        signals=shot_frame_signals(frame(5)),
    )

    assert resolved is None
    assert candidate is not None
    assert candidate.event_id == original_event_id
    assert candidate.latest_shot_frame_index == 145
    assert next_event_id == 2


def test_isolated_early_jump_shot_blip_does_not_anchor_later_made_event_start():
    candidate = None
    next_event_id = 1

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=0,
        timestamp_seconds=0.0,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None
    assert candidate is not None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=1,
        timestamp_seconds=0.0333,
        signals=shot_frame_signals(frame()),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=18,
        timestamp_seconds=0.6,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None
    assert candidate is not None
    assert candidate.start_frame_index == 18

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=20,
        timestamp_seconds=0.6667,
        signals=shot_frame_signals(frame(1)),
    )

    assert resolved is not None
    assert resolved["outcome"] == "made"
    assert resolved["start_frame_index"] == 18
    assert resolved["ball_in_basket_frame_index"] == 20
    assert next_event_id == 2


def test_low_confidence_first_shot_burst_is_dropped_but_later_high_confidence_burst_survives():
    candidate = None
    next_event_id = 1

    low_burst_frames = (
        (0, 0.0, detection_frame({"class_id": 5, "class": "jump_shot", "confidence": 0.473})),
        (4, 0.1333, frame()),
        (8, 0.2667, detection_frame({"class_id": 5, "class": "jump_shot", "confidence": 0.91})),
        (12, 0.4, detection_frame({"class_id": 5, "class": "jump_shot", "confidence": 0.88})),
        (18, 0.6, frame(1)),
    )
    for frame_index, timestamp_seconds, result in low_burst_frames:
        candidate, resolved, next_event_id = update_shot_event_candidate(
            candidate,
            next_event_id=next_event_id,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            signals=shot_frame_signals(result),
        )
        assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=224,
        timestamp_seconds=7.4667,
        signals=shot_frame_signals(
            detection_frame({"class_id": 5, "class": "jump_shot", "confidence": 0.72})
        ),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=228,
        timestamp_seconds=7.6,
        signals=shot_frame_signals(
            detection_frame({"class_id": 5, "class": "jump_shot", "confidence": 0.81})
        ),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=232,
        timestamp_seconds=7.7333,
        signals=shot_frame_signals(frame(1)),
    )

    assert resolved is not None
    assert resolved["outcome"] == "made"
    assert resolved["event_id"] == 1
    assert resolved["ball_in_basket_frame_index"] == 232
    assert resolved["start_frame_index"] == 228
    assert next_event_id == 2


def test_shot_becomes_missed_after_five_seconds_without_ball_in_basket():
    candidate = None
    next_event_id = 1

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=100,
        timestamp_seconds=4.0,
        signals=shot_frame_signals(frame(6)),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=225,
        timestamp_seconds=9.0,
        signals=shot_frame_signals(frame()),
    )

    assert candidate is None
    assert resolved is not None
    assert resolved["outcome"] == "missed"
    assert resolved["resolution_reason"] == "shot_timeout"
    assert resolved["start_frame_index"] == 100
    assert resolved["resolved_frame_index"] == 225


def test_new_shot_outside_same_event_window_after_reset_misses_previous_and_starts_new_one():
    candidate = None
    next_event_id = 1

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=100,
        timestamp_seconds=4.0,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=101,
        timestamp_seconds=4.1,
        signals=shot_frame_signals(frame()),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=130,
        timestamp_seconds=5.0,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is not None
    assert resolved["event_id"] == 1
    assert resolved["outcome"] == "missed"
    assert resolved["resolution_reason"] == "new_shot_started"
    assert candidate is not None
    assert candidate.event_id == 2
    assert candidate.start_frame_index == 130
    assert next_event_id == 3


def test_made_shot_duplicate_within_post_made_suppression_window_is_ignored():
    candidate = None
    next_event_id = 1

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=100,
        timestamp_seconds=4.0,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=130,
        timestamp_seconds=5.2,
        signals=shot_frame_signals(frame(1)),
    )
    assert resolved is not None
    assert resolved["outcome"] == "made"
    assert next_event_id == 2

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=166,
        timestamp_seconds=6.6,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None
    assert next_event_id == 2
    assert flush_open_shot_event(candidate, frame_index=166, timestamp_seconds=6.6) is None


def test_post_made_suppression_blocks_observed_duplicate_window():
    candidate = None
    next_event_id = 1

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=34,
        timestamp_seconds=1.1333,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=68,
        timestamp_seconds=2.2667,
        signals=shot_frame_signals(frame(1)),
    )
    assert resolved is not None
    assert resolved["outcome"] == "made"

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=112,
        timestamp_seconds=3.7333,
        signals=shot_frame_signals(frame(5)),
    )

    assert resolved is None
    assert next_event_id == 2
    assert flush_open_shot_event(candidate, frame_index=152, timestamp_seconds=5.0667) is None


def test_made_shot_requires_reset_before_opening_new_event():
    candidate = None
    next_event_id = 1

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=100,
        timestamp_seconds=4.0,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=130,
        timestamp_seconds=5.2,
        signals=shot_frame_signals(frame(1)),
    )
    assert resolved is not None
    assert resolved["outcome"] == "made"

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=170,
        timestamp_seconds=7.3,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None
    assert next_event_id == 2

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=171,
        timestamp_seconds=7.4,
        signals=shot_frame_signals(frame()),
    )
    assert resolved is None

    candidate, resolved, next_event_id = update_shot_event_candidate(
        candidate,
        next_event_id=next_event_id,
        frame_index=180,
        timestamp_seconds=7.8,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None
    assert candidate is not None
    assert candidate.event_id == 2
    assert candidate.start_frame_index == 180
    assert next_event_id == 3


def test_open_shot_flushes_as_unknown_at_end_of_clip():
    candidate, resolved, _ = update_shot_event_candidate(
        None,
        next_event_id=1,
        frame_index=100,
        timestamp_seconds=4.0,
        signals=shot_frame_signals(frame(5)),
    )
    assert resolved is None

    final_event = flush_open_shot_event(candidate, frame_index=100, timestamp_seconds=4.0)

    assert final_event is not None
    assert final_event["outcome"] == "unknown"
    assert final_event["resolution_reason"] == "end_of_clip"


def test_shot_signal_trace_record_reports_active_and_resolved_states():
    candidate, resolved, _ = update_shot_event_candidate(
        None,
        next_event_id=1,
        frame_index=100,
        timestamp_seconds=4.0,
        signals=shot_frame_signals(frame(5)),
    )
    active_trace = shot_signal_trace_record(
        frame_index=100,
        timestamp_seconds=4.0,
        signals=shot_frame_signals(frame(5)),
        candidate=candidate,
        resolved_event=resolved,
    )
    assert active_trace == {
        "frame_index": 100,
        "timestamp_seconds": 4.0,
        "has_jump_shot": True,
        "has_ball_in_basket": False,
        "has_layup_dunk": False,
        "shot_type_signal": "jump_shot",
        "event_state": "active",
        "event_id": 1,
        "reason": "tracking_shot_signal",
    }

    candidate, resolved, _ = update_shot_event_candidate(
        candidate,
        next_event_id=2,
        frame_index=130,
        timestamp_seconds=5.2,
        signals=shot_frame_signals(frame(1)),
    )
    resolved_trace = shot_signal_trace_record(
        frame_index=130,
        timestamp_seconds=5.2,
        signals=shot_frame_signals(frame(1)),
        candidate=candidate,
        resolved_event=resolved,
    )
    assert resolved_trace == {
        "frame_index": 130,
        "timestamp_seconds": 5.2,
        "has_jump_shot": False,
        "has_ball_in_basket": True,
        "has_layup_dunk": False,
        "shot_type_signal": None,
        "event_state": "cooldown",
        "reason": "resolved_made",
    }


def test_run_pipeline_writes_shot_events_and_versions_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import numpy as np
    from pipeline.runner import run_pipeline

    segments = [
        Segment(frame_indices=[0, 1], frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)], is_last=False),
        Segment(frame_indices=[2, 3], frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)], is_last=True),
    ]
    tracking_detector_results = [
        {
            "image": {"width": 16, "height": 16},
            "predictions": [
                {"class": "player", "confidence": 0.9, "x": 8, "y": 8, "width": 4, "height": 4},
            ]
        },
        {
            "image": {"width": 16, "height": 16},
            "predictions": [
                {"class": "player", "confidence": 0.9, "x": 8, "y": 8, "width": 4, "height": 4},
            ]
        },
    ]
    shot_detector_results = [
        {
            "image": {"width": 16, "height": 16},
            "predictions": [{"class_id": 5, "class": "jump_shot"}],
        },
        {
            "image": {"width": 16, "height": 16},
            "predictions": [{"class_id": 1, "class": "ball_in_basket"}],
        },
    ]
    class FakeDetector:
        def __init__(self, results):
            self._results = results
            self.call_count = 0

        def detect(self, _frame):
            result = self._results[self.call_count]
            self.call_count += 1
            return result

    class FakeTracker:
        device = "cpu"

        def propagate_segment(self, frames, seed_rows):
            assert len(frames) == 2
            return {1: [{**dict(row), "track_state": "propagated"} for row in seed_rows]}

    class FakeEmbedder:
        name = "none"

    class FakeTeamEmbedder:
        name = "none"

        def embed(self, _frame, samples):
            from pipeline.team_embedder import TeamFeatures

            return [TeamFeatures(color_signature=None, embedding=None) for _ in samples]

    class FakeNumberPredictor:
        name = "dummy"

        def predict(self, _sample):
            from pipeline.number_predictor import NumberPrediction

            return NumberPrediction(raw_text=None, score=None)

    requested_model_ids: list[str] = []
    iter_segment_calls: list[dict] = []

    def fake_build_detector(config, *, model_id=None):
        requested = model_id or config.player_model_id
        requested_model_ids.append(requested)
        if requested == config.shot_model_id:
            results = shot_detector_results
        else:
            results = tracking_detector_results
        return FakeDetector(results)

    monkeypatch.setattr(
        "pipeline.runner.probe",
        lambda _path: VideoInfo(fps=30.0, frame_count=4, width=16, height=16),
    )
    def fake_iter_segments(*_args, **kwargs):
        iter_segment_calls.append(dict(kwargs))
        return iter(segments)

    monkeypatch.setattr("pipeline.runner.iter_segments", fake_iter_segments)
    monkeypatch.setattr("pipeline.runner.build_detector", fake_build_detector)
    monkeypatch.setattr(
        "pipeline.runner.detections_from_result",
        lambda result, **_kwargs: [
            {
                "class_name": prediction["class"],
                "confidence": prediction.get("confidence", 1.0),
                "xyxy": [6, 6, 10, 10],
            }
            for prediction in result["predictions"]
            if prediction.get("class") == "player"
        ],
    )
    monkeypatch.setattr("pipeline.runner.detection_rows", lambda rows: [dict(row) for row in rows])
    monkeypatch.setattr("pipeline.runner.build_embedder", lambda _backend: FakeEmbedder())
    monkeypatch.setattr("pipeline.runner.build_team_embedder", lambda *_args, **_kwargs: FakeTeamEmbedder())
    monkeypatch.setattr("pipeline.runner.build_number_predictor", lambda _config: FakeNumberPredictor())

    config = PipelineConfig(
        roboflow_api_key="test-key",
        appearance_weight=0.0,
        reid_backend="none",
        team_embedder_backend="none",
        min_team_samples_per_track=1,
        team_sample_stride=1,
        max_team_samples_per_track=1,
        prefetch_detections=False,
        team_debug_enabled=False,
        number_predictor_backend="dummy",
    )

    results = run_pipeline(tmp_path / "dummy.mp4", config, tmp_path / "out", tracker=FakeTracker())

    assert results.schema_version == 5
    assert results.shot_events_file == "shot_events.jsonl"
    assert results.shot_evidence_file == "shot_evidence.jsonl"
    assert requested_model_ids == [
        "basketball-player-detection-3-ycjdo/16",
        config.shot_model_id,
    ]

    results_payload = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert results_payload["schema_version"] == 5
    assert results_payload["shot_events_file"] == "shot_events.jsonl"
    assert results_payload["shot_evidence_file"] == "shot_evidence.jsonl"
    assert results_payload["ball_provenance"]["backend"] == "hosted_detector"
    assert results_payload["ball_provenance"]["validation"]["valid"] is False
    assert results_payload["ball_provenance"]["validation"]["metadata"]["missing_frame_count"] == 2
    assert results_payload["config"]["player_model_id"] == "basketball-player-detection-3-ycjdo/16"
    assert results_payload["config"]["shot_model_id"] == config.shot_model_id
    assert results_payload["config"]["shot_target_fps"] is None
    assert results_payload["config"]["shot_frame_step"] == 2
    assert results_payload["config"]["shot_frame_stride"] == 1
    assert [call["frame_step"] for call in iter_segment_calls] == [2, 2]

    shot_events = [
        json.loads(line)
        for line in (tmp_path / "out" / "shot_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    shot_signals = [
        json.loads(line)
        for line in (tmp_path / "out" / "shot_signals.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    shot_raw_predictions = [
        json.loads(line)
        for line in (tmp_path / "out" / "shot_raw_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert shot_events == [
        {
            "event_id": 1,
            "shot_type": "jump_shot",
            "outcome": "made",
            "start_frame_index": 0,
            "end_frame_index": 2,
            "resolved_frame_index": 2,
            "start_timestamp_seconds": 0.0,
            "end_timestamp_seconds": 2 / 30,
            "resolved_timestamp_seconds": 2 / 30,
            "ball_in_basket_frame_index": 2,
            "ball_in_basket_timestamp_seconds": 2 / 30,
            "start_image_xy": None,
            "start_shot_xyxy": None,
            "resolution_reason": "ball_in_basket",
        }
    ]
    shot_evidence = [
        json.loads(line)
        for line in (tmp_path / "out" / "shot_evidence.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert shot_evidence[0]["ball_evidence"]["status"] == "missing"
    assert shot_evidence[0]["ball_evidence"]["coverage_status"] == "missing"
    assert shot_evidence[0]["player_identity"]["status"] == "unknown"
    assert shot_signals == [
        {
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "has_jump_shot": True,
            "has_ball_in_basket": False,
            "has_layup_dunk": False,
            "shot_type_signal": "jump_shot",
            "event_state": "active",
            "event_id": 1,
            "reason": "tracking_shot_signal",
        },
        {
            "frame_index": 2,
            "timestamp_seconds": 2 / 30,
            "has_jump_shot": False,
            "has_ball_in_basket": True,
            "has_layup_dunk": False,
            "shot_type_signal": None,
            "event_state": "cooldown",
            "reason": "resolved_made",
        },
    ]
    assert shot_raw_predictions == [
        {
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "model_id": config.shot_model_id,
            "predictions": [{"class_id": 5, "class": "jump_shot"}],
        },
        {
            "frame_index": 2,
            "timestamp_seconds": 2 / 30,
            "model_id": config.shot_model_id,
            "predictions": [{"class_id": 1, "class": "ball_in_basket"}],
        },
    ]


def test_run_pipeline_uses_independent_shot_target_fps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import numpy as np
    from pipeline.runner import run_pipeline

    segments = [
        Segment(frame_indices=[0, 2], frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)], is_last=False),
        Segment(frame_indices=[2, 4], frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)], is_last=True),
    ]
    tracking_detector_results = [
        {
            "image": {"width": 16, "height": 16},
            "predictions": [
                {"class": "player", "confidence": 0.9, "x": 8, "y": 8, "width": 4, "height": 4},
            ]
        },
        {
            "image": {"width": 16, "height": 16},
            "predictions": [
                {"class": "player", "confidence": 0.9, "x": 8, "y": 8, "width": 4, "height": 4},
            ]
        },
    ]
    shot_detector_results = [
        {
            "image": {"width": 16, "height": 16},
            "predictions": [{"class_id": 5, "class": "jump_shot"}],
        },
        {
            "image": {"width": 16, "height": 16},
            "predictions": [{"class_id": 1, "class": "ball_in_basket"}],
        },
    ]

    class FakeDetector:
        def __init__(self, results):
            self._results = results
            self.call_count = 0

        def detect(self, _frame):
            result = self._results[self.call_count]
            self.call_count += 1
            return result

    class FakeTracker:
        device = "cpu"

        def propagate_segment(self, frames, seed_rows):
            assert len(frames) == 2
            return {1: [{**dict(row), "track_state": "propagated"} for row in seed_rows]}

    class FakeEmbedder:
        name = "none"

    class FakeTeamEmbedder:
        name = "none"

        def embed(self, _frame, samples):
            from pipeline.team_embedder import TeamFeatures

            return [TeamFeatures(color_signature=None, embedding=None) for _ in samples]

    class FakeNumberPredictor:
        name = "dummy"

        def predict(self, _sample):
            from pipeline.number_predictor import NumberPrediction

            return NumberPrediction(raw_text=None, score=None)

    iter_segment_calls: list[dict] = []

    def fake_build_detector(config, *, model_id=None):
        requested = model_id or config.player_model_id
        results = shot_detector_results if requested == config.shot_model_id else tracking_detector_results
        return FakeDetector(results)

    def fake_iter_segments(*_args, **kwargs):
        iter_segment_calls.append(dict(kwargs))
        return iter(segments)

    monkeypatch.setattr(
        "pipeline.runner.probe",
        lambda _path: VideoInfo(fps=30.0, frame_count=5, width=16, height=16),
    )
    monkeypatch.setattr("pipeline.runner.iter_segments", fake_iter_segments)
    monkeypatch.setattr("pipeline.runner.build_detector", fake_build_detector)
    monkeypatch.setattr(
        "pipeline.runner.detections_from_result",
        lambda result, **_kwargs: [
            {
                "class_name": prediction["class"],
                "confidence": prediction.get("confidence", 1.0),
                "xyxy": [6, 6, 10, 10],
            }
            for prediction in result["predictions"]
            if prediction.get("class") == "player"
        ],
    )
    monkeypatch.setattr("pipeline.runner.detection_rows", lambda rows: [dict(row) for row in rows])
    monkeypatch.setattr("pipeline.runner.build_embedder", lambda _backend: FakeEmbedder())
    monkeypatch.setattr("pipeline.runner.build_team_embedder", lambda *_args, **_kwargs: FakeTeamEmbedder())
    monkeypatch.setattr("pipeline.runner.build_number_predictor", lambda _config: FakeNumberPredictor())

    config = PipelineConfig(
        roboflow_api_key="test-key",
        target_fps=15.0,
        shot_target_fps=7.5,
        appearance_weight=0.0,
        reid_backend="none",
        team_embedder_backend="none",
        min_team_samples_per_track=1,
        team_sample_stride=1,
        max_team_samples_per_track=1,
        prefetch_detections=False,
        team_debug_enabled=False,
        number_predictor_backend="dummy",
    )

    run_pipeline(tmp_path / "dummy.mp4", config, tmp_path / "out", tracker=FakeTracker())

    assert [call["frame_step"] for call in iter_segment_calls] == [2, 4]
    results_payload = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert results_payload["config"]["frame_step"] == 2
    assert results_payload["config"]["shot_target_fps"] == 7.5
    assert results_payload["config"]["shot_frame_step"] == 4


@pytest.mark.parametrize("homography_available", [True, False])
def test_run_pipeline_handles_projected_artifacts_based_on_homography_availability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, homography_available: bool
):
    import numpy as np
    from pipeline.runner import run_pipeline

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = [
        Segment(frame_indices=[0, 2], frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)], is_last=False),
        Segment(frame_indices=[2, 4], frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)], is_last=True),
    ]
    tracking_detector_results = [
        {
            "image": {"width": 16, "height": 16},
            "predictions": [
                {"class": "player", "confidence": 0.9, "x": 8, "y": 8, "width": 4, "height": 4},
            ],
        },
        {
            "image": {"width": 16, "height": 16},
            "predictions": [
                {"class": "player", "confidence": 0.9, "x": 8, "y": 8, "width": 4, "height": 4},
            ],
        },
    ]
    shot_detector_results = [
        {
            "image": {"width": 16, "height": 16},
            "predictions": [
                {"class_id": 5, "class": "jump_shot", "x": 8, "y": 8, "width": 4, "height": 4},
            ],
        },
        {
            "image": {"width": 16, "height": 16},
            "predictions": [{"class_id": 1, "class": "ball_in_basket"}],
        },
    ]
    court_detector_results = [
        {"predictions": [{"class": "09", "x": 1, "y": 1, "confidence": 0.9}]},
        {"predictions": [{"class": "09", "x": 1, "y": 1, "confidence": 0.9}]},
    ]

    class FakeDetector:
        def __init__(self, results):
            self._results = results
            self.call_count = 0

        def detect(self, _frame):
            result = self._results[self.call_count]
            self.call_count += 1
            return result

    class FakeTracker:
        device = "cpu"

        def propagate_segment(self, frames, seed_rows):
            assert len(frames) == 2
            return {1: [{**dict(row), "track_state": "propagated"} for row in seed_rows]}

    class FakeEmbedder:
        name = "none"

    class FakeTeamEmbedder:
        name = "none"

        def embed(self, _frame, samples):
            from pipeline.team_embedder import TeamFeatures

            return [TeamFeatures(color_signature=None, embedding=None) for _ in samples]

    class FakeNumberPredictor:
        name = "dummy"

        def predict(self, _sample):
            from pipeline.number_predictor import NumberPrediction

            return NumberPrediction(raw_text=None, score=None)

    def fake_build_detector(config, *, model_id=None):
        requested = model_id or config.player_model_id
        if requested == config.shot_model_id:
            results = shot_detector_results
        elif requested == config.court_model_id:
            results = court_detector_results
        else:
            results = tracking_detector_results
        return FakeDetector(results)

    monkeypatch.setattr(
        "pipeline.runner.probe",
        lambda _path: VideoInfo(fps=30.0, frame_count=5, width=16, height=16),
    )
    monkeypatch.setattr("pipeline.runner.iter_segments", lambda *_args, **_kwargs: iter(segments))
    monkeypatch.setattr("pipeline.runner.build_detector", fake_build_detector)
    monkeypatch.setattr(
        "pipeline.runner.detections_from_result",
        lambda result, **_kwargs: [
            {
                "class_name": prediction["class"],
                "confidence": prediction.get("confidence", 1.0),
                "xyxy": [6, 6, 10, 10],
            }
            for prediction in result["predictions"]
            if prediction.get("class") == "player"
        ],
    )
    monkeypatch.setattr("pipeline.runner.detection_rows", lambda rows: [dict(row) for row in rows])
    monkeypatch.setattr("pipeline.runner.build_embedder", lambda _backend: FakeEmbedder())
    monkeypatch.setattr("pipeline.runner.build_team_embedder", lambda *_args, **_kwargs: FakeTeamEmbedder())
    monkeypatch.setattr("pipeline.runner.build_number_predictor", lambda _config: FakeNumberPredictor())
    monkeypatch.setattr(
        "pipeline.runner.parse_court_keypoints",
        lambda *_args, **_kwargs: [
            {"label": "01", "x": 0.0, "y": 0.0, "confidence": 0.9},
            {"label": "08", "x": 0.0, "y": 15.0, "confidence": 0.9},
            {"label": "34", "x": 28.0, "y": 0.0, "confidence": 0.9},
            {"label": "41", "x": 28.0, "y": 15.0, "confidence": 0.9},
        ],
    )

    class FakeHomography:
        matrix = "unused"
        inliers_count = 4
        reprojection_mean_m = 0.0

    monkeypatch.setattr(
        "pipeline.runner.compute_frame_homography",
        lambda *_args, **_kwargs: FakeHomography() if homography_available else None,
    )
    monkeypatch.setattr(
        "pipeline.runner.project_points",
        lambda _matrix, points: [[float(point[0]) - 3.5, float(point[1]) - 8.75] for point in points],
    )

    config = PipelineConfig(
        roboflow_api_key="test-key",
        court_model_id="court-model/1",
        appearance_weight=0.0,
        reid_backend="none",
        team_embedder_backend="none",
        min_team_samples_per_track=1,
        team_sample_stride=1,
        max_team_samples_per_track=1,
        prefetch_detections=False,
        team_debug_enabled=False,
        number_predictor_backend="dummy",
    )

    results = run_pipeline(tmp_path / "dummy.mp4", config, output_dir, tracker=FakeTracker())

    results_payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    if homography_available:
        assert results.projected_tracks_file == "projected_tracks.jsonl"
        assert results.projected_shots_file == "projected_shots.json"
        assert results.projected_shots_image_file == "projected_shots.png"
        projected_tracks = [
            json.loads(line)
            for line in (output_dir / "projected_tracks.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert projected_tracks[0]["frame_index"] == 0
        assert projected_tracks[0]["track_id"] == 1
        assert projected_tracks[0]["court_xy_m"] == [4.5, 1.25]
        projected_shots_payload = json.loads((output_dir / "projected_shots.json").read_text(encoding="utf-8"))
        assert projected_shots_payload["shots"][0]["matched_projected_frame_index"] == 0
        assert projected_shots_payload["shots"][0]["shooter_track_id"] == 1
        assert projected_shots_payload["shots"][0]["shot_origin_court_xy_m"] == [4.5, 1.25]
        assert (output_dir / "projected_shots.png").exists()
        assert results_payload["projected_tracks_file"] == "projected_tracks.jsonl"
        assert results_payload["projected_shots_file"] == "projected_shots.json"
        assert results_payload["projected_shots_image_file"] == "projected_shots.png"
    else:
        assert results.projected_tracks_file is None
        assert results.projected_shots_file is None
        assert results.projected_shots_image_file is None
        assert not (output_dir / "projected_tracks.jsonl").exists()
        assert not (output_dir / "projected_shots.json").exists()
        assert not (output_dir / "projected_shots.png").exists()
        assert "projected_tracks_file" not in results_payload
        assert "projected_shots_file" not in results_payload
        assert "projected_shots_image_file" not in results_payload
