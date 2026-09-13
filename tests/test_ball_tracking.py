from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.ball import (
    ball_track_record,
    candidate_to_ball_row,
    filter_ball_predictions,
    reconcile_ball_track_id,
    select_seed_ball_rows,
)
from pipeline.annotate import team_color
from pipeline.config import PipelineConfig
from pipeline.video import Segment, VideoInfo


def test_sam2_uses_distinct_object_ids_for_colliding_public_track_ids(monkeypatch):
    from pipeline.tracking import Sam2SegmentTracker

    added = []

    class FakePredictor:
        def init_state(self, **_kwargs):
            return object()

        def add_new_points_or_box(self, **kwargs):
            added.append(kwargs["obj_id"])

        def propagate_in_video(self, *_args, **_kwargs):
            masks = np.ones((2, 1, 2, 2), dtype=float)
            yield 1, added, masks

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return nullcontext()

    tracker = Sam2SegmentTracker.__new__(Sam2SegmentTracker)
    tracker._predictor = FakePredictor()
    tracker._torch = FakeTorch()
    tracker._autocast = nullcontext()
    tracker._offload = True
    monkeypatch.setattr(tracker, "_write_segment_frames", lambda *_args: None)

    propagated = tracker.propagate_segment(
        [object(), object()],
        [
            {"track_id": 1, "class_name": "player", "confidence": 0.9, "xyxy": [0, 0, 2, 2]},
            {"track_id": 1, "class_name": "ball", "confidence": 0.8, "xyxy": [3, 3, 5, 5]},
        ],
    )

    assert len(added) == 2
    assert len(set(added)) == 2
    assert [(row["class_name"], row["track_id"]) for row in propagated[1]] == [
        ("player", 1),
        ("ball", 1),
    ]


def test_filter_ball_predictions_matches_ball_aliases_case_insensitively():
    result = {
        "predictions": [
            {"class": "Basketball", "confidence": 0.9},
            {"class": " player ", "confidence": 0.8},
            {"class": "BALL", "confidence": 0.7},
        ]
    }

    filtered = filter_ball_predictions(result, ("basketball", "ball"))

    assert [prediction["class"] for prediction in filtered["predictions"]] == ["Basketball", "BALL"]


def test_select_seed_ball_rows_prefers_detection_overlapping_active_ball():
    active_rows = [{"track_id": 1, "xyxy": [10, 10, 20, 20], "class_name": "ball"}]
    rows = [
        {"xyxy": [12, 12, 22, 22], "confidence": 0.6, "class_name": "ball"},
        {"xyxy": [40, 40, 50, 50], "confidence": 0.99, "class_name": "ball"},
    ]

    chosen = select_seed_ball_rows(rows, active_rows=active_rows, min_match_iou=0.1)

    assert chosen == [rows[0]]


def test_select_seed_ball_rows_uses_highest_confidence_without_active_ball():
    rows = [
        {"xyxy": [10, 10, 20, 20], "confidence": 0.7, "class_name": "ball"},
        {"xyxy": [50, 50, 60, 60], "confidence": 0.95, "class_name": "ball"},
    ]

    chosen = select_seed_ball_rows(rows, active_rows=[], min_match_iou=0.1)

    assert chosen == [rows[1]]


def test_tracknet_temporal_buffer_requires_eight_frames():
    from scripts.ball.tracknet import TemporalFrameBuffer

    buffer = TemporalFrameBuffer(8)
    for index in range(7):
        buffer.push(index, np.zeros((2, 2, 3), dtype=np.uint8))
    assert not buffer.ready
    buffer.push(7, np.zeros((2, 2, 3), dtype=np.uint8))
    assert buffer.ready
    assert buffer.frame_indices() == tuple(range(8))


def test_tracknet_candidate_maps_to_clipped_conservative_box():
    candidate = type("Candidate", (), {"x": 3.0, "y": 4.0, "score": 0.9})()

    row = candidate_to_ball_row(candidate, frame_size=(20, 16), radius=6)

    assert row["xyxy"] == [0.0, 0.0, 9.0, 10.0]
    assert row["confidence"] == 0.9


def test_tracknet_reseed_keeps_recent_public_ball_id():
    rows, next_id = reconcile_ball_track_id(
        [{"xyxy": [12, 12, 20, 20]}],
        [{"track_id": 4, "xyxy": [10, 10, 18, 18]}],
        5,
    )

    assert rows[0]["track_id"] == 4
    assert next_id == 5


def test_ball_track_record_preserves_row_and_adds_center():
    record = ball_track_record(
        4,
        {
            "track_id": 7,
            "track_state": "propagated",
            "class_name": "ball",
            "confidence": 0.8,
            "xyxy": [2, 4, 6, 8],
        },
    )

    assert record == {
        "frame_index": 4,
        "timestamp_seconds": None,
        "track_id": 7,
        "track_state": "propagated",
        "class_name": "ball",
        "confidence": 0.8,
        "xyxy": [2.0, 4.0, 6.0, 8.0],
        "center_xy": [4.0, 6.0],
    }


def test_dense_tracknet_stream_uses_ball_cadence_and_sam2_fallback(tmp_path: Path, monkeypatch):
    from pipeline.runner import _collect_tracknet_rows
    from scripts.ball.tracknet import BallHeatmapPrediction

    segments = [
        Segment(
            frame_indices=list(range(6)),
            frames=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(6)],
            is_last=True,
        )
    ]
    calls = []
    monkeypatch.setattr(
        "pipeline.runner.iter_segments",
        lambda _path, **kwargs: calls.append(kwargs) or iter(segments),
    )

    class TrackNet:
        def predict(self, _frames, frame_indices):
            heatmap = np.zeros((2, 2), dtype=np.float32)
            if frame_indices[-1] == 0:
                heatmap[0, 0] = 0.9
            return BallHeatmapPrediction(
                heatmap=heatmap,
                input_size=(512, 288),
                source_frame_size=(8, 8),
                frame_indices=tuple(frame_indices),
                backend_name="fake",
            )

    class Sam2:
        def propagate_segment(self, _frames, seed_rows):
            return {1: [{**seed_rows[0], "track_state": "fallback"}]}

    config = PipelineConfig(
        roboflow_api_key="test-key",
        target_fps=15,
        ball_target_fps=30,
        tracknet_fallback_budget=5,
    )
    output = tmp_path / "ball.jsonl"
    count = _collect_tracknet_rows(
        tmp_path / "clip.mp4",
        info=VideoInfo(30.0, 6, 8, 8),
        config=config,
        detector=TrackNet(),
        tracker=Sam2(),
        player_step=2,
        output_path=output,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == 6
    assert calls[0]["frame_step"] == 1  # 30 FPS ball stream; player step is 2.
    assert calls[0]["max_frames"] is None  # full clip when no player max is configured
    assert [row["frame_index"] for row in rows] == list(range(6))
    assert rows[0]["track_state"] == "detected"
    assert [row["track_state"] for row in rows[1:]] == ["fallback"] * 5
    assert rows[3]["timestamp_seconds"] == pytest.approx(3 / 30)


@pytest.mark.parametrize("source_fps", [30.0, 60.0])
def test_tracknet_all_frame_mode_ignores_declared_fps_and_keeps_compatibility(
    tmp_path: Path, monkeypatch, source_fps: float
):
    from pipeline.runner import _collect_tracknet_rows
    from scripts.ball.tracknet import BallHeatmapPrediction

    frame_count = 12
    segments_seen = []

    def fake_segments(_path, **kwargs):
        segments_seen.append(kwargs)
        indices = list(range(0, frame_count, kwargs["frame_step"]))
        return iter(
            [
                Segment(
                    frame_indices=indices,
                    frames=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in indices],
                    is_last=True,
                )
            ]
        )

    class TrackNet:
        def __init__(self):
            self.calls = []

        def predict(self, _frames, frame_indices):
            self.calls.append(tuple(frame_indices))
            return BallHeatmapPrediction(
                heatmap=np.zeros((2, 2), dtype=np.float32),
                input_size=(512, 288),
                source_frame_size=(8, 8),
                frame_indices=tuple(frame_indices),
                backend_name="fake",
            )

    class Tracker:
        def propagate_segment(self, *_args):
            return {}

    monkeypatch.setattr("pipeline.runner.iter_segments", fake_segments)
    detector = TrackNet()
    output = tmp_path / "ball.jsonl"
    config = PipelineConfig(
        roboflow_api_key="test-key",
        ball_all_frames=True,
        ball_target_fps=1.0,
    )

    _collect_tracknet_rows(
        tmp_path / "clip.mp4",
        info=VideoInfo(source_fps, frame_count, 8, 8),
        config=config,
        detector=detector,
        tracker=Tracker(),
        player_step=2,
        output_path=output,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["frame_index"] for row in rows] == list(range(frame_count))
    assert len(detector.calls) == frame_count
    assert segments_seen[0]["frame_step"] == 1
    assert [row["timestamp_seconds"] for row in rows] == pytest.approx(
        [index / source_fps for index in range(frame_count)]
    )

    output = tmp_path / "compatibility.jsonl"
    config = PipelineConfig(
        roboflow_api_key="test-key",
        ball_all_frames=False,
        ball_target_fps=source_fps / 2,
    )
    segments_seen.clear()
    detector.calls.clear()
    _collect_tracknet_rows(
        tmp_path / "clip.mp4",
        info=VideoInfo(source_fps, frame_count, 8, 8),
        config=config,
        detector=detector,
        tracker=Tracker(),
        player_step=2,
        output_path=output,
    )
    assert segments_seen[0]["frame_step"] == 2
    assert [row["frame_index"] for row in json.loads("[" + ",".join(output.read_text().splitlines()) + "]")] == list(range(0, frame_count, 2))


def test_tracknet_lost_row_suppresses_sam2_fallback_at_same_frame(tmp_path: Path):
    from pipeline.runner import _BallRecordReader

    path = tmp_path / "ball_tracks.jsonl"
    path.write_text(
        json.dumps(
            {
                "frame_index": 4,
                "track_id": 1,
                "track_state": "lost",
                "class_name": "ball",
                "confidence": None,
                "xyxy": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reader = _BallRecordReader(path)
    try:
        present, authoritative_rows = reader.take(4)
    finally:
        reader.close()

    sam2_fallback = [{"frame_index": 4, "track_state": "fallback", "xyxy": [1, 1, 2, 2]}]
    merged_rows = authoritative_rows if present else sam2_fallback
    assert present is True
    assert authoritative_rows == []
    assert merged_rows == []


def test_annotated_frame_rows_preserves_players_and_appends_ball_rows():
    from pipeline.runner import _annotated_frame_rows

    player_rows = [
        {"track_id": 1, "class_name": "player", "xyxy": [0, 0, 10, 10], "confidence": 0.9},
    ]
    ball_rows = [
        {"track_id": 7, "class_name": "ball", "xyxy": [2, 2, 4, 4], "confidence": 0.8},
    ]

    rows = _annotated_frame_rows(
        player_rows,
        confirmed_numbers_by_track_id={1: "23"},
        ball_rows=ball_rows,
    )

    assert [row["class_name"] for row in rows] == ["player", "ball"]
    assert rows[0]["confirmed_number"] == "23"
    assert "confirmed_number" not in rows[1]
    assert "confirmed_number" not in player_rows[0]


def test_render_annotated_video_combines_player_and_ball_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from pipeline.runner import _iter_track_rows_by_frame, _render_annotated_video

    tracks_path = tmp_path / "tracks.jsonl"
    tracks_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "track_id": 1,
                "track_state": "detected",
                "class_name": "player",
                "confidence": 0.9,
                "xyxy": [0, 0, 10, 20],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ball_tracks_path = tmp_path / "ball_tracks.jsonl"
    ball_tracks_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "track_id": 1,
                "track_state": "detected",
                "class_name": "ball",
                "confidence": 0.8,
                "xyxy": [2, 2, 6, 6],
                "center_xy": [4.0, 4.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    numbers_path = tmp_path / "numbers.json"
    numbers_path.write_text(
        json.dumps({"tracks": [{"track_id": 1, "status": "confirmed", "number": "23"}]}),
        encoding="utf-8",
    )

    rendered_rows = []

    class FakeAnnotatedVideoWriter:
        def __init__(self, output_path, info, fps):
            del output_path, info, fps

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, _frame):
            return None

    def fake_draw_tracks(frame, rows):
        rendered_rows.append([dict(row) for row in rows])
        return frame

    segment = Segment(
        frame_indices=[0],
        frames=[np.zeros((16, 16, 3), dtype=np.uint8)],
        is_last=True,
    )

    monkeypatch.setattr("pipeline.runner.iter_segments", lambda *_args, **_kwargs: iter([segment]))
    monkeypatch.setattr("pipeline.runner.AnnotatedVideoWriter", FakeAnnotatedVideoWriter)
    monkeypatch.setattr("pipeline.annotate.draw_tracks", fake_draw_tracks)

    _render_annotated_video(
        tmp_path / "dummy.mp4",
        tracks_path,
        numbers_path,
        tmp_path / "annotated.mp4",
        info=VideoInfo(fps=30.0, frame_count=1, width=16, height=16),
        frame_stride=1,
        max_frames=None,
        frame_step=1,
        fps=30.0,
        ball_tracks_path=ball_tracks_path,
    )

    assert rendered_rows
    assert [row["class_name"] for row in rendered_rows[0]] == ["player", "ball"]
    assert rendered_rows[0][0]["confirmed_number"] == "23"
    assert "confirmed_number" not in rendered_rows[0][1]


def test_render_annotated_video_reads_ball_rows_forward_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from pipeline.runner import _iter_track_rows_by_frame, _render_annotated_video

    tracks_path = tmp_path / "tracks.jsonl"
    tracks_path.write_text(
        json.dumps({"frame_index": 0, "track_id": 1, "class_name": "player", "confidence": 0.9, "track_state": "detected", "xyxy": [0, 0, 2, 2]})
        + "\n",
        encoding="utf-8",
    )
    ball_tracks_path = tmp_path / "ball_tracks.jsonl"
    ball_tracks_path.write_text("placeholder\n", encoding="utf-8")
    rendered = False
    consumed = 0

    def bounded_ball_rows(_path):
        nonlocal consumed
        for frame_index in range(1000):
            consumed += 1
            if not rendered and consumed > 2:
                raise AssertionError("renderer consumed the full dense ball stream before writing")
            yield frame_index, [{
                "frame_index": frame_index,
                "class_name": "ball",
                "track_id": 2,
                "confidence": 0.8,
                "track_state": "detected",
                "xyxy": [1, 1, 2, 2],
            }]

    def fake_rows(path):
        return bounded_ball_rows(path) if path == ball_tracks_path else _iter_track_rows_by_frame(path)

    class FakeWriter:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, _frame):
            nonlocal rendered
            rendered = True

    segment = Segment(frame_indices=[0], frames=[np.zeros((4, 4, 3), dtype=np.uint8)], is_last=True)
    monkeypatch.setattr("pipeline.runner._iter_track_rows_by_frame", fake_rows)
    monkeypatch.setattr("pipeline.runner.iter_segments", lambda *_args, **_kwargs: iter([segment]))
    monkeypatch.setattr("pipeline.runner.AnnotatedVideoWriter", FakeWriter)

    _render_annotated_video(
        tmp_path / "dummy.mp4",
        tracks_path,
        tmp_path / "numbers.json",
        tmp_path / "annotated.mp4",
        info=VideoInfo(fps=30.0, frame_count=1000, width=4, height=4),
        frame_stride=1,
        max_frames=None,
        frame_step=1,
        fps=30.0,
        ball_tracks_path=ball_tracks_path,
    )

    assert consumed == 2


def test_render_annotated_video_emits_only_player_frames_and_exact_ball_join(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from pipeline.runner import _render_annotated_video

    tracks_path = tmp_path / "tracks.jsonl"
    tracks_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "frame_index": index,
                    "track_id": 1,
                    "track_state": "detected",
                    "class_name": "player",
                    "confidence": 0.9,
                    "xyxy": [0, 0, 2, 2],
                }
            )
            for index in (0, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    ball_tracks_path = tmp_path / "ball_tracks.jsonl"
    ball_tracks_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "frame_index": index,
                    "track_id": 2,
                    "track_state": "detected",
                    "class_name": "ball",
                    "confidence": 0.8,
                    "xyxy": [index, index, index + 1, index + 1],
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    rendered = []

    class FakeWriter:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, frame):
            rendered.append(frame)

    joined_rows = []
    monkeypatch.setattr("pipeline.runner.AnnotatedVideoWriter", FakeWriter)
    monkeypatch.setattr(
        "pipeline.annotate.draw_tracks",
        lambda frame, rows: joined_rows.append(rows) or frame,
    )
    monkeypatch.setattr(
        "pipeline.runner.iter_segments",
        lambda *_args, **_kwargs: iter(
            [
                Segment(
                    frame_indices=[0, 2],
                    frames=[np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)],
                    is_last=True,
                )
            ]
        ),
    )

    _render_annotated_video(
        tmp_path / "dummy.mp4",
        tracks_path,
        tmp_path / "numbers.json",
        tmp_path / "annotated.mp4",
        info=VideoInfo(60.0, 3, 4, 4),
        frame_stride=1,
        max_frames=None,
        frame_step=2,
        fps=30.0,
        ball_tracks_path=ball_tracks_path,
    )

    assert len(rendered) == 2
    assert [[row["frame_index"] for row in rows if row["class_name"] == "ball"] for rows in joined_rows] == [[0], [2]]


def test_render_annotated_video_renders_lost_ball_row_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from pipeline.runner import _render_annotated_video

    tracks_path = tmp_path / "tracks.jsonl"
    tracks_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "track_id": 1,
                "track_state": "detected",
                "class_name": "player",
                "confidence": 0.9,
                "xyxy": [0, 0, 10, 10],
                "team_id": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ball_tracks_path = tmp_path / "ball_tracks.jsonl"
    ball_tracks_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "track_id": 2,
                "track_state": "lost",
                "class_name": "ball",
                "confidence": None,
                "xyxy": None,
                "center_xy": None,
                "team_id": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    written_frames = []

    class FakeAnnotatedVideoWriter:
        def __init__(self, output_path, info, fps):
            del output_path, info, fps

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, frame):
            written_frames.append(frame)

    segment = Segment(
        frame_indices=[0],
        frames=[np.zeros((16, 16, 3), dtype=np.uint8)],
        is_last=True,
    )
    monkeypatch.setattr("pipeline.runner.iter_segments", lambda *_args, **_kwargs: iter([segment]))
    monkeypatch.setattr("pipeline.runner.AnnotatedVideoWriter", FakeAnnotatedVideoWriter)

    _render_annotated_video(
        tmp_path / "dummy.mp4",
        tracks_path,
        tmp_path / "numbers.json",
        tmp_path / "annotated.mp4",
        info=VideoInfo(fps=30.0, frame_count=1, width=16, height=16),
        frame_stride=1,
        max_frames=None,
        frame_step=1,
        fps=30.0,
        ball_tracks_path=ball_tracks_path,
    )

    assert len(written_frames) == 1
    assert tuple(int(value) for value in written_frames[0][0, 0]) == team_color(0)


def test_run_pipeline_writes_ball_tracks_from_same_detection_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
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
                {"class": "basketball", "confidence": 0.7, "x": 4, "y": 4, "width": 2, "height": 2},
            ],
        },
        {
            "image": {"width": 16, "height": 16},
            "predictions": [
                {"class": "player", "confidence": 0.9, "x": 9, "y": 8, "width": 4, "height": 4},
                {"class": "ball", "confidence": 0.8, "x": 5, "y": 4, "width": 2, "height": 2},
            ],
        },
    ]
    shot_detector_results = [
        {"image": {"width": 16, "height": 16}, "predictions": []},
        {"image": {"width": 16, "height": 16}, "predictions": []},
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
            return {
                1: [
                    {
                        **dict(row),
                        "track_state": "propagated",
                        "xyxy": [value + 1.0 for value in row["xyxy"]],
                    }
                    for row in seed_rows
                ]
            }

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
        results = shot_detector_results if requested == config.shot_model_id else tracking_detector_results
        return FakeDetector(results)

    def fake_detections_from_result(result, *, min_confidence=0.0, allowed_classes=None):
        rows = []
        for prediction in result.get("predictions", []):
            confidence = float(prediction.get("confidence", 1.0))
            if confidence < min_confidence:
                continue
            class_name = prediction.get("class", "player")
            if allowed_classes and class_name not in allowed_classes:
                continue
            x = float(prediction.get("x", 0.0))
            y = float(prediction.get("y", 0.0))
            width = float(prediction.get("width", 0.0))
            height = float(prediction.get("height", 0.0))
            rows.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "xyxy": [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0],
                }
            )
        return rows

    monkeypatch.setattr(
        "pipeline.runner.probe",
        lambda _path: VideoInfo(fps=30.0, frame_count=5, width=16, height=16),
    )
    monkeypatch.setattr("pipeline.runner.iter_segments", lambda *_args, **_kwargs: iter(segments))
    monkeypatch.setattr("pipeline.runner.build_detector", fake_build_detector)
    monkeypatch.setattr("pipeline.runner.detections_from_result", fake_detections_from_result)
    monkeypatch.setattr("pipeline.runner.detection_rows", lambda rows: [dict(row, track_id=None, track_state="detected") for row in rows])
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

    ball_tracks = [
        json.loads(line)
        for line in (tmp_path / "out" / "ball_tracks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results_payload = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))

    assert results.ball_tracks_file == "ball_tracks.jsonl"
    assert results_payload["ball_tracks_file"] == "ball_tracks.jsonl"
    assert [row["frame_index"] for row in ball_tracks] == [0, 2, 4]
    assert [row["track_id"] for row in ball_tracks] == [1, 1, 1]
    assert [row["track_state"] for row in ball_tracks] == ["detected", "detected", "propagated"]


def test_run_pipeline_tracknet_temporal_fallback_budget_loss_and_reseed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from pipeline.ball import TRACKNET_CONTEXT_SIZE
    from pipeline.runner import run_pipeline
    from scripts.ball.tracknet import BallHeatmapPrediction

    segments = [
        Segment(
            frame_indices=list(range(8)),
            frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(8)],
            is_last=False,
        ),
        Segment(
            frame_indices=list(range(7, 12)),
            frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)],
            is_last=True,
        ),
    ]
    player_segments = [
        Segment(
            frame_indices=[index, index + 2],
            frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)],
            is_last=False,
        )
        for index in range(0, 10, 2)
    ] + [Segment(frame_indices=[10], frames=[np.zeros((16, 16, 3), dtype=np.uint8)], is_last=True)]

    class FakeHostedDetector:
        call_count = 0

        def detect(self, _frame):
            self.call_count += 1
            return {"image": {"width": 16, "height": 16}, "predictions": []}

    class FakeTrackNet:
        def __init__(self):
            self.calls = []

        def predict(self, frames, frame_indices):
            assert len(frames) == TRACKNET_CONTEXT_SIZE
            assert len(frame_indices) == TRACKNET_CONTEXT_SIZE
            assert all(left <= right for left, right in zip(frame_indices, frame_indices[1:]))
            assert frame_indices[-1] <= len(self.calls)
            self.calls.append(tuple(frame_indices))
            heatmap = np.zeros((4, 4), dtype=np.float32)
            if len(self.calls) in {1, 3}:
                heatmap[1, 1] = 0.9
            return BallHeatmapPrediction(
                heatmap=heatmap,
                input_size=(512, 288),
                source_frame_size=(16, 16),
                frame_indices=tuple(frame_indices),
                backend_name="fake",
            )

    fake_tracknet = FakeTrackNet()

    class FakeTracker:
        device = "cpu"

        def propagate_segment(self, frames, seed_rows):
            return {
                offset: [
                        {
                            **row,
                            "track_state": (
                                "fallback"
                                if row.get("class_name") == "ball"
                                else "propagated"
                            ),
                    }
                    for row in seed_rows
                ]
                for offset in range(1, len(frames))
            }

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

    hosted_detectors = []

    def fake_build_detector(_config, *, model_id=None):
        detector = FakeHostedDetector()
        hosted_detectors.append((model_id, detector))
        return detector

    monkeypatch.setattr("pipeline.runner.probe", lambda _path: VideoInfo(30.0, 12, 16, 16))
    def fake_iter_segments(_path, **kwargs):
        return iter(segments if kwargs["frame_step"] == 1 else player_segments)

    monkeypatch.setattr("pipeline.runner.iter_segments", fake_iter_segments)
    monkeypatch.setattr("pipeline.runner.build_detector", fake_build_detector)
    monkeypatch.setattr(
        "pipeline.runner.build_tracknet_detector",
        lambda _path: pytest.fail("injected TrackNet detector must bypass construction"),
    )
    monkeypatch.setattr(
        "pipeline.runner.detections_from_result",
        lambda result, **_kwargs: [],
    )
    monkeypatch.setattr("pipeline.runner.detection_rows", lambda rows: [])
    monkeypatch.setattr("pipeline.runner.build_embedder", lambda _backend: FakeEmbedder())
    monkeypatch.setattr("pipeline.runner.build_team_embedder", lambda *_args, **_kwargs: FakeTeamEmbedder())
    monkeypatch.setattr("pipeline.runner.build_number_predictor", lambda _config: FakeNumberPredictor())

    config = PipelineConfig(
        roboflow_api_key="test-key",
        tracknet_enabled=True,
        tracknet_checkpoint="fake.pt",
        tracknet_seed_radius=2,
        tracknet_fallback_budget=2,
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

    run_pipeline(
        tmp_path / "dummy.mp4",
        config,
        tmp_path / "out",
        tracker=FakeTracker(),
        tracknet_detector=fake_tracknet,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "ball_tracks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(fake_tracknet.calls) == 12
    assert fake_tracknet.calls[0] == (0,) * TRACKNET_CONTEXT_SIZE
    from scripts.ball.tracknet import padded_window

    assert fake_tracknet.calls == [
        padded_window(range(12), position, TRACKNET_CONTEXT_SIZE) for position in range(12)
    ]
    assert len({row["frame_index"] for row in rows}) == len(rows)
    assert [row["frame_index"] for row in rows if row["track_state"] == "detected"] == [0, 2]
    assert [row["frame_index"] for row in rows if row["track_state"] == "fallback"] == [1, 3, 4]
    assert [row["frame_index"] for row in rows if row["track_state"] == "lost"] == [5, 6, 7, 8, 9, 10, 11]
    assert all(row["track_id"] == 1 for row in rows)
    assert rows[0]["xyxy"] == [4.0, 4.0, 8.0, 8.0]
    assert rows[-1]["xyxy"] is None and rows[-1]["center_xy"] is None
    assert sum(row["track_state"] == "fallback" for row in rows) == 3


def test_run_pipeline_keeps_dense_tracknet_lost_authoritative_over_sam2_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from pipeline.runner import run_pipeline
    from scripts.ball.tracknet import BallHeatmapPrediction

    dense_segment = Segment(
        frame_indices=[0, 1, 2],
        frames=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        is_last=True,
    )
    player_segment = Segment(
        frame_indices=[0, 2],
        frames=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)],
        is_last=True,
    )

    class FakeTrackNet:
        def predict(self, _frames, frame_indices):
            heatmap = np.zeros((2, 2), dtype=np.float32)
            if frame_indices[-1] == 0:
                heatmap[0, 0] = 0.9
            return BallHeatmapPrediction(
                heatmap=heatmap,
                input_size=(512, 288),
                source_frame_size=(8, 8),
                frame_indices=tuple(frame_indices),
                backend_name="fake",
            )

    class FakeHostedDetector:
        call_count = 0

        def detect(self, _frame):
            self.call_count += 1
            return {"image": {"width": 8, "height": 8}, "predictions": []}

    class FakeTracker:
        device = "cpu"

        def propagate_segment(self, _frames, seed_rows):
            return {1: [{**row, "track_state": "fallback"} for row in seed_rows]}

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

    monkeypatch.setattr("pipeline.runner.probe", lambda _path: VideoInfo(30.0, 3, 8, 8))
    monkeypatch.setattr(
        "pipeline.runner.iter_segments",
        lambda _path, **kwargs: iter([dense_segment if kwargs["frame_step"] == 1 else player_segment]),
    )
    monkeypatch.setattr("pipeline.runner.build_detector", lambda *_args, **_kwargs: FakeHostedDetector())
    monkeypatch.setattr("pipeline.runner.detections_from_result", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("pipeline.runner.detection_rows", lambda rows: [])
    monkeypatch.setattr("pipeline.runner.build_embedder", lambda _backend: FakeEmbedder())
    monkeypatch.setattr("pipeline.runner.build_team_embedder", lambda *_args, **_kwargs: FakeTeamEmbedder())
    monkeypatch.setattr("pipeline.runner.build_number_predictor", lambda _config: FakeNumberPredictor())

    rendered_rows = []

    class FakeAnnotatedVideoWriter:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, _frame):
            return None

    monkeypatch.setattr("pipeline.runner.AnnotatedVideoWriter", FakeAnnotatedVideoWriter)
    monkeypatch.setattr(
        "pipeline.annotate.draw_tracks",
        lambda frame, rows: rendered_rows.append([dict(row) for row in rows]) or frame,
    )

    config = PipelineConfig(
        roboflow_api_key="test-key",
        tracknet_enabled=True,
        tracknet_checkpoint="fake.pt",
        tracknet_fallback_budget=1,
        appearance_weight=0.0,
        reid_backend="none",
        team_embedder_backend="none",
        min_team_samples_per_track=1,
        team_sample_stride=1,
        max_team_samples_per_track=1,
        prefetch_detections=False,
        team_debug_enabled=False,
        number_predictor_backend="dummy",
        write_annotated_video=True,
    )

    run_pipeline(
        tmp_path / "dummy.mp4",
        config,
        tmp_path / "out",
        tracker=FakeTracker(),
        tracknet_detector=FakeTrackNet(),
    )

    ball_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "ball_tracks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["frame_index"] for row in ball_rows] == [0, 1, 2]
    assert len({row["frame_index"] for row in ball_rows}) == 3
    assert ball_rows[2]["track_state"] == "lost"
    assert ball_rows[2]["xyxy"] is None
    assert len(rendered_rows) == 2
    assert not [
        row
        for row in rendered_rows[1]
        if row.get("class_name") == "ball" and row.get("track_state") == "fallback"
    ]


def test_run_pipeline_tracknet_enabled_requires_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from pipeline.runner import run_pipeline

    monkeypatch.setattr("pipeline.runner.probe", lambda _path: VideoInfo(30.0, 1, 16, 16))
    config = PipelineConfig(roboflow_api_key="test-key", tracknet_enabled=True)

    with pytest.raises(RuntimeError, match="TRACKNET_CHECKPOINT"):
        run_pipeline(tmp_path / "dummy.mp4", config, tmp_path / "out", tracker=object())


def test_run_pipeline_tracknet_short_clip_uses_padded_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from pipeline.runner import run_pipeline
    from scripts.ball.tracknet import BallHeatmapPrediction, padded_window

    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    segments = [Segment(frame_indices=[0, 1, 2], frames=frames, is_last=True)]

    class FakeTrackNet:
        def __init__(self):
            self.calls = []

        def predict(self, window_frames, window_indices):
            self.calls.append((window_frames, tuple(window_indices)))
            return BallHeatmapPrediction(
                heatmap=np.zeros((2, 2), dtype=np.float32),
                input_size=(512, 288),
                source_frame_size=(8, 8),
                frame_indices=tuple(window_indices),
                backend_name="fake",
            )

    class FakeHostedDetector:
        call_count = 0

        def detect(self, _frame):
            self.call_count += 1
            return {"predictions": []}

    class FakeTracker:
        device = "cpu"

        def propagate_segment(self, _frames, _seed_rows):
            return {}

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

    tracknet = FakeTrackNet()
    monkeypatch.setattr("pipeline.runner.probe", lambda _path: VideoInfo(30.0, 3, 8, 8))
    monkeypatch.setattr("pipeline.runner.iter_segments", lambda *_args, **_kwargs: iter(segments))
    monkeypatch.setattr("pipeline.runner.build_detector", lambda *_args, **_kwargs: FakeHostedDetector())
    monkeypatch.setattr("pipeline.runner.detections_from_result", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("pipeline.runner.detection_rows", lambda rows: rows)
    monkeypatch.setattr("pipeline.runner.build_embedder", lambda _backend: type("E", (), {"name": "none"})())
    monkeypatch.setattr("pipeline.runner.build_team_embedder", lambda *_args, **_kwargs: FakeTeamEmbedder())
    monkeypatch.setattr("pipeline.runner.build_number_predictor", lambda _config: FakeNumberPredictor())

    config = PipelineConfig(
        roboflow_api_key="test-key",
        tracknet_enabled=True,
        tracknet_checkpoint="fake.pt",
        appearance_weight=0.0,
        reid_backend="none",
        team_embedder_backend="none",
        min_team_samples_per_track=1,
        team_sample_stride=1,
        max_team_samples_per_track=1,
        prefetch_detections=False,
        team_debug_enabled=False,
        number_predictor_backend="dummy",
        max_frames=3,
    )

    run_pipeline(
        tmp_path / "dummy.mp4",
        config,
        tmp_path / "out",
        tracker=FakeTracker(),
        tracknet_detector=tracknet,
    )

    assert [indices for _frames, indices in tracknet.calls] == [
        padded_window(range(3), position, 8) for position in range(3)
    ]
