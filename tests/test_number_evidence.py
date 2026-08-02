"""Jersey-number evidence tests.

    python -m pytest tests/test_number_evidence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.number_evidence import (  # noqa: E402
    NumberPredictionRecord,
    associate_number_rows,
    summarize_number_evidence,
    write_number_prediction_artifacts,
)
from pipeline.number_predictor import (  # noqa: E402
    NumberPrediction,
    build_number_predictor,
    normalize_number_prediction,
)
from pipeline.storage import R2Storage  # noqa: E402
from pipeline.video import Segment, VideoInfo  # noqa: E402


def _run_number_prediction_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    name: str,
    detections: list[dict],
    predict,
):
    import numpy as np
    from pipeline.runner import run_pipeline

    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    segment = Segment(frame_indices=[0, 1], frames=frames, is_last=True)

    class FakeDetector:
        def __init__(self, rows):
            self._rows = rows
            self.call_count = 0

        def detect(self, _frame):
            self.call_count += 1
            return [dict(row) for row in self._rows]

    class FakeTracker:
        device = "cpu"

        def propagate_segment(self, frames, seed_rows):
            assert len(frames) == 2
            return {1: [{**dict(row), "track_state": "propagated"} for row in seed_rows]}

    class FakeEmbedder:
        name = "none"

    class FakeTeamEmbedder:
        name = "colour-test"

        def embed(self, frame, samples):
            del frame
            from pipeline.team_embedder import TeamFeatures

            return [
                TeamFeatures(
                    color_signature=[1.0, 0.0] if sample.xyxy[0] < 10 else [0.0, 1.0],
                    embedding=None,
                )
                for sample in samples
            ]

    class FakeNumberPredictor:
        name = "dummy"

        def __init__(self):
            self.calls = []

        def predict(self, sample):
            self.calls.append(sample)
            return predict(sample)

    monkeypatch.setattr(
        "pipeline.runner.probe",
        lambda _path: VideoInfo(fps=30.0, frame_count=2, width=32, height=32),
    )
    monkeypatch.setattr("pipeline.runner.iter_segments", lambda *_args, **_kwargs: iter([segment]))
    monkeypatch.setattr("pipeline.runner.build_detector", lambda _config: FakeDetector(detections))
    monkeypatch.setattr("pipeline.runner.detections_from_result", lambda result, **_kwargs: result)
    monkeypatch.setattr(
        "pipeline.runner.detection_rows",
        lambda rows: [dict(row) for row in rows],
    )
    monkeypatch.setattr("pipeline.runner.build_embedder", lambda _backend: FakeEmbedder())
    monkeypatch.setattr(
        "pipeline.runner.build_team_embedder",
        lambda *_args, **_kwargs: FakeTeamEmbedder(),
    )
    predictor = FakeNumberPredictor()
    monkeypatch.setattr("pipeline.runner.build_number_predictor", lambda _backend: predictor)

    output_dir = tmp_path / name
    results = run_pipeline(
        video_path=tmp_path / "dummy.mp4",
        config=PipelineConfig(
            roboflow_api_key="test-key",
            appearance_weight=0.0,
            reid_backend="none",
            team_embedder_backend="colour",
            min_team_samples_per_track=1,
            team_sample_stride=1,
            max_team_samples_per_track=4,
            prefetch_detections=False,
            team_debug_enabled=False,
            number_predictor_backend="dummy",
        ),
        output_dir=output_dir,
        tracker=FakeTracker(),
    )

    assert results.summary.detector_calls == 1
    tracks = [json.loads(line) for line in (output_dir / "tracks.jsonl").read_text(encoding="utf-8").splitlines()]
    teams = json.loads((output_dir / "teams.json").read_text(encoding="utf-8"))
    results_payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    results_payload["summary"].pop("wall_seconds", None)
    predictions = json.loads((output_dir / "number_predictions.json").read_text(encoding="utf-8"))["predictions"]
    return tracks, teams, results_payload, predictions, predictor.calls


def test_associate_number_rows_matches_unique_best_overlaps():
    import numpy as np

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[10:20, 15:25] = 255
    frame[10:20, 75:85] = 127

    player_rows = [
        {"track_id": 11, "track_state": "detected", "confidence": 0.9, "xyxy": [0, 0, 50, 60]},
        {"track_id": 22, "track_state": "detected", "confidence": 0.8, "xyxy": [60, 0, 110, 60]},
    ]
    number_rows = [
        {"confidence": 0.7, "xyxy": [12, 8, 28, 24]},
        {"confidence": 0.6, "xyxy": [72, 8, 88, 24]},
        {"confidence": 0.5, "xyxy": [40, 10, 70, 30]},
    ]

    samples = associate_number_rows(frame, player_rows, number_rows, frame_index=5)

    assert [sample.track_id for sample in samples] == [11, 22]
    assert samples[0].contains_number_center is True
    assert samples[0].crop_image.shape == (16, 16, 3)
    assert samples[1].number_confidence == pytest.approx(0.6)


def test_summarize_number_evidence_counts_samples():
    import numpy as np

    samples = associate_number_rows(
        np.zeros((40, 40, 3), dtype=np.uint8),
        [{"track_id": 7, "track_state": "detected", "confidence": 0.9, "xyxy": [0, 0, 30, 30]}],
        [{"confidence": 0.8, "xyxy": [5, 5, 15, 15]}],
        frame_index=9,
    )

    summary = summarize_number_evidence(samples)

    assert summary["sample_count"] == 1
    assert summary["mean_overlap_ratio"] == 1.0
    assert summary["samples"][0]["frame_index"] == 9


def test_normalize_number_prediction_preserves_complete_confident_digits():
    accepted = ["0", "00", "23", "123"]

    assert [
        normalize_number_prediction(NumberPrediction(raw_text=value, is_confident=True))
        for value in accepted
    ] == [
        ("0", True),
        ("00", True),
        ("23", True),
        ("123", True),
    ]


def test_normalize_number_prediction_rejects_incomplete_or_uncertain_values():
    rejected = [
        NumberPrediction(raw_text="2?", is_confident=True),
        NumberPrediction(raw_text="AB", is_confident=True),
        NumberPrediction(raw_text="1234", is_confident=True),
        NumberPrediction(raw_text="23", is_confident=False),
        NumberPrediction(raw_text=None, is_confident=True),
    ]

    assert [normalize_number_prediction(prediction) for prediction in rejected] == [
        ("unknown", False),
        ("unknown", False),
        ("unknown", False),
        ("unknown", False),
        ("unknown", False),
    ]


def test_build_number_predictor_supports_dummy_only():
    predictor = build_number_predictor("dummy")

    assert predictor.name == "dummy"

    with pytest.raises(ValueError, match="Unknown number_predictor backend"):
        build_number_predictor("ocr")


def test_write_number_prediction_artifacts_records_every_box(tmp_path: Path):
    records = [
        NumberPredictionRecord(
            frame_index=5,
            track_id=11,
            track_state="detected",
            raw_text="23",
            tentative_number="23",
            accepted=True,
        ),
        NumberPredictionRecord(
            frame_index=6,
            track_id=22,
            track_state="propagated",
            raw_text="2?",
            tentative_number="unknown",
            accepted=False,
        ),
    ]

    write_number_prediction_artifacts(tmp_path, records)

    predictions = json.loads((tmp_path / "number_predictions.json").read_text(encoding="utf-8"))
    cases = [
        json.loads(line)
        for line in (tmp_path / "number_prediction_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert predictions == {"predictions": [record.to_dict() for record in records]}
    assert cases == [record.to_dict() for record in records]


def test_write_number_prediction_artifacts_writes_empty_files(tmp_path: Path):
    write_number_prediction_artifacts(tmp_path, [])

    assert json.loads((tmp_path / "number_predictions.json").read_text(encoding="utf-8")) == {
        "predictions": []
    }
    assert (tmp_path / "number_prediction_cases.jsonl").read_text(encoding="utf-8") == ""


def test_upload_directory_keeps_nested_paths(tmp_path: Path):
    nested = tmp_path / "out" / "number_debug" / "track_0001"
    nested.mkdir(parents=True)
    (tmp_path / "out" / "results.json").write_text("{}", encoding="utf-8")
    (nested / "000_frame_000001.png").write_bytes(b"png")

    uploaded = []
    storage = object.__new__(R2Storage)
    storage.upload = lambda source, key, content_type=None: uploaded.append((source.name, key, content_type)) or key

    keys = R2Storage.upload_directory(storage, tmp_path / "out", "jobs/run-1")

    assert keys == [
        "jobs/run-1/number_debug/track_0001/000_frame_000001.png",
        "jobs/run-1/results.json",
    ]
    assert uploaded[0][2] is None
    assert uploaded[1][2] == "application/json"


def test_run_pipeline_ignores_number_detections_in_tracks_and_teams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import numpy as np
    from pipeline.number_predictor import NumberPrediction
    from pipeline.runner import run_pipeline

    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    segment = Segment(frame_indices=[0, 1], frames=frames, is_last=True)
    player_rows = [
        {
            "xyxy": [0, 0, 10, 20],
            "confidence": 0.9,
            "class_name": "player",
            "track_id": None,
            "track_state": "detected",
        },
        {
            "xyxy": [20, 0, 30, 20],
            "confidence": 0.8,
            "class_name": "player",
            "track_id": None,
            "track_state": "detected",
        },
    ]
    number_row = {
        "xyxy": [2, 2, 6, 8],
        "confidence": 0.7,
        "class_name": "number",
        "track_id": None,
        "track_state": "detected",
    }

    class FakeDetector:
        def __init__(self, detections):
            self._detections = detections
            self.call_count = 0

        def detect(self, _frame):
            self.call_count += 1
            return [dict(row) for row in self._detections]

    class FakeTracker:
        device = "cpu"

        def propagate_segment(self, frames, seed_rows):
            assert len(frames) == 2
            return {
                1: [
                    {
                        **dict(row),
                        "track_state": "propagated",
                    }
                    for row in seed_rows
                ]
            }

    class FakeEmbedder:
        name = "none"

    class FakeTeamEmbedder:
        name = "colour-test"

        def embed(self, frame, samples):
            del frame
            from pipeline.team_embedder import TeamFeatures

            return [
                TeamFeatures(
                    color_signature=[1.0, 0.0] if sample.xyxy[0] < 10 else [0.0, 1.0],
                    embedding=None,
                )
                for sample in samples
            ]

    class FakeNumberPredictor:
        name = "dummy"

        def __init__(self):
            self.calls = []

        def predict(self, sample):
            self.calls.append(sample)
            return NumberPrediction(raw_text="23", is_confident=True)

    def run_case(name: str, detections: list[dict]) -> tuple[list[dict], dict, dict, list[dict], list[dict]]:
        monkeypatch.setattr(
            "pipeline.runner.probe",
            lambda _path: VideoInfo(fps=30.0, frame_count=2, width=32, height=32),
        )
        monkeypatch.setattr("pipeline.runner.iter_segments", lambda *_args, **_kwargs: iter([segment]))
        monkeypatch.setattr("pipeline.runner.build_detector", lambda _config: FakeDetector(detections))
        monkeypatch.setattr("pipeline.runner.detections_from_result", lambda result, **_kwargs: result)
        monkeypatch.setattr(
            "pipeline.runner.detection_rows",
            lambda detections: [dict(row) for row in detections],
        )
        monkeypatch.setattr("pipeline.runner.build_embedder", lambda _backend: FakeEmbedder())
        monkeypatch.setattr(
            "pipeline.runner.build_team_embedder",
            lambda *_args, **_kwargs: FakeTeamEmbedder(),
        )
        predictor = FakeNumberPredictor()
        monkeypatch.setattr("pipeline.runner.build_number_predictor", lambda _backend: predictor)

        output_dir = tmp_path / name
        results = run_pipeline(
            video_path=tmp_path / "dummy.mp4",
            config=PipelineConfig(
                roboflow_api_key="test-key",
                appearance_weight=0.0,
                reid_backend="none",
                team_embedder_backend="colour",
                min_team_samples_per_track=1,
                team_sample_stride=1,
                max_team_samples_per_track=4,
                prefetch_detections=False,
                team_debug_enabled=False,
                number_predictor_backend="dummy",
            ),
            output_dir=output_dir,
            tracker=FakeTracker(),
        )
        assert results.summary.detector_calls == 1
        tracks = [
            json.loads(line)
            for line in (output_dir / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        teams = json.loads((output_dir / "teams.json").read_text(encoding="utf-8"))
        results_payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        predictions = json.loads((output_dir / "number_predictions.json").read_text(encoding="utf-8"))[
            "predictions"
        ]
        return tracks, teams, results_payload, predictions, predictor.calls

    player_only_tracks, player_only_teams, player_only_results, player_only_predictions, player_only_calls = run_case(
        "player_only", player_rows
    )
    with_numbers_tracks, with_numbers_teams, with_numbers_results, with_numbers_predictions, with_numbers_calls = run_case(
        "with_numbers", [*player_rows, number_row]
    )

    assert with_numbers_tracks == player_only_tracks
    assert with_numbers_teams == player_only_teams
    player_only_results["summary"].pop("wall_seconds", None)
    with_numbers_results["summary"].pop("wall_seconds", None)
    assert with_numbers_results == player_only_results
    assert [row["class_name"] for row in with_numbers_tracks] == ["player", "player", "player", "player"]
    assert with_numbers_teams["assignments_by_track_id"] == {"1": 1, "2": 0}
    assert player_only_predictions == []
    assert player_only_calls == []
    assert len(with_numbers_calls) == 1
    assert with_numbers_predictions == [
        {
            "frame_index": 0,
            "track_id": 1,
            "track_state": "detected",
            "raw_text": "23",
            "tentative_number": "23",
            "accepted": True,
        }
    ]


def test_run_pipeline_predicts_each_of_two_associated_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    player_rows = [
        {
            "xyxy": [0, 0, 10, 20],
            "confidence": 0.9,
            "class_name": "player",
            "track_id": None,
            "track_state": "detected",
        },
        {
            "xyxy": [20, 0, 30, 20],
            "confidence": 0.8,
            "class_name": "player",
            "track_id": None,
            "track_state": "detected",
        },
    ]
    number_rows = [
        {
            "xyxy": [2, 2, 6, 8],
            "confidence": 0.7,
            "class_name": "number",
            "track_id": None,
            "track_state": "detected",
        },
        {
            "xyxy": [22, 2, 26, 8],
            "confidence": 0.6,
            "class_name": "number",
            "track_id": None,
            "track_state": "detected",
        },
    ]

    _, _, _, predictions, calls = _run_number_prediction_case(
        monkeypatch,
        tmp_path,
        name="two_numbers",
        detections=[*player_rows, *number_rows],
        predict=lambda sample: NumberPrediction(raw_text=str(sample.track_id), is_confident=True),
    )

    assert [sample.track_id for sample in calls] == [1, 2]
    assert predictions == [
        {
            "frame_index": 0,
            "track_id": 1,
            "track_state": "detected",
            "raw_text": "1",
            "tentative_number": "1",
            "accepted": True,
        },
        {
            "frame_index": 0,
            "track_id": 2,
            "track_state": "detected",
            "raw_text": "2",
            "tentative_number": "2",
            "accepted": True,
        },
    ]


def test_run_pipeline_prediction_seam_is_observational_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    detections = [
        {
            "xyxy": [0, 0, 10, 20],
            "confidence": 0.9,
            "class_name": "player",
            "track_id": None,
            "track_state": "detected",
        },
        {
            "xyxy": [20, 0, 30, 20],
            "confidence": 0.8,
            "class_name": "player",
            "track_id": None,
            "track_state": "detected",
        },
        {
            "xyxy": [2, 2, 6, 8],
            "confidence": 0.7,
            "class_name": "number",
            "track_id": None,
            "track_state": "detected",
        },
    ]

    disabled_tracks, disabled_teams, disabled_results, disabled_predictions, _ = _run_number_prediction_case(
        monkeypatch,
        tmp_path,
        name="prediction_disabled",
        detections=detections,
        predict=lambda _sample: NumberPrediction(raw_text=None, is_confident=False),
    )
    enabled_tracks, enabled_teams, enabled_results, enabled_predictions, _ = _run_number_prediction_case(
        monkeypatch,
        tmp_path,
        name="prediction_enabled",
        detections=detections,
        predict=lambda _sample: NumberPrediction(raw_text="23", is_confident=True),
    )

    assert enabled_tracks == disabled_tracks
    assert enabled_teams == disabled_teams
    assert enabled_results == disabled_results
    assert disabled_predictions != enabled_predictions
