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
    aggregate_track_number_predictions,
    associate_number_rows,
    summarize_number_evidence,
    write_number_confirmation_artifacts,
    write_number_prediction_artifacts,
)
from pipeline.number_predictor import (  # noqa: E402
    NumberPrediction,
    build_number_predictor,
    normalize_number_prediction,
)
from pipeline.results import RESULTS_SCHEMA_VERSION  # noqa: E402
from pipeline.storage import R2Storage  # noqa: E402
from pipeline.video import Segment, VideoInfo  # noqa: E402


def _prediction_record(
    track_id: int,
    tentative_number: str,
    *,
    frame_index: int,
    accepted: bool = True,
    raw_text: str | None = None,
    track_state: str = "detected",
    score: float | None = None,
    raw_response: dict | None = None,
) -> NumberPredictionRecord:
    return NumberPredictionRecord(
        frame_index=frame_index,
        track_id=track_id,
        track_state=track_state,
        raw_text=raw_text if raw_text is not None else tentative_number,
        tentative_number=tentative_number,
        accepted=accepted,
        score=score,
        raw_response=raw_response,
    )


def _run_number_prediction_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    name: str,
    detections: list[dict],
    detector_results: list[list[dict]] | None = None,
    predict,
    config: PipelineConfig | None = None,
    predictor_factory=None,
):
    import numpy as np
    from pipeline.runner import run_pipeline

    if detector_results is None:
        detector_results = [detections]
    segments = []
    for index, rows in enumerate(detector_results):
        del rows
        frame_indices = [index, index + 1]
        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
        segments.append(Segment(frame_indices=frame_indices, frames=frames, is_last=index == len(detector_results) - 1))

    class FakeDetector:
        def __init__(self, results_by_call):
            self._results_by_call = results_by_call
            self.call_count = 0

        def detect(self, _frame):
            result_index = min(self.call_count, len(self._results_by_call) - 1)
            self.call_count += 1
            return [dict(row) for row in self._results_by_call[result_index]]

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
        lambda _path: VideoInfo(fps=30.0, frame_count=segments[-1].frame_indices[-1] + 1, width=32, height=32),
    )
    monkeypatch.setattr("pipeline.runner.iter_segments", lambda *_args, **_kwargs: iter(segments))
    monkeypatch.setattr("pipeline.runner.build_detector", lambda _config: FakeDetector(detector_results))
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
    monkeypatch.setattr(
        "pipeline.runner.build_number_predictor",
        predictor_factory or (lambda _config: predictor),
    )

    if config is None:
        config = PipelineConfig(
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
        )

    output_dir = tmp_path / name
    results = run_pipeline(
        video_path=tmp_path / "dummy.mp4",
        config=config,
        output_dir=output_dir,
        tracker=FakeTracker(),
    )

    assert results.summary.detector_calls == len(detector_results)
    tracks = [json.loads(line) for line in (output_dir / "tracks.jsonl").read_text(encoding="utf-8").splitlines()]
    teams = json.loads((output_dir / "teams.json").read_text(encoding="utf-8"))
    results_payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert results_payload["schema_version"] == RESULTS_SCHEMA_VERSION
    assert results_payload["numbers_file"] == "numbers.json"
    assert results_payload["shot_events_file"] == "shot_events.jsonl"
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
    assert samples[0].crop_image.shape == (34, 36, 3)
    assert samples[1].number_confidence == pytest.approx(0.6)


def test_associate_number_rows_clamps_ocr_padding_to_frame_edges():
    import numpy as np

    samples = associate_number_rows(
        np.zeros((32, 32, 3), dtype=np.uint8),
        [{"track_id": 7, "track_state": "detected", "confidence": 0.9, "xyxy": [0, 0, 20, 20]}],
        [{"confidence": 0.8, "xyxy": [2, 2, 10, 10]}],
        frame_index=3,
    )

    assert samples[0].crop_image.shape == (20, 20, 3)


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


def test_normalize_number_prediction_preserves_score_qualified_clean_digits():
    accepted = ["0", "00", "23"]

    assert [
        normalize_number_prediction(NumberPrediction(raw_text=value, score=0.7))
        for value in accepted
    ] == [
        ("0", True),
        ("00", True),
        ("23", True),
    ]


def test_normalize_number_prediction_accepts_clean_digits_without_score():
    accepted = ["0", "00", "23"]

    assert [
        normalize_number_prediction(NumberPrediction(raw_text=value, score=None))
        for value in accepted
    ] == [
        ("0", True),
        ("00", True),
        ("23", True),
    ]


def test_normalize_number_prediction_rejects_invalid_or_low_score_values():
    rejected = [
        NumberPrediction(raw_text="100", score=0.9),
        NumberPrediction(raw_text="7A", score=0.9),
        NumberPrediction(raw_text=" 23 ", score=0.9),
        NumberPrediction(raw_text="", score=0.9),
        NumberPrediction(raw_text="23", score=0.69),
        NumberPrediction(raw_text=None, score=0.9),
    ]

    assert [normalize_number_prediction(prediction) for prediction in rejected] == [
        ("unknown", False),
        ("unknown", False),
        ("unknown", False),
        ("unknown", False),
        ("unknown", False),
        ("unknown", False),
    ]


def test_build_number_predictor_supports_dummy_and_hosted(monkeypatch: pytest.MonkeyPatch):
    predictor = build_number_predictor(PipelineConfig(roboflow_api_key="test-key", number_predictor_backend="dummy"))

    assert predictor.name == "dummy"

    created = {}

    class FakeInferenceHTTPClient:
        def __init__(self, *, api_url: str, api_key: str):
            created["api_url"] = api_url
            created["api_key"] = api_key

        def infer(self, *_args, **_kwargs):
            return {"top": "23", "confidence": 0.91}

    import pipeline.number_predictor as number_predictor_module

    monkeypatch.setattr(number_predictor_module, "InferenceHTTPClient", FakeInferenceHTTPClient, raising=False)
    hosted = build_number_predictor(
        PipelineConfig(
            roboflow_api_key="hosted-key",
            number_predictor_backend="roboflow-hosted",
        )
    )

    assert hosted.name == "roboflow-hosted"
    assert created == {
        "api_url": "https://serverless.roboflow.com",
        "api_key": "hosted-key",
    }

    with pytest.raises(ValueError, match="Unknown number_predictor_backend"):
        build_number_predictor(PipelineConfig(roboflow_api_key="test-key", number_predictor_backend="ocr"))


def test_hosted_number_predictor_parses_text_score_and_raw_response(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    class FakeInferenceHTTPClient:
        def __init__(self, *, api_url: str, api_key: str):
            calls["api_url"] = api_url
            calls["api_key"] = api_key

        def infer(self, inference_input, model_id: str):
            calls["inference_input_shape"] = tuple(inference_input.shape)
            calls["model_id"] = model_id
            return {
                "predictions": [{"class": "00", "confidence": 0.82}],
                "top": "00",
            }

    import numpy as np
    import pipeline.number_predictor as number_predictor_module

    monkeypatch.setattr(number_predictor_module, "InferenceHTTPClient", FakeInferenceHTTPClient, raising=False)
    predictor = build_number_predictor(
        PipelineConfig(
            roboflow_api_key="hosted-key",
            number_predictor_backend="roboflow-hosted",
        )
    )

    prediction = predictor.predict(
        associate_number_rows(
            np.zeros((32, 32, 3), dtype=np.uint8),
            [{"track_id": 7, "track_state": "detected", "confidence": 0.9, "xyxy": [0, 0, 20, 20]}],
            [{"confidence": 0.8, "xyxy": [2, 2, 10, 10]}],
            frame_index=3,
        )[0]
    )

    assert calls == {
        "api_url": "https://serverless.roboflow.com",
        "api_key": "hosted-key",
        "inference_input_shape": (20, 20, 3),
        "model_id": "basketball-jersey-numbers-ocr/6",
    }
    assert prediction.raw_text == "00"
    assert prediction.score == pytest.approx(0.82)
    assert prediction.raw_response == {
        "predictions": [{"class": "00", "confidence": 0.82}],
        "top": "00",
    }


def test_hosted_number_predictor_parses_modal_response_payload(monkeypatch: pytest.MonkeyPatch):
    class FakeInferenceHTTPClient:
        def __init__(self, *, api_url: str, api_key: str):
            del api_url, api_key

        def infer(self, *_args, **_kwargs):
            return {
                "inference_id": "abc",
                "response": {">": "23"},
            }

    import numpy as np
    import pipeline.number_predictor as number_predictor_module

    monkeypatch.setattr(number_predictor_module, "InferenceHTTPClient", FakeInferenceHTTPClient, raising=False)
    predictor = build_number_predictor(
        PipelineConfig(
            roboflow_api_key="hosted-key",
            number_predictor_backend="roboflow-hosted",
        )
    )

    prediction = predictor.predict(
        associate_number_rows(
            np.zeros((32, 32, 3), dtype=np.uint8),
            [{"track_id": 7, "track_state": "detected", "confidence": 0.9, "xyxy": [0, 0, 20, 20]}],
            [{"confidence": 0.8, "xyxy": [2, 2, 10, 10]}],
            frame_index=3,
        )[0]
    )

    assert prediction.raw_text == "23"
    assert prediction.score is None
    assert prediction.raw_response == {
        "inference_id": "abc",
        "response": {">": "23"},
    }


def test_hosted_number_predictor_preserves_empty_modal_response_text(monkeypatch: pytest.MonkeyPatch):
    class FakeInferenceHTTPClient:
        def __init__(self, *, api_url: str, api_key: str):
            del api_url, api_key

        def infer(self, *_args, **_kwargs):
            return {
                "inference_id": "abc",
                "response": {">": ""},
            }

    import numpy as np
    import pipeline.number_predictor as number_predictor_module

    monkeypatch.setattr(number_predictor_module, "InferenceHTTPClient", FakeInferenceHTTPClient, raising=False)
    predictor = build_number_predictor(
        PipelineConfig(
            roboflow_api_key="hosted-key",
            number_predictor_backend="roboflow-hosted",
        )
    )

    prediction = predictor.predict(
        associate_number_rows(
            np.zeros((32, 32, 3), dtype=np.uint8),
            [{"track_id": 7, "track_state": "detected", "confidence": 0.9, "xyxy": [0, 0, 20, 20]}],
            [{"confidence": 0.8, "xyxy": [2, 2, 10, 10]}],
            frame_index=3,
        )[0]
    )

    assert prediction.raw_text == ""
    assert prediction.score is None


def test_hosted_number_predictor_preserves_explicit_zero_score(monkeypatch: pytest.MonkeyPatch):
    class FakeInferenceHTTPClient:
        def __init__(self, *, api_url: str, api_key: str):
            del api_url, api_key

        def infer(self, *_args, **_kwargs):
            return {
                "predictions": [{"class": "23", "confidence": 0.0}],
                "top": "23",
            }

    import numpy as np
    import pipeline.number_predictor as number_predictor_module

    monkeypatch.setattr(number_predictor_module, "InferenceHTTPClient", FakeInferenceHTTPClient, raising=False)
    predictor = build_number_predictor(
        PipelineConfig(
            roboflow_api_key="hosted-key",
            number_predictor_backend="roboflow-hosted",
        )
    )

    prediction = predictor.predict(
        associate_number_rows(
            np.zeros((32, 32, 3), dtype=np.uint8),
            [{"track_id": 7, "track_state": "detected", "confidence": 0.9, "xyxy": [0, 0, 20, 20]}],
            [{"confidence": 0.8, "xyxy": [2, 2, 10, 10]}],
            frame_index=3,
        )[0]
    )

    assert prediction.raw_text == "23"
    assert prediction.score == 0.0
    assert normalize_number_prediction(prediction) == ("unknown", False)


def test_write_number_prediction_artifacts_records_every_box(tmp_path: Path):
    records = [
        NumberPredictionRecord(
            frame_index=5,
            track_id=11,
            track_state="detected",
            raw_text="23",
            tentative_number="23",
            accepted=True,
            score=0.91,
            raw_response={"top": "23", "confidence": 0.91},
        ),
        NumberPredictionRecord(
            frame_index=6,
            track_id=22,
            track_state="propagated",
            raw_text="2?",
            tentative_number="unknown",
            accepted=False,
            score=0.42,
            raw_response={"top": "2?", "confidence": 0.42},
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


def test_aggregate_track_number_predictions_confirms_strongest_number():
    tracks = aggregate_track_number_predictions(
        [
            _prediction_record(11, "23", frame_index=5),
            _prediction_record(11, "23", frame_index=7, track_state="propagated"),
            _prediction_record(11, "45", frame_index=9),
        ]
    )

    assert tracks == [
        {
            "track_id": 11,
            "status": "confirmed",
            "number": "23",
            "support": [
                _prediction_record(11, "23", frame_index=5).to_dict(),
                _prediction_record(11, "23", frame_index=7, track_state="propagated").to_dict(),
            ],
            "conflicts": [_prediction_record(11, "45", frame_index=9).to_dict()],
            "ignored": [],
        }
    ]


def test_aggregate_track_number_predictions_marks_ties_ambiguous():
    tracks = aggregate_track_number_predictions(
        [
            _prediction_record(11, "23", frame_index=5),
            _prediction_record(11, "45", frame_index=7),
        ]
    )

    assert tracks == [
        {
            "track_id": 11,
            "status": "ambiguous",
            "number": "unknown",
            "support": [],
            "conflicts": [
                _prediction_record(11, "23", frame_index=5).to_dict(),
                _prediction_record(11, "45", frame_index=7).to_dict(),
            ],
            "ignored": [],
        }
    ]


def test_aggregate_track_number_predictions_confirms_single_accepted_number_without_conflicts():
    tracks = aggregate_track_number_predictions(
        [
            _prediction_record(11, "23", frame_index=5),
        ]
    )

    assert tracks == [
        {
            "track_id": 11,
            "status": "confirmed",
            "number": "23",
            "support": [_prediction_record(11, "23", frame_index=5).to_dict()],
            "conflicts": [],
            "ignored": [],
        }
    ]


def test_aggregate_track_number_predictions_keeps_unknown_without_decisive_reads():
    tracks = aggregate_track_number_predictions(
        [
            _prediction_record(11, "unknown", frame_index=5, accepted=False, raw_text="2?"),
            _prediction_record(11, "23", frame_index=7, accepted=False),
        ]
    )

    assert tracks == [
        {
            "track_id": 11,
            "status": "unknown",
            "number": "unknown",
            "support": [],
            "conflicts": [],
            "ignored": [
                _prediction_record(11, "unknown", frame_index=5, accepted=False, raw_text="2?").to_dict(),
                _prediction_record(11, "23", frame_index=7, accepted=False).to_dict(),
            ],
        }
    ]


def test_write_number_confirmation_artifacts_sorts_tracks_and_preserves_traceability(tmp_path: Path):
    records = [
        _prediction_record(22, "unknown", frame_index=8, accepted=False, raw_text="2?"),
        _prediction_record(22, "45", frame_index=6),
        _prediction_record(11, "23", frame_index=2),
        _prediction_record(22, "23", frame_index=4),
        _prediction_record(22, "45", frame_index=5, track_state="propagated"),
    ]

    write_number_confirmation_artifacts(tmp_path, records)

    assert json.loads((tmp_path / "numbers.json").read_text(encoding="utf-8")) == {
        "tracks": [
            {
                "track_id": 11,
                "status": "confirmed",
                "number": "23",
                "support": [_prediction_record(11, "23", frame_index=2).to_dict()],
                "conflicts": [],
                "ignored": [],
            },
            {
                "track_id": 22,
                "status": "confirmed",
                "number": "45",
                "support": [
                    _prediction_record(22, "45", frame_index=5, track_state="propagated").to_dict(),
                    _prediction_record(22, "45", frame_index=6).to_dict(),
                ],
                "conflicts": [_prediction_record(22, "23", frame_index=4).to_dict()],
                "ignored": [_prediction_record(22, "unknown", frame_index=8, accepted=False, raw_text="2?").to_dict()],
            },
        ]
    }


def test_write_number_confirmation_artifacts_writes_empty_tracks_list(tmp_path: Path):
    write_number_confirmation_artifacts(tmp_path, [])

    assert json.loads((tmp_path / "numbers.json").read_text(encoding="utf-8")) == {"tracks": []}


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
            return NumberPrediction(raw_text="23", score=0.9, raw_response={"top": "23", "confidence": 0.9})

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
        monkeypatch.setattr("pipeline.runner.build_number_predictor", lambda _config: predictor)

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
            "score": 0.9,
            "raw_response": {"top": "23", "confidence": 0.9},
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
        predict=lambda sample: NumberPrediction(
            raw_text=str(sample.track_id),
            score=0.9,
            raw_response={"top": str(sample.track_id), "confidence": 0.9},
        ),
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
            "score": 0.9,
            "raw_response": {"top": "1", "confidence": 0.9},
        },
        {
            "frame_index": 0,
            "track_id": 2,
            "track_state": "detected",
            "raw_text": "2",
            "tentative_number": "2",
            "accepted": True,
            "score": 0.9,
            "raw_response": {"top": "2", "confidence": 0.9},
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
        predict=lambda _sample: NumberPrediction(raw_text=None, score=None, raw_response={"top": None}),
    )
    enabled_tracks, enabled_teams, enabled_results, enabled_predictions, _ = _run_number_prediction_case(
        monkeypatch,
        tmp_path,
        name="prediction_enabled",
        detections=detections,
        predict=lambda _sample: NumberPrediction(raw_text="23", score=0.9, raw_response={"top": "23", "confidence": 0.9}),
    )

    assert enabled_tracks == disabled_tracks
    assert enabled_teams == disabled_teams
    assert enabled_results == disabled_results
    assert disabled_predictions != enabled_predictions


def test_run_pipeline_writes_confirmed_numbers_without_changing_existing_outputs(
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
            "xyxy": [2, 2, 6, 8],
            "confidence": 0.7,
            "class_name": "number",
            "track_id": None,
            "track_state": "detected",
        },
    ]

    disabled_tracks, disabled_teams, disabled_results, _, _ = _run_number_prediction_case(
        monkeypatch,
        tmp_path,
        name="confirmed_numbers_disabled",
        detections=detections,
        predict=lambda _sample: NumberPrediction(raw_text=None, score=None, raw_response={"top": None}),
    )
    enabled_tracks, enabled_teams, enabled_results, enabled_predictions, _ = _run_number_prediction_case(
        monkeypatch,
        tmp_path,
        name="confirmed_numbers_enabled",
        detections=detections,
        predict=lambda _sample: NumberPrediction(raw_text="23", score=0.9, raw_response={"top": "23", "confidence": 0.9}),
    )

    assert enabled_tracks == disabled_tracks
    assert enabled_teams == disabled_teams
    assert enabled_results == disabled_results
    assert enabled_predictions == [
        {
            "frame_index": 0,
            "track_id": 1,
            "track_state": "detected",
            "raw_text": "23",
            "tentative_number": "23",
            "accepted": True,
            "score": 0.9,
            "raw_response": {"top": "23", "confidence": 0.9},
        }
    ]
    assert json.loads(
        (tmp_path / "confirmed_numbers_enabled" / "numbers.json").read_text(encoding="utf-8")
    ) == {
        "tracks": [
            {
                "track_id": 1,
                "status": "confirmed",
                "number": "23",
                "support": enabled_predictions,
                "conflicts": [],
                "ignored": [],
            }
        ]
    }


def test_run_pipeline_confirms_most_frequent_valid_number_without_scores(
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
            "xyxy": [2, 2, 6, 8],
            "confidence": 0.7,
            "class_name": "number",
            "track_id": None,
            "track_state": "detected",
        },
    ]
    predictions = iter(
        [
            NumberPrediction(raw_text="28", score=None, raw_response={"top": "28"}),
            NumberPrediction(raw_text="28", score=None, raw_response={"top": "28"}),
            NumberPrediction(raw_text="11", score=None, raw_response={"top": "11"}),
        ]
    )

    _, _, _, prediction_records, _ = _run_number_prediction_case(
        monkeypatch,
        tmp_path,
        name="scoreless_majority_numbers",
        detections=detections,
        detector_results=[detections, detections, detections],
        predict=lambda _sample: next(predictions),
    )

    assert prediction_records == [
        {
            "frame_index": 0,
            "track_id": 1,
            "track_state": "detected",
            "raw_text": "28",
            "tentative_number": "28",
            "accepted": True,
            "score": None,
            "raw_response": {"top": "28"},
        },
        {
            "frame_index": 1,
            "track_id": 1,
            "track_state": "detected",
            "raw_text": "28",
            "tentative_number": "28",
            "accepted": True,
            "score": None,
            "raw_response": {"top": "28"},
        },
        {
            "frame_index": 2,
            "track_id": 1,
            "track_state": "detected",
            "raw_text": "11",
            "tentative_number": "11",
            "accepted": True,
            "score": None,
            "raw_response": {"top": "11"},
        },
    ]
    assert json.loads(
        (tmp_path / "scoreless_majority_numbers" / "numbers.json").read_text(encoding="utf-8")
    ) == {
        "tracks": [
            {
                "track_id": 1,
                "status": "confirmed",
                "number": "28",
                "support": prediction_records[:2],
                "conflicts": prediction_records[2:],
                "ignored": [],
            }
        ]
    }


def test_run_pipeline_numbers_artifact_keeps_support_conflict_and_ignored_records(
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
            "xyxy": [2, 2, 6, 8],
            "confidence": 0.7,
            "class_name": "number",
            "track_id": None,
            "track_state": "detected",
        },
    ]
    predictions = iter(
        [
            NumberPrediction(raw_text="23", score=0.95, raw_response={"top": "23", "confidence": 0.95}),
            NumberPrediction(raw_text="45", score=0.91, raw_response={"top": "45", "confidence": 0.91}),
            NumberPrediction(raw_text="2?", score=0.9, raw_response={"top": "2?", "confidence": 0.9}),
        ]
    )

    _, _, _, prediction_records, _ = _run_number_prediction_case(
        monkeypatch,
        tmp_path,
        name="traceability_numbers",
        detections=detections,
        detector_results=[detections, detections, detections],
        predict=lambda _sample: next(predictions),
    )

    assert prediction_records == [
        {
            "frame_index": 0,
            "track_id": 1,
            "track_state": "detected",
            "raw_text": "23",
            "tentative_number": "23",
            "accepted": True,
            "score": 0.95,
            "raw_response": {"top": "23", "confidence": 0.95},
        },
        {
            "frame_index": 1,
            "track_id": 1,
            "track_state": "detected",
            "raw_text": "45",
            "tentative_number": "45",
            "accepted": True,
            "score": 0.91,
            "raw_response": {"top": "45", "confidence": 0.91},
        },
        {
            "frame_index": 2,
            "track_id": 1,
            "track_state": "detected",
            "raw_text": "2?",
            "tentative_number": "unknown",
            "accepted": False,
            "score": 0.9,
            "raw_response": {"top": "2?", "confidence": 0.9},
        },
    ]


def test_run_pipeline_annotated_video_uses_confirmed_number_labels_without_rewriting_tracks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import numpy as np
    from pipeline.number_predictor import NumberPrediction
    from pipeline.runner import run_pipeline

    detections = [
        {
            "xyxy": [0, 0, 10, 20],
            "confidence": 0.9,
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
    segments = [
        Segment(frame_indices=[0, 1], frames=[np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)], is_last=False),
        Segment(frame_indices=[1, 2], frames=[np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)], is_last=True),
    ]

    class FakeDetector:
        def __init__(self):
            self.call_count = 0

        def detect(self, _frame):
            self.call_count += 1
            return [dict(row) for row in detections]

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

            return [TeamFeatures(color_signature=[1.0, 0.0], embedding=None) for _sample in samples]

    class FakeNumberPredictor:
        name = "dummy"

        def predict(self, _sample):
            return NumberPrediction(raw_text="23", score=0.9, raw_response={"top": "23", "confidence": 0.9})

    class FakeAnnotatedVideoWriter:
        def __init__(self, output_path, info, fps):
            del output_path, info, fps

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, _frame):
            return None

    rendered_rows = []

    def fake_draw_tracks(frame, rows):
        rendered_rows.append([dict(row) for row in rows])
        return frame

    monkeypatch.setattr(
        "pipeline.runner.probe",
        lambda _path: VideoInfo(fps=30.0, frame_count=3, width=32, height=32),
    )
    monkeypatch.setattr("pipeline.runner.iter_segments", lambda *_args, **_kwargs: iter(segments))
    monkeypatch.setattr("pipeline.runner.build_detector", lambda _config: FakeDetector())
    monkeypatch.setattr("pipeline.runner.detections_from_result", lambda result, **_kwargs: result)
    monkeypatch.setattr("pipeline.runner.detection_rows", lambda rows: [dict(row) for row in rows])
    monkeypatch.setattr("pipeline.runner.build_embedder", lambda _backend: FakeEmbedder())
    monkeypatch.setattr("pipeline.runner.build_team_embedder", lambda *_args, **_kwargs: FakeTeamEmbedder())
    monkeypatch.setattr("pipeline.runner.build_number_predictor", lambda _config: FakeNumberPredictor())
    monkeypatch.setattr("pipeline.runner.AnnotatedVideoWriter", FakeAnnotatedVideoWriter)
    monkeypatch.setattr("pipeline.annotate.draw_tracks", fake_draw_tracks)

    run_pipeline(
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
            write_annotated_video=True,
        ),
        output_dir=tmp_path / "annotated_numbers",
        tracker=FakeTracker(),
    )

    assert rendered_rows
    assert rendered_rows[0][0]["track_id"] == 1
    assert rendered_rows[0][0]["confirmed_number"] == "23"
    tracks = [
        json.loads(line)
        for line in (tmp_path / "annotated_numbers" / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all("confirmed_number" not in row for row in tracks)


def test_run_pipeline_does_not_write_results_before_annotated_render_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import numpy as np

    from pipeline.number_predictor import NumberPrediction
    from pipeline.runner import run_pipeline

    detections = [
        {
            "xyxy": [0, 0, 10, 20],
            "confidence": 0.9,
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

            return [TeamFeatures(color_signature=[1.0, 0.0], embedding=None) for _sample in samples]

    class FakeNumberPredictor:
        name = "dummy"

        def predict(self, _sample):
            return NumberPrediction(raw_text="23", score=0.9, raw_response={"top": "23", "confidence": 0.9})

    class FakeDetector:
        def __init__(self):
            self.call_count = 0

        def detect(self, _frame):
            self.call_count += 1
            return [dict(row) for row in detections]

    segment = Segment(
        frame_indices=[0, 1],
        frames=[np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)],
        is_last=True,
    )

    monkeypatch.setattr(
        "pipeline.runner.probe",
        lambda _path: VideoInfo(fps=30.0, frame_count=2, width=32, height=32),
    )
    monkeypatch.setattr(
        "pipeline.runner.iter_segments",
        lambda *_args, **_kwargs: iter([segment]),
    )
    monkeypatch.setattr("pipeline.runner.build_detector", lambda _config: FakeDetector())
    monkeypatch.setattr("pipeline.runner.detections_from_result", lambda result, **_kwargs: result)
    monkeypatch.setattr("pipeline.runner.detection_rows", lambda rows: [dict(row) for row in rows])
    monkeypatch.setattr("pipeline.runner.build_embedder", lambda _backend: FakeEmbedder())
    monkeypatch.setattr("pipeline.runner.build_team_embedder", lambda *_args, **_kwargs: FakeTeamEmbedder())
    monkeypatch.setattr("pipeline.runner.build_number_predictor", lambda _config: FakeNumberPredictor())
    monkeypatch.setattr(
        "pipeline.runner._render_annotated_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    output_dir = tmp_path / "annotated_render_failure"
    with pytest.raises(RuntimeError, match="render failed"):
        run_pipeline(
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
                write_annotated_video=True,
            ),
            output_dir=output_dir,
            tracker=FakeTracker(),
        )

    assert (output_dir / "numbers.json").exists()
    assert not (output_dir / "results.json").exists()


def test_run_pipeline_hosted_backend_uses_configured_credentials_and_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls = []

    def predict(sample):
        return NumberPrediction(raw_text="23", score=0.88, raw_response={"top": "23", "confidence": 0.88})

    def build_predictor(config: PipelineConfig):
        calls.append(
            {
                "backend": config.number_predictor_backend,
                "api_key": config.roboflow_api_key,
                "api_url": config.roboflow_api_url,
                "model_id": config.number_predictor_roboflow_model_id,
                "threshold": config.number_predictor_score_threshold,
            }
        )

        class FakeHostedNumberPredictor:
            name = "roboflow-hosted"

            def __init__(self):
                self.samples = []

            def predict(self, sample):
                self.samples.append(sample)
                return predict(sample)

        return FakeHostedNumberPredictor()

    _, _, _, predictions, _ = _run_number_prediction_case(
        monkeypatch,
        tmp_path,
        name="hosted_backend",
        detections=[
            {"xyxy": [0, 0, 10, 20], "confidence": 0.9, "class_name": "player", "track_id": None, "track_state": "detected"},
            {"xyxy": [2, 2, 6, 8], "confidence": 0.7, "class_name": "number", "track_id": None, "track_state": "detected"},
        ],
        predict=predict,
        config=PipelineConfig(
            roboflow_api_key="hosted-key",
            roboflow_api_url="https://serverless.roboflow.com",
            appearance_weight=0.0,
            reid_backend="none",
            team_embedder_backend="colour",
            min_team_samples_per_track=1,
            team_sample_stride=1,
            max_team_samples_per_track=4,
            prefetch_detections=False,
            team_debug_enabled=False,
            number_predictor_backend="roboflow-hosted",
        ),
        predictor_factory=build_predictor,
    )

    assert calls == [
        {
            "backend": "roboflow-hosted",
            "api_key": "hosted-key",
            "api_url": "https://serverless.roboflow.com",
            "model_id": "basketball-jersey-numbers-ocr/6",
            "threshold": 0.7,
        }
    ]
    assert predictions[0]["score"] == 0.88
    assert predictions[0]["raw_response"] == {"top": "23", "confidence": 0.88}
