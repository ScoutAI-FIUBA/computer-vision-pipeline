"""Single-frame jersey-number prediction seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

InferenceHTTPClient = None

from pipeline.number_evidence import NumberEvidenceSample


@dataclass(frozen=True)
class NumberPrediction:
    raw_text: str | None
    score: float | None = None
    raw_response: dict | None = None


class NumberPredictor(Protocol):
    name: str

    def predict(self, sample: NumberEvidenceSample) -> NumberPrediction: ...


class DummyNumberPredictor:
    name = "dummy"

    def predict(self, sample: NumberEvidenceSample) -> NumberPrediction:
        del sample
        return NumberPrediction(raw_text=None)


def _coerce_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _coerce_score(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_first_text(mapping: dict, *keys: str) -> str | None:
    for key in keys:
        if key in mapping:
            value = _coerce_text(mapping[key])
            if value is not None:
                return value
    return None


def _first_score(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _parse_hosted_response(raw_response: dict | None) -> tuple[str | None, float | None]:
    if not isinstance(raw_response, dict):
        return None, None

    fallback_score = _coerce_score(
        raw_response.get("confidence") if "confidence" in raw_response else raw_response.get("score")
    )

    predictions = raw_response.get("predictions")
    first_prediction = predictions[0] if isinstance(predictions, list) and predictions else None
    prediction_score = None
    if isinstance(first_prediction, dict):
        prediction_score = _coerce_score(
            first_prediction.get("confidence") if "confidence" in first_prediction else first_prediction.get("score")
        )

    top = raw_response.get("top")
    if isinstance(top, dict):
        return (
            _coerce_first_text(top, "class", "class_name", "text", "value", "prediction"),
            _first_score(
                _coerce_score(top.get("confidence") if "confidence" in top else top.get("score")),
                prediction_score,
                fallback_score,
            ),
        )
    if top is not None:
        return _coerce_text(top), _first_score(prediction_score, fallback_score)

    if isinstance(first_prediction, dict):
        first = first_prediction
        if isinstance(first, dict):
            return (
                _coerce_first_text(first, "class", "class_name", "text", "value", "prediction"),
                _first_score(prediction_score, fallback_score),
            )

    predicted_classes = raw_response.get("predicted_classes")
    if isinstance(predicted_classes, list) and predicted_classes:
        return _coerce_text(predicted_classes[0]), _first_score(prediction_score, fallback_score)

    response = raw_response.get("response")
    if isinstance(response, dict):
        return (
            _coerce_first_text(response, ">", "class", "class_name", "text", "value", "prediction"),
            _first_score(
                _coerce_score(response.get("confidence") if "confidence" in response else response.get("score")),
                prediction_score,
                fallback_score,
            ),
        )
    if response is not None:
        return _coerce_text(response), _first_score(prediction_score, fallback_score)

    return None, _first_score(prediction_score, fallback_score)


class HostedRoboflowNumberPredictor:
    name = "roboflow-hosted"

    def __init__(self, *, api_key: str, api_url: str, model_id: str) -> None:
        client_class = InferenceHTTPClient
        if client_class is None:
            from inference_sdk import InferenceHTTPClient as client_class

        self._client = client_class(api_url=api_url, api_key=api_key)
        self._model_id = model_id

    def predict(self, sample: NumberEvidenceSample) -> NumberPrediction:
        raw_response = self._client.infer(sample.crop_image, model_id=self._model_id)
        raw_text, score = _parse_hosted_response(raw_response)
        return NumberPrediction(raw_text=raw_text, score=score, raw_response=raw_response)


def normalize_number_prediction(prediction: NumberPrediction, threshold: float = 0.7) -> tuple[str, bool]:
    raw_text = prediction.raw_text
    if prediction.score is not None and prediction.score < threshold:
        return "unknown", False
    if raw_text is None or not raw_text.isdigit() or len(raw_text) not in {1, 2}:
        return "unknown", False
    return raw_text, True


def build_number_predictor(config) -> NumberPredictor:
    backend = config if isinstance(config, str) else config.number_predictor_backend
    if backend == "dummy":
        return DummyNumberPredictor()
    if backend == "roboflow-hosted" and not isinstance(config, str):
        return HostedRoboflowNumberPredictor(
            api_key=config.roboflow_api_key,
            api_url=config.roboflow_api_url,
            model_id=config.number_predictor_roboflow_model_id,
        )
    raise ValueError(f"Unknown number_predictor backend: {backend!r} (dummy, roboflow-hosted)")
