"""Single-frame jersey-number prediction seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pipeline.number_evidence import NumberEvidenceSample


@dataclass(frozen=True)
class NumberPrediction:
    raw_text: str | None
    is_confident: bool


class NumberPredictor(Protocol):
    name: str

    def predict(self, sample: NumberEvidenceSample) -> NumberPrediction: ...


class DummyNumberPredictor:
    name = "dummy"

    def predict(self, sample: NumberEvidenceSample) -> NumberPrediction:
        del sample
        return NumberPrediction(raw_text=None, is_confident=False)


def normalize_number_prediction(prediction: NumberPrediction) -> tuple[str, bool]:
    raw_text = prediction.raw_text
    if not prediction.is_confident or raw_text is None or not raw_text.isdigit() or len(raw_text) not in {1, 2, 3}:
        return "unknown", False
    return raw_text, True


def build_number_predictor(backend: str) -> NumberPredictor:
    if backend == "dummy":
        return DummyNumberPredictor()
    raise ValueError(f"Unknown number_predictor backend: {backend!r} (dummy)")
