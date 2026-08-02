"""Detector-frame jersey-number evidence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def _intersection_xyxy(left_xyxy: list[float], right_xyxy: list[float]) -> tuple[float, float, float, float] | None:
    x1 = max(float(left_xyxy[0]), float(right_xyxy[0]))
    y1 = max(float(left_xyxy[1]), float(right_xyxy[1]))
    x2 = min(float(left_xyxy[2]), float(right_xyxy[2]))
    y2 = min(float(left_xyxy[3]), float(right_xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _area(xyxy: list[float]) -> float:
    return max(0.0, float(xyxy[2]) - float(xyxy[0])) * max(0.0, float(xyxy[3]) - float(xyxy[1]))


def crop_xyxy(frame, xyxy: list[float]):
    x1 = max(0, int(round(float(xyxy[0]))))
    y1 = max(0, int(round(float(xyxy[1]))))
    x2 = max(x1, int(round(float(xyxy[2]))))
    y2 = max(y1, int(round(float(xyxy[3]))))
    return frame[y1:y2, x1:x2].copy()


@dataclass
class NumberEvidenceSample:
    track_id: int
    frame_index: int
    track_state: str
    player_xyxy: tuple[float, float, float, float]
    number_xyxy: tuple[float, float, float, float]
    player_confidence: float | None = None
    number_confidence: float | None = None
    overlap_ratio: float = 0.0
    player_overlap_ratio: float = 0.0
    contains_number_center: bool = False
    crop_image: object | None = None
    confidence_hints: dict[str, float | bool | None] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame_index": int(self.frame_index),
            "track_state": self.track_state,
            "track_id": int(self.track_id),
            "player_xyxy": [round(float(value), 2) for value in self.player_xyxy],
            "number_xyxy": [round(float(value), 2) for value in self.number_xyxy],
            "player_confidence": None if self.player_confidence is None else round(float(self.player_confidence), 4),
            "number_confidence": None if self.number_confidence is None else round(float(self.number_confidence), 4),
            "overlap_ratio": round(float(self.overlap_ratio), 4),
            "player_overlap_ratio": round(float(self.player_overlap_ratio), 4),
            "contains_number_center": bool(self.contains_number_center),
            "confidence_hints": dict(self.confidence_hints),
        }


@dataclass(frozen=True)
class NumberPredictionRecord:
    frame_index: int
    track_id: int
    track_state: str
    raw_text: str | None
    tentative_number: str
    accepted: bool

    def to_dict(self) -> dict:
        return {
            "frame_index": int(self.frame_index),
            "track_id": int(self.track_id),
            "track_state": self.track_state,
            "raw_text": self.raw_text,
            "tentative_number": self.tentative_number,
            "accepted": bool(self.accepted),
        }


def associate_number_rows(frame, player_rows: list[dict], number_rows: list[dict], *, frame_index: int) -> list[NumberEvidenceSample]:
    """Greedily match number boxes to player detections for one detector frame."""

    candidates = []
    for player_index, player_row in enumerate(player_rows):
        player_xyxy = [float(value) for value in player_row["xyxy"]]
        player_area = _area(player_xyxy)
        if player_area <= 0.0 or player_row.get("track_id") is None:
            continue
        for number_index, number_row in enumerate(number_rows):
            number_xyxy = [float(value) for value in number_row["xyxy"]]
            number_area = _area(number_xyxy)
            if number_area <= 0.0:
                continue
            intersection = _intersection_xyxy(player_xyxy, number_xyxy)
            if intersection is None:
                continue
            inter_area = _area(list(intersection))
            number_center_x = (number_xyxy[0] + number_xyxy[2]) / 2.0
            number_center_y = (number_xyxy[1] + number_xyxy[3]) / 2.0
            contains_center = (
                player_xyxy[0] <= number_center_x <= player_xyxy[2]
                and player_xyxy[1] <= number_center_y <= player_xyxy[3]
            )
            overlap_ratio = inter_area / number_area
            player_overlap_ratio = inter_area / player_area
            candidates.append(
                {
                    "player_index": player_index,
                    "number_index": number_index,
                    "score": (
                        1 if contains_center else 0,
                        overlap_ratio,
                        player_overlap_ratio,
                        inter_area,
                        -player_index,
                        -number_index,
                    ),
                    "sample": NumberEvidenceSample(
                        track_id=int(player_row["track_id"]),
                        frame_index=frame_index,
                        track_state=str(player_row.get("track_state", "detected")),
                        player_xyxy=tuple(player_xyxy),
                        number_xyxy=tuple(number_xyxy),
                        player_confidence=None
                        if player_row.get("confidence") is None
                        else float(player_row["confidence"]),
                        number_confidence=None
                        if number_row.get("confidence") is None
                        else float(number_row["confidence"]),
                        overlap_ratio=overlap_ratio,
                        player_overlap_ratio=player_overlap_ratio,
                        contains_number_center=contains_center,
                        crop_image=crop_xyxy(frame, number_xyxy),
                        confidence_hints={
                            "contains_number_center": contains_center,
                            "overlap_ratio": round(overlap_ratio, 4),
                            "player_overlap_ratio": round(player_overlap_ratio, 4),
                            "player_confidence": player_row.get("confidence"),
                            "number_confidence": number_row.get("confidence"),
                        },
                    ),
                }
            )

    assigned_players: set[int] = set()
    assigned_numbers: set[int] = set()
    matches: list[NumberEvidenceSample] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if candidate["player_index"] in assigned_players or candidate["number_index"] in assigned_numbers:
            continue
        assigned_players.add(candidate["player_index"])
        assigned_numbers.add(candidate["number_index"])
        matches.append(candidate["sample"])

    matches.sort(key=lambda sample: (sample.frame_index, sample.track_id))
    return matches


def summarize_number_evidence(samples: list[NumberEvidenceSample]) -> dict:
    return {
        "sample_count": len(samples),
        "mean_overlap_ratio": round(
            sum(float(sample.overlap_ratio) for sample in samples) / len(samples), 4
        )
        if samples
        else None,
        "mean_player_overlap_ratio": round(
            sum(float(sample.player_overlap_ratio) for sample in samples) / len(samples), 4
        )
        if samples
        else None,
        "samples": [sample.to_dict() for sample in samples],
    }


def write_number_prediction_artifacts(output_dir: Path, records: list[NumberPredictionRecord]) -> None:
    (output_dir / "number_predictions.json").write_text(
        json.dumps({"predictions": [record.to_dict() for record in records]}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "number_prediction_cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")


def write_number_debug_artifacts(output_dir: Path, evidence_by_track_id: dict[int, list[NumberEvidenceSample]]) -> None:
    import cv2

    debug_dir = output_dir / "number_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "mode": "detector_frames_only",
        "tracks": {
            str(track_id): summarize_number_evidence(samples)
            for track_id, samples in sorted(evidence_by_track_id.items())
        },
    }
    (output_dir / "number_debug.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for track_id, samples in sorted(evidence_by_track_id.items()):
        track_dir = debug_dir / f"track_{track_id:04d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            if sample.crop_image is None:
                continue
            filename = track_dir / f"{index:03d}_frame_{sample.frame_index:06d}.png"
            cv2.imwrite(str(filename), sample.crop_image)
