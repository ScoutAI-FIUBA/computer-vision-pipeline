"""Detector-frame jersey-number evidence helpers."""

from __future__ import annotations

import json
from collections import Counter
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


def crop_xyxy(frame, xyxy: list[float], *, px: int = 0, py: int = 0):
    height, width = frame.shape[:2]
    x1 = max(0, int(round(float(xyxy[0]))) - px)
    y1 = max(0, int(round(float(xyxy[1]))) - py)
    x2 = min(width, max(x1, int(round(float(xyxy[2]))) + px))
    y2 = min(height, max(y1, int(round(float(xyxy[3]))) + py))
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
    score: float | None = None
    raw_response: dict | None = None

    def to_dict(self) -> dict:
        return {
            "frame_index": int(self.frame_index),
            "track_id": int(self.track_id),
            "track_state": self.track_state,
            "raw_text": self.raw_text,
            "tentative_number": self.tentative_number,
            "accepted": bool(self.accepted),
            "score": None if self.score is None else round(float(self.score), 4),
            "raw_response": self.raw_response,
        }


def _sorted_prediction_records(records: list[NumberPredictionRecord]) -> list[NumberPredictionRecord]:
    return sorted(records, key=lambda record: (record.frame_index, record.track_id, record.track_state))


def _is_decisive_number_record(record: NumberPredictionRecord) -> bool:
    return bool(record.accepted) and record.tentative_number != "unknown"


def aggregate_track_number_predictions(records: list[NumberPredictionRecord]) -> list[dict]:
    tracks_by_id: dict[int, list[NumberPredictionRecord]] = {}
    for record in _sorted_prediction_records(records):
        tracks_by_id.setdefault(record.track_id, []).append(record)

    tracks: list[dict] = []
    for track_id in sorted(tracks_by_id):
        track_records = tracks_by_id[track_id]
        decisive_records = [record for record in track_records if _is_decisive_number_record(record)]
        decisive_counts = Counter(record.tentative_number for record in decisive_records)
        winning_number = "unknown"
        status = "unknown"
        support: list[dict] = []
        conflicts: list[dict] = []

        if decisive_counts:
            highest_count = max(decisive_counts.values())
            winners = sorted(number for number, count in decisive_counts.items() if count == highest_count)
            if len(winners) == 1 and (highest_count > 1 or len(decisive_counts) == 1):
                winning_number = winners[0]
                status = "confirmed"
                support = [
                    record.to_dict()
                    for record in decisive_records
                    if record.tentative_number == winning_number
                ]
                conflicts = [
                    record.to_dict()
                    for record in decisive_records
                    if record.tentative_number != winning_number
                ]
            else:
                status = "ambiguous"
                conflicts = [record.to_dict() for record in decisive_records]

        ignored = [record.to_dict() for record in track_records if not _is_decisive_number_record(record)]
        tracks.append(
            {
                "track_id": track_id,
                "status": status,
                "number": winning_number,
                "support": support,
                "conflicts": conflicts,
                "ignored": ignored,
            }
        )

    return tracks


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
                        crop_image=crop_xyxy(frame, number_xyxy, px=10, py=10),
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


def write_number_confirmation_artifacts(output_dir: Path, records: list[NumberPredictionRecord]) -> None:
    (output_dir / "numbers.json").write_text(
        json.dumps({"tracks": aggregate_track_number_predictions(records)}, indent=2),
        encoding="utf-8",
    )


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
