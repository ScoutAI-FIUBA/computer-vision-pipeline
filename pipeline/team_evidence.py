"""Team-evidence selection and debug artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pipeline.team_id import compute_track_coherence


@dataclass
class TeamEvidenceSample:
    """One team-identification sample plus enough metadata to inspect it later."""

    track_id: int
    frame_index: int
    track_state: str
    quality_score: float
    accepted: bool
    rejection_reasons: list[str] = field(default_factory=list)
    selected: bool = False
    color_signature: list[float] | None = None
    embedding: list[float] | None = None
    used_mask: bool = False
    uniform_xyxy: tuple[int, int, int, int] | None = None
    uniform_pixel_count: int = 0
    total_pixel_count: int = 0
    quality_details: dict[str, float] = field(default_factory=dict)
    crop_image: object | None = None
    mask_image: object | None = None

    def to_dict(self) -> dict:
        return {
            "frame_index": int(self.frame_index),
            "track_state": self.track_state,
            "quality_score": round(float(self.quality_score), 4),
            "accepted": bool(self.accepted),
            "selected": bool(self.selected),
            "rejection_reasons": list(self.rejection_reasons),
            "used_mask": bool(self.used_mask),
            "uniform_xyxy": None if self.uniform_xyxy is None else [int(value) for value in self.uniform_xyxy],
            "uniform_pixel_count": int(self.uniform_pixel_count),
            "total_pixel_count": int(self.total_pixel_count),
            "has_color_signature": self.color_signature is not None,
            "has_embedding": self.embedding is not None,
            "quality_details": {
                key: round(float(value), 4) for key, value in sorted(self.quality_details.items())
            },
        }


def select_top_k_evidence(samples: list[TeamEvidenceSample], max_samples: int) -> list[TeamEvidenceSample]:
    """Keep the highest-quality accepted samples for aggregation."""

    if max_samples < 1:
        return []

    candidates = [
        sample
        for sample in samples
        if sample.accepted and (sample.color_signature is not None or sample.embedding is not None)
    ]
    ranked = sorted(candidates, key=lambda sample: (-sample.quality_score, sample.frame_index))
    selected_ids = {id(sample) for sample in ranked[:max_samples]}
    for sample in samples:
        sample.selected = id(sample) in selected_ids
    return [sample for sample in ranked[:max_samples]]


def summarize_track_evidence(samples: list[TeamEvidenceSample]) -> dict:
    """Return per-track evidence counts and coherence for debugging."""

    selected = [sample for sample in samples if sample.selected]
    accepted = [sample for sample in samples if sample.accepted]
    rejected = [sample for sample in samples if not sample.accepted]
    color_vectors = [sample.color_signature for sample in selected if sample.color_signature is not None]
    embedding_vectors = [sample.embedding for sample in selected if sample.embedding is not None]
    quality_scores = [float(sample.quality_score) for sample in accepted]
    return {
        "accepted_samples": len(accepted),
        "rejected_samples": len(rejected),
        "selected_samples": len(selected),
        "used_mask_samples": sum(1 for sample in samples if sample.used_mask),
        "accepted_quality_mean": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else None,
        "color_coherence": compute_track_coherence(color_vectors),
        "embedding_coherence": compute_track_coherence(embedding_vectors),
    }


def write_debug_artifacts(output_dir: Path, evidence_by_track_id: dict[int, list[TeamEvidenceSample]]) -> None:
    """Write sample metadata plus crop/mask images for offline inspection."""

    import cv2
    import json
    import numpy as np

    debug_dir = output_dir / "team_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        str(track_id): {
            **summarize_track_evidence(samples),
            "samples": [sample.to_dict() for sample in samples],
        }
        for track_id, samples in sorted(evidence_by_track_id.items())
    }
    (output_dir / "team_debug.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for track_id, samples in sorted(evidence_by_track_id.items()):
        track_dir = debug_dir / f"track_{track_id:04d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            prefix = f"{index:03d}_frame_{sample.frame_index:06d}_{'accepted' if sample.accepted else 'rejected'}"
            if sample.crop_image is not None:
                cv2.imwrite(str(track_dir / f"{prefix}.png"), sample.crop_image)
            if sample.mask_image is not None:
                mask = np.asarray(sample.mask_image, dtype=np.uint8)
                cv2.imwrite(str(track_dir / f"{prefix}_mask.png"), mask)
