"""Validation-only SAM 3.1 V2 boundary for normalized player observations.

The repository has real contracts for ReID and team features, but TrackNet,
court projection, and shot detection currently consume video/model streams
rather than normalized player rows. This module wires only the contracts that
fit this boundary; later V2 orchestration owns the remaining stages.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pipeline.reid import Embedder
from pipeline.team_embedder import TeamEmbedder, TeamEmbeddingSample
from pipeline.team_id import KMeansTeamClusterer, TeamClusterer, TeamObservation, identify_teams


IdentityResolver = Callable[[list[dict[str, Any]]], Mapping[str, str]]
_DERIVED_OBSERVATION_FIELDS = frozenset(
    {"stable_player_id", "reid_embedding", "reid_evidence", "team_id"}
)


@dataclass(frozen=True)
class MaskAppearanceCrop:
    """Deterministic, mask-gated evidence for one full-frame observation."""

    image: np.ndarray
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    metadata: dict[str, Any]


class MaskAppearanceAdapter:
    """Create appearance crops where the full-frame mask is authoritative."""

    def crop(self, frame: Any, bbox_xyxy: Any, full_mask: Any) -> MaskAppearanceCrop | None:
        """Return a zeroed ROI, or ``None`` when mask evidence is unusable."""

        try:
            image = np.asarray(frame)
            mask = np.asarray(full_mask)
        except (TypeError, ValueError):
            return None

        if image.ndim < 2 or image.size == 0 or mask.ndim != 2:
            return None
        height, width = image.shape[:2]
        if mask.shape != (height, width) or not np.issubdtype(mask.dtype, np.number) and mask.dtype != np.bool_:
            return None
        try:
            if not np.isfinite(mask).all():
                return None
        except TypeError:
            return None
        full_mask_bool = mask.astype(bool, copy=False)
        if not full_mask_bool.any():
            return None

        try:
            coordinates = tuple(float(value) for value in bbox_xyxy)
        except (TypeError, ValueError, OverflowError):
            return None
        if len(coordinates) != 4 or not all(np.isfinite(value) for value in coordinates):
            return None
        x1, y1, x2, y2 = (int(round(value)) for value in coordinates)
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        crop_mask = full_mask_bool[y1:y2, x1:x2].copy()
        if not crop_mask.any():
            return None
        crop_image = image[y1:y2, x1:x2].copy()
        crop_image[~crop_mask] = 0
        roi = (x1, y1, x2, y2)
        metadata = {
            "source_shape": (height, width),
            "roi_xyxy": roi,
            "crop_shape": crop_mask.shape,
            "mask_pixel_count": int(crop_mask.sum()),
        }
        return MaskAppearanceCrop(crop_image, crop_mask, roi, metadata)


@dataclass(frozen=True)
class RunnerV2Seams:
    """Existing repository contracts available at the normalized-row boundary."""

    identity: IdentityResolver | None = None
    reid: Embedder | None = None
    team_embedder: TeamEmbedder | None = None
    team_clusterer: TeamClusterer | None = None
    team_count: int = 2


@dataclass
class RunnerV2Result:
    observations: list[dict[str, Any]]
    downstream: dict[str, Any] = field(default_factory=dict)
    acceptance: dict[str, str] = field(default_factory=dict)
    acceptance_reasons: list[str] = field(default_factory=list)


def _copy_observations(observations: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for observation in observations:
        row = dict(observation)
        raw_id = row.get("raw_track_id")
        if type(raw_id) is not int:
            raise ValueError("raw_track_id must be an integer")
        row["raw_track_id"] = raw_id
        rows.append(row)
    return rows


def _resolve_identity(
    rows: list[dict[str, Any]], resolver: IdentityResolver | None
) -> list[str]:
    if not rows:
        return ["empty_observations"]
    if resolver is None:
        return [f"missing_stable_player_id:{row.get('observation_key', '')}" for row in rows]

    resolved = resolver(rows)
    reasons: list[str] = []
    stable_by_raw: dict[int, set[str]] = defaultdict(set)
    stable_by_frame: dict[int, dict[str, int]] = defaultdict(dict)
    for row in rows:
        key = str(row.get("observation_key", ""))
        stable_id = resolved.get(key)
        if not isinstance(stable_id, str) or not stable_id.strip():
            reasons.append(f"missing_stable_player_id:{key}")
            continue
        existing = row.get("stable_player_id")
        if existing is not None and existing != stable_id:
            reasons.append(f"conflicting_stable_player_id:{key}")
        row["stable_player_id"] = stable_id
        stable_by_raw[row["raw_track_id"]].add(stable_id)
        frame_index = int(row["frame_index"])
        previous_raw = stable_by_frame[frame_index].get(stable_id)
        if previous_raw is not None and previous_raw != row["raw_track_id"]:
            reasons.append(
                f"conflicting_stable_player_id:frame={frame_index}:stable_player_id={stable_id}"
            )
        stable_by_frame[frame_index][stable_id] = row["raw_track_id"]

    for raw_id, stable_ids in stable_by_raw.items():
        if len(stable_ids) > 1:
            reasons.append(f"conflicting_stable_player_id:raw_track_id={raw_id}")
    return list(dict.fromkeys(reasons))


def _group_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame_index"])].append(row)
    return grouped


def _scrub_derived_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key not in _DERIVED_OBSERVATION_FIELDS}
        for row in rows
    ]


def _normalize_embedding(embedding: Any) -> list[float] | None:
    if embedding is None:
        return None
    try:
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all() or norm <= 1e-12:
        return None
    return (vector / norm).tolist()


def _run_reid(
    rows: list[dict[str, Any]],
    embedder: Embedder,
    frames_by_index: Mapping[int, Any],
    masks_by_observation: Mapping[str, Any] | None,
) -> tuple[int, dict[str, MaskAppearanceCrop]]:
    adapter = MaskAppearanceAdapter()
    prepared: dict[str, MaskAppearanceCrop] = {}
    count = 0
    for row in rows:
        key = str(row.get("observation_key", ""))
        frame_index = int(row["frame_index"])
        crop = None
        if frame_index in frames_by_index and masks_by_observation is not None:
            crop = adapter.crop(frames_by_index[frame_index], row["bbox_xyxy"], masks_by_observation.get(key))
        if crop is None:
            row["reid_evidence"] = {"status": "invalid", "reason": "invalid_mask"}
            continue

        height, width = crop.image.shape[:2]
        embeddings = embedder.embed(crop.image, [[0.0, 0.0, float(width), float(height)]])
        if len(embeddings) != 1:
            raise ValueError("ReID returned a different number of embeddings")
        embedding = _normalize_embedding(embeddings[0])
        if embedding is None:
            row["reid_evidence"] = {"status": "invalid", "reason": "invalid_embedding"}
            continue
        row["reid_embedding"] = embedding
        row["reid_evidence"] = {"status": "valid"}
        prepared[key] = crop
        count += 1
    return count, prepared


def _run_team_id(
    rows: list[dict[str, Any]],
    team_embedder: TeamEmbedder,
    clusterer: TeamClusterer,
    team_count: int,
    frames_by_index: Mapping[int, Any],
    crops_by_observation: Mapping[str, MaskAppearanceCrop] | None = None,
) -> tuple[dict[str, int], list[str]]:
    observations: list[TeamObservation] = []
    next_detection_id = 1
    for frame_index, frame_rows in _group_by_frame(rows).items():
        if frame_index not in frames_by_index:
            raise ValueError(f"missing frame for team embedding: {frame_index}")
        for row in frame_rows:
            crop = None if crops_by_observation is None else crops_by_observation.get(str(row["observation_key"]))
            frame = frames_by_index[frame_index] if crop is None else crop.image
            box = tuple(float(value) for value in row["bbox_xyxy"]) if crop is None else (
                0.0,
                0.0,
                float(crop.image.shape[1]),
                float(crop.image.shape[0]),
            )
            sample = TeamEmbeddingSample(
                detection_id=next_detection_id,
                track_id=row["raw_track_id"],
                xyxy=box,
                player_mask=None if crop is None else crop.mask,
            )
            features = team_embedder.embed(frame, [sample])
            if len(features) != 1:
                raise ValueError("team embedder returned a different number of features")
            feature = features[0]
            observations.append(
                TeamObservation(
                    detection_id=sample.detection_id,
                    track_id=sample.track_id,
                    color_signature=feature.color_signature,
                    embedding=feature.embedding,
                )
            )
            next_detection_id += 1

    result = identify_teams(
        observations,
        clusterer=clusterer,
        team_count=team_count,
        backend=team_embedder.name,
        minimum_samples=1,
    )
    raw_to_stable = {row["raw_track_id"]: row["stable_player_id"] for row in rows}
    team_ids_by_stable: dict[str, set[int]] = defaultdict(set)
    for assignment in result.assignments:
        if assignment.track_id is None or assignment.team_id is None:
            continue
        team_ids_by_stable[raw_to_stable[assignment.track_id]].add(int(assignment.team_id))

    conflicts = [
        f"conflicting_team_assignment:stable_player_id={stable_id}"
        for stable_id, team_ids in sorted(team_ids_by_stable.items())
        if len(team_ids) > 1
    ]
    assignments = {
        stable_id: next(iter(team_ids))
        for stable_id, team_ids in team_ids_by_stable.items()
        if len(team_ids) == 1
    }
    return assignments, conflicts


class RunnerV2:
    """Validate identity evidence and invoke supported production stage contracts.

    A resolver assigns candidate identities, but this boundary never treats that
    assignment alone as acceptance. A structurally valid result is reported as
    ``unvalidated`` until a later identity-authority stage supplies acceptance.
    """

    def __init__(self, seams: RunnerV2Seams | None = None) -> None:
        self.seams = seams or RunnerV2Seams()

    def run(
        self,
        observations: Iterable[Mapping[str, Any]],
        *,
        frames_by_index: Mapping[int, Any] | None = None,
        masks_by_observation: Mapping[str, Any] | None = None,
    ) -> RunnerV2Result:
        rows = _copy_observations(observations)
        reasons = _resolve_identity(rows, self.seams.identity)
        structurally_valid = not reasons
        frames = frames_by_index or {}
        downstream: dict[str, Any] = {}

        prepared_crops: dict[str, MaskAppearanceCrop] = {}
        if structurally_valid and self.seams.reid is not None:
            downstream["reid_records"], prepared_crops = _run_reid(
                rows, self.seams.reid, frames, masks_by_observation
            )
        if structurally_valid and self.seams.team_embedder is not None:
            team_rows = rows
            if masks_by_observation is not None:
                team_rows = [row for row in rows if str(row.get("observation_key", "")) in prepared_crops]
            if team_rows:
                team_assignments, team_reasons = _run_team_id(
                    team_rows,
                    self.seams.team_embedder,
                    self.seams.team_clusterer or KMeansTeamClusterer(),
                    self.seams.team_count,
                    frames,
                    prepared_crops if masks_by_observation is not None else None,
                )
                reasons.extend(team_reasons)
                if not team_reasons:
                    downstream["team_assignments"] = team_assignments

        structurally_valid = not reasons
        if not structurally_valid:
            # ReID may have run before a later team check failed. Never expose
            # that partial work as consumable output from a rejected run.
            downstream = {}
            rows = _scrub_derived_observations(rows)

        acceptance = {
            "tracking_identity": "unvalidated" if structurally_valid else "rejected",
            "statistics": "deferred",
        }
        return RunnerV2Result(rows, downstream, acceptance, reasons)


def run_v2(
    observations: Iterable[Mapping[str, Any]],
    *,
    seams: RunnerV2Seams | None = None,
    frames_by_index: Mapping[int, Any] | None = None,
    masks_by_observation: Mapping[str, Any] | None = None,
) -> RunnerV2Result:
    return RunnerV2(seams).run(
        observations,
        frames_by_index=frames_by_index,
        masks_by_observation=masks_by_observation,
    )
