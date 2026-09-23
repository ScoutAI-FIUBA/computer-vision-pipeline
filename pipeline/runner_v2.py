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
from pipeline.bbox_sanitizer import sanitize_temporal_bboxes
from pipeline.team_embedder import TeamEmbedder, TeamEmbeddingSample
from pipeline.team_id import (
    STABLE_PLAYER_TEAM_VOTE_SHARE_THRESHOLD,
    KMeansTeamClusterer,
    TeamClusterer,
    TeamObservation,
    identify_teams,
)


IdentityResolver = Callable[[list[dict[str, Any]]], Mapping[str, str]]
IdentityAuthority = Callable[
    [list[dict[str, Any]], "RuntimeStageReceipts"], "IdentityAuthorityDecision"
]
_DERIVED_OBSERVATION_FIELDS = frozenset(
    {"stable_player_id", "reid_embedding", "reid_evidence", "team_id"}
)
_RUNTIME_RECEIPT_TOKEN = object()


@dataclass(frozen=True)
class MaskAppearanceCrop:
    """Deterministic, mask-gated evidence for one full-frame observation."""

    image: np.ndarray
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    metadata: dict[str, Any]


class MaskAppearanceAdapter:
    """Create appearance crops where the full-frame mask is authoritative."""

    def crop(
        self,
        frame: Any,
        bbox_xyxy: Any,
        full_mask: Any,
        *,
        coordinate_space: str | None = None,
    ) -> MaskAppearanceCrop | None:
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
        if coordinate_space not in (None, "pixel", "normalized"):
            return None
        if coordinate_space == "normalized":
            if not all(0.0 <= value <= 1.0 for value in coordinates):
                return None
            coordinates = (
                coordinates[0] * width,
                coordinates[1] * height,
                coordinates[2] * width,
                coordinates[3] * height,
            )
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
    identity_authority: IdentityAuthority | None = None
    team_count: int = 2


@dataclass
class RunnerV2Result:
    observations: list[dict[str, Any]]
    downstream: dict[str, Any] = field(default_factory=dict)
    acceptance: dict[str, Any] = field(default_factory=dict)
    acceptance_reasons: list[str] = field(default_factory=list)
    runtime_diagnostics: dict[str, Any] = field(default_factory=dict)
    cleaned_masks_by_observation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IdentityAuthorityDecision:
    accepted: bool
    reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeStageReceipts:
    stable_identity: frozenset[str] = frozenset()
    reid: frozenset[str] = frozenset()
    team: frozenset[str] = frozenset()
    _token: object = field(default=None, repr=False, compare=False)

    def _is_runner_provenance(self) -> bool:
        return self._token is _RUNTIME_RECEIPT_TOKEN

    def _with(
        self,
        *,
        stable_identity: frozenset[str] | None = None,
        reid: frozenset[str] | None = None,
        team: frozenset[str] | None = None,
    ) -> "RuntimeStageReceipts":
        return RuntimeStageReceipts(
            stable_identity=self.stable_identity if stable_identity is None else stable_identity,
            reid=self.reid if reid is None else reid,
            team=self.team if team is None else team,
            _token=self._token,
        )


def _mint_runtime_stage_receipts() -> RuntimeStageReceipts:
    return RuntimeStageReceipts(_token=_RUNTIME_RECEIPT_TOKEN)


class RuntimeIdentityAuthority:
    """Accept only rows with every runtime identity authority check complete."""

    def __call__(
        self, rows: list[dict[str, Any]], receipts: RuntimeStageReceipts
    ) -> IdentityAuthorityDecision:
        if not isinstance(receipts, RuntimeStageReceipts) or not receipts._is_runner_provenance():
            return IdentityAuthorityDecision(False, ("identity_authority:invalid_stage_receipts",))
        reasons: list[str] = []
        for row in rows:
            key = str(row.get("observation_key", ""))
            if key not in receipts.stable_identity:
                reasons.append(f"identity_authority:missing_stage_receipt:stable_identity:{key}")
            if key not in receipts.reid:
                reasons.append(f"identity_authority:missing_stage_receipt:reid:{key}")
            if key not in receipts.team:
                reasons.append(f"identity_authority:missing_stage_receipt:team:{key}")
            if not isinstance(row.get("stable_player_id"), str) or not row["stable_player_id"].strip():
                reasons.append(f"identity_authority:missing_stable_player_id:{key}")
            evidence = row.get("reid_evidence")
            if not isinstance(evidence, Mapping) or evidence.get("status") != "valid":
                reasons.append(f"identity_authority:invalid_reid_evidence:{key}")
            if _normalize_embedding(row.get("reid_embedding")) is None:
                reasons.append(f"identity_authority:missing_reid_embedding:{key}")
            team_id = row.get("team_id")
            if type(team_id) is not int:
                reasons.append(f"identity_authority:missing_team_id:{key}")
        if reasons:
            return IdentityAuthorityDecision(False, tuple(dict.fromkeys(reasons)))
        return IdentityAuthorityDecision(
            True,
            (
                "stable_identity_resolved",
                "mask_authoritative_reid_verified",
                "team_id_resolved",
            ),
            {
                "authority": "runtime_identity",
                "checks": {
                    "stable_identity": "passed",
                    "mask_authoritative_reid": "passed",
                    "team_id": "passed",
                },
            },
        )


@dataclass
class _IdentityFragment:
    first_frame: int
    last_frame: int
    raw_track_id: int
    rows: list[dict[str, Any]]
    embeddings: list[np.ndarray]

    @property
    def prototype(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        vector = np.mean(self.embeddings, axis=0)
        norm = float(np.linalg.norm(vector))
        return None if norm <= 1e-12 else vector / norm


class ConservativeStableIdentityResolver:
    """Resolve raw fragments with a high-evidence, fragmentation-first policy."""

    def __init__(self, similarity_threshold: float = 0.90, max_frame_gap: int = 5) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if max_frame_gap < 0:
            raise ValueError("max_frame_gap must be non-negative")
        self.similarity_threshold = similarity_threshold
        self.max_frame_gap = max_frame_gap

    def __call__(self, rows: list[dict[str, Any]]) -> Mapping[str, str]:
        fragments = self._fragments(rows)
        identities: list[_IdentityFragment] = []
        resolved: dict[str, str] = {}
        for fragment in fragments:
            candidates: list[tuple[float, int]] = []
            for index, identity in enumerate(identities):
                if self._eligible(fragment, identity):
                    candidates.append((self._similarity(fragment, identity), index))
            candidates = [candidate for candidate in candidates if candidate[0] >= self.similarity_threshold]
            if candidates:
                candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
                best_score, best_index = candidates[0]
                tied = len(candidates) > 1 and np.isclose(best_score, candidates[1][0], atol=1e-12)
                if not tied:
                    identity = identities[best_index]
                    identity.rows.extend(fragment.rows)
                    identity.embeddings.extend(fragment.embeddings)
                    identity.last_frame = fragment.last_frame
                    stable_id = f"player-{best_index + 1:03d}"
                    for row in fragment.rows:
                        resolved[str(row.get("observation_key", ""))] = stable_id
                    continue

            identities.append(fragment)
            stable_id = f"player-{len(identities):03d}"
            for row in fragment.rows:
                resolved[str(row.get("observation_key", ""))] = stable_id
        return resolved

    def _fragments(self, rows: list[dict[str, Any]]) -> list[_IdentityFragment]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["raw_track_id"])].append(row)
        fragments = []
        for raw_track_id, fragment_rows in grouped.items():
            fragment_rows.sort(key=lambda row: (int(row["frame_index"]), str(row.get("observation_key", ""))))
            embeddings = []
            for row in fragment_rows:
                embedding = _normalize_embedding(row.get("reid_embedding"))
                if embedding is not None:
                    embeddings.append(np.asarray(embedding, dtype=np.float32))
            frames = [int(row["frame_index"]) for row in fragment_rows]
            fragments.append(_IdentityFragment(min(frames), max(frames), raw_track_id, fragment_rows, embeddings))
        return sorted(fragments, key=lambda fragment: (fragment.first_frame, fragment.raw_track_id))

    def _eligible(self, fragment: _IdentityFragment, identity: _IdentityFragment) -> bool:
        if fragment.first_frame <= identity.last_frame:
            return False
        if fragment.first_frame - identity.last_frame - 1 > self.max_frame_gap:
            return False
        return not any(
            int(row["frame_index"]) == fragment.first_frame for row in identity.rows
        )

    @staticmethod
    def _similarity(fragment: _IdentityFragment, identity: _IdentityFragment) -> float:
        first, second = fragment.prototype, identity.prototype
        if first is None or second is None or first.shape != second.shape:
            return -1.0
        return float(np.dot(first, second))


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


def _duplicate_observation_key_reasons(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        key = str(row.get("observation_key", ""))
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    return [f"duplicate_observation_key:{key}" for key in duplicates]


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


def _prepare_masked_crops(
    rows: list[dict[str, Any]],
    frames_by_index: Mapping[int, Any],
    masks_by_observation: Mapping[str, Any],
) -> dict[str, MaskAppearanceCrop]:
    adapter = MaskAppearanceAdapter()
    prepared: dict[str, MaskAppearanceCrop] = {}
    for row in rows:
        key = str(row.get("observation_key", ""))
        frame_index = int(row["frame_index"])
        if frame_index not in frames_by_index:
            continue
        crop = adapter.crop(
            frames_by_index[frame_index],
            row["bbox_xyxy"],
            masks_by_observation.get(key),
            coordinate_space=row.get("bbox_coordinate_space"),
        )
        if crop is not None:
            prepared[key] = crop
    return prepared


def _run_reid(
    rows: list[dict[str, Any]],
    embedder: Embedder,
    frames_by_index: Mapping[int, Any],
    masks_by_observation: Mapping[str, Any] | None,
) -> tuple[int, dict[str, MaskAppearanceCrop], frozenset[str]]:
    prepared = _prepare_masked_crops(rows, frames_by_index, masks_by_observation) if masks_by_observation is not None else {}
    count = 0
    receipt_keys: set[str] = set()
    for row in rows:
        key = str(row.get("observation_key", ""))
        crop = prepared.get(key)
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
        receipt_keys.add(key)
    return count, prepared, frozenset(receipt_keys)


def _run_team_id(
    rows: list[dict[str, Any]],
    team_embedder: TeamEmbedder,
    clusterer: TeamClusterer,
    team_count: int,
    frames_by_index: Mapping[int, Any],
    crops_by_observation: Mapping[str, MaskAppearanceCrop] | None = None,
) -> tuple[dict[str, int], list[str], dict[str, Any]]:
    observations: list[TeamObservation] = []
    next_detection_id = 1
    stable_ids = sorted(
        {
            str(row["stable_player_id"])
            for row in rows
            if isinstance(row.get("stable_player_id"), str) and row["stable_player_id"].strip()
        }
    )
    stable_to_track = {stable_id: index for index, stable_id in enumerate(stable_ids, start=1)}
    for frame_index, frame_rows in _group_by_frame(rows).items():
        if frame_index not in frames_by_index:
            raise ValueError(f"missing frame for team embedding: {frame_index}")
        for row in frame_rows:
            stable_id = row.get("stable_player_id")
            if not isinstance(stable_id, str) or not stable_id.strip():
                continue
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
                # team_id aggregates by stable identity; raw_track_id remains row provenance.
                track_id=stable_to_track[stable_id],
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
        minimum_vote_share=STABLE_PLAYER_TEAM_VOTE_SHARE_THRESHOLD,
    )
    conflicts = []
    vote_diagnostics: dict[str, dict[str, Any]] = {}
    for stable_id in stable_ids:
        diagnostics = result.track_diagnostics.get(stable_to_track[stable_id], {})
        if diagnostics:
            vote_diagnostics[stable_id] = {
                "cluster_votes": dict(diagnostics.get("cluster_votes", {})),
                "total_samples": int(diagnostics.get("eligible_samples", 0)),
            }
        if diagnostics.get("vote_decision") != "assigned":
            conflicts.append(f"conflicting_team_assignment:stable_player_id={stable_id}")

    assignments = {
        stable_id: int(assignment.team_id)
        for assignment in result.assignments
        if assignment.track_id in stable_to_track.values()
        and assignment.team_id is not None
        for stable_id, track_id in stable_to_track.items()
        if track_id == assignment.track_id
    }
    return assignments, conflicts, vote_diagnostics


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
        bbox_sanitizer_config: Any | None = None,
    ) -> RunnerV2Result:
        rows = _copy_observations(observations)
        sanitizer_report = None
        downstream_masks = masks_by_observation
        if bbox_sanitizer_config is not None:
            sanitizer_report = sanitize_temporal_bboxes(
                rows, bbox_sanitizer_config, masks_by_observation=masks_by_observation
            )
            rows = sanitizer_report.rows
            if masks_by_observation is not None:
                downstream_masks = dict(masks_by_observation)
                downstream_masks.update(sanitizer_report.cleaned_masks_by_observation)
        reasons = _duplicate_observation_key_reasons(rows)
        structurally_valid = not reasons
        frames = frames_by_index or {}
        downstream: dict[str, Any] = {}
        runtime_diagnostics: dict[str, Any] = {}
        if sanitizer_report is not None and sanitizer_report.diagnostics:
            runtime_diagnostics["bbox_sanitizer"] = sanitizer_report.diagnostics
        acceptance_reasons = reasons
        stage_receipts = _mint_runtime_stage_receipts()

        appearance_requested = self.seams.reid is not None or self.seams.team_embedder is not None
        if structurally_valid and appearance_requested and downstream_masks is None:
            reasons.extend(
                f"missing_masks_by_observation:{row.get('observation_key', '')}" for row in rows
            )
            structurally_valid = False

        prepared_crops: dict[str, MaskAppearanceCrop] = {}
        if structurally_valid and self.seams.reid is not None:
            downstream["reid_records"], prepared_crops, reid_receipts = _run_reid(
                rows, self.seams.reid, frames, downstream_masks
            )
            stage_receipts = stage_receipts._with(reid=reid_receipts)
        elif structurally_valid and self.seams.team_embedder is not None and downstream_masks is not None:
            prepared_crops = _prepare_masked_crops(rows, frames, downstream_masks)

        if structurally_valid:
            reasons.extend(_resolve_identity(rows, self.seams.identity))
            structurally_valid = not reasons
            if structurally_valid:
                stage_receipts = stage_receipts._with(
                    stable_identity=frozenset(str(row["observation_key"]) for row in rows),
                )

        if structurally_valid and self.seams.team_embedder is not None:
            team_rows = rows
            if downstream_masks is not None:
                team_rows = [
                    row
                    for row in rows
                    if str(row.get("observation_key", "")) in prepared_crops
                    and isinstance(row.get("stable_player_id"), str)
                    and row["stable_player_id"].strip()
                ]
            if team_rows:
                team_assignments, team_reasons, vote_diagnostics = _run_team_id(
                    team_rows,
                    self.seams.team_embedder,
                    self.seams.team_clusterer or KMeansTeamClusterer(),
                    self.seams.team_count,
                    frames,
                    prepared_crops if downstream_masks is not None else None,
                )
                runtime_diagnostics.update({
                    "debug_runtime_only": True,
                    "team_vote_diagnostics": vote_diagnostics,
                })
                reasons.extend(team_reasons)
                if not team_reasons:
                    downstream["team_assignments"] = team_assignments
                    stage_receipts = stage_receipts._with(
                        team=frozenset(
                            str(row["observation_key"])
                            for row in team_rows
                            if row.get("stable_player_id") in team_assignments
                        ),
                    )
                    for row in rows:
                        stable_id = row.get("stable_player_id")
                        if stable_id in team_assignments:
                            row["team_id"] = team_assignments[stable_id]

        authority_decision: IdentityAuthorityDecision | None = None
        if structurally_valid and self.seams.identity_authority is not None:
            authority_decision = self.seams.identity_authority(rows, stage_receipts)
            if not isinstance(authority_decision, IdentityAuthorityDecision):
                raise TypeError("identity_authority must return IdentityAuthorityDecision")
            if authority_decision.accepted and not reasons:
                acceptance_reasons = list(authority_decision.reasons)
            else:
                reasons.extend(authority_decision.reasons or ("identity_authority_rejected",))

        structurally_valid = not reasons
        if not structurally_valid:
            # ReID may have run before a later team check failed. Never expose
            # that partial work as consumable output from a rejected run.
            downstream = {}
            rows = _scrub_derived_observations(rows)

        acceptance = {"tracking_identity": "unvalidated", "statistics": "deferred"}
        if not structurally_valid:
            acceptance["tracking_identity"] = "rejected"
        elif authority_decision is not None:
            acceptance.update(dict(authority_decision.metadata))
            acceptance["tracking_identity"] = "accepted"
        for row in rows:
            row.pop("bbox_coordinate_space", None)
        if not structurally_valid:
            acceptance_reasons = reasons
        return RunnerV2Result(
            rows,
            downstream,
            acceptance,
            acceptance_reasons,
            runtime_diagnostics,
            {} if sanitizer_report is None else sanitizer_report.cleaned_masks_by_observation,
        )


def run_v2(
    observations: Iterable[Mapping[str, Any]],
    *,
    seams: RunnerV2Seams | None = None,
    frames_by_index: Mapping[int, Any] | None = None,
    masks_by_observation: Mapping[str, Any] | None = None,
    bbox_sanitizer_config: Any | None = None,
) -> RunnerV2Result:
    return RunnerV2(seams).run(
        observations,
        frames_by_index=frames_by_index,
        masks_by_observation=masks_by_observation,
        bbox_sanitizer_config=bbox_sanitizer_config,
    )
