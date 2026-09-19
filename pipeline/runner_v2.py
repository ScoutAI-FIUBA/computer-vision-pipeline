"""Truthful SAM 3.1 V2 boundary for normalized player observations.

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

from pipeline.reid import Embedder
from pipeline.team_embedder import TeamEmbedder, TeamEmbeddingSample
from pipeline.team_id import KMeansTeamClusterer, TeamClusterer, TeamObservation, identify_teams


IdentityResolver = Callable[[list[dict[str, Any]]], Mapping[str, str]]


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
        if isinstance(raw_id, bool):
            raise ValueError("raw_track_id must be an integer")
        try:
            row["raw_track_id"] = int(raw_id)
        except (TypeError, ValueError):
            raise ValueError("raw_track_id must be an integer") from None
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

    for raw_id, stable_ids in stable_by_raw.items():
        if len(stable_ids) > 1:
            reasons.append(f"conflicting_stable_player_id:raw_track_id={raw_id}")
    return list(dict.fromkeys(reasons))


def _group_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame_index"])].append(row)
    return grouped


def _run_reid(rows: list[dict[str, Any]], embedder: Embedder, frames_by_index: Mapping[int, Any]) -> int:
    count = 0
    for frame_index, frame_rows in _group_by_frame(rows).items():
        if frame_index not in frames_by_index:
            raise ValueError(f"missing frame for ReID: {frame_index}")
        embeddings = embedder.embed(frames_by_index[frame_index], [row["bbox_xyxy"] for row in frame_rows])
        if len(embeddings) != len(frame_rows):
            raise ValueError("ReID returned a different number of embeddings")
        for row, embedding in zip(frame_rows, embeddings):
            row["reid_embedding"] = embedding
            count += 1
    return count


def _run_team_id(
    rows: list[dict[str, Any]],
    team_embedder: TeamEmbedder,
    clusterer: TeamClusterer,
    team_count: int,
    frames_by_index: Mapping[int, Any],
) -> dict[str, int]:
    observations: list[TeamObservation] = []
    next_detection_id = 1
    for frame_index, frame_rows in _group_by_frame(rows).items():
        if frame_index not in frames_by_index:
            raise ValueError(f"missing frame for team embedding: {frame_index}")
        samples = [
            TeamEmbeddingSample(
                detection_id=index,
                track_id=row["raw_track_id"],
                xyxy=tuple(float(value) for value in row["bbox_xyxy"]),
            )
            for index, row in enumerate(frame_rows, start=next_detection_id)
        ]
        features = team_embedder.embed(frames_by_index[frame_index], samples)
        if len(features) != len(samples):
            raise ValueError("team embedder returned a different number of features")
        for sample, feature in zip(samples, features):
            observations.append(
                TeamObservation(
                    detection_id=sample.detection_id,
                    track_id=sample.track_id,
                    color_signature=feature.color_signature,
                    embedding=feature.embedding,
                )
            )
        next_detection_id += len(samples)

    result = identify_teams(
        observations,
        clusterer=clusterer,
        team_count=team_count,
        backend=team_embedder.name,
        minimum_samples=1,
    )
    raw_to_stable = {row["raw_track_id"]: row["stable_player_id"] for row in rows}
    return {
        raw_to_stable[assignment.track_id]: int(assignment.team_id)
        for assignment in result.assignments
        if assignment.track_id is not None and assignment.team_id is not None
    }


class RunnerV2:
    """Normalize identity and invoke only supported production stage contracts."""

    def __init__(self, seams: RunnerV2Seams | None = None) -> None:
        self.seams = seams or RunnerV2Seams()

    def run(
        self,
        observations: Iterable[Mapping[str, Any]],
        *,
        frames_by_index: Mapping[int, Any] | None = None,
    ) -> RunnerV2Result:
        rows = _copy_observations(observations)
        reasons = _resolve_identity(rows, self.seams.identity)
        accepted = not reasons
        frames = frames_by_index or {}
        downstream: dict[str, Any] = {}

        if accepted and self.seams.reid is not None:
            downstream["reid_records"] = _run_reid(rows, self.seams.reid, frames)
        if accepted and self.seams.team_embedder is not None:
            downstream["team_assignments"] = _run_team_id(
                rows,
                self.seams.team_embedder,
                self.seams.team_clusterer or KMeansTeamClusterer(),
                self.seams.team_count,
                frames,
            )

        acceptance = {
            "tracking_identity": "accepted" if accepted else "rejected",
            "statistics": "deferred",
        }
        return RunnerV2Result(rows, downstream, acceptance, reasons)


def run_v2(
    observations: Iterable[Mapping[str, Any]],
    *,
    seams: RunnerV2Seams | None = None,
    frames_by_index: Mapping[int, Any] | None = None,
) -> RunnerV2Result:
    return RunnerV2(seams).run(observations, frames_by_index=frames_by_index)
