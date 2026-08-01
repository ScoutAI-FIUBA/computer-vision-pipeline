"""Pure team-identification structures and global clustering wrapper."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class TeamObservation:
    """One accepted team-evidence sample that may be eligible for clustering."""

    detection_id: int
    track_id: int | None
    color_signature: list[float] | None = None
    embedding: list[float] | None = None
    color_sample_count: int = 1
    embedding_sample_count: int = 1


@dataclass(frozen=True)
class TeamAssignment:
    """Cluster label assigned to one observation."""

    detection_id: int
    track_id: int | None
    team_id: int | None


@dataclass(frozen=True)
class TeamIdentificationResult:
    """Serializable output for one team-identification pass."""

    backend: str
    clusterer: str
    requested_teams: int
    assignments: list[TeamAssignment]
    skipped_detection_ids: list[int]
    track_diagnostics: dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "clusterer": self.clusterer,
            "requested_teams": self.requested_teams,
            "assignments": [asdict(assignment) for assignment in self.assignments],
            "skipped_detection_ids": list(self.skipped_detection_ids),
        }


class TeamClusterer(Protocol):
    """Minimal contract for a future K-means-like clusterer."""

    name: str

    def fit_predict(self, embeddings: list[list[float]], team_count: int) -> list[int]:
        """Return one cluster label per embedding."""
        ...


class KMeansTeamClusterer:
    """Small deterministic K-means for track-level team grouping."""

    name = "kmeans"

    def __init__(self, max_iterations: int = 16) -> None:
        self.max_iterations = max_iterations

    def fit_predict(self, embeddings: list[list[float]], team_count: int) -> list[int]:
        import numpy as np

        if not embeddings:
            return []

        data = np.asarray(embeddings, dtype=np.float32)
        cluster_count = min(max(1, team_count), len(data))
        if cluster_count == 1:
            return [0] * len(data)

        centers = _initial_centers(data, cluster_count)
        labels = np.zeros(len(data), dtype=np.int32)
        for _ in range(self.max_iterations):
            distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            next_labels = distances.argmin(axis=1)
            if np.array_equal(labels, next_labels):
                break
            labels = next_labels
            for index in range(cluster_count):
                members = data[labels == index]
                if len(members) > 0:
                    centers[index] = members.mean(axis=0)

        remap = {
            old: new
            for new, old in enumerate(
                sorted(range(cluster_count), key=lambda index: tuple(float(value) for value in centers[index]))
            )
        }
        return [remap[int(label)] for label in labels]


def mean_embedding(embeddings: list[list[float]]) -> list[float] | None:
    """Backwards-compatible wrapper around the track-level aggregator."""

    embedding, _ = aggregate_track_embedding(embeddings)
    return embedding


def aggregate_track_embedding(embeddings: list[list[float]]) -> tuple[list[float] | None, int]:
    """Merge one track's embeddings, trimming samples that disagree with the pack."""

    import numpy as np

    if not embeddings:
        return None, 0

    data = np.asarray(embeddings, dtype=np.float32)
    norms = np.clip(np.linalg.norm(data, axis=1, keepdims=True), 1e-12, None)
    normalized = data / norms

    if len(normalized) == 1:
        return normalized[0].tolist(), 1

    similarities = normalized @ normalized.T
    centrality = similarities.mean(axis=1)
    keep = centrality >= float(np.median(centrality))
    kept = normalized[keep]
    merged = kept.mean(axis=0)
    norm = float(np.linalg.norm(merged))
    if norm <= 0:
        return None, int(keep.sum())
    return (merged / norm).tolist(), int(keep.sum())


def compute_track_coherence(embeddings: list[list[float]]) -> float | None:
    """Mean cosine similarity of a track's samples to its merged centre."""

    import numpy as np

    if len(embeddings) < 2:
        return None

    center, _ = aggregate_track_embedding(embeddings)
    if center is None:
        return None

    data = np.asarray(embeddings, dtype=np.float32)
    norms = np.clip(np.linalg.norm(data, axis=1, keepdims=True), 1e-12, None)
    normalized = data / norms
    center_array = np.asarray(center, dtype=np.float32)
    return round(float((normalized @ center_array).mean()), 4)


def _initial_centers(data, cluster_count: int):
    import numpy as np

    centers = [data[0]]
    while len(centers) < cluster_count:
        distances = np.asarray(
            [min(float(((row - center) ** 2).sum()) for center in centers) for row in data],
            dtype=np.float32,
        )
        centers.append(data[int(distances.argmax())])
    return np.asarray(centers, dtype=np.float32)


def identify_teams(
    observations: list[TeamObservation],
    clusterer: TeamClusterer,
    team_count: int = 2,
    backend: str = "unknown",
    minimum_samples: int = 1,
    assignment_margin: float = 0.03,
) -> TeamIdentificationResult:
    """Cluster accepted samples globally, then vote one team label per track."""

    if team_count < 1:
        raise ValueError("team_count must be at least 1")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be at least 1")
    if assignment_margin < 0:
        raise ValueError("assignment_margin must be at least 0")

    track_ids = sorted({int(observation.track_id) for observation in observations if observation.track_id is not None})
    eligible = [
        observation
        for observation in observations
        if observation.track_id is not None and _observation_vector(observation) is not None
    ]
    labels_by_track_id: dict[int, list[int]] = {track_id: [] for track_id in track_ids}

    if eligible:
        labels = clusterer.fit_predict(
            _reduce_embeddings([_observation_vector(observation) for observation in eligible]),
            team_count,
        )
        if len(labels) != len(eligible):
            raise ValueError("clusterer returned a different number of labels than embeddings")
        for observation, label in zip(eligible, labels):
            labels_by_track_id[int(observation.track_id)].append(int(label))

    assignments: list[TeamAssignment] = []
    track_diagnostics: dict[int, dict] = {}
    skipped: list[int] = []
    for track_id in track_ids:
        vote_labels = labels_by_track_id.get(track_id, [])
        if not vote_labels:
            skipped.append(track_id)
            continue
        team_id, diagnostics = _vote_for_track(vote_labels, minimum_samples, assignment_margin)
        track_diagnostics[track_id] = diagnostics
        assignments.append(TeamAssignment(detection_id=track_id, track_id=track_id, team_id=team_id))

    return TeamIdentificationResult(
        backend=backend,
        clusterer=f"global-umap:{clusterer.name}:track-vote",
        requested_teams=team_count,
        assignments=assignments,
        skipped_detection_ids=skipped,
        track_diagnostics=track_diagnostics,
    )


def _observation_vector(observation: TeamObservation) -> list[float] | None:
    return observation.embedding or observation.color_signature


def _reduce_embeddings(embeddings: list[list[float]]) -> list[list[float]]:
    import numpy as np

    if len(embeddings) < 6:
        return embeddings

    data = np.asarray(embeddings, dtype=np.float32)
    if data.shape[1] <= 2:
        return data.tolist()

    try:
        import umap
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in integration
        raise ModuleNotFoundError("umap-learn is required for global team clustering") from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(15, len(data) - 1)),
        min_dist=0.0,
        metric="cosine",
        random_state=0,
        transform_seed=0,
    )
    reduced = reducer.fit_transform(data)
    return np.asarray(reduced, dtype=np.float32).tolist()


def _vote_for_track(vote_labels: list[int], minimum_samples: int, assignment_margin: float) -> tuple[int | None, dict]:
    vote_counts = Counter(int(label) for label in vote_labels)
    ranked = vote_counts.most_common()
    top_label, top_count = ranked[0]
    next_count = ranked[1][1] if len(ranked) > 1 else 0
    sample_count = len(vote_labels)
    vote_margin = (top_count - next_count) / sample_count
    majority_share = top_count / sample_count

    if sample_count < minimum_samples:
        team_id = None
        reason = "under_sampled"
    elif top_count * 2 <= sample_count:
        team_id = None
        reason = "no_majority"
    elif vote_margin <= assignment_margin:
        team_id = None
        reason = "ambiguous_vote"
    else:
        team_id = int(top_label)
        reason = "assigned"

    return team_id, {
        "eligible_samples": sample_count,
        "cluster_votes": {str(label): count for label, count in ranked},
        "majority_share": round(float(majority_share), 4),
        "vote_margin": round(float(vote_margin), 4),
        "vote_decision": reason,
    }
