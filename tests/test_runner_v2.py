from __future__ import annotations

import pytest

from pipeline.runner_v2 import RunnerV2, RunnerV2Seams
from pipeline.team_embedder import TeamFeatures


def row(frame_index=60, raw_track_id=3, stable_player_id=None):
    return {
        "observation_key": f"{frame_index}:{raw_track_id}",
        "frame_index": frame_index,
        "raw_track_id": raw_track_id,
        "stable_player_id": stable_player_id,
        "bbox_xyxy": [1, 2, 11, 22],
        "track_state": "tracked",
    }


def resolve(rows):
    return {item["observation_key"]: f"player-{item['raw_track_id']}" for item in rows}


def test_empty_input_is_rejected_without_statistics():
    result = RunnerV2(RunnerV2Seams(identity=resolve)).run([])

    assert result.acceptance == {"tracking_identity": "rejected", "statistics": "deferred"}
    assert result.acceptance_reasons == ["empty_observations"]


@pytest.mark.parametrize("raw_track_id", ["3", 3.0, True, float("nan"), None])
def test_malformed_raw_id_is_rejected_at_the_boundary(raw_track_id):
    with pytest.raises(ValueError, match="raw_track_id must be an integer"):
        RunnerV2(RunnerV2Seams(identity=resolve)).run([row(raw_track_id=raw_track_id)])


def test_missing_identity_mapping_rejects_acceptance_and_statistics():
    result = RunnerV2(RunnerV2Seams(identity=lambda rows: {})).run([row()])

    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.acceptance_reasons == ["missing_stable_player_id:60:3"]
    assert "statistics" not in result.downstream


def test_conflicting_stable_ids_for_one_raw_track_reject_acceptance():
    observations = [row(raw_track_id=3), row(frame_index=61, raw_track_id=3)]

    result = RunnerV2(
        RunnerV2Seams(
            identity=lambda rows: {"60:3": "player-3", "61:3": "player-9"}
        )
    ).run(observations)

    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.acceptance_reasons == ["conflicting_stable_player_id:raw_track_id=3"]


def test_same_frame_raw_tracks_cannot_share_one_stable_player():
    observations = [row(raw_track_id=3), row(raw_track_id=4)]

    result = RunnerV2(
        RunnerV2Seams(
            identity=lambda rows: {"60:3": "player-7", "60:4": "player-7"}
        )
    ).run(observations)

    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.acceptance_reasons == [
        "conflicting_stable_player_id:frame=60:stable_player_id=player-7"
    ]


def test_reid_uses_the_repository_embedder_contract_and_stable_rows():
    calls = []

    class FakeEmbedder:
        def embed(self, frame, boxes):
            calls.append((frame, boxes))
            return [[0.25, 0.75] for _ in boxes]

    result = RunnerV2(
        RunnerV2Seams(identity=resolve, reid=FakeEmbedder())
    ).run([row()], frames_by_index={60: "frame-60"})

    assert calls == [("frame-60", [[1, 2, 11, 22]])]
    assert result.observations[0]["stable_player_id"] == "player-3"
    assert result.observations[0]["reid_embedding"] == [0.25, 0.75]
    assert result.acceptance["tracking_identity"] == "unvalidated"


def test_team_embedder_and_team_id_use_production_contracts_then_map_to_stable_ids():
    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, frame, samples):
            return [
                TeamFeatures(color_signature=[float(sample.track_id)], embedding=None)
                for sample in samples
            ]

    class FakeClusterer:
        name = "fake-clusterer"

        def fit_predict(self, embeddings, team_count):
            return [0, 1]

    observations = [row(raw_track_id=3), row(frame_index=60, raw_track_id=4)]
    result = RunnerV2(
        RunnerV2Seams(
            identity=resolve,
            team_embedder=FakeTeamEmbedder(),
            team_clusterer=FakeClusterer(),
        )
    ).run(observations, frames_by_index={60: "frame-60"})

    assert result.downstream["team_assignments"] == {
        "player-3": 0,
        "player-4": 1,
    }
    assert result.acceptance["tracking_identity"] == "unvalidated"


def test_same_stable_player_with_conflicting_team_assignments_is_rejected():
    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, frame, samples):
            return [TeamFeatures(color_signature=[1.0], embedding=None) for _ in samples]

    class ConflictingClusterer:
        name = "conflicting-clusterer"

        def fit_predict(self, embeddings, team_count):
            return [0, 1]

    observations = [row(frame_index=60, raw_track_id=3), row(frame_index=61, raw_track_id=4)]
    result = RunnerV2(
        RunnerV2Seams(
            identity=lambda rows: {item["observation_key"]: "player-7" for item in rows},
            team_embedder=FakeTeamEmbedder(),
            team_clusterer=ConflictingClusterer(),
        )
    ).run(observations, frames_by_index={60: "frame-60", 61: "frame-61"})

    assert "team_assignments" not in result.downstream
    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.acceptance_reasons == [
        "conflicting_team_assignment:stable_player_id=player-7"
    ]


def test_same_stable_player_with_consistent_team_assignments_is_retained():
    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, frame, samples):
            return [TeamFeatures(color_signature=[1.0], embedding=None) for _ in samples]

    class ConsistentClusterer:
        name = "consistent-clusterer"

        def fit_predict(self, embeddings, team_count):
            return [1, 1]

    observations = [row(frame_index=60, raw_track_id=3), row(frame_index=61, raw_track_id=4)]
    result = RunnerV2(
        RunnerV2Seams(
            identity=lambda rows: {item["observation_key"]: "player-7" for item in rows},
            team_embedder=FakeTeamEmbedder(),
            team_clusterer=ConsistentClusterer(),
        )
    ).run(observations, frames_by_index={60: "frame-60", 61: "frame-61"})

    assert result.downstream["team_assignments"] == {"player-7": 1}
    assert result.acceptance["tracking_identity"] == "unvalidated"
