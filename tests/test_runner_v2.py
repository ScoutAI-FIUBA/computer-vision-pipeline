from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import numpy as np

from pipeline.runner_v2 import (
    ConservativeStableIdentityResolver,
    MaskAppearanceAdapter,
    RunnerV2,
    RunnerV2Seams,
)
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


def identity_row(frame_index, raw_track_id, embedding, stable_player_id=None):
    return {
        **row(frame_index=frame_index, raw_track_id=raw_track_id, stable_player_id=stable_player_id),
        "reid_embedding": embedding,
        "reid_evidence": {"status": "valid"},
    }


def test_conservative_identity_merges_high_similarity_fragments_within_gap():
    observations = [
        identity_row(10, 3, [1.0, 0.0]),
        identity_row(12, 8, [0.99, 0.01]),
    ]

    resolved = ConservativeStableIdentityResolver(similarity_threshold=0.98, max_frame_gap=2)(
        observations
    )

    assert [resolved[item["observation_key"]] for item in observations] == [
        "player-001",
        "player-001",
    ]


@pytest.mark.parametrize(
    ("second_frame", "second_embedding"),
    [(12, [0.7, 0.3]), (20, [1.0, 0.0])],
)
def test_conservative_identity_fragments_weak_or_long_gap_evidence(second_frame, second_embedding):
    observations = [
        identity_row(10, 3, [1.0, 0.0]),
        identity_row(second_frame, 8, second_embedding),
    ]

    resolved = ConservativeStableIdentityResolver(similarity_threshold=0.98, max_frame_gap=2)(
        observations
    )

    assert resolved["10:3"] == "player-001"
    assert resolved[f"{second_frame}:8"] == "player-002"


def test_conservative_identity_does_not_merge_same_frame_fragments():
    observations = [
        identity_row(10, 3, [1.0, 0.0]),
        identity_row(10, 8, [1.0, 0.0]),
    ]

    resolved = ConservativeStableIdentityResolver()(observations)

    assert resolved == {"10:3": "player-001", "10:8": "player-002"}


def test_conservative_identity_order_is_deterministic_independent_of_input_order():
    observations = [
        identity_row(20, 8, [1.0, 0.0]),
        identity_row(10, 3, [0.0, 1.0]),
    ]

    resolved = ConservativeStableIdentityResolver()(list(reversed(observations)))

    assert resolved == {"10:3": "player-001", "20:8": "player-002"}


def test_rejected_conservative_identity_run_scrubs_derived_fields_atomically():
    observations = [identity_row(10, 3, [1.0, 0.0], stable_player_id="wrong")]

    result = RunnerV2(RunnerV2Seams(identity=ConservativeStableIdentityResolver())).run(observations)

    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.downstream == {}
    assert result.observations == [{key: value for key, value in observations[0].items()
                                    if key not in {"stable_player_id", "reid_embedding", "reid_evidence", "team_id"}}]


def test_runner_without_identity_authority_remains_rejected_without_synthetic_ids():
    result = RunnerV2().run([row()])

    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.acceptance_reasons == ["missing_stable_player_id:60:3"]
    assert "stable_player_id" not in result.observations[0]


def test_duplicate_observation_keys_are_rejected_before_identity_authority_runs():
    def should_not_run(_rows):
        raise AssertionError("identity authority must not receive duplicate keys")

    observations = [row(raw_track_id=3), {**row(raw_track_id=4), "observation_key": "60:3"}]
    result = RunnerV2(RunnerV2Seams(identity=should_not_run)).run(observations)

    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.acceptance_reasons == ["duplicate_observation_key:60:3"]
    assert all("stable_player_id" not in observation for observation in result.observations)


def test_conservative_identity_accepts_exact_similarity_threshold_boundary():
    observations = [
        identity_row(10, 3, [1.0, 0.0]),
        identity_row(11, 8, [0.9, np.sqrt(1.0 - 0.9**2)]),
    ]

    resolved = ConservativeStableIdentityResolver()(observations)

    assert resolved["10:3"] == resolved["11:8"] == "player-001"


@pytest.mark.parametrize(
    ("second_frame", "expected_same_identity"),
    [(16, True), (17, False)],
)
def test_conservative_identity_enforces_exact_default_gap_boundary(second_frame, expected_same_identity):
    observations = [
        identity_row(10, 3, [1.0, 0.0]),
        identity_row(second_frame, 8, [1.0, 0.0]),
    ]

    resolved = ConservativeStableIdentityResolver()(observations)

    assert (resolved["10:3"] == resolved[f"{second_frame}:8"]) is expected_same_identity


def test_conservative_identity_rejects_negative_gap_configuration():
    with pytest.raises(ValueError, match="max_frame_gap must be non-negative"):
        ConservativeStableIdentityResolver(max_frame_gap=-1)


def test_mask_adapter_returns_masked_roi_and_deterministic_metadata():
    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 1

    crop = MaskAppearanceAdapter().crop(frame, [1, 0, 5, 4], mask)

    assert crop is not None
    assert crop.bbox_xyxy == (1, 0, 5, 4)
    assert crop.image.shape == (4, 4, 3)
    assert crop.mask.tolist() == [
        [False] * 4,
        [False, True, True, False],
        [False, True, True, False],
        [False] * 4,
    ]
    assert crop.metadata == {
        "source_shape": (4, 5),
        "roi_xyxy": (1, 0, 5, 4),
        "crop_shape": (4, 4),
        "mask_pixel_count": 4,
    }


def test_mask_adapter_zeros_every_pixel_outside_mask():
    frame = np.full((3, 3, 3), 7, dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    crop = MaskAppearanceAdapter().crop(frame, [0, 0, 3, 3], mask)

    assert crop is not None
    assert np.array_equal(crop.image[1, 1], [7, 7, 7])
    assert int(np.count_nonzero(crop.image)) == 3


def test_mask_adapter_clips_roi_to_frame_bounds():
    frame = np.ones((4, 5, 3), dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[0, 0] = 1

    crop = MaskAppearanceAdapter().crop(frame, [-2, -1, 2, 3], mask)

    assert crop is not None
    assert crop.bbox_xyxy == (0, 0, 2, 3)
    assert crop.image.shape == (3, 2, 3)
    assert crop.metadata["mask_pixel_count"] == 1


@pytest.mark.parametrize(
    "mask",
    [None, np.zeros((3, 3), dtype=np.uint8), np.zeros((3, 3, 1), dtype=np.uint8), np.zeros((2, 3), dtype=np.uint8), [["bad"]]],
)
def test_mask_adapter_rejects_invalid_masks_without_bbox_fallback(mask):
    frame = np.full((3, 3, 3), 9, dtype=np.uint8)

    assert MaskAppearanceAdapter().crop(frame, [0, 0, 3, 3], mask) is None


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_mask_adapter_rejects_non_finite_mask_values(value):
    frame = np.ones((2, 2, 3), dtype=np.uint8)
    mask = np.ones((2, 2), dtype=float)
    mask[0, 0] = value

    assert MaskAppearanceAdapter().crop(frame, [0, 0, 2, 2], mask) is None


def test_mask_adapter_rejects_non_empty_mask_outside_roi():
    frame = np.full((4, 4, 3), 5, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[3, 3] = True

    assert MaskAppearanceAdapter().crop(frame, [0, 0, 2, 2], mask) is None


@pytest.mark.parametrize(
    "bbox",
    [[1, 1, 1, 3], [3, 3, 1, 1], [-5, -5, -1, -1], [5, 5, 8, 8]],
)
def test_mask_adapter_rejects_degenerate_inverted_or_out_of_frame_rois(bbox):
    frame = np.ones((4, 4, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)

    assert MaskAppearanceAdapter().crop(frame, bbox, mask) is None


def test_mask_adapter_is_deterministic_for_identical_inputs():
    frame = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:4, 2:5] = True
    adapter = MaskAppearanceAdapter()

    first = adapter.crop(frame, [1, 0, 6, 5], mask)
    second = adapter.crop(frame, [1, 0, 6, 5], mask)

    assert first is not None and second is not None
    assert first.metadata == second.metadata
    assert first.bbox_xyxy == second.bbox_xyxy
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)


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

    frame = np.ones((24, 12, 3), dtype=np.uint8)
    result = RunnerV2(RunnerV2Seams(identity=resolve, reid=FakeEmbedder())).run(
        [row()], frames_by_index={60: frame}, masks_by_observation={"60:3": np.ones((24, 12), dtype=bool)}
    )

    assert calls[0][0].shape == (20, 10, 3)
    assert calls[0][1] == [[0.0, 0.0, 10.0, 20.0]]
    assert result.observations[0]["stable_player_id"] == "player-3"
    assert result.observations[0]["reid_embedding"] == pytest.approx([0.25 / 0.7905694, 0.75 / 0.7905694])
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
    ).run(
        observations,
        frames_by_index={60: np.ones((24, 12, 3), dtype=np.uint8)},
        masks_by_observation={key: np.ones((24, 12), dtype=bool) for key in ("60:3", "60:4")},
    )

    assert result.downstream["team_assignments"] == {
        "player-3": 0,
        "player-4": 1,
    }
    assert result.acceptance["tracking_identity"] == "unvalidated"


def test_team_assignments_are_serialized_on_each_stable_observation():
    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, frame, samples):
            return [TeamFeatures(color_signature=[float(sample.track_id)], embedding=None) for sample in samples]

    class FakeClusterer:
        name = "fake-clusterer"

        def fit_predict(self, embeddings, team_count):
            return [1]

    result = RunnerV2(
        RunnerV2Seams(identity=resolve, team_embedder=FakeTeamEmbedder(), team_clusterer=FakeClusterer())
    ).run(
        [row()],
        frames_by_index={60: np.ones((24, 12, 3), dtype=np.uint8)},
        masks_by_observation={"60:3": np.ones((24, 12), dtype=bool)},
    )

    assert result.observations[0]["raw_track_id"] == 3
    assert result.observations[0]["stable_player_id"] == "player-3"
    assert result.observations[0]["team_id"] == 1


def test_rejected_runner_result_keeps_explicit_rejection_state():
    result = RunnerV2(RunnerV2Seams(identity=lambda _rows: {})).run([row()])

    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.acceptance_reasons == ["missing_stable_player_id:60:3"]


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
    ).run(
        observations,
        frames_by_index={
            60: np.ones((24, 12, 3), dtype=np.uint8),
            61: np.ones((24, 12, 3), dtype=np.uint8),
        },
        masks_by_observation={
            "60:3": np.ones((24, 12), dtype=bool),
            "61:4": np.ones((24, 12), dtype=bool),
        },
    )

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
    ).run(
        observations,
        frames_by_index={
            60: np.ones((24, 12, 3), dtype=np.uint8),
            61: np.ones((24, 12, 3), dtype=np.uint8),
        },
        masks_by_observation={
            "60:3": np.ones((24, 12), dtype=bool),
            "61:4": np.ones((24, 12), dtype=bool),
        },
    )

    assert result.downstream["team_assignments"] == {"player-7": 1}
    assert result.acceptance["tracking_identity"] == "unvalidated"


def test_team_id_runs_after_identity_and_groups_fragments_by_stable_player():
    calls = []
    team_calls = []

    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, frame, samples):
            calls.append([(sample.track_id, sample.player_mask.shape) for sample in samples])
            team_calls.append((np.array(frame, copy=True), samples))
            return [TeamFeatures(color_signature=[1.0], embedding=None) for _ in samples]

    class RecordingClusterer:
        name = "recording-clusterer"

        def fit_predict(self, embeddings, team_count):
            return [0, 0, 1]

    observations = [
        {**row(frame_index=60, raw_track_id=3), "bbox_xyxy": [1, 1, 5, 5]},
        {**row(frame_index=61, raw_track_id=4), "bbox_xyxy": [1, 1, 5, 5]},
        {**row(frame_index=62, raw_track_id=5), "bbox_xyxy": [1, 1, 5, 5]},
    ]
    events = []

    def identity(rows):
        events.append("identity")
        return {
            "60:3": "player-7",
            "61:4": "player-7",
            "62:5": "player-8",
        }

    class OrderedTeamEmbedder(FakeTeamEmbedder):
        def embed(self, frame, samples):
            events.append("team")
            return super().embed(frame, samples)

    result = RunnerV2(
        RunnerV2Seams(
            identity=identity,
            team_embedder=OrderedTeamEmbedder(),
            team_clusterer=RecordingClusterer(),
        )
    ).run(
        observations,
        frames_by_index={
            index: np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
            for index in (60, 61, 62)
        },
        masks_by_observation={
            item["observation_key"]: np.pad(
                np.ones((2, 2), dtype=bool), ((2, 1), (2, 2))
            )
            for item in observations
        },
    )

    assert events[0] == "identity"
    assert events[1:] == ["team", "team", "team"]
    assert calls == [[(1, (4, 4))], [(1, (4, 4))], [(2, (4, 4))]]
    expected_frame = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)[1:5, 1:5].copy()
    expected_mask = np.zeros((4, 4), dtype=bool)
    expected_mask[1:3, 1:3] = True
    expected_frame[~expected_mask] = 0
    assert all(np.array_equal(frame, expected_frame) for frame, _ in team_calls)
    assert all(
        sample.xyxy == (0.0, 0.0, 4.0, 4.0)
        and np.array_equal(sample.player_mask, expected_mask)
        for _, samples in team_calls
        for sample in samples
    )
    assert result.downstream["team_assignments"] == {"player-7": 0, "player-8": 1}


def test_team_id_rejects_conflicting_labels_after_stable_aggregation_even_with_raw_majority():
    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, frame, samples):
            return [TeamFeatures(color_signature=[1.0], embedding=None) for _ in samples]

    class ConflictingClusterer:
        name = "conflicting-clusterer"

        def fit_predict(self, embeddings, team_count):
            # The first raw fragment has a local tie; stable aggregation must
            # still reject the stable player instead of keeping the other raw fragment's label.
            return [0, 1, 0]

    observations = [
        row(frame_index=60, raw_track_id=3),
        row(frame_index=61, raw_track_id=3),
        row(frame_index=62, raw_track_id=4),
    ]
    result = RunnerV2(
        RunnerV2Seams(
            identity=lambda rows: {item["observation_key"]: "player-7" for item in rows},
            team_embedder=FakeTeamEmbedder(),
            team_clusterer=ConflictingClusterer(),
        )
    ).run(
        observations,
        frames_by_index={index: np.ones((24, 12, 3), dtype=np.uint8) for index in (60, 61, 62)},
        masks_by_observation={
            item["observation_key"]: np.ones((24, 12), dtype=bool) for item in observations
        },
    )

    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.downstream == {}
    assert result.acceptance_reasons == [
        "conflicting_team_assignment:stable_player_id=player-7"
    ]
    assert all(
        not {"stable_player_id", "team_id"}.intersection(observation)
        for observation in result.observations
    )


def test_team_id_leaves_fragments_without_stable_identity_unassigned():
    team_calls = []

    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, frame, samples):
            team_calls.append(samples)
            return [TeamFeatures(color_signature=[1.0], embedding=None) for _ in samples]

    result = RunnerV2(
        RunnerV2Seams(
            identity=lambda rows: {"60:3": "player-7"},
            team_embedder=FakeTeamEmbedder(),
        )
    ).run(
        [row(frame_index=60, raw_track_id=3), row(frame_index=61, raw_track_id=4)],
        frames_by_index={60: np.ones((24, 12, 3), dtype=np.uint8), 61: np.ones((24, 12, 3), dtype=np.uint8)},
        masks_by_observation={"60:3": np.ones((24, 12), dtype=bool), "61:4": np.ones((24, 12), dtype=bool)},
    )

    assert team_calls == []
    assert result.downstream == {}
    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.acceptance_reasons == ["missing_stable_player_id:61:4"]


def test_v2_team_changes_do_not_modify_legacy_entrypoint():
    assert Path("modal_app.py").read_text() == subprocess.check_output(
        ["git", "show", "HEAD:modal_app.py"], text=True
    )


def test_conflicting_team_validation_discards_reid_output_atomically():
    class FakeEmbedder:
        def embed(self, frame, boxes):
            return [[0.25, 0.75] for _ in boxes]

    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, frame, samples):
            return [TeamFeatures(color_signature=[1.0], embedding=None) for _ in samples]

    class ConflictingClusterer:
        name = "conflicting-clusterer"

        def fit_predict(self, embeddings, team_count):
            return [0, 1]

    observations = [row(frame_index=60, raw_track_id=3), row(frame_index=61, raw_track_id=4)]
    observations[0]["team_id"] = 1
    result = RunnerV2(
        RunnerV2Seams(
            identity=lambda rows: {item["observation_key"]: "player-7" for item in rows},
            reid=FakeEmbedder(),
            team_embedder=FakeTeamEmbedder(),
            team_clusterer=ConflictingClusterer(),
        )
    ).run(observations, frames_by_index={60: "frame-60", 61: "frame-61"})

    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.acceptance["statistics"] == "deferred"
    assert result.downstream == {}
    derived_fields = {"stable_player_id", "reid_embedding", "team_id"}
    assert all(not derived_fields.intersection(observation) for observation in result.observations)


def test_reid_backend_receives_only_masked_roi_pixels_and_local_box():
    calls = []

    class FakeEmbedder:
        def embed(self, frame, boxes):
            calls.append((np.array(frame, copy=True), boxes))
            return [[3.0, 4.0] for _ in boxes]

    frame = np.full((4, 5, 3), 9, dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True
    masked_row = {**row(), "bbox_xyxy": [1, 0, 5, 4]}
    result = RunnerV2(RunnerV2Seams(identity=resolve, reid=FakeEmbedder())).run(
        [masked_row], frames_by_index={60: frame}, masks_by_observation={"60:3": mask}
    )

    assert len(calls) == 1
    received_frame, received_boxes = calls[0]
    assert received_frame.shape == (4, 4, 3)
    assert np.array_equal(received_frame[1:3, 1:3], np.full((2, 2, 3), 9, dtype=np.uint8))
    assert int(np.count_nonzero(received_frame[:1])) == 0
    assert received_boxes == [[0.0, 0.0, 4.0, 4.0]]
    assert result.observations[0]["reid_embedding"] == pytest.approx([0.6, 0.8])
    assert result.observations[0]["reid_evidence"] == {"status": "valid"}


@pytest.mark.parametrize("mask", [None, np.zeros((4, 5), dtype=bool), np.zeros((3, 5), dtype=bool)])
def test_invalid_masks_skip_reid_and_team_evidence(mask):
    reid_calls = []
    team_calls = []

    class FakeEmbedder:
        def embed(self, frame, boxes):
            reid_calls.append((frame, boxes))
            return [[1.0, 0.0] for _ in boxes]

    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, frame, samples):
            team_calls.append((frame, samples))
            return [TeamFeatures(color_signature=[1.0], embedding=None) for _ in samples]

    result = RunnerV2(
        RunnerV2Seams(identity=resolve, reid=FakeEmbedder(), team_embedder=FakeTeamEmbedder())
    ).run([row()], frames_by_index={60: np.ones((4, 5, 3), dtype=np.uint8)}, masks_by_observation={"60:3": mask})

    assert reid_calls == []
    assert team_calls == []
    assert result.observations == [{**row(), "stable_player_id": "player-3", "reid_evidence": {"status": "invalid", "reason": "invalid_mask"}}]
    assert "reid_embedding" not in result.observations[0]
    assert "team_id" not in result.observations[0]


def test_mask_aware_team_mode_never_falls_back_to_full_frame_bbox():
    team_calls = []

    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, frame, samples):
            team_calls.append((frame, samples))
            return [TeamFeatures(color_signature=[1.0], embedding=None) for _ in samples]

    result = RunnerV2(RunnerV2Seams(identity=resolve, team_embedder=FakeTeamEmbedder())).run(
        [row()],
        frames_by_index={60: np.full((4, 5, 3), 7, dtype=np.uint8)},
        masks_by_observation={"60:3": None},
    )

    assert team_calls == []
    assert result.downstream == {}
    assert result.observations[0]["raw_track_id"] == 3


def test_omitted_masks_fail_closed_before_reid_or_team_evidence():
    reid_calls = []
    team_calls = []

    class FakeEmbedder:
        def embed(self, frame, boxes):
            reid_calls.append((frame, boxes))
            return [[1.0, 0.0] for _ in boxes]

    class FakeTeamEmbedder:
        name = "fake-team"

        def embed(self, frame, samples):
            team_calls.append((frame, samples))
            return [TeamFeatures(color_signature=[1.0], embedding=None) for _ in samples]

    result = RunnerV2(
        RunnerV2Seams(identity=resolve, reid=FakeEmbedder(), team_embedder=FakeTeamEmbedder())
    ).run([row()], frames_by_index={60: np.ones((4, 5, 3), dtype=np.uint8)})

    assert reid_calls == []
    assert team_calls == []
    assert result.downstream == {}
    assert result.acceptance["tracking_identity"] == "rejected"
    assert result.acceptance_reasons == ["missing_masks_by_observation:60:3"]


def test_reid_embedding_contract_is_normalized_and_one_result_preserves_raw_track_id():
    class FakeEmbedder:
        def embed(self, frame, boxes):
            return [[10.0, 0.0] for _ in boxes]

    frame = np.ones((4, 5, 3), dtype=np.uint8)
    masks = {"60:3": np.ones((4, 5), dtype=bool)}
    result = RunnerV2(RunnerV2Seams(identity=resolve, reid=FakeEmbedder())).run(
        [row()], frames_by_index={60: frame}, masks_by_observation=masks
    )

    assert len(result.observations) == 1
    assert result.observations[0]["raw_track_id"] == 3
    assert result.observations[0]["reid_embedding"] == [1.0, 0.0]
