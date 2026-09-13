"""Team-identification structure tests.

These cover only the isolated pure-Python scaffolding added ahead of actual
SigLIP or K-means integration.

    python -m pytest tests/test_team_id.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.appearance import ReidConfig  # noqa: E402
from pipeline.team_id import (  # noqa: E402
    TeamIdentificationResult,
    TeamObservation,
    compute_track_coherence,
    identify_teams,
)
from pipeline.team_embedder import (  # noqa: E402
    ColourTeamEmbedder,
    NullTeamEmbedder,
    SiglipTeamEmbedder,
    TeamEmbeddingSample,
    TeamFeatures,
    extract_team_features,
    build_team_embedder,
)
from pipeline.team_id import KMeansTeamClusterer, aggregate_track_embedding, mean_embedding  # noqa: E402


class FakeClusterer:
    name = "fake"

    def fit_predict(self, embeddings: list[list[float]], team_count: int) -> list[int]:
        assert team_count == 2
        return [index % 2 for index, _ in enumerate(embeddings)]


def test_null_team_embedder_returns_one_none_per_sample():
    embedder = NullTeamEmbedder()
    samples = [
        TeamEmbeddingSample(detection_id=10, track_id=1, xyxy=(0.0, 0.0, 10.0, 20.0)),
        TeamEmbeddingSample(detection_id=11, track_id=None, xyxy=(5.0, 5.0, 15.0, 25.0)),
    ]
    assert embedder.embed(None, samples) == [
        TeamFeatures(color_signature=None, embedding=None),
        TeamFeatures(color_signature=None, embedding=None),
    ]


def test_build_team_embedder_selects_none_backend():
    assert isinstance(build_team_embedder("none"), NullTeamEmbedder)


def test_build_team_embedder_selects_fashion_siglip_backend_by_default(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeModel:
        def to(self, device: str):
            assert device != "meta"
            captured["device"] = device
            return self

        def eval(self):
            captured["eval"] = True
            return self

    class FakeOpenClip:
        @staticmethod
        def create_model_and_transforms(model_id):
            captured["model"] = model_id
            return FakeModel(), None, lambda image: image

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    monkeypatch.setattr(
        "pipeline.team_embedder._load_siglip_components",
        lambda: (FakeTorch, object(), object()),
    )
    monkeypatch.setitem(sys.modules, "open_clip", FakeOpenClip)

    assert isinstance(build_team_embedder(), SiglipTeamEmbedder)
    assert captured["model"] == "hf-hub:Marqo/marqo-fashionSigLIP"
    assert captured["device"] == "cpu"
    assert captured["eval"] is True


def test_build_team_embedder_selects_legacy_timm_backend_when_explicit(monkeypatch: pytest.MonkeyPatch):
    class FakeModel:
        def to(self, device: str):
            return self

        def eval(self):
            return self

        def __call__(self, inputs):
            return [[1.0, 0.0]] * len(inputs)

    class FakeTimm:
        @staticmethod
        def create_model(model_id: str, pretrained: bool, num_classes: int):
            assert pretrained is True
            assert num_classes == 0
            return FakeModel()

        class data:
            @staticmethod
            def resolve_model_data_config(model):
                return {}

            @staticmethod
            def create_transform(is_training: bool, **kwargs):
                assert is_training is False
                return lambda image: image

    class FakeImage:
        @staticmethod
        def fromarray(array):
            return array

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        @staticmethod
        def inference_mode():
            class _Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Context()

    monkeypatch.setattr(
        "pipeline.team_embedder._load_siglip_components",
        lambda: (FakeTorch, FakeTimm, FakeImage),
    )

    assert isinstance(build_team_embedder("siglip", model_id="vit_base_patch16_siglip_224"), SiglipTeamEmbedder)


def test_build_team_embedder_passes_custom_siglip_model_id(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeModel:
        def to(self, device: str):
            return self

        def eval(self):
            return self

    class FakeTimm:
        @staticmethod
        def create_model(model_id: str, pretrained: bool, num_classes: int):
            captured["model_id"] = model_id
            return FakeModel()

        class data:
            @staticmethod
            def resolve_model_data_config(model):
                return {}

            @staticmethod
            def create_transform(is_training: bool, **kwargs):
                return lambda image: image

    class FakeImage:
        @staticmethod
        def fromarray(array):
            return array

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    monkeypatch.setattr(
        "pipeline.team_embedder._load_siglip_components",
        lambda: (FakeTorch, FakeTimm, FakeImage),
    )

    build_team_embedder("siglip", model_id="custom-siglip")

    assert captured["model_id"] == "custom-siglip"


def test_build_team_embedder_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown team_embedder backend"):
        build_team_embedder("wat")


def test_siglip_team_embedder_uses_visible_torso_crop(monkeypatch: pytest.MonkeyPatch):
    import numpy as np

    captured = {}

    class FakeModel:
        def to(self, device: str):
            return self

        def eval(self):
            return self

        def __call__(self, inputs):
            return [[3.0, 4.0]]

    class FakeTimm:
        @staticmethod
        def create_model(model_id: str, pretrained: bool, num_classes: int):
            return FakeModel()

        class data:
            @staticmethod
            def resolve_model_data_config(model):
                return {}

            @staticmethod
            def create_transform(is_training: bool, **kwargs):
                return lambda image: image

    class FakeImage:
        @staticmethod
        def fromarray(array):
            captured.setdefault("images", []).append(array)
            return array

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class _Batch(list):
            def to(self, device: str):
                return self

        @staticmethod
        def stack(items):
            return FakeTorch._Batch(items)

        @staticmethod
        def inference_mode():
            class _Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Context()

    monkeypatch.setattr(
        "pipeline.team_embedder._load_siglip_components",
        lambda: (FakeTorch, FakeTimm, FakeImage),
    )

    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    embedder = SiglipTeamEmbedder(model_id="vit_base_patch16_siglip_224")
    samples = [TeamEmbeddingSample(detection_id=10, track_id=1, xyxy=(10.0, 20.0, 50.0, 80.0))]

    vectors = embedder.embed(frame, samples)

    assert captured["images"][0].shape == (16, 22, 3)
    assert vectors[0].embedding == pytest.approx([0.6, 0.8])


def test_fashion_siglip_uses_huggingface_image_features_without_downloads(monkeypatch: pytest.MonkeyPatch):
    import numpy as np

    captured = {}

    class FakeTensor:
        def to(self, device):
            captured["input_device"] = device
            return self

    class FakeModel:
        def to(self, device):
            assert device != "meta"
            captured["model_device"] = device
            return self

        def eval(self):
            captured["eval"] = True
            return self

        def encode_image(self, inputs):
            captured["model_inputs"] = inputs
            return np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)

    class FakeOpenClip:
        @staticmethod
        def create_model_and_transforms(model_id):
            captured["model_load"] = model_id
            return FakeModel(), None, lambda image: FakeTensor()

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        @staticmethod
        def inference_mode():
            class _Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Context()

        class _Batch(list):
            def to(self, device):
                captured["input_device"] = device
                return self

        @staticmethod
        def stack(items):
            return FakeTorch._Batch(items)

    monkeypatch.setattr(
        "pipeline.team_embedder._load_siglip_components",
        lambda: (FakeTorch, object(), types.SimpleNamespace(fromarray=lambda array: array)),
    )
    monkeypatch.setitem(sys.modules, "open_clip", FakeOpenClip)

    embedder = SiglipTeamEmbedder(model_id="Marqo/marqo-fashionSigLIP")
    vectors = embedder.embed_crops([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)

    assert captured["model_load"] == "hf-hub:Marqo/marqo-fashionSigLIP"
    assert captured["model_device"] == "cpu"
    assert captured["input_device"] == "cpu"
    assert captured["eval"] is True
    assert captured["model_inputs"] is not None
    assert np.asarray(vectors).shape == (2, 2)
    assert np.asarray(vectors) == pytest.approx(np.asarray([[0.6, 0.8], [0.0, 1.0]]))


def test_default_fashion_siglip_embed_extracts_crops_and_propagates_features(monkeypatch: pytest.MonkeyPatch):
    import numpy as np

    captured = {}

    class FakeTensor:
        def to(self, device):
            captured["input_device"] = device
            return self

    class FakeModel:
        def to(self, device):
            assert device != "meta"
            captured["model_device"] = device
            return self

        def eval(self):
            captured["eval"] = True
            return self

        def encode_image(self, inputs):
            captured["model_inputs"] = inputs
            return np.asarray([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)

    class FakeOpenClip:
        @staticmethod
        def create_model_and_transforms(model_id):
            captured["model_load"] = model_id
            def transform(image):
                captured.setdefault("images", []).append(image)
                return FakeTensor()

            return FakeModel(), None, transform

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        @staticmethod
        def inference_mode():
            class _Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Context()

        class _Batch(list):
            def to(self, device):
                captured["input_device"] = device
                return self

        @staticmethod
        def stack(items):
            return FakeTorch._Batch(items)

    import pipeline.team_embedder as team_embedder_module

    monkeypatch.setattr(
        team_embedder_module,
        "_load_siglip_components",
        lambda: (FakeTorch, object(), types.SimpleNamespace(fromarray=lambda array: array)),
    )
    monkeypatch.setitem(sys.modules, "open_clip", FakeOpenClip)

    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    frame[20:80, 10:50] = (0, 0, 255)
    samples = [
        TeamEmbeddingSample(detection_id=10, track_id=1, xyxy=(10.0, 20.0, 50.0, 80.0)),
        TeamEmbeddingSample(detection_id=11, track_id=2, xyxy=(50.0, 20.0, 80.0, 80.0)),
    ]

    embedder = build_team_embedder()
    assert isinstance(embedder, SiglipTeamEmbedder)
    features = embedder.embed(frame, samples)

    assert captured["model_load"] == "hf-hub:Marqo/marqo-fashionSigLIP"
    assert captured["images"]
    assert captured["images"][0].shape == (16, 22, 3)
    assert captured["model_inputs"] is not None
    assert captured["model_device"] == captured["input_device"] == "cpu"
    assert captured["eval"] is True
    assert len(features) == 2
    assert all(isinstance(feature, TeamFeatures) for feature in features)
    assert np.asarray([feature.embedding for feature in features]) == pytest.approx(
        np.asarray([[0.6, 0.8], [5.0 / 13.0, 12.0 / 13.0]])
    )
    assert all(feature.color_signature is not None for feature in features)
    assert all(feature.quality_score > 0 for feature in features)
    assert all(feature.debug_crop is not None for feature in features)


def test_extract_team_features_prefers_masked_uniform_region():
    import numpy as np

    frame = np.zeros((50, 40, 3), dtype=np.uint8)
    frame[15:30, 9:21] = (0, 0, 255)
    mask = np.zeros((32, 20), dtype=bool)
    mask[5:20, 4:16] = True

    features = extract_team_features(
        frame,
        TeamEmbeddingSample(detection_id=1, track_id=7, xyxy=(5.0, 10.0, 25.0, 42.0), player_mask=mask),
        [],
        ReidConfig(),
    )

    assert features.used_mask is True
    assert features.debug_mask is not None
    assert features.uniform_pixel_count > 0
    assert features.color_signature is not None


def test_compute_track_coherence_reports_similarity_to_center():
    coherence = compute_track_coherence([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]])
    assert coherence is not None
    assert coherence > 0.99


def test_mean_embedding_normalizes_average_vector():
    merged = mean_embedding([[1.0, 0.0], [1.0, 0.0]])
    assert merged == [1.0, 0.0]


def test_aggregate_track_embedding_trims_outlier_samples():
    merged, sample_count = aggregate_track_embedding(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    )

    assert sample_count == 2
    assert merged == pytest.approx([0.99998724, 0.00505031], abs=1e-5)


def test_kmeans_clusterer_separates_two_groups():
    labels = KMeansTeamClusterer().fit_predict(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        team_count=2,
    )
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_identify_teams_keeps_shape_but_leaves_embedding_only_tracks_unknown():
    result = identify_teams(
        [
            TeamObservation(detection_id=100, track_id=7, embedding=[1.0, 0.0]),
            TeamObservation(detection_id=101, track_id=8, embedding=None),
            TeamObservation(detection_id=102, track_id=None, embedding=[0.0, 1.0]),
        ],
        clusterer=FakeClusterer(),
        team_count=2,
        backend="siglip-placeholder",
        minimum_samples=2,
    )

    assert isinstance(result, TeamIdentificationResult)
    assert [assignment.detection_id for assignment in result.assignments] == [7]
    assert [assignment.team_id for assignment in result.assignments] == [None]
    assert result.skipped_detection_ids == [8]
    assert result.to_dict() == {
        "backend": "siglip-placeholder",
        "clusterer": "global-umap:fake:track-vote",
        "requested_teams": 2,
        "assignments": [
            {"detection_id": 7, "track_id": 7, "team_id": None},
        ],
        "skipped_detection_ids": [8],
    }


def test_identify_teams_votes_per_track_after_global_clustering():
    result = identify_teams(
        [
            TeamObservation(detection_id=100, track_id=7, embedding=[1.0, 0.0]),
            TeamObservation(detection_id=101, track_id=7, embedding=[0.9, 0.1]),
            TeamObservation(detection_id=102, track_id=8, embedding=[0.0, 1.0]),
            TeamObservation(detection_id=103, track_id=8, embedding=[0.1, 0.9]),
        ],
        clusterer=KMeansTeamClusterer(),
        team_count=2,
        assignment_margin=0.03,
        minimum_samples=2,
    )

    assigned = {assignment.track_id: assignment.team_id for assignment in result.assignments}
    assert assigned[7] is not None
    assert assigned[8] is not None
    assert assigned[7] != assigned[8]


def test_identify_teams_keeps_unknown_when_track_vote_is_tied():
    class SplitTrackClusterer:
        name = "split-track"

        def fit_predict(self, embeddings: list[list[float]], team_count: int) -> list[int]:
            return [0, 1, 0, 0]

    result = identify_teams(
        [
            TeamObservation(detection_id=100, track_id=7, embedding=[1.0, 0.0]),
            TeamObservation(detection_id=101, track_id=7, embedding=[0.0, 1.0]),
            TeamObservation(detection_id=102, track_id=8, embedding=[1.0, 0.0]),
            TeamObservation(detection_id=103, track_id=8, embedding=[1.0, 0.0]),
        ],
        clusterer=SplitTrackClusterer(),
        team_count=2,
        assignment_margin=0.03,
        minimum_samples=2,
    )

    assigned = {assignment.track_id: assignment.team_id for assignment in result.assignments}
    assert assigned[7] is None
    assert assigned[8] == 0


def test_identify_teams_returns_empty_assignments_when_nothing_is_embedded():
    result = identify_teams(
        [TeamObservation(detection_id=100, track_id=7, embedding=None)],
        clusterer=FakeClusterer(),
        backend="none",
    )

    assert result.assignments == []
    assert result.skipped_detection_ids == [7]


def test_identify_teams_skips_tracks_below_minimum_samples():
    result = identify_teams(
        [
            TeamObservation(
                detection_id=100,
                track_id=7,
                embedding=[1.0, 0.0],
            ),
            TeamObservation(
                detection_id=101,
                track_id=8,
                embedding=[0.0, 1.0],
            ),
        ],
        clusterer=FakeClusterer(),
        team_count=2,
        minimum_samples=2,
    )

    assert [assignment.detection_id for assignment in result.assignments] == [7, 8]
    assert [assignment.team_id for assignment in result.assignments] == [None, None]
    assert result.skipped_detection_ids == []


def test_identify_teams_excludes_under_sampled_tracks_from_clustering():
    captured = {}

    class CapturingClusterer:
        name = "capturing"

        def fit_predict(self, embeddings: list[list[float]], team_count: int) -> list[int]:
            captured["embeddings"] = embeddings
            assert team_count == 2
            return [0, 0, 1, 1]

    result = identify_teams(
        [
            TeamObservation(detection_id=100, track_id=7, embedding=[1.0, 0.0]),
            TeamObservation(detection_id=101, track_id=7, embedding=[0.9, 0.1]),
            TeamObservation(detection_id=102, track_id=8, embedding=[0.0, 1.0]),
            TeamObservation(detection_id=103, track_id=8, embedding=[0.1, 0.9]),
            TeamObservation(detection_id=104, track_id=99, embedding=[5.0, 5.0]),
        ],
        clusterer=CapturingClusterer(),
        team_count=2,
        minimum_samples=2,
    )

    assert captured["embeddings"] == [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    assigned = {assignment.track_id: assignment.team_id for assignment in result.assignments}
    assert assigned[7] == 0
    assert assigned[8] == 1
    assert assigned[99] is None


def test_identify_teams_uses_colour_when_siglip_embedding_is_missing():
    result = identify_teams(
        [
            TeamObservation(detection_id=100, track_id=7, color_signature=[1.0, 0.0]),
            TeamObservation(detection_id=101, track_id=7, color_signature=[0.9, 0.1]),
            TeamObservation(detection_id=102, track_id=8, color_signature=[0.0, 1.0]),
            TeamObservation(detection_id=103, track_id=8, color_signature=[0.1, 0.9]),
        ],
        clusterer=KMeansTeamClusterer(),
        team_count=2,
        minimum_samples=2,
    )

    assigned = {assignment.track_id: assignment.team_id for assignment in result.assignments}
    assert assigned[7] is not None
    assert assigned[8] is not None
    assert assigned[7] != assigned[8]


def test_identify_teams_rejects_invalid_team_count():
    with pytest.raises(ValueError, match="team_count must be at least 1"):
        identify_teams([], clusterer=FakeClusterer(), team_count=0)


def test_identify_teams_rejects_invalid_minimum_samples():
    with pytest.raises(ValueError, match="minimum_samples must be at least 1"):
        identify_teams([], clusterer=FakeClusterer(), minimum_samples=0)


def test_identify_teams_rejects_invalid_assignment_margin():
    with pytest.raises(ValueError, match="assignment_margin must be at least 0"):
        identify_teams([], clusterer=FakeClusterer(), assignment_margin=-0.01)


def test_identify_teams_rejects_clusterers_with_wrong_label_count():
    class BadClusterer:
        name = "bad"

        def fit_predict(self, embeddings: list[list[float]], team_count: int) -> list[int]:
            return [0]

    with pytest.raises(ValueError, match="different number of labels"):
        identify_teams(
            [
                TeamObservation(detection_id=100, track_id=7, embedding=[1.0, 0.0]),
                TeamObservation(detection_id=101, track_id=8, embedding=[0.0, 1.0]),
            ],
            clusterer=BadClusterer(),
        )
