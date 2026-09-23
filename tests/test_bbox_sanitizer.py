from __future__ import annotations

import numpy as np

from pipeline.bbox_sanitizer import BBoxSanitizerConfig, sanitize_temporal_bboxes
from pipeline.config import PipelineConfig
from pipeline.runner_v2 import RunnerV2Seams, run_v2


def box_row(frame: int, box, track: int = 7, **extra):
    return {
        "observation_key": f"{frame}:{track}",
        "frame_index": frame,
        "raw_track_id": track,
        "bbox_xyxy": list(box),
        "stable_player_id": "player-007",
        "team_id": 1,
        **extra,
    }


def enabled(**kwargs):
    return BBoxSanitizerConfig(enabled=True, **kwargs)


def test_single_frame_spike_is_interpolated_without_mutating_identity_or_raw_bbox():
    rows = [
        box_row(0, (10, 10, 20, 30)),
        box_row(1, (100, 100, 180, 190)),
        box_row(2, (12, 10, 22, 30)),
    ]

    result = sanitize_temporal_bboxes(rows, enabled())

    fixed = result.rows[1]
    assert fixed["bbox_xyxy"] == [11.0, 10.0, 21.0, 30.0]
    assert fixed["raw_bbox"] == [100, 100, 180, 190]
    assert fixed["bbox_corrected"] is True
    assert fixed["bbox_correction_method"] == "linear_interpolation"
    assert fixed["stable_player_id"] == "player-007"
    assert fixed["team_id"] == 1
    assert result.corrected_count == 1


def test_normal_movement_is_not_flagged():
    rows = [box_row(frame, (10 + frame * 2, 10, 20 + frame * 2, 30)) for frame in range(5)]

    result = sanitize_temporal_bboxes(rows, enabled())

    assert result.corrected_count == 0
    assert result.unresolved_count == 0
    assert all(not row.get("bbox_outlier", False) for row in result.rows)


def test_two_frame_short_glitch_is_interpolated():
    rows = [
        box_row(0, (10, 10, 20, 30)),
        box_row(1, (100, 100, 180, 190)),
        box_row(2, (110, 100, 190, 190)),
        box_row(3, (16, 10, 26, 30)),
    ]

    result = sanitize_temporal_bboxes(rows, enabled(max_interpolation_gap=2))

    assert [row["bbox_xyxy"] for row in result.rows[1:3]] == [
        [12.0, 10.0, 22.0, 30.0],
        [14.0, 10.0, 24.0, 30.0],
    ]
    assert result.corrected_count == 2


def test_long_corruption_is_preserved_and_marked_unresolved():
    rows = [box_row(0, (10, 10, 20, 30))]
    rows.extend(box_row(frame, (100, 100, 180, 190)) for frame in range(1, 5))
    rows.append(box_row(5, (20, 10, 30, 30)))

    result = sanitize_temporal_bboxes(rows, enabled(max_interpolation_gap=2))

    assert result.corrected_count == 0
    assert result.unresolved_count == 4
    assert all(row["bbox_corrected"] is False for row in result.rows[1:5])
    assert all(row["bbox_unresolved"] is True for row in result.rows[1:5])


def test_ambiguous_neighbors_are_not_corrected():
    rows = [
        box_row(0, (10, 10, 20, 30)),
        box_row(1, (100, 100, 180, 190)),
        box_row(2, (300, 10, 310, 30)),
    ]

    result = sanitize_temporal_bboxes(rows, enabled())

    assert result.rows[1]["bbox_xyxy"] == [100, 100, 180, 190]
    assert result.rows[1]["bbox_unresolved"] is True
    assert result.corrected_count == 0


def test_disabled_config_returns_rows_unchanged():
    rows = [box_row(0, (10, 10, 20, 30)), box_row(1, (100, 100, 180, 190))]

    result = sanitize_temporal_bboxes(rows, BBoxSanitizerConfig())

    assert result.rows == rows
    assert result.corrected_count == 0
    assert result.unresolved_count == 0


def test_runner_v2_sanitizes_before_downstream_without_changing_identity_fields():
    rows = [
        box_row(0, (10, 10, 20, 30)),
        box_row(1, (100, 100, 180, 190)),
        box_row(2, (12, 10, 22, 30)),
    ]

    result = run_v2(
        rows,
        seams=RunnerV2Seams(
            identity=lambda observations: {
                f"{item['frame_index']}:{item['raw_track_id']}": "player-007"
                for item in observations
            }
        ),
        bbox_sanitizer_config=enabled(),
    )

    assert result.observations[1]["bbox_xyxy"] == [11.0, 10.0, 21.0, 30.0]
    assert result.observations[1]["raw_track_id"] == 7
    assert result.observations[1]["stable_player_id"] == "player-007"
    assert result.observations[1]["team_id"] == 1
    assert result.runtime_diagnostics["bbox_sanitizer"] == {
        "corrected_count": 1,
        "unresolved_count": 0,
    }


def test_v2_environment_config_reaches_sanitizer_before_reid(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("BBOX_SANITIZER_ENABLED", "1")
    monkeypatch.setenv("BBOX_HISTORY_SIZE", "7")
    monkeypatch.setenv("BBOX_MAX_INTERPOLATION_GAP", "2")
    monkeypatch.setenv("BBOX_SIGNAL_THRESHOLD", "2.5")
    config = PipelineConfig.from_env()
    rows = [box_row(10, (10, 10, 20, 30)), box_row(11, (100, 100, 180, 190)), box_row(12, (12, 10, 22, 30))]
    frames = {index: np.full((40, 40, 3), index - 9, dtype=np.uint8) for index in (10, 11, 12)}
    masks = {row["observation_key"]: np.ones((40, 40), dtype=bool) for row in rows}
    reid_crops = []

    class FakeEmbedder:
        def embed(self, image, boxes):
            reid_crops.append((int(image[0, 0, 0]), boxes))
            return [[3.0, 4.0]]

    result = run_v2(
        rows,
        seams=RunnerV2Seams(
            identity=lambda items: {item["observation_key"]: "player-007" for item in items},
            reid=FakeEmbedder(),
        ),
        frames_by_index=frames,
        masks_by_observation=masks,
        bbox_sanitizer_config=config,
    )

    assert config.bbox_sanitizer_enabled is True
    assert config.max_interpolation_gap == 2
    assert result.observations[1]["bbox_corrected"] is True
    assert reid_crops[1] == (2, [[0.0, 0.0, 10.0, 20.0]])


def test_explicit_disabled_pipeline_config_preserves_v2_rows(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    monkeypatch.setenv("BBOX_SANITIZER_ENABLED", "0")
    row = box_row(0, (0.1, 0.1, 0.2, 0.3), bbox_coordinate_space="normalized")

    result = run_v2(
        [row],
        seams=RunnerV2Seams(identity=lambda items: {item["observation_key"]: "player-007" for item in items}),
        bbox_sanitizer_config=PipelineConfig.from_env(),
    )

    assert result.observations == [{key: value for key, value in row.items() if key != "bbox_coordinate_space"}]


def test_sanitizer_preserves_raw_bbox_and_coordinate_metadata_for_alternate_and_malformed_boxes():
    rows = [
        box_row(10, (0.1, 0.1, 0.2, 0.3), bbox_coordinate_space="normalized", raw_bbox=["original"]),
        box_row(12, (0.12, 0.1, 0.22, 0.3), bbox_coordinate_space="normalized"),
        {**box_row(14, (0.1, 0.1, 0.2, 0.3), bbox_coordinate_space="normalized"), "bbox": [0.1, 0.1, 0.2, 0.3]},
        {**box_row(16, (0, 0, 1, 1), bbox_coordinate_space="pixel"), "bbox_xyxy": "malformed"},
    ]
    rows[2].pop("bbox_xyxy")

    result = sanitize_temporal_bboxes(rows, enabled(max_interpolation_gap=2))

    assert result.rows[0]["raw_bbox"] == ["original"]
    assert result.rows[0]["bbox_coordinate_space"] == "normalized"
    assert result.rows[2]["bbox_xyxy"] == [0.1, 0.1, 0.2, 0.3]
    assert result.rows[3]["bbox_unresolved"] is False
    assert result.rows[3]["bbox_xyxy"] == "malformed"


def test_non_consecutive_frames_use_frame_gap_not_row_count():
    rows = [
        box_row(10, (10, 10, 20, 30)),
        box_row(12, (100, 100, 180, 190)),
        box_row(14, (12, 10, 22, 30)),
    ]

    result = sanitize_temporal_bboxes(rows, enabled(max_interpolation_gap=1))

    assert result.corrected_count == 0
    assert result.rows[1]["bbox_unresolved"] is True


def test_corrected_bbox_reaches_reid_boundary_before_identity_processing():
    captured = []
    stage_events = []

    class FakeEmbedder:
        def embed(self, image, boxes):
            marker = int(image[0, 0, 0])
            captured.append((marker, image.copy(), tuple(image.shape[:2]), boxes))
            stage_events.append(f"reid:{marker}")
            return [[3.0, 4.0]]

    rows = [
        box_row(0, (10, 10, 18, 28)),
        box_row(1, (100, 100, 180, 190), mask_ref="masks/raw-1.png"),
        box_row(2, (12, 10, 24, 32)),
    ]
    frames = {index: np.full((40, 40, 3), index + 1, dtype=np.uint8) for index in range(3)}
    middle_mask = _mask(rects=((11, 10, 21, 30),))
    middle_mask[16:22, 14:18] = 0
    middle_mask[18:20, 15:17] = 1
    masks = {
        "0:7": _mask(rects=((10, 10, 18, 28), (35, 35, 38, 38))),
        "1:7": middle_mask,
        "2:7": _mask(rects=((12, 10, 24, 32), (35, 35, 38, 38))),
    }

    def resolve(items):
        stage_events.append("identity")
        assert len(captured) == 3
        return {item["observation_key"]: "player-007" for item in items}

    result = run_v2(
        rows,
        seams=RunnerV2Seams(
            identity=resolve,
            reid=FakeEmbedder(),
        ),
        frames_by_index=frames,
        masks_by_observation=masks,
        bbox_sanitizer_config=enabled(),
    )

    assert [row["bbox_xyxy"] for row in result.observations] == [
        [10, 10, 18, 28],
        [11.0, 10.0, 21.0, 30.0],
        [12, 10, 24, 32],
    ]
    assert result.observations[1]["raw_bbox"] == [100, 100, 180, 190]
    assert result.observations[1]["raw_mask_ref"] == "masks/raw-1.png"
    assert [(marker, shape, boxes) for marker, _image, shape, boxes in captured] == [
        (1, (18, 8), [[0.0, 0.0, 8.0, 18.0]]),
        (2, (20, 10), [[0.0, 0.0, 10.0, 20.0]]),
        (3, (22, 12), [[0.0, 0.0, 12.0, 22.0]]),
    ]
    assert captured[1][1].shape == (20, 10, 3)
    assert int(captured[1][1].sum()) == 1056
    assert captured[1][1][9, 5].tolist() == [0, 0, 0]
    assert masks["1:7"].sum() == 180
    assert result.cleaned_masks_by_observation["1:7"].sum() == 176
    assert stage_events == ["reid:1", "reid:2", "reid:3", "identity"]


def _mask(shape=(40, 40), rects=()):
    mask = np.zeros(shape, dtype=np.uint8)
    for x1, y1, x2, y2 in rects:
        mask[y1:y2, x1:x2] = 1
    return mask


def test_mask_cleanup_keeps_main_component_and_removes_remote_blob():
    rows = [
        box_row(0, (10, 10, 20, 30)),
        box_row(1, (100, 100, 180, 190)),
        box_row(2, (12, 10, 22, 30)),
    ]
    masks = {row["observation_key"]: _mask(rects=((11, 10, 21, 30), (35, 35, 38, 38))) for row in rows}

    result = sanitize_temporal_bboxes(rows, enabled(), masks_by_observation=masks)

    fixed = result.rows[1]
    assert fixed["bbox_xyxy"] == [11.0, 10.0, 21.0, 30.0]
    assert fixed["bbox_correction_method"] == "mask_cleanup"
    assert fixed["bbox_corrected"] is True
    assert result.cleaned_masks_by_observation["1:7"].sum() == 200
    assert masks["1:7"].sum() == 209


def test_mask_cleanup_count_includes_mask_only_corrections():
    row = box_row(0, (5, 5, 15, 15))
    mask = _mask(rects=((5, 5, 15, 15), (30, 30, 32, 32)))

    result = sanitize_temporal_bboxes([row], enabled(), masks_by_observation={"0:7": mask})

    assert result.rows[0]["bbox_xyxy"] == [5.0, 5.0, 15.0, 15.0]
    assert result.rows[0]["mask_cleanup_corrected"] is True
    assert result.diagnostics["mask_cleanup_count"] == 1


def test_mask_cleanup_does_not_choose_between_two_ambiguous_large_components():
    rows = [box_row(0, (8, 8, 20, 28)), box_row(1, (100, 100, 180, 190)), box_row(2, (22, 8, 34, 28))]
    masks = {row["observation_key"]: _mask(rects=((8, 8, 20, 28), (22, 8, 34, 28))) for row in rows}

    result = sanitize_temporal_bboxes(rows, enabled(), masks_by_observation=masks)

    assert result.rows[1]["bbox_xyxy"] == [100, 100, 180, 190]
    assert result.rows[1]["bbox_correction_method"] == "unresolved_outlier"
    assert result.rows[1]["bbox_unresolved"] is True
    assert result.rows[1]["bbox_outlier_diagnostics"]["mask_cleanup"]["reason"] == "ambiguous_components"


def test_clean_single_component_recalculates_bbox_and_preserves_raw_mask():
    rows = [box_row(0, (0, 0, 40, 40), bbox_coordinate_space="pixel", mask_ref="masks/raw.png")]
    raw_mask = _mask(rects=((5, 7, 15, 19),))

    result = sanitize_temporal_bboxes(rows, enabled(), masks_by_observation={"0:7": raw_mask})

    assert result.rows[0]["bbox_xyxy"] == [5.0, 7.0, 15.0, 19.0]
    assert result.rows[0]["raw_bbox"] == [0, 0, 40, 40]
    assert result.rows[0]["raw_mask_ref"] == "masks/raw.png"
    assert np.array_equal(raw_mask, _mask(rects=((5, 7, 15, 19),)))


def test_mask_cleanup_converts_pixel_bbox_to_normalized_coordinates():
    row = box_row(0, (0.0, 0.0, 1.0, 1.0), bbox_coordinate_space="normalized")
    mask = _mask(shape=(20, 40), rects=((4, 5, 14, 15),))

    result = sanitize_temporal_bboxes([row], enabled(), masks_by_observation={"0:7": mask})

    assert result.rows[0]["bbox_xyxy"] == [0.1, 0.25, 0.35, 0.75]


def test_mask_cleanup_falls_back_to_temporal_interpolation_when_mask_is_unusable():
    rows = [box_row(0, (10, 10, 20, 30)), box_row(1, (100, 100, 180, 190)), box_row(2, (12, 10, 22, 30))]
    masks = {"1:7": np.zeros((40, 40), dtype=np.uint8)}

    result = sanitize_temporal_bboxes(rows, enabled(), masks_by_observation=masks)

    assert result.rows[1]["bbox_xyxy"] == [11.0, 10.0, 21.0, 30.0]
    assert result.rows[1]["bbox_correction_method"] == "linear_interpolation"


def test_runner_reid_receives_clean_mask_geometry():
    rows = [box_row(0, (100, 100, 180, 190), mask_ref="masks/raw.png")]
    mask = _mask(rects=((5, 7, 15, 19),))
    captured = []

    class Embedder:
        def embed(self, image, boxes):
            captured.append((image.shape[:2], boxes))
            return [[1.0, 0.0]]

    result = run_v2(
        rows,
        seams=RunnerV2Seams(identity=lambda items: {items[0]["observation_key"]: "player-007"}, reid=Embedder()),
        frames_by_index={0: np.zeros((40, 40, 3), dtype=np.uint8)},
        masks_by_observation={"0:7": mask},
        bbox_sanitizer_config=enabled(),
    )

    assert captured == [((12, 10), [[0.0, 0.0, 10.0, 12.0]])]
    assert result.cleaned_masks_by_observation["0:7"].sum() == 120


def test_multiple_components_without_temporal_evidence_are_unresolved():
    row = box_row(0, (1, 1, 20, 20))
    mask = _mask(rects=((2, 2, 12, 12), (25, 2, 32, 9)))

    result = sanitize_temporal_bboxes([row], enabled(), masks_by_observation={"0:7": mask})

    assert result.rows[0]["bbox_xyxy"] == [1, 1, 20, 20]
    assert result.rows[0]["bbox_correction_method"] == "unresolved_mask"
    assert result.rows[0]["bbox_unresolved"] is True
    assert result.rows[0]["bbox_outlier_diagnostics"]["mask_cleanup"]["reason"] == "ambiguous_components_no_temporal_evidence"
    assert result.unresolved_count == 1
