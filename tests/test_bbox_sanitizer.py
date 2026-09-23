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
            captured.append((marker, tuple(image.shape[:2]), boxes))
            stage_events.append(f"reid:{marker}")
            return [[3.0, 4.0]]

    rows = [
        box_row(0, (10, 10, 18, 28)),
        box_row(1, (100, 100, 180, 190)),
        box_row(2, (12, 10, 24, 32)),
    ]
    frames = {index: np.full((40, 40, 3), index + 1, dtype=np.uint8) for index in range(3)}
    masks = {row["observation_key"]: np.ones((40, 40), dtype=bool) for row in rows}

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
    assert captured == [
        (1, (18, 8), [[0.0, 0.0, 8.0, 18.0]]),
        (2, (20, 10), [[0.0, 0.0, 10.0, 20.0]]),
        (3, (22, 12), [[0.0, 0.0, 12.0, 22.0]]),
    ]
    assert stage_events == ["reid:1", "reid:2", "reid:3", "identity"]
