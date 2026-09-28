from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.sam31_segments import (
    assert_segment_object_limit,
    deduplicate_segment_observations,
    extract_segment_video,
    namespace_raw_track_id,
    plan_segment_windows,
    remap_segment_rows,
    stitch_segment_seams,
)


def _write_video(path: Path, frame_count: int = 5) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (16, 12))
    for index in range(frame_count):
        writer.write(np.full((12, 16, 3), index, dtype=np.uint8))
    writer.release()


def test_planner_has_global_coverage_and_later_overlap_ownership():
    windows = plan_segment_windows(10, 24, window=6, overlap=1)
    assert [(item.global_start, item.global_end, item.owned_start) for item in windows] == [
        (10, 15, 10), (15, 20, 15), (20, 24, 20)
    ]
    owned = [frame for item in windows for frame in range(item.owned_start, item.global_end + 1)]
    assert set(owned) == set(range(10, 25))


def test_remap_namespaces_reused_ids_and_masks():
    first = plan_segment_windows(0, 3, window=4)[0]
    rows, masks = remap_segment_rows(
        [{"frame_index": 0, "raw_track_id": 7, "observation_key": "0:7"}],
        {"0:7": "mask-a"},
        first,
    )
    assert rows[0]["raw_track_id"] == namespace_raw_track_id(0, 7)
    assert rows[0]["local_raw_track_id"] == 7
    assert rows[0]["raw_observation_key"] == "0:7"
    assert masks == {rows[0]["observation_key"]: "mask-a"}

    second = plan_segment_windows(0, 6, window=4, overlap=1)[1]
    rows2, _ = remap_segment_rows(
        [{"frame_index": 0, "raw_track_id": 7, "observation_key": "0:7"}], {}, second
    )
    assert rows2[0]["raw_track_id"] != rows[0]["raw_track_id"]
    assert rows2[0]["raw_observation_key"] == "0:7"


def test_deduplication_keeps_later_segment_and_aligned_mask():
    rows = [
        {"frame_index": 3, "local_raw_track_id": 7, "segment_index": 0, "raw_track_id": 7, "raw_observation_key": "0:7", "observation_key": "3:7"},
        {"frame_index": 3, "local_raw_track_id": 7, "segment_index": 1, "raw_track_id": 1_000_007, "raw_observation_key": "0:7", "observation_key": "3:1000007"},
    ]
    final, masks = deduplicate_segment_observations(rows, {"3:7": "old", "3:1000007": "new"})
    assert [row["observation_key"] for row in final] == ["3:1000007"]
    assert final[0]["raw_observation_key"] == "0:7"
    assert masks == {"3:1000007": "new"}


def test_deduplication_reports_overlap_ownership():
    diagnostics = []
    rows = [
        {"frame_index": 3, "local_raw_track_id": 7, "segment_index": 0, "raw_track_id": 7, "observation_key": "3:7"},
        {"frame_index": 3, "local_raw_track_id": 7, "segment_index": 1, "raw_track_id": 1_000_007, "observation_key": "3:1000007"},
    ]

    deduplicate_segment_observations(
        rows, {"3:7": "old", "3:1000007": "new"}, diagnostics=diagnostics
    )

    assert diagnostics == [{
        "event": "ownership",
        "frame_index": 3,
        "canonical_raw_track_id": 7,
        "owner_segment_index": 1,
        "replaced_segment_index": 0,
        "observation_key": "3:1000007",
        "replaced_observation_key": "3:7",
    }]


def test_later_owner_without_mask_drops_earlier_mask_and_reports_truthful_provenance():
    from pipeline.bbox_sanitizer import BBoxSanitizerConfig, sanitize_temporal_bboxes

    rows = [
        {
            "frame_index": 3,
            "raw_track_id": 7,
            "raw_observation_key": "3:7",
            "canonical_raw_track_id": 7,
            "local_raw_track_id": 7,
            "segment_index": 0,
            "observation_key": "3:7",
        },
        {
            "frame_index": 3,
            "raw_track_id": 1_000_020,
            "raw_observation_key": "3:1000020",
            "canonical_raw_track_id": 7,
            "local_raw_track_id": 20,
            "segment_index": 1,
            "observation_key": "3:1000020",
        },
    ]
    diagnostics = []
    final, final_masks = deduplicate_segment_observations(
        rows,
        {"3:7": np.ones((2, 2), dtype=np.uint8)},
        diagnostics=diagnostics,
    )
    assert [row["raw_track_id"] for row in final] == [1_000_020]
    assert final[0]["raw_observation_key"] == "3:1000020"
    assert final[0]["mask_status"] == "missing"
    assert final_masks == {}
    assert diagnostics == [
        {
            "event": "ownership",
            "frame_index": 3,
            "canonical_raw_track_id": 7,
            "owner_segment_index": 1,
            "replaced_segment_index": 0,
            "observation_key": "3:1000020",
            "replaced_observation_key": "3:7",
        },
        {
            "event": "owned_observation_missing_mask",
            "frame_index": 3,
            "canonical_raw_track_id": 7,
            "raw_track_id": 1_000_020,
            "observation_key": "3:1000020",
            "raw_observation_key": "3:1000020",
        },
    ]
    sanitized = sanitize_temporal_bboxes(
        final,
        BBoxSanitizerConfig(enabled=True),
        masks_by_observation=final_masks,
    )
    assert sanitized.cleaned_masks_by_observation == {}


def _seam_rows(previous_ids, current_ids, boxes):
    rows = []
    masks = {}
    for segment_index, ids in enumerate((previous_ids, current_ids)):
        for raw_id, box in zip(ids, boxes[segment_index]):
            key = f"3:{raw_id}"
            rows.append({
                "frame_index": 3,
                "raw_track_id": raw_id,
                "local_raw_track_id": raw_id,
                "segment_index": segment_index,
                "observation_key": key,
                "bbox_xyxy": box,
            })
            mask = np.zeros((12, 16), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 1
            masks[key] = mask
    return rows, masks


def test_seam_stitching_matches_reversed_local_ids_by_mask_and_preserves_provenance():
    rows, masks = _seam_rows([7, 8], [20, 10], [([1, 1, 4, 5], [10, 1, 13, 5]), ([10, 1, 13, 5], [1, 1, 4, 5])])
    stitched, stitched_masks = stitch_segment_seams(rows, masks)
    current = [row for row in stitched if row["segment_index"] == 1]
    assert {row["raw_track_id"] for row in current} == {10, 20}
    assert {row["canonical_raw_track_id"] for row in current} == {7, 8}
    assert {row["local_raw_track_id"] for row in current} == {10, 20}
    assert {row["observation_key"] for row in current} == {"3:10", "3:20"}
    assert set(stitched_masks) == {row["observation_key"] for row in stitched}
    final, final_masks = deduplicate_segment_observations(stitched, stitched_masks)
    assert {row["raw_track_id"] for row in final} == {10, 20}
    assert len({row["observation_key"] for row in final}) == len(final)
    assert set(final_masks) == {row["observation_key"] for row in final}


def test_seam_stitching_never_pairs_rows_from_different_overlap_frames():
    rows = [
        {"frame_index": 3, "raw_track_id": 7, "local_raw_track_id": 7, "segment_index": 0, "observation_key": "3:7"},
        {"frame_index": 4, "raw_track_id": 7, "local_raw_track_id": 7, "segment_index": 0, "observation_key": "4:7"},
        {"frame_index": 3, "raw_track_id": 20, "local_raw_track_id": 20, "segment_index": 1, "observation_key": "3:20"},
        {"frame_index": 4, "raw_track_id": 20, "local_raw_track_id": 20, "segment_index": 1, "observation_key": "4:20"},
    ]
    masks = {
        "3:7": np.pad(np.ones((2, 2), dtype=np.uint8), ((0, 2), (0, 2))),
        "4:7": np.pad(np.ones((2, 2), dtype=np.uint8), ((2, 0), (2, 0))),
        "3:20": np.pad(np.ones((2, 2), dtype=np.uint8), ((2, 0), (2, 0))),
        "4:20": np.pad(np.ones((2, 2), dtype=np.uint8), ((0, 2), (0, 2))),
    }
    stitched, _ = stitch_segment_seams(rows, masks)
    assert [row["raw_track_id"] for row in stitched if row["segment_index"] == 1] == [20, 20]
    assert [row["canonical_raw_track_id"] for row in stitched if row["segment_index"] == 1] == [20, 20]


def test_empty_masks_use_bbox_fallback_and_threshold_boundaries():
    rows, masks = _seam_rows([7], [20], [([1, 1, 4, 5],), ([1, 1, 4, 5],)])
    masks = {key: np.zeros_like(value) for key, value in masks.items()}
    stitched, _ = stitch_segment_seams(rows, masks, min_score=1.0)
    assert stitched[1]["raw_track_id"] == 20
    assert stitched[1]["canonical_raw_track_id"] == 7
    with pytest.raises(ValueError):
        stitch_segment_seams(rows, masks, min_score=1.00001)


@pytest.mark.parametrize("bbox", [[1, 1, 1, 5], [1, 1, float("nan"), 5], [1, 1, 4]])
def test_invalid_or_zero_area_boxes_cannot_stitch_even_with_masks(bbox):
    rows = [
        {"frame_index": 3, "raw_track_id": 7, "segment_index": 0,
         "observation_key": "3:7", "bbox_xyxy": bbox},
        {"frame_index": 3, "raw_track_id": 20, "segment_index": 1,
         "observation_key": "3:20", "bbox_xyxy": [1, 1, 4, 5]},
    ]
    masks = {key: np.ones((12, 16), dtype=np.uint8) for key in ("3:7", "3:20")}

    stitched, _ = stitch_segment_seams(rows, masks, min_score=0.0)

    assert stitched[0]["canonical_raw_track_id"] == 7
    assert stitched[1]["canonical_raw_track_id"] == 20


@pytest.mark.parametrize("ambiguity_margin", [float("nan"), float("inf"), float("-inf")])
def test_seam_stitching_rejects_non_finite_ambiguity_margin(ambiguity_margin):
    rows, masks = _seam_rows([7], [20], [([1, 1, 4, 5],), ([1, 1, 4, 5],)])

    with pytest.raises(ValueError, match="invalid seam matching thresholds"):
        stitch_segment_seams(rows, masks, ambiguity_margin=ambiguity_margin)


def test_segment_seam_environment_and_explicit_overrides(monkeypatch):
    from modal_app_v2 import _segment_float_setting

    monkeypatch.setenv("SAM31_V2_TEST_SCORE", "0.73")
    assert _segment_float_setting(None, "SAM31_V2_TEST_SCORE", 0.5) == pytest.approx(0.73)
    assert _segment_float_setting(0.91, "SAM31_V2_TEST_SCORE", 0.5) == pytest.approx(0.91)


def test_seam_stitching_fails_closed_on_ambiguous_crossing():
    rows, masks = _seam_rows([7, 8], [20, 10], [([1, 1, 5, 5], [7, 1, 11, 5]), ([4, 1, 8, 5], [4, 1, 8, 5])])
    masks = {key: np.ones((12, 16), dtype=np.uint8) for key in masks}
    diagnostics = []
    stitched, _ = stitch_segment_seams(rows, masks, ambiguity_margin=0.1, diagnostics=diagnostics)
    assert {row["raw_track_id"] for row in stitched if row["segment_index"] == 1} == {10, 20}
    assert {row["canonical_raw_track_id"] for row in stitched if row["segment_index"] == 1} == {10, 20}
    assert any(item["event"] == "ambiguous" for item in diagnostics)


def test_seam_stitching_keeps_missing_overlap_as_separate_fragment():
    rows, masks = _seam_rows([7], [20], [([1, 1, 4, 5],), ([1, 1, 4, 5],)])
    rows[1]["frame_index"] = 4
    stitched, _ = stitch_segment_seams(rows, masks)
    assert [row["raw_track_id"] for row in stitched] == [7, 20]
    assert [row["canonical_raw_track_id"] for row in stitched] == [7, 20]


def test_stitched_identity_survives_runner_v2_with_sanitizer_and_deterministic_reid():
    from pipeline.bbox_sanitizer import BBoxSanitizerConfig
    from pipeline.runner_v2 import RunnerV2, RunnerV2Seams

    rows, masks = _seam_rows([7], [20], [([1, 1, 4, 5],), ([1, 1, 4, 5],)])
    stitched, stitched_masks = stitch_segment_seams(rows, masks)
    final, final_masks = deduplicate_segment_observations(stitched, stitched_masks)

    class FakeEmbedder:
        def embed(self, _frame, boxes):
            return [[1.0, 0.0] for _ in boxes]

    result = RunnerV2(RunnerV2Seams(
        identity=lambda items: {item["observation_key"]: f'player-{item["canonical_raw_track_id"]}' for item in items},
        reid=FakeEmbedder(),
    )).run(
        final,
        frames_by_index={3: np.zeros((12, 16, 3), dtype=np.uint8)},
        masks_by_observation=final_masks,
        bbox_sanitizer_config=BBoxSanitizerConfig(enabled=True),
    )
    assert [row["stable_player_id"] for row in result.observations] == ["player-7"]
    assert result.observations[0]["local_raw_track_id"] in {7, 20}


def test_runner_and_sanitizer_group_multiple_stitched_fragments():
    from pipeline.bbox_sanitizer import BBoxSanitizerConfig
    from pipeline.runner_v2 import RunnerV2, RunnerV2Seams

    rows = []
    masks = {}
    for frame_index, segment_index, raw_id in [(2, 0, 7), (3, 0, 7), (3, 1, 20), (4, 1, 20)]:
        key = f"{frame_index}:{raw_id}"
        rows.append({
            "frame_index": frame_index,
            "raw_track_id": raw_id,
            "raw_observation_key": key,
            "local_raw_track_id": raw_id,
            "segment_index": segment_index,
            "observation_key": key,
            "bbox_xyxy": [1, 1, 4, 5],
        })
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[1:5, 1:4] = 1
        masks[key] = mask

    stitched, stitched_masks = stitch_segment_seams(rows, masks)
    final, final_masks = deduplicate_segment_observations(stitched, stitched_masks)

    class FakeEmbedder:
        def embed(self, _frame, boxes):
            return [[1.0, 0.0] for _ in boxes]

    result = RunnerV2(RunnerV2Seams(
        identity=lambda items: {item["observation_key"]: f'player-{item["canonical_raw_track_id"]}' for item in items},
        reid=FakeEmbedder(),
    )).run(
        final,
        frames_by_index={index: np.zeros((12, 16, 3), dtype=np.uint8) for index in (2, 3, 4)},
        masks_by_observation=final_masks,
        bbox_sanitizer_config=BBoxSanitizerConfig(enabled=True),
    )
    assert [row["frame_index"] for row in result.observations] == [2, 3, 4]
    assert {row["raw_track_id"] for row in result.observations} == {7, 20}
    assert {row["canonical_raw_track_id"] for row in result.observations} == {7}
    assert {row["stable_player_id"] for row in result.observations} == {"player-7"}
    assert set(result.cleaned_masks_by_observation) == set(final_masks)


def test_segment_object_limit_is_a_hard_guard():
    rows = [{"raw_track_id": index} for index in range(17)]
    with pytest.raises(ValueError, match="maximum is 16"):
        assert_segment_object_limit(rows)


def test_extract_segment_preserves_dimensions_fps_and_frame_count(tmp_path: Path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "segment.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 20, (32, 24))
    for index in range(8):
        writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    writer.release()

    window = plan_segment_windows(2, 5, window=4)[0]
    extract_segment_video(source, output, window)
    capture = cv2.VideoCapture(str(output))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 32
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 24
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(20, abs=1)
    finally:
        capture.release()


def test_local_propagation_request_uses_clip_bounds(tmp_path: Path):
    from modal_app_v2 import _track_video

    requests = []

    class Model:
        def init_state(self, resource_path, **kwargs):
            return {"session_id": "segment"}

    class Predictor:
        model = Model()

        def handle_request(self, request):
            requests.append(request)
            return {"session_id": "segment"} if request["type"] == "start_session" else None

        def handle_stream_request(self, request):
            requests.append(request)
            return iter([])

    _track_video(Predictor(), tmp_path / "clip.mp4", "player", 0, 4, local_propagation=True)
    assert next(request for request in requests if request["type"] == "propagate_in_video") == {
        "type": "propagate_in_video",
        "session_id": "segment",
        "start_frame_index": 0,
        "max_frame_num_to_track": 5,
    }


def test_track_segmented_video_remaps_deduplicates_and_aligns_masks(tmp_path: Path):
    from modal_app_v2 import _track_segmented_video

    source = tmp_path / "source.mp4"
    _write_video(source)
    prompts: list[dict] = []
    propagated: list[dict] = []
    closed: list[str] = []

    class Model:
        def init_state(self, resource_path, **_kwargs):
            return {"resource_path": resource_path}

    class Predictor:
        model = Model()

        def handle_request(self, request):
            if request["type"] == "start_session":
                return {"session_id": Path(request["resource_path"]).stem}
            if request["type"] == "add_prompt":
                prompts.append(request)
            elif request["type"] == "close_session":
                closed.append(request["session_id"])
            return None

        def handle_stream_request(self, request):
            propagated.append(request)
            segment_index = int(request["session_id"].split("-")[-1])
            frame_count = request["max_frame_num_to_track"]
            for local_frame in range(frame_count):
                mask = np.ones((2, 2), dtype=np.uint8)
                yield {
                    "frame_index": local_frame,
                    "out_obj_ids": [7],
                    "out_boxes_xywh": [[0, 0, 1, 1]],
                    "out_binary_masks": [mask],
                }

    rows, masks = _track_segmented_video(
        Predictor(), source, "player", 0, 4, tmp_path,
        window=3, overlap=1,
    )

    assert [request["frame_index"] for request in prompts] == [0, 0]
    assert [request["start_frame_index"] for request in propagated] == [0, 0]
    assert [request["max_frame_num_to_track"] for request in propagated] == [3, 3]
    assert closed == ["segment-0000", "segment-0001"]
    assert [row["frame_index"] for row in rows] == [0, 1, 2, 3, 4]
    assert [row["raw_track_id"] for row in rows] == [7, 7, 1_000_007, 1_000_007, 1_000_007]
    assert [row["canonical_raw_track_id"] for row in rows] == [7, 7, 7, 7, 7]
    assert [row["raw_observation_key"] for row in rows] == ["0:7", "1:7", "0:7", "1:7", "2:7"]
    assert rows[2]["segment_index"] == 1
    assert set(masks) == {row["observation_key"] for row in rows}
    assert int(masks[rows[2]["observation_key"]][0, 0]) == 1


def test_track_segmented_video_enforces_sam_object_limit(tmp_path: Path):
    from modal_app_v2 import _track_segmented_video

    source = tmp_path / "source.mp4"
    _write_video(source, frame_count=1)

    class Model:
        def init_state(self, **_kwargs):
            return None

    class Predictor:
        model = Model()

        def handle_request(self, request):
            return {"session_id": "segment-0000"} if request["type"] == "start_session" else None

        def handle_stream_request(self, _request):
            yield {
                "frame_index": 0,
                "out_obj_ids": list(range(17)),
                "out_boxes_xywh": [[0, 0, 1, 1]] * 17,
            }

    with pytest.raises(ValueError, match="maximum is 16"):
        _track_segmented_video(Predictor(), source, "player", 0, 0, tmp_path, window=1, overlap=0)
