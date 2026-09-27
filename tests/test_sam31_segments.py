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
    assert masks == {rows[0]["observation_key"]: "mask-a"}

    second = plan_segment_windows(0, 6, window=4, overlap=1)[1]
    rows2, _ = remap_segment_rows(
        [{"frame_index": 0, "raw_track_id": 7, "observation_key": "0:7"}], {}, second
    )
    assert rows2[0]["raw_track_id"] != rows[0]["raw_track_id"]


def test_deduplication_keeps_later_segment_and_aligned_mask():
    rows = [
        {"frame_index": 3, "local_raw_track_id": 7, "segment_index": 0, "raw_track_id": 7, "observation_key": "3:7"},
        {"frame_index": 3, "local_raw_track_id": 7, "segment_index": 1, "raw_track_id": 1_000_007, "observation_key": "3:1000007"},
    ]
    final, masks = deduplicate_segment_observations(rows, {"3:7": "old", "3:1000007": "new"})
    assert [row["observation_key"] for row in final] == ["3:1000007"]
    assert masks == {"3:1000007": "new"}


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
                mask = np.full((2, 2), segment_index, dtype=np.uint8)
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
