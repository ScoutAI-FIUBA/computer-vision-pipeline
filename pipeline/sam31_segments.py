"""CPU-side planning and stitching helpers for segmented SAM3.1 V2 runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SAM31_MAX_NUM_OBJECTS = 16
_RAW_ID_BASE = 1_000_000


@dataclass(frozen=True)
class SegmentWindow:
    """Inclusive global and local bounds for one temporary SAM video."""

    ordinal: int
    global_start: int
    global_end: int
    owned_start: int

    @property
    def frame_count(self) -> int:
        return self.global_end - self.global_start + 1

    def local_index(self, global_frame: int) -> int:
        if not self.global_start <= global_frame <= self.global_end:
            raise ValueError("global frame is outside segment")
        return global_frame - self.global_start


def plan_segment_windows(
    frame_start: int,
    frame_end: int,
    *,
    window: int,
    overlap: int = 1,
) -> list[SegmentWindow]:
    """Plan complete inclusive coverage; shared frames belong to the later window."""

    if type(frame_start) is not int or type(frame_end) is not int:
        raise ValueError("frame bounds must be integers")
    if frame_start < 0 or frame_end < frame_start:
        raise ValueError("frame bounds must be non-negative and ordered")
    if type(window) is not int or window <= 0:
        raise ValueError("window must be a positive integer")
    if type(overlap) is not int or overlap < 0 or overlap >= window:
        raise ValueError("overlap must be non-negative and smaller than window")

    result: list[SegmentWindow] = []
    start = frame_start
    ordinal = 0
    while start <= frame_end:
        end = min(frame_end, start + window - 1)
        result.append(
            SegmentWindow(
                ordinal=ordinal,
                global_start=start,
                global_end=end,
                owned_start=frame_start if ordinal == 0 else start,
            )
        )
        if end == frame_end:
            break
        start = end - overlap + 1
        ordinal += 1
    return result


def namespace_raw_track_id(segment_ordinal: int, local_raw_track_id: int) -> int:
    """Encode a segment-local SAM ID without allowing sessions to merge it."""

    if type(segment_ordinal) is not int or segment_ordinal < 0:
        raise ValueError("segment ordinal must be a non-negative integer")
    if type(local_raw_track_id) is not int or local_raw_track_id < 0:
        raise ValueError("local raw track ID must be a non-negative integer")
    if local_raw_track_id >= _RAW_ID_BASE:
        raise ValueError("local raw track ID exceeds namespace capacity")
    return segment_ordinal * _RAW_ID_BASE + local_raw_track_id


def remap_segment_rows(
    rows: Iterable[Mapping[str, Any]],
    masks: Mapping[str, Any],
    segment: SegmentWindow,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert local SAM rows/masks to globally keyed, namespaced observations."""

    remapped: list[dict[str, Any]] = []
    remapped_masks: dict[str, Any] = {}
    for source in rows:
        row = dict(source)
        global_frame = segment.global_start + int(row["frame_index"])
        if not segment.owned_start <= global_frame <= segment.global_end:
            continue
        local_id = int(row["raw_track_id"])
        namespaced_id = namespace_raw_track_id(segment.ordinal, local_id)
        key = f"{global_frame}:{namespaced_id}"
        row["frame_index"] = global_frame
        row["raw_track_id"] = namespaced_id
        row["observation_key"] = key
        row["segment_index"] = segment.ordinal
        row["local_raw_track_id"] = local_id
        remapped.append(row)
        local_key = str(source.get("observation_key", f'{int(source["frame_index"])}:{local_id}'))
        if local_key in masks:
            remapped_masks[key] = masks[local_key]
    return remapped, remapped_masks


def deduplicate_segment_observations(
    rows: Iterable[Mapping[str, Any]], masks: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the later segment's row for an overlap frame/local SAM ID."""

    chosen: dict[tuple[int, int], dict[str, Any]] = {}
    chosen_masks: dict[tuple[int, int], Any] = {}
    for source in rows:
        row = dict(source)
        key = (int(row["frame_index"]), int(row["local_raw_track_id"]))
        previous = chosen.get(key)
        if previous is None or int(row.get("segment_index", -1)) >= int(previous.get("segment_index", -1)):
            chosen[key] = row
            observation_key = str(row["observation_key"])
            if observation_key in masks:
                chosen_masks[key] = masks[observation_key]
    ordered = sorted(chosen.values(), key=lambda row: (int(row["frame_index"]), str(row["observation_key"])))
    final_masks = {str(row["observation_key"]): chosen_masks[key] for key, row in chosen.items() if key in chosen_masks}
    return ordered, final_masks


def assert_segment_object_limit(rows: Iterable[Mapping[str, Any]], *, max_objects: int = SAM31_MAX_NUM_OBJECTS) -> None:
    """Reject a segment that asks SAM for more active objects than configured."""

    ids = {int(row["raw_track_id"]) for row in rows}
    if len(ids) > max_objects:
        raise ValueError(f"SAM3.1 segment has {len(ids)} active objects; maximum is {max_objects}")


def extract_segment_video(source: Path, destination: Path, segment: SegmentWindow) -> None:
    """Write a temporary clip with source dimensions/FPS and the exact frame window."""

    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create segment video: {destination}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, segment.global_start)
        for frame_index in range(segment.global_start, segment.global_end + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read source frame {frame_index}")
            writer.write(frame)
    finally:
        writer.release()
        capture.release()
