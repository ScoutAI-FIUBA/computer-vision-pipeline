"""CPU-side planning and stitching helpers for segmented SAM3.1 V2 runs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


SAM31_MAX_NUM_OBJECTS = 16
SAM31_SEGMENT_GRACE_MAX_OBJECTS = 22
_RAW_ID_BASE = 1_000_000
DEFAULT_SEAM_MIN_SCORE = 0.5
DEFAULT_SEAM_AMBIGUITY_MARGIN = 0.05


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
        local_key = str(
            source.get(
                "raw_observation_key",
                source.get("observation_key", f'{int(source["frame_index"])}:{local_id}'),
            )
        )
        namespaced_id = namespace_raw_track_id(segment.ordinal, local_id)
        key = f"{global_frame}:{namespaced_id}"
        row["frame_index"] = global_frame
        row["raw_track_id"] = namespaced_id
        row["observation_key"] = key
        row["raw_observation_key"] = local_key
        row["segment_index"] = segment.ordinal
        row["local_raw_track_id"] = local_id
        remapped.append(row)
        mask_key = str(source.get("observation_key", local_key))
        if mask_key in masks:
            remapped_masks[key] = masks[mask_key]
    return remapped, remapped_masks


def deduplicate_segment_observations(
    rows: Iterable[Mapping[str, Any]],
    masks: Mapping[str, Any],
    *,
    diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the later segment's row for an overlap frame/canonical SAM ID."""

    chosen: dict[tuple[int, int], dict[str, Any]] = {}
    chosen_masks: dict[tuple[int, int], Any] = {}
    for source in rows:
        row = dict(source)
        identity = row.get("canonical_raw_track_id", row.get("local_raw_track_id", row["raw_track_id"]))
        key = (int(row["frame_index"]), int(identity))
        previous = chosen.get(key)
        if previous is None or int(row.get("segment_index", -1)) >= int(previous.get("segment_index", -1)):
            if previous is not None and diagnostics is not None:
                diagnostics.append({
                    "event": "ownership",
                    "frame_index": int(row["frame_index"]),
                    "canonical_raw_track_id": int(identity),
                    "owner_segment_index": int(row.get("segment_index", -1)),
                    "replaced_segment_index": int(previous.get("segment_index", -1)),
                    "observation_key": str(row["observation_key"]),
                    "replaced_observation_key": str(previous["observation_key"]),
                })
            chosen[key] = row
            observation_key = str(row["observation_key"])
            if observation_key in masks:
                chosen_masks[key] = masks[observation_key]
                row["mask_status"] = "present"
            else:
                chosen_masks.pop(key, None)
                row["mask_status"] = "missing"
                if diagnostics is not None:
                    diagnostics.append({
                        "event": "owned_observation_missing_mask",
                        "frame_index": int(row["frame_index"]),
                        "canonical_raw_track_id": int(identity),
                        "raw_track_id": int(row["raw_track_id"]),
                        "observation_key": observation_key,
                        "raw_observation_key": str(row.get("raw_observation_key", observation_key)),
                    })
    ordered = sorted(chosen.values(), key=lambda row: (int(row["frame_index"]), str(row["observation_key"])))
    final_masks = {str(row["observation_key"]): chosen_masks[key] for key, row in chosen.items() if key in chosen_masks}
    return ordered, final_masks


def _valid_bbox(value: Any) -> bool:
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(item) for item in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1


def _bbox_iou(left: Any, right: Any) -> float:
    if not _valid_bbox(left) or not _valid_bbox(right):
        return 0.0
    try:
        lx1, ly1, lx2, ly2 = (float(value) for value in left)
        rx1, ry1, rx2, ry2 = (float(value) for value in right)
    except (TypeError, ValueError):
        return 0.0
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _bbox_center_score(left: Any, right: Any) -> float:
    if not _valid_bbox(left) or not _valid_bbox(right):
        return 0.0
    try:
        lx1, ly1, lx2, ly2 = (float(value) for value in left)
        rx1, ry1, rx2, ry2 = (float(value) for value in right)
    except (TypeError, ValueError):
        return 0.0
    dx = (lx1 + lx2 - rx1 - rx2) / 2.0
    dy = (ly1 + ly2 - ry1 - ry2) / 2.0
    diagonal = max(((max(lx2, rx2) - min(lx1, rx1)) ** 2 + (max(ly2, ry2) - min(ly1, ry1)) ** 2) ** 0.5, 1e-6)
    return 1.0 / (1.0 + (dx * dx + dy * dy) ** 0.5 / diagonal)


def _mask_iou(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        left_bool = left.astype(bool)
        right_bool = right.astype(bool)
        if left_bool.shape != right_bool.shape:
            return None
        if not left_bool.any() or not right_bool.any():
            return None
        union = (left_bool | right_bool).sum()
        return float((left_bool & right_bool).sum() / union) if union else 0.0
    except (AttributeError, TypeError, ValueError):
        return None


def _seam_score(left: Mapping[str, Any], right: Mapping[str, Any], masks: Mapping[str, Any]) -> float:
    if not _valid_bbox(left.get("bbox_xyxy")) or not _valid_bbox(right.get("bbox_xyxy")):
        return 0.0
    mask_score = _mask_iou(masks.get(str(left["observation_key"])), masks.get(str(right["observation_key"])))
    if mask_score is not None:
        return mask_score
    bbox_iou = _bbox_iou(left.get("bbox_xyxy"), right.get("bbox_xyxy"))
    return max(bbox_iou, _bbox_center_score(left.get("bbox_xyxy"), right.get("bbox_xyxy")))


def stitch_segment_seams(
    rows: Iterable[Mapping[str, Any]],
    masks: Mapping[str, Any],
    *,
    min_score: float = DEFAULT_SEAM_MIN_SCORE,
    ambiguity_margin: float = DEFAULT_SEAM_AMBIGUITY_MARGIN,
    diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Canonicalize adjacent segment IDs only when physical seam evidence is unique."""

    if not 0.0 <= min_score <= 1.0 or not math.isfinite(ambiguity_margin) or ambiguity_margin < 0.0:
        raise ValueError("invalid seam matching thresholds")
    stitched = [dict(row) for row in rows]
    stitched_masks = dict(masks)
    for row in stitched:
        row["canonical_raw_track_id"] = int(row["raw_track_id"])
    by_segment: dict[int, list[dict[str, Any]]] = {}
    for row in stitched:
        by_segment.setdefault(int(row.get("segment_index", 0)), []).append(row)

    def report(event: str, **details: Any) -> None:
        if diagnostics is not None:
            diagnostics.append({"event": event, **details})

    for segment_index in sorted(by_segment):
        if segment_index == 0 or segment_index - 1 not in by_segment:
            continue
        previous = by_segment[segment_index - 1]
        current = by_segment[segment_index]
        shared_frames = {int(row["frame_index"]) for row in previous} & {int(row["frame_index"]) for row in current}
        previous_ids = {int(row["raw_track_id"]) for row in previous if int(row["frame_index"]) in shared_frames}
        current_ids = {int(row["raw_track_id"]) for row in current if int(row["frame_index"]) in shared_frames}
        candidates: list[tuple[float, int, int]] = []
        for previous_id in previous_ids:
            for current_id in current_ids:
                aligned_scores = []
                for frame_index in sorted(shared_frames):
                    previous_rows = [
                        row for row in previous
                        if int(row["raw_track_id"]) == previous_id and int(row["frame_index"]) == frame_index
                    ]
                    current_rows = [
                        row for row in current
                        if int(row["raw_track_id"]) == current_id and int(row["frame_index"]) == frame_index
                    ]
                    for left in previous_rows:
                        for right in current_rows:
                            if not _valid_bbox(left.get("bbox_xyxy")) or not _valid_bbox(right.get("bbox_xyxy")):
                                aligned_scores = []
                                break
                            aligned_scores.append(_seam_score(left, right, stitched_masks))
                        if not aligned_scores and (previous_rows or current_rows):
                            break
                    if not aligned_scores:
                        break
                score = sum(aligned_scores) / len(aligned_scores) if aligned_scores else 0.0
                if aligned_scores and score >= min_score:
                    candidates.append((score, previous_id, current_id))

        accepted: list[tuple[float, int, int]] = []
        for score, previous_id, current_id in candidates:
            previous_scores = sorted((item[0] for item in candidates if item[1] == previous_id), reverse=True)
            current_scores = sorted((item[0] for item in candidates if item[2] == current_id), reverse=True)
            if (len(previous_scores) > 1 and previous_scores[0] - previous_scores[1] < ambiguity_margin) or (
                len(current_scores) > 1 and current_scores[0] - current_scores[1] < ambiguity_margin
            ):
                report("ambiguous", segment_index=segment_index, previous_raw_track_id=previous_id, current_raw_track_id=current_id)
                continue
            accepted.append((score, previous_id, current_id))

        used_previous: set[int] = set()
        used_current: set[int] = set()
        for score, previous_id, current_id in sorted(accepted, reverse=True):
            if previous_id in used_previous or current_id in used_current:
                continue
            used_previous.add(previous_id)
            used_current.add(current_id)
            canonical_id = next(
                int(row.get("canonical_raw_track_id", previous_id))
                for row in previous
                if int(row["raw_track_id"]) == previous_id
            )
            for row in previous:
                if int(row["raw_track_id"]) == previous_id:
                    row["canonical_raw_track_id"] = canonical_id
            for row in current:
                if int(row["raw_track_id"]) != current_id:
                    continue
                row["canonical_raw_track_id"] = canonical_id
            report("matched", segment_index=segment_index, previous_raw_track_id=previous_id, current_raw_track_id=current_id, score=score)
    return stitched, stitched_masks


def assert_segment_object_limit(rows: Iterable[Mapping[str, Any]], *, max_objects: int = SAM31_MAX_NUM_OBJECTS) -> None:
    """Reject a segment that asks SAM for more active objects than configured."""

    ids = {int(row["raw_track_id"]) for row in rows}
    if len(ids) > max_objects:
        raise ValueError(f"SAM3.1 segment has {len(ids)} active objects; maximum is {max_objects}")


def prune_segment_object_overflow(
    rows: Iterable[Mapping[str, Any]],
    *,
    diagnostics: list[dict[str, Any]] | None = None,
    max_objects: int = SAM31_MAX_NUM_OBJECTS,
    grace_max_objects: int = SAM31_SEGMENT_GRACE_MAX_OBJECTS,
    segment_index: int | None = None,
) -> list[dict[str, Any]]:
    """Keep the strongest 16 IDs when a segmented session briefly overflows."""

    materialized = [dict(row) for row in rows]
    by_id: dict[int, list[dict[str, Any]]] = {}
    for row in materialized:
        by_id.setdefault(int(row["raw_track_id"]), []).append(row)
    before_count = len(by_id)
    if before_count <= max_objects:
        return materialized
    if before_count > grace_max_objects:
        raise ValueError(
            f"SAM3.1 segment has {before_count} active objects; grace maximum is {grace_max_objects}"
        )

    def finite_values(track_rows: list[dict[str, Any]], names: tuple[str, ...]) -> list[float]:
        values = []
        for row in track_rows:
            for name in names:
                value = row.get(name)
                if type(value) in {int, float} and math.isfinite(float(value)):
                    values.append(float(value))
        return values

    def bbox_iou_for_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
        if "bbox_xyxy" not in left or "bbox_xyxy" not in right:
            return None
        return _bbox_iou(left["bbox_xyxy"], right["bbox_xyxy"])

    max_frame_count = max(len({int(row["frame_index"]) for row in track_rows}) for track_rows in by_id.values())
    ranking: list[dict[str, Any]] = []
    for raw_id, track_rows in by_id.items():
        track_rows = sorted(track_rows, key=lambda row: (int(row["frame_index"]), str(row.get("observation_key", ""))))
        frames = sorted({int(row["frame_index"]) for row in track_rows})
        confidence_values = finite_values(track_rows, ("confidence", "score", "probability"))
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        frame_count = len(frames)
        consecutive = sum(right - left == 1 for left, right in zip(frames, frames[1:]))
        continuity = consecutive / max(1, frame_count - 1)
        ious = [
            score for left, right in zip(track_rows, track_rows[1:])
            if (score := bbox_iou_for_rows(left, right)) is not None
        ]
        spatial_consistency = sum(ious) / len(ious) if ious else continuity
        temporal_consistency = (continuity + spatial_consistency) / 2.0
        duration_score = frame_count / max_frame_count
        quality_score = 0.5 * confidence + 0.3 * duration_score + 0.2 * temporal_consistency
        ranking.append({
            "raw_id": raw_id,
            "quality_score": round(quality_score, 12),
            "confidence_score": round(confidence, 12),
            "frame_count": frame_count,
            "duration_score": round(duration_score, 12),
            "temporal_consistency": round(temporal_consistency, 12),
        })

    ranking.sort(key=lambda item: (
        item["quality_score"], item["confidence_score"], item["frame_count"],
        item["temporal_consistency"], -item["raw_id"],
    ))
    drop_count = before_count - max_objects
    dropped = ranking[:drop_count]
    dropped_ids = {item["raw_id"] for item in dropped}
    for item in ranking:
        item["dropped"] = item["raw_id"] in dropped_ids
        item["reason"] = (
            "short-lived/low-confidence/temporally-inconsistent"
            if item["dropped"] else "retained by deterministic quality ranking"
        )
    if diagnostics is not None:
        diagnostics.append({
            "event": "segment_object_overflow",
            **({"segment_index": segment_index} if segment_index is not None else {}),
            "before_count": before_count,
            "after_count": max_objects,
            "dropped_raw_ids": [item["raw_id"] for item in dropped],
            "grace_applied": True,
            "grace_max_objects": grace_max_objects,
            "ranking": ranking,
        })
    return sorted(
        (row for row in materialized if int(row["raw_track_id"]) not in dropped_ids),
        key=lambda row: (int(row["frame_index"]), str(row.get("observation_key", "")), int(row["raw_track_id"])),
    )


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
