"""Detector-seeded SAM2 tracking.

The algorithm is unchanged from the original local script: every `frame_stride`
frames the detector runs, its boxes are matched to the currently-live tracks by
IoU to preserve identities, and SAM2 propagates those boxes forward until the
next detector call.

What changed is how SAM2 is fed. The original called
`init_state(video_path=<whole video>)` once, which decodes and holds the entire
video as float32 tensors (~12 MB per frame). Since the script calls
`reset_state()` every `frame_stride` frames and only ever propagates within one
stride, it was loading the whole video to use 16 frames at a time.

Here SAM2 is initialised per segment instead. Memory becomes flat in video
length, which is what makes long videos possible at all and lets the job run on
a cheap GPU rather than an A100.
"""

from __future__ import annotations

import tempfile
from contextlib import nullcontext
from pathlib import Path

from pipeline import appearance


def xyxy_iou(left_xyxy: list[float], right_xyxy: list[float]) -> float:
    """Intersection-over-union of two boxes in [x1, y1, x2, y2] form."""

    left_x1, left_y1, left_x2, left_y2 = left_xyxy
    right_x1, right_y1, right_x2, right_y2 = right_xyxy
    inter_w = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    inter_h = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    inter_area = inter_w * inter_h
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - inter_area
    return inter_area / union if union > 0 else 0.0


def mask_to_xyxy(mask_logits) -> list[float] | None:
    """Tightest box around a SAM2 mask, or None when the mask is empty."""

    import numpy as np

    mask = mask_logits > 0.0
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask_array = np.asarray(mask)
    if mask_array.ndim > 2:
        mask_array = mask_array.squeeze()
    if mask_array.ndim != 2 or not mask_array.any():
        return None

    ys, xs = np.where(mask_array)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def mask_to_crop(mask_logits):
    """Small binary crop aligned to `mask_to_xyxy`, or None when empty."""

    import numpy as np

    mask = mask_logits > 0.0
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask_array = np.asarray(mask)
    if mask_array.ndim > 2:
        mask_array = mask_array.squeeze()
    if mask_array.ndim != 2 or not mask_array.any():
        return None

    ys, xs = np.where(mask_array)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return mask_array[y1:y2, x1:x2].astype(bool)


def detection_rows(detections) -> list[dict]:
    """Convert supervision Detections into the pipeline's row dicts."""

    class_names = detections.data.get("class_name", ["player"] * len(detections))
    confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
    return [
        {
            "xyxy": [float(value) for value in xyxy],
            "confidence": None if confidence is None else float(confidence),
            "class_name": str(class_name),
            "track_id": None,
            "track_state": "detected",
        }
        for xyxy, confidence, class_name in zip(detections.xyxy, confidences, class_names)
    ]


# Cost assigned to pairs that fail the IoU gate. Any value far above the maximum
# real cost (1.0) works; it keeps the pair out of the optimal assignment.
_IMPOSSIBLE = 1e6


def assign_track_ids(
    current_rows: list[dict],
    active_rows: list[dict],
    next_track_id: int,
    min_match_iou: float,
    *,
    appearance_by_track_id: dict[int, list[float]] | None = None,
    appearance_weight: float = 0.0,
    min_appearance_match: float = 0.7,
) -> tuple[list[dict], int]:
    """Give each fresh detector box either a live track's ID or a brand new one.

    Matching preserves identity across a segment boundary, since SAM2's own
    memory is discarded when each segment ends.

    Two properties matter here:

    *Optimal, not greedy.* The original implementation walked detections in order
    and let each grab its best available track. That can be globally wrong: an
    early detection takes a track a later detection needed more. Optimal
    assignment (Hungarian) minimises total cost across all pairs at once.

    *Appearance breaks ties geometry cannot.* When two players cross, both
    detections overlap both tracks and IoU alone is close to a coin flip —
    the measured cause of players swapping numbers. Colour similarity is folded
    into the cost when available, which is what actually separates them.

    A pair becomes a candidate when EITHER the boxes overlap by `min_match_iou`,
    OR the colours match strongly (`min_appearance_match`) and the boxes overlap
    at all. That second route matters: during a real crossing each player moves
    close to a full box-width between detector calls, so the *correct* pairing
    can drop below the IoU gate while the *swapped* one sits well above it.
    Gating on geometry first would throw away the right answer before colour was
    ever consulted.

    Requiring non-zero overlap on the appearance route keeps the search local, so
    two players in matching kit at opposite ends of the court can never be
    confused. Detection rows may carry an "appearance" key; pairs missing a
    signature on either side fall back to pure geometry.
    """

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not current_rows:
        return current_rows, next_track_id

    if not active_rows:
        for row in current_rows:
            row["track_id"] = next_track_id
            next_track_id += 1
        return current_rows, next_track_id

    appearance_by_track_id = appearance_by_track_id or {}
    cost = np.full((len(current_rows), len(active_rows)), _IMPOSSIBLE, dtype=float)

    for i, detection in enumerate(current_rows):
        for j, track in enumerate(active_rows):
            iou = xyxy_iou(detection["xyxy"], track["xyxy"])
            geometry_cost = 1.0 - iou

            if appearance_weight <= 0.0:
                if iou >= min_match_iou:
                    cost[i, j] = geometry_cost
                continue

            score = appearance.similarity(
                detection.get("appearance"),
                appearance_by_track_id.get(int(track["track_id"])),
            )
            colour_vouches = score is not None and score >= min_appearance_match and iou > 0.0
            if iou < min_match_iou and not colour_vouches:
                continue

            if score is None:
                cost[i, j] = geometry_cost
            else:
                cost[i, j] = (1.0 - appearance_weight) * geometry_cost + appearance_weight * (1.0 - score)

    detection_indices, track_indices = linear_sum_assignment(cost)
    matched = {
        int(i): active_rows[int(j)]
        for i, j in zip(detection_indices, track_indices)
        if cost[i, j] < _IMPOSSIBLE
    }

    for index, row in enumerate(current_rows):
        track = matched.get(index)
        if track is None:
            row["track_id"] = next_track_id
            next_track_id += 1
        else:
            row["track_id"] = int(track["track_id"])

    return current_rows, next_track_id


def carry_unmatched_tracks(
    seed_rows: list[dict],
    active_rows: list[dict],
    missed_counts: dict[int, int],
    max_missed: int,
) -> list[dict]:
    """Keep tracks the detector missed alive, instead of deleting them.

    SAM2 is re-seeded from scratch at every detector call. Seeding it with only
    the current detections means one missed detection permanently destroys a
    track — and when the player is picked up again they are issued a new ID.
    That single behaviour, not tracking drift, is the main cause of ID churn:
    a detector that alternates between finding 9 and 10 players mints a new ID
    on every recovery.

    So any live track that no detection claimed is re-seeded from its last known
    box ("coasted") for up to `max_missed` consecutive detector calls. That
    reserves its ID long enough to survive a brief occlusion. After that it is
    retired, so players who genuinely leave do not linger forever.

    Mutates `missed_counts` in place. Returns the rows to seed SAM2 with.
    """

    if max_missed <= 0:
        missed_counts.clear()
        return seed_rows

    matched_ids = {int(row["track_id"]) for row in seed_rows}
    for track_id in list(missed_counts):
        if track_id in matched_ids:
            del missed_counts[track_id]

    carried = []
    for row in active_rows:
        track_id = int(row["track_id"])
        if track_id in matched_ids:
            continue

        misses = missed_counts.get(track_id, 0) + 1
        if misses > max_missed:
            missed_counts.pop(track_id, None)
            continue

        missed_counts[track_id] = misses
        carried.append({**row, "track_state": "coasted"})

    return seed_rows + carried


class Sam2SegmentTracker:
    """Loads SAM2 once, then propagates boxes through one segment at a time.

    Model loading is separated from per-segment work on purpose: on Modal the
    load happens in `@modal.enter()` so it is paid once per container, not once
    per job.
    """

    def __init__(self, model_id: str) -> None:
        try:
            import torch
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError as error:
            raise RuntimeError(
                "SAM2 tracking requires PyTorch and the sam2 package. "
                "Install PyTorch first, then `pip install -r requirements-sam2.txt`."
            ) from error

        self._torch = torch
        self._predictor = SAM2VideoPredictor.from_pretrained(model_id)
        self.model_id = model_id

        self._autocast = nullcontext()
        if torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            self._autocast = torch.autocast("cuda", dtype=torch.bfloat16)

        # SAM2 keeps decoded frames on the GPU when one is available, and spills
        # to CPU RAM otherwise. Safe either way here because a segment is only
        # ~16 frames; this flag is what made the original whole-video load
        # explode on GPU.
        self._offload = not torch.cuda.is_available()

    @property
    def device(self) -> str:
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def propagate_segment(self, frames: list, seed_rows: list[dict]) -> dict[int, list[dict]]:
        """Propagate `seed_rows` (boxes on frames[0]) through the rest of `frames`.

        Returns a map of local frame offset -> rows, covering offsets 1..len-1.
        Offset 0 is the detector frame and is not re-derived from masks.
        """

        import numpy as np

        if not seed_rows or len(frames) < 2:
            return {}

        # One namespace per SAM2 state; enumerate the combined player + ball
        # seeds so ball objects cannot collide with player objects internally.
        seed_by_obj_id = {obj_id: row for obj_id, row in enumerate(seed_rows, start=1)}

        with self._torch.inference_mode(), self._autocast:
            with tempfile.TemporaryDirectory(prefix="sam2seg_") as temp_dir:
                self._write_segment_frames(frames, Path(temp_dir))
                state = self._predictor.init_state(
                    video_path=temp_dir,
                    offload_video_to_cpu=self._offload,
                    offload_state_to_cpu=self._offload,
                    async_loading_frames=False,
                )

                for obj_id, row in seed_by_obj_id.items():
                    self._predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=np.asarray(row["xyxy"], dtype=np.float32),
                    )

                propagated: dict[int, list[dict]] = {}
                for offset, obj_ids, mask_logits in self._predictor.propagate_in_video(
                    state,
                    start_frame_idx=0,
                    max_frame_num_to_track=len(frames) - 1,
                    reverse=False,
                ):
                    if offset <= 0 or offset >= len(frames):
                        continue

                    rows = []
                    for mask_index, obj_id in enumerate(obj_ids):
                        seed = seed_by_obj_id[int(obj_id)]
                        xyxy = mask_to_xyxy(mask_logits[mask_index])
                        if xyxy is None:
                            continue
                        player_mask = mask_to_crop(mask_logits[mask_index])
                        rows.append(
                            {
                                "xyxy": xyxy,
                                "confidence": seed["confidence"],
                                "class_name": seed["class_name"],
                                "track_id": int(seed["track_id"]),
                                "track_state": "fallback" if seed.get("track_state") == "fallback" else "propagated",
                                "player_mask": player_mask,
                            }
                        )
                    propagated[offset] = rows

                return propagated

    @staticmethod
    def _write_segment_frames(frames: list, directory: Path) -> None:
        """Write segment frames as JPEGs named by index.

        SAM2's loader takes either a video file or a directory of numbered JPEGs.
        We use the directory form so SAM2 never opens the video itself — the
        pipeline decodes once and SAM2 reads only the segment it needs.

        Note: this re-encodes to JPEG, so masks can differ imperceptibly from
        feeding SAM2 the original video. Worth revisiting if it ever shows up in
        output quality; the encode is also ~10% of segment runtime.
        """

        import cv2

        for offset, frame in enumerate(frames):
            if not cv2.imwrite(str(directory / f"{offset}.jpg"), frame):
                raise RuntimeError(f"Could not write SAM2 segment frame {offset}")
