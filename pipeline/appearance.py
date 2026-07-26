"""Torso-colour appearance features for re-identification.

Ported from `scripts/tracking/track_players_video.py`, keeping the same feature
definition and the same tuning knobs as `config/reid.toml` so results stay
comparable with the older BoxMOT path and the team only has one set of concepts
to learn.

The feature is a colour histogram of a player's torso: HSV hue/saturation plus
LAB a/b, L1-normalised then L2-normalised so similarity is a plain dot product.
Deliberately cheap — this runs on CPU while the GPU is busy, and colour is
enough to separate opposing teams.

What it cannot do: separate two players on the SAME team. Identical jerseys give
near-identical histograms. That needs learned re-ID embeddings.

Extraction refuses rather than guesses when the torso is occluded. A colour
sampled through another player's body is worse than no colour at all, because it
would be confidently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class ReidConfig:
    """Tuning for torso crops and quality gates.

    Defaults mirror `config/reid.toml`. They are inlined rather than read from
    the TOML so the pipeline has no file dependency at runtime; change both
    together if you retune.
    """

    enabled: bool = True

    # Crop geometry, as fractions of the player box. Trims the sides, skips the
    # head, and stops above the legs so shorts and shoes do not pollute the
    # jersey colour.
    x_margin_fraction: float = 0.05
    y_start_fraction: float = 0.15
    y_end_fraction: float = 0.68

    # If this much of the torso is covered by other players, give up entirely.
    heavy_overlap_fraction: float = 0.45
    # Even below that, the largest unobstructed piece must be this big.
    min_visible_fraction: float = 0.35

    # Boxes and crops smaller than these are too noisy to describe.
    min_box_width: int = 16
    min_box_height: int = 32
    min_crop_width: int = 8
    min_crop_height: int = 14


def intersect(left: Box, right: Box) -> Box | None:
    x1 = max(int(left[0]), int(right[0]))
    y1 = max(int(left[1]), int(right[1]))
    x2 = min(int(left[2]), int(right[2]))
    y2 = min(int(left[3]), int(right[3]))
    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


def area(box: Box) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def subtract(source: Box, cut: Box) -> list[Box]:
    """Split `source` into the rectangles left over after removing `cut`."""

    overlap = intersect(source, cut)
    if overlap is None:
        return [tuple(int(value) for value in source)]  # type: ignore[return-value]

    sx1, sy1, sx2, sy2 = (int(value) for value in source)
    cx1, cy1, cx2, cy2 = overlap
    pieces = [
        (sx1, sy1, sx2, cy1),  # above
        (sx1, cy2, sx2, sy2),  # below
        (sx1, cy1, cx1, cy2),  # left
        (cx2, cy1, sx2, cy2),  # right
    ]
    return [piece for piece in pieces if piece[2] > piece[0] and piece[3] > piece[1]]


def torso_box(frame_shape, xyxy, config: ReidConfig) -> Box | None:
    """The torso region of a player box, clamped to the frame."""

    height, width = frame_shape[:2]
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1

    tx1 = max(0, min(width, int(round(x1 + box_width * config.x_margin_fraction))))
    tx2 = max(0, min(width, int(round(x2 - box_width * config.x_margin_fraction))))
    ty1 = max(0, min(height, int(round(y1 + box_height * config.y_start_fraction))))
    ty2 = max(0, min(height, int(round(y1 + box_height * config.y_end_fraction))))
    return None if tx2 <= tx1 or ty2 <= ty1 else (tx1, ty1, tx2, ty2)


def visible_torso(frame_shape, xyxy, other_boxes, config: ReidConfig) -> Box | None:
    """Largest unobstructed piece of the torso, or None if too occluded.

    Returning None is the important behaviour: during a crossing, both players'
    torsos are mutually covered, and a histogram sampled there would describe the
    wrong person. Better to have no feature and fall back to geometry.
    """

    torso = torso_box(frame_shape, xyxy, config)
    if torso is None:
        return None

    regions = [torso]
    for other in other_boxes:
        overlap = intersect(torso, other)
        if overlap is None:
            continue
        regions = [piece for region in regions for piece in subtract(region, overlap)]
        if not regions:
            return None

    torso_area = area(torso)
    visible_area = sum(area(region) for region in regions)
    if torso_area <= 0 or visible_area <= 0:
        return None
    if 1.0 - (visible_area / torso_area) >= config.heavy_overlap_fraction:
        return None

    best = max(regions, key=area)
    if area(best) / torso_area < config.min_visible_fraction:
        return None
    if best[2] - best[0] < config.min_crop_width or best[3] - best[1] < config.min_crop_height:
        return None
    return best


def extract(frame, xyxy, other_boxes, config: ReidConfig | None = None) -> list[float] | None:
    """Colour signature for one player box, or None when it cannot be trusted."""

    import cv2
    import numpy as np

    config = config or ReidConfig()
    if not config.enabled:
        return None

    x1, y1, x2, y2 = xyxy
    if (x2 - x1) < config.min_box_width or (y2 - y1) < config.min_box_height:
        return None

    region = visible_torso(frame.shape, xyxy, other_boxes, config)
    if region is None:
        return None

    crop = frame[region[1]:region[3], region[0]:region[2]]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hsv_hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
    lab_hist = cv2.calcHist([lab], [1, 2], None, [4, 4], [0, 256, 0, 256]).flatten()

    features = np.concatenate([hsv_hist, lab_hist]).astype(np.float32)
    total = float(features.sum())
    if total <= 0:
        return None
    features /= total

    norm = float(np.linalg.norm(features))
    if norm <= 0:
        return None
    return (features / norm).tolist()


def similarity(left: list[float] | None, right: list[float] | None) -> float | None:
    """Cosine similarity in [0, 1], or None when either side is missing."""

    import numpy as np

    if left is None or right is None:
        return None
    return float(
        np.clip(
            np.dot(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)),
            0.0,
            1.0,
        )
    )


def find_better_match(
    signature: list[float] | None,
    assigned_track_id: int,
    signatures: dict[int, list[float]],
    margin: float = 0.03,
) -> tuple[int, float, float] | None:
    """Flag a detection that looks more like some OTHER track than its own.

    This is the swap detector. A track ID swap does not create or destroy IDs, so
    it is invisible to counts like `distinct_track_ids` — a clip where all ten
    players trade numbers still reports exactly ten. The only automatic signal we
    have is that after a swap, a track's box suddenly contains someone whose
    colours match a *different* track's history.

    Returns `(better_track_id, own_similarity, other_similarity)` when another
    track is a better colour match by more than `margin`, else None. Comparisons
    use signatures from BEFORE this frame's update, so a swap is caught before it
    can teach itself the wrong colour.
    """

    own = similarity(signature, signatures.get(assigned_track_id))
    if own is None:
        return None

    best_id = None
    best_score = own + margin
    for track_id, other in signatures.items():
        if track_id == assigned_track_id:
            continue
        score = similarity(signature, other)
        if score is not None and score > best_score:
            best_id, best_score = track_id, score

    return None if best_id is None else (best_id, own, best_score)


def blend(previous: list[float] | None, current: list[float], momentum: float = 0.7) -> list[float]:
    """Exponential moving average of a track's signature.

    Smooths out lighting changes and partial occlusions so one bad-but-passing
    observation cannot redefine a player's colour.
    """

    import numpy as np

    if previous is None:
        return current

    merged = momentum * np.asarray(previous, dtype=np.float32) + (1.0 - momentum) * np.asarray(
        current, dtype=np.float32
    )
    norm = float(np.linalg.norm(merged))
    return current if norm <= 0 else (merged / norm).tolist()
