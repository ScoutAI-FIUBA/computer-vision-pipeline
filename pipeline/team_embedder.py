"""Team feature backends for jersey/team identification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pipeline.appearance import ReidConfig, visible_torso

DEFAULT_SIGLIP_MODEL_ID = "vit_base_patch16_siglip_224"


@dataclass(frozen=True)
class TeamEmbeddingSample:
    """One candidate player crop to describe for team clustering."""

    detection_id: int
    track_id: int | None
    xyxy: tuple[float, float, float, float]
    player_mask: object | None = None


@dataclass(frozen=True)
class TeamFeatures:
    """Cheap colour prior plus optional learned embedding for one sample."""

    color_signature: list[float] | None
    embedding: list[float] | None
    quality_score: float = 0.0
    quality_details: dict[str, float] = field(default_factory=dict)
    rejection_reasons: list[str] = field(default_factory=list)
    used_mask: bool = False
    uniform_xyxy: tuple[int, int, int, int] | None = None
    uniform_pixel_count: int = 0
    total_pixel_count: int = 0
    debug_crop: object | None = None
    debug_mask: object | None = None


class TeamEmbedder(Protocol):
    """Turns player samples into team-oriented features."""

    name: str

    def embed(self, frame, samples: list[TeamEmbeddingSample]) -> list[TeamFeatures]:
        """One feature bundle per sample."""
        ...


class NullTeamEmbedder:
    """Placeholder backend until a real team embedder is wired in."""

    name = "none"

    def embed(self, frame, samples: list[TeamEmbeddingSample]) -> list[TeamFeatures]:
        return [TeamFeatures(color_signature=None, embedding=None) for _ in samples]


class ColourTeamEmbedder:
    """Cheap team backend based on visible jersey colour histograms."""

    name = "colour-histogram"

    def __init__(self, config: ReidConfig | None = None) -> None:
        self._config = config or ReidConfig()

    def embed(self, frame, samples: list[TeamEmbeddingSample]) -> list[TeamFeatures]:
        boxes = [list(sample.xyxy) for sample in samples]
        return [
            extract_team_features(
                frame,
                samples[index],
                [other for position, other in enumerate(boxes) if position != index],
                self._config,
            )
            for index, box in enumerate(boxes)
        ]


def _load_siglip_components():
    import torch
    import timm
    from PIL import Image

    return torch, timm, Image


class SiglipTeamEmbedder:
    """Jersey colour prior plus optional SigLIP refinement features."""

    def __init__(self, model_id: str = DEFAULT_SIGLIP_MODEL_ID, config: ReidConfig | None = None) -> None:
        import cv2
        import numpy as np

        torch, timm, image_module = _load_siglip_components()
        self._torch = torch
        self._cv2 = cv2
        self._np = np
        self._image = image_module
        self._config = config or ReidConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = timm.create_model(model_id, pretrained=True, num_classes=0).to(self.device)
        data_config = timm.data.resolve_model_data_config(self._model)
        self._transform = timm.data.create_transform(**data_config, is_training=False)
        self._model.eval()
        self.name = f"siglip:{model_id}"

    def embed(self, frame, samples: list[TeamEmbeddingSample]) -> list[TeamFeatures]:
        if not samples:
            return []

        boxes = [list(sample.xyxy) for sample in samples]
        results = [TeamFeatures(color_signature=None, embedding=None) for _ in samples]
        crops = []
        keep = []

        for index, sample in enumerate(samples):
            features = extract_team_features(
                frame,
                sample,
                [other for position, other in enumerate(boxes) if position != index],
                self._config,
            )
            results[index] = features
            if features.debug_crop is None:
                continue
            crops.append(
                self._image.fromarray(self._cv2.cvtColor(features.debug_crop, self._cv2.COLOR_BGR2RGB))
            )
            keep.append(index)

        if not keep:
            return results

        inputs = self._torch.stack([self._transform(crop) for crop in crops]).to(self.device)
        with self._torch.inference_mode():
            features = self._model(inputs)
        array = self._np.asarray(
            features.detach().cpu() if hasattr(features, "detach") else features,
            dtype=self._np.float32,
        )
        norms = self._np.clip(self._np.linalg.norm(array, axis=1, keepdims=True), 1e-12, None)
        normalized = array / norms

        for position, index in enumerate(keep):
            results[index] = TeamFeatures(
                color_signature=results[index].color_signature,
                embedding=normalized[position].tolist(),
                quality_score=results[index].quality_score,
                quality_details=results[index].quality_details,
                rejection_reasons=results[index].rejection_reasons,
                used_mask=results[index].used_mask,
                uniform_xyxy=results[index].uniform_xyxy,
                uniform_pixel_count=results[index].uniform_pixel_count,
                total_pixel_count=results[index].total_pixel_count,
                debug_crop=results[index].debug_crop,
                debug_mask=results[index].debug_mask,
            )
        return results


def extract_team_features(frame, sample: TeamEmbeddingSample, other_boxes, config: ReidConfig) -> TeamFeatures:
    crop, mask, uniform_xyxy, quality_details, rejection_reasons, used_mask = _extract_uniform_region(
        frame,
        sample.xyxy,
        sample.player_mask,
        other_boxes,
        config,
    )
    if crop is None:
        return TeamFeatures(
            color_signature=None,
            embedding=None,
            quality_score=0.0,
            quality_details=quality_details,
            rejection_reasons=rejection_reasons,
            used_mask=used_mask,
            uniform_xyxy=uniform_xyxy,
        )

    quality_score = _compute_sample_quality(crop, mask, used_mask, quality_details)
    color_signature = _colour_signature_from_crop(crop, mask)
    if color_signature is None:
        rejection_reasons = list(rejection_reasons) + ["empty_signature"]
    return TeamFeatures(
        color_signature=color_signature,
        embedding=None,
        quality_score=quality_score,
        quality_details=quality_details,
        rejection_reasons=rejection_reasons,
        used_mask=used_mask,
        uniform_xyxy=uniform_xyxy,
        uniform_pixel_count=int(mask.sum()) if mask is not None else int(crop.shape[0] * crop.shape[1]),
        total_pixel_count=int(crop.shape[0] * crop.shape[1]),
        debug_crop=_apply_mask_to_crop(crop, mask),
        debug_mask=None if mask is None else (mask.astype("uint8") * 255),
    )


def _extract_colour_signature(frame, xyxy, other_boxes, config: ReidConfig) -> list[float] | None:
    x1, y1, x2, y2 = xyxy
    if (x2 - x1) < config.min_box_width or (y2 - y1) < config.min_box_height:
        return None

    region = visible_torso(frame.shape, xyxy, other_boxes, config)
    if region is None:
        return None

    jersey = _jersey_region(region, config)
    crop = frame[jersey[1]:jersey[3], jersey[0]:jersey[2]]
    return None if crop.size == 0 else _colour_signature_from_crop(crop)


def _extract_uniform_region(frame, xyxy, player_mask, other_boxes, config: ReidConfig):
    import numpy as np

    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    x1 = max(0, min(frame.shape[1], x1))
    x2 = max(0, min(frame.shape[1], x2))
    y1 = max(0, min(frame.shape[0], y1))
    y2 = max(0, min(frame.shape[0], y2))
    quality_details: dict[str, float] = {}
    rejection_reasons: list[str] = []

    if x2 - x1 < config.min_box_width or y2 - y1 < config.min_box_height:
        return None, None, None, quality_details, ["box_too_small"], False

    if player_mask is not None:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None, None, quality_details, ["empty_crop"], True
        mask = np.asarray(player_mask).astype(bool)
        if mask.shape[:2] != crop.shape[:2]:
            return None, None, None, quality_details, ["mask_shape_mismatch"], True
        height, width = crop.shape[:2]
        rx1 = max(0, int(round(width * 0.18)))
        rx2 = min(width, int(round(width * 0.82)))
        ry1 = max(0, int(round(height * 0.15)))
        ry2 = min(height, int(round(height * 0.62)))
        if rx2 <= rx1 or ry2 <= ry1:
            return None, None, None, quality_details, ["uniform_region_invalid"], True
        uniform_crop = crop[ry1:ry2, rx1:rx2]
        uniform_mask = mask[ry1:ry2, rx1:rx2]
        quality_details["mask_fill_ratio"] = float(uniform_mask.mean()) if uniform_mask.size else 0.0
        if uniform_mask.any():
            return uniform_crop, uniform_mask, (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2), quality_details, rejection_reasons, True
        rejection_reasons.append("empty_uniform_mask")

    region = visible_torso(frame.shape, xyxy, other_boxes, config)
    if region is None:
        return None, None, None, quality_details, ["torso_not_visible"], False

    jersey = _jersey_region(region, config)
    crop = frame[jersey[1]:jersey[3], jersey[0]:jersey[2]]
    if crop.size == 0:
        return None, None, None, quality_details, ["empty_crop"], False
    return crop, None, jersey, quality_details, rejection_reasons, False


def _compute_sample_quality(crop, mask, used_mask: bool, quality_details: dict[str, float]) -> float:
    pixel_count = int(mask.sum()) if mask is not None else int(crop.shape[0] * crop.shape[1])
    total_pixels = int(crop.shape[0] * crop.shape[1])
    area_score = min(1.0, pixel_count / 400.0)
    fill_score = 0.75 if mask is None else min(1.0, pixel_count / max(total_pixels * 0.35, 1.0))
    mask_score = 1.0 if used_mask else 0.7
    quality_details["pixel_count"] = float(pixel_count)
    quality_details["area_score"] = float(area_score)
    quality_details["fill_score"] = float(fill_score)
    quality_details["mask_score"] = float(mask_score)
    return round(0.45 * area_score + 0.35 * fill_score + 0.20 * mask_score, 4)


def _apply_mask_to_crop(crop, mask):
    import numpy as np

    if mask is None:
        return crop
    result = crop.copy()
    result[~np.asarray(mask).astype(bool)] = 0
    return result


def _colour_signature_from_crop(crop, mask=None) -> list[float] | None:
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hist_mask = None if mask is None else mask.astype("uint8") * 255
    hsv_hist = cv2.calcHist([hsv], [0, 1], hist_mask, [8, 4], [0, 180, 0, 256]).flatten()
    lab_hist = cv2.calcHist([lab], [1, 2], hist_mask, [4, 4], [0, 256, 0, 256]).flatten()

    features = np.concatenate([hsv_hist, lab_hist]).astype(np.float32)
    total = float(features.sum())
    if total <= 0:
        return None
    features /= total

    norm = float(np.linalg.norm(features))
    if norm <= 0:
        return None
    return (features / norm).tolist()


def _jersey_region(region: tuple[int, int, int, int], config: ReidConfig) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = region
    width = x2 - x1
    height = y2 - y1
    trimmed = (
        int(round(x1 + width * 0.20)),
        int(round(y1 + height * 0.08)),
        int(round(x2 - width * 0.20)),
        int(round(y1 + height * 0.60)),
    )
    if (
        trimmed[2] - trimmed[0] < config.min_crop_width
        or trimmed[3] - trimmed[1] < config.min_crop_height
    ):
        return region
    return trimmed


def build_team_embedder(
    backend: str = "siglip", model_id: str = DEFAULT_SIGLIP_MODEL_ID
) -> TeamEmbedder:
    """Construct the requested team embedding backend."""

    if backend == "none":
        return NullTeamEmbedder()
    if backend == "siglip":
        return SiglipTeamEmbedder(model_id=model_id)
    if backend == "colour":
        return ColourTeamEmbedder()
    raise ValueError(f"Unknown team_embedder backend: {backend!r} (siglip, colour, none)")
