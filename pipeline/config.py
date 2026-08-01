"""Pipeline configuration.

One frozen dataclass holds every knob the pipeline has. Nothing else in the
package reads environment variables directly, so a caller (CLI, Modal, a
notebook) can build a config however it likes and the rest of the code stays
pure.

Cost-relevant defaults are marked COST. Read those comments before changing them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # dotenv is a convenience, not a requirement
    def load_dotenv() -> bool:
        return False


# COST: which SAM2 checkpoint to run. The image encoder runs on EVERY processed
# frame, so checkpoint choice matters, but target_fps is still the biggest cost
# lever because it decides how many frames we process at all.
#   hiera-tiny  ~ fastest, lowest mask quality
#   hiera-small ~ roughly 2-3x faster than large, small quality drop  <-- default
#   hiera-large ~ best quality, what the original local script used
# Start on small while iterating; switch to large only for final demo runs.
DEFAULT_SAM2_MODEL_ID = "facebook/sam2-hiera-small"

# COST: run the detector every N frames. SAM2 propagates masks in between.
# Roboflow's free plan is metered per call, so this directly sets your API spend:
# at 30fps and stride 15 you make 2 calls per second of footage.
# Lower  = more detector calls and more chances to refresh tracks, but also more
# API spend and more hard resets of propagated tracks.
# Higher = fewer detector calls, longer propagation windows, and more pressure
# on SAM2 to hold identity between refreshes.
DEFAULT_FRAME_STRIDE = 15

# COST: process the video at this rate rather than its native one. Measured
# throughput is ~3.3 frames/sec end to end on an L4, so 60fps source footage
# costs ~3 GPU-hours per 10 minutes if every frame is processed. At 15fps that
# drops to ~45 minutes. Set to None to disable downsampling.
DEFAULT_TARGET_FPS = 15.0

# Minimum IoU to keep an existing track ID when re-seeding from detector boxes.
# Matches the original local script's default; changing it changes tracking output.
DEFAULT_MATCH_IOU = 0.15

# Consecutive missed detector calls tolerated before a track is retired. Higher
# values preserve identity through longer occlusions, but risk a coasted box
# latching onto the wrong player.
DEFAULT_MAX_MISSED_DETECTIONS = 2

# Weight of torso-colour similarity relative to box overlap when matching.
# 0.5 treats them equally. Appearance is only ever a tie-breaker among pairs that
# already pass the IoU gate, so this cannot cause long-range false matches.
DEFAULT_APPEARANCE_WEIGHT = 0.5

# Colour similarity that lets a barely-overlapping pair be considered at all.
# Cosine similarity on normalised histograms; opposing team kits typically score
# well below this against each other.
DEFAULT_MIN_APPEARANCE_MATCH = 0.7

# Appearance backend: "osnet" (learned re-ID), "colour" (histogram), "none".
DEFAULT_REID_BACKEND = "osnet"
DEFAULT_TEAM_EMBEDDER_BACKEND = "siglip"
DEFAULT_TEAM_EMBEDDER_MODEL_ID = "vit_base_patch16_siglip_224"
DEFAULT_TEAM_COUNT = 2
DEFAULT_MIN_TEAM_SAMPLES_PER_TRACK = 3
DEFAULT_TEAM_ASSIGNMENT_MARGIN = 0.03
DEFAULT_TEAM_SAMPLE_STRIDE = 5
DEFAULT_MIN_TEAM_SAMPLE_QUALITY = 0.0
DEFAULT_MAX_TEAM_SAMPLES_PER_TRACK = 12

DEFAULT_ROBOFLOW_API_URL = "https://serverless.roboflow.com"
DEFAULT_PLAYER_MODEL_ID = "basketball-player-detection-3-ycjdo/16"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


@dataclass(frozen=True)
class PipelineConfig:
    """Every tunable the pipeline has. Build with `from_env()` or directly."""

    # --- Detector (Roboflow hosted API) ---
    roboflow_api_key: str
    roboflow_api_url: str = DEFAULT_ROBOFLOW_API_URL
    player_model_id: str = DEFAULT_PLAYER_MODEL_ID

    # --- Sampling ---
    # COST: the biggest lever there is. SAM2's image encoder runs on every
    # processed frame at roughly 4 frames/sec on an L4, so runtime and cost are
    # proportional to how many frames we process — not to how long the video is.
    # Source footage is 60fps; at 15fps we process one frame in four, for a 4x
    # saving. Player positions 66ms apart are more than enough for scouting
    # stats. Set to None to process every frame (4x the cost on 60fps video).
    target_fps: float | None = DEFAULT_TARGET_FPS

    # Run the detector every N *processed* frames. Note this is in processed
    # frames, so its meaning in wall-clock depends on target_fps: at 15fps a
    # stride of 15 means one detector call per second of footage.
    # Detector calls are still much cheaper than SAM2, but this is not a free
    # accuracy knob: lower stride means more re-seeding events, which can help or
    # hurt identity depending on the clip.
    frame_stride: int = DEFAULT_FRAME_STRIDE

    # Cap on processed frames, not source frames — it bounds what you spend.
    max_frames: int | None = None
    confidence: float = 0.0
    allowed_classes: tuple[str, ...] = ("player",)

    # --- SAM2 ---
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID
    sam2_match_iou: float = DEFAULT_MATCH_IOU

    # How many consecutive detector calls a track may go unmatched before it is
    # retired. Detectors routinely miss a player for one frame (occlusion, an
    # awkward pose, motion blur). Without a grace period that single miss erases
    # the track, and when the player is found again they get a brand new ID —
    # which is the dominant source of ID churn, not tracking drift.
    # 0 restores the original script's behaviour (retire immediately).
    max_missed_detections: int = DEFAULT_MAX_MISSED_DETECTIONS

    # How much torso colour counts against box overlap when matching detections
    # to tracks. Pure geometry cannot separate two players occupying the same
    # space, which is why crossing players swap numbers; colour can, as long as
    # they are on opposing teams.
    # Negligible cost: histograms on small crops, CPU-only, while the GPU works.
    # 0.0 disables appearance entirely and matches on geometry alone.
    appearance_weight: float = DEFAULT_APPEARANCE_WEIGHT

    # Colour similarity strong enough to consider a pair whose boxes barely
    # overlap. During a crossing the correct pairing can fall below the IoU gate
    # while the swapped one sits above it, so colour needs to be able to rescue
    # it. The boxes must still overlap at all, keeping matches local.
    min_appearance_match: float = DEFAULT_MIN_APPEARANCE_MATCH

    # Which appearance backend describes players for matching.
    #   "osnet"  learned person re-ID (osnet_x1_0_msmt17 via boxmot). Default.
    #   "colour" torso colour histogram; CPU-only, cannot separate teammates.
    #   "none"   geometry only.
    # Measured negligible against SAM2: ~10 small crops per detector call on a
    # GPU that is already warm.
    reid_backend: str = DEFAULT_REID_BACKEND

    # Team-identification feature backend.
    #   "siglip" jersey crop + SigLIP embedding, with the colour signature kept
    #            alongside it for debugging.
    #   "colour" jersey colour histogram only.
    #   "none"   disable team features and leave every track unassigned.
    team_embedder_backend: str = DEFAULT_TEAM_EMBEDDER_BACKEND

    # SigLIP checkpoint to load when `team_embedder_backend="siglip"`.
    team_embedder_model_id: str = DEFAULT_TEAM_EMBEDDER_MODEL_ID

    # Basketball has two teams, but make it explicit rather than baking it into
    # the runner so experiments stay in config like the rest of the pipeline.
    team_count: int = DEFAULT_TEAM_COUNT

    # Tracks with fewer than this many selected team samples are left unassigned
    # instead of being guessed from a short or contradictory crop history.
    min_team_samples_per_track: int = DEFAULT_MIN_TEAM_SAMPLES_PER_TRACK

    # Sample propagated frames this often for team evidence. Detector frames are
    # always sampled; this only controls the in-between propagated frames.
    team_sample_stride: int = DEFAULT_TEAM_SAMPLE_STRIDE

    # Keep only evidence at or above this quality score before per-track top-K
    # selection. The default 0.0 means "accept any non-rejected sample".
    min_team_sample_quality: float = DEFAULT_MIN_TEAM_SAMPLE_QUALITY

    # Cap selected evidence per track so one long clean track does not dominate
    # the global clustering pass.
    max_team_samples_per_track: int = DEFAULT_MAX_TEAM_SAMPLES_PER_TRACK

    # Leave a track unassigned when its per-track vote barely beats the runner-up
    # cluster after the global clustering pass.
    team_assignment_margin: float = DEFAULT_TEAM_ASSIGNMENT_MARGIN

    # COST: writing an annotated MP4 means encoding every single frame, which can
    # be a large fraction of total runtime and produces a file nobody reads when
    # the real output is the track JSON. Off by default; turn on to debug.
    write_annotated_video: bool = False

    # Write per-sample crops, masks, and accept/reject metadata for team ID.
    # Helpful for debugging, noisy for routine runs.
    team_debug_enabled: bool = True

    # COST: the detector call for segment N+1 is fired in a background thread
    # while SAM2 is still working on segment N, so the ~0.3s HTTP round-trip
    # overlaps with GPU compute instead of leaving the GPU idle and billing.
    # One worker is enough: there is only ever one call in flight.
    prefetch_detections: bool = True

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be at least 1 when set")
        if not 0.0 <= self.sam2_match_iou <= 1.0:
            raise ValueError("sam2_match_iou must be between 0 and 1")
        if self.team_count < 1:
            raise ValueError("team_count must be at least 1")
        if self.min_team_samples_per_track < 1:
            raise ValueError("min_team_samples_per_track must be at least 1")
        if self.team_sample_stride < 1:
            raise ValueError("team_sample_stride must be at least 1")
        if self.min_team_sample_quality < 0:
            raise ValueError("min_team_sample_quality must be at least 0")
        if self.max_team_samples_per_track < 1:
            raise ValueError("max_team_samples_per_track must be at least 1")
        if self.team_assignment_margin < 0:
            raise ValueError("team_assignment_margin must be at least 0")
        if not self.roboflow_api_key:
            raise ValueError("roboflow_api_key is required")

    @classmethod
    def from_env(cls, **overrides: object) -> "PipelineConfig":
        """Build config from environment variables, with explicit overrides on top.

        Reads a local `.env` when python-dotenv is installed. On Modal the values
        arrive as real environment variables from a Secret, so the same code path
        works in both places.
        """

        load_dotenv()
        values: dict[str, object] = {
            "roboflow_api_key": os.getenv("ROBOFLOW_API_KEY", ""),
            "roboflow_api_url": os.getenv("ROBOFLOW_API_URL", DEFAULT_ROBOFLOW_API_URL),
            "player_model_id": os.getenv("PLAYER_MODEL_ID", DEFAULT_PLAYER_MODEL_ID),
            "sam2_model_id": os.getenv("SAM2_MODEL_ID", DEFAULT_SAM2_MODEL_ID),
            "team_embedder_backend": os.getenv(
                "TEAM_EMBEDDER_BACKEND", DEFAULT_TEAM_EMBEDDER_BACKEND
            ),
            "team_embedder_model_id": os.getenv(
                "TEAM_EMBEDDER_MODEL_ID", DEFAULT_TEAM_EMBEDDER_MODEL_ID
            ),
            "team_count": _env_int("TEAM_COUNT", DEFAULT_TEAM_COUNT),
            "min_team_samples_per_track": _env_int(
                "MIN_TEAM_SAMPLES_PER_TRACK", DEFAULT_MIN_TEAM_SAMPLES_PER_TRACK
            ),
            "team_sample_stride": _env_int("TEAM_SAMPLE_STRIDE", DEFAULT_TEAM_SAMPLE_STRIDE),
            "min_team_sample_quality": _env_float(
                "MIN_TEAM_SAMPLE_QUALITY", DEFAULT_MIN_TEAM_SAMPLE_QUALITY
            ),
            "max_team_samples_per_track": _env_int(
                "MAX_TEAM_SAMPLES_PER_TRACK", DEFAULT_MAX_TEAM_SAMPLES_PER_TRACK
            ),
            "team_assignment_margin": _env_float(
                "TEAM_ASSIGNMENT_MARGIN", DEFAULT_TEAM_ASSIGNMENT_MARGIN
            ),
            "team_debug_enabled": os.getenv("TEAM_DEBUG_ENABLED", "1") not in {"0", "false", "False"},
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)  # type: ignore[arg-type]
