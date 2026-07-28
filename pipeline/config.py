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


# COST: which SAM2 checkpoint to run. The image encoder runs on EVERY frame, so
# this is the single biggest driver of GPU time.
#   hiera-tiny  ~ fastest, lowest mask quality
#   hiera-small ~ roughly 2-3x faster than large, small quality drop  <-- default
#   hiera-large ~ best quality, what the original local script used
# Start on small while iterating; switch to large only for final demo runs.
DEFAULT_SAM2_MODEL_ID = "facebook/sam2-hiera-small"

# COST: run the detector every N processed frames. SAM2 propagates in between.
# Roboflow's free plan is metered per call, so this also sets your API spend.
#
# Identity quality is NOT monotonic in this value — it is U-shaped, and 10 was
# measured as the minimum. On the test clip (146 processed frames, 10 players),
# distinct track IDs came out as:
#
#     stride  4 -> 18      too many detector calls; each is a chance to drop a
#                          track the detector momentarily missed
#     stride 10 -> 10      correct, and flat across the whole clip
#     stride 15 -> 11      long propagation lets identities drift
#
# Do not "improve" this by lowering it. That was tried and measured worse.
DEFAULT_FRAME_STRIDE = 10

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

DEFAULT_ROBOFLOW_API_URL = "https://serverless.roboflow.com"
DEFAULT_PLAYER_MODEL_ID = "basketball-player-detection-3-ycjdo/16"


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
    # Detector calls are cheap relative to SAM2 (6 calls cost ~2s of a 27s run),
    # so if identity switches get worse, lowering this is nearly free.
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

    # COST: writing an annotated MP4 means encoding every single frame, which can
    # be a large fraction of total runtime and produces a file nobody reads when
    # the real output is the track JSON. Off by default; turn on to debug.
    write_annotated_video: bool = False

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
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)  # type: ignore[arg-type]
