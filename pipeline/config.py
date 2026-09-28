"""Pipeline configuration.

One frozen dataclass holds every knob the pipeline has. Nothing else in the
package reads environment variables directly, so a caller (CLI, Modal, a
notebook) can build a config however it likes and the rest of the code stays
pure.

Cost-relevant defaults are marked COST. Read those comments before changing them.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass

from pipeline.court_projection import DEFAULT_COURT_MODEL_ID

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
# TrackNet's primary mode is source-frame based, not FPS based. The optional FPS
# value remains only as a compatibility override when ball_all_frames is false.
DEFAULT_BALL_TARGET_FPS = None
DEFAULT_BALL_ALL_FRAMES = True

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
DEFAULT_TEAM_EMBEDDER_MODEL_ID = "Marqo/marqo-fashionSigLIP"
DEFAULT_TEAM_COUNT = 2
DEFAULT_MIN_TEAM_SAMPLES_PER_TRACK = 3
DEFAULT_TEAM_ASSIGNMENT_MARGIN = 0.03
DEFAULT_TEAM_SAMPLE_STRIDE = 5
DEFAULT_MIN_TEAM_SAMPLE_QUALITY = 0.0
DEFAULT_MAX_TEAM_SAMPLES_PER_TRACK = 12
DEFAULT_NUMBER_PREDICTOR_BACKEND = "dummy"
DEFAULT_NUMBER_PREDICTOR_ROBOFLOW_MODEL_ID = "basketball-jersey-numbers-ocr/6"
DEFAULT_NUMBER_PREDICTOR_SCORE_THRESHOLD = 0.7
DEFAULT_TRACKNET_CONTEXT_SIZE = 8
DEFAULT_TRACKNET_INPUT_SIZE = (512, 288)
DEFAULT_TRACKNET_SEED_RADIUS = 12.0
DEFAULT_TRACKNET_HEATMAP_THRESHOLD = 0.5
DEFAULT_TRACKNET_FALLBACK_BUDGET = 12

DEFAULT_ROBOFLOW_API_URL = "https://serverless.roboflow.com"
DEFAULT_PLAYER_MODEL_ID = "basketball-player-detection-3-ycjdo/16"
DEFAULT_SHOT_MODEL_ID = "basketball-player-detection-3-ycjdo/1"
DEFAULT_SHOT_FRAME_STRIDE = 1
DEFAULT_COURT_KEYPOINT_CONFIDENCE = 0.7
DEFAULT_SAM31_PROMPT = "basketball player"
DEFAULT_SAM31_FRAME_START = 60
DEFAULT_SAM31_FRAME_END = 130
DEFAULT_SAM31_RETENTION_FRAMES = 2
DEFAULT_SAM31_OUTPUT_PREFIX = None
DEFAULT_SAM31_ACCEPTANCE_REQUIRED = True
DEFAULT_BBOX_SANITIZER_ENABLED = False
DEFAULT_BBOX_HISTORY_SIZE = 15
DEFAULT_BBOX_MAX_INTERPOLATION_GAP = 2
DEFAULT_BBOX_SIGNAL_THRESHOLD = 3.0
DEFAULT_POSSESSION_TRANSITION_MAX_GAP = 15
DEFAULT_POSSESSION_ENGINE = "legacy"
DEFAULT_POSSESSION_V3_GRACE_SECONDS = 0.6
DEFAULT_POSSESSION_V3_HYSTERESIS_MARGIN = 0.05
DEFAULT_POSSESSION_V3_PERSISTENCE_SECONDS = 0.2
DEFAULT_POSSESSION_V3_EVENT_WINDOW_SECONDS = 1.0
DEFAULT_POSSESSION_V3_CALIBRATION_PROFILE = "default"
DEFAULT_POSSESSION_V3_INPUT_CONTRACT_VERSION = 1
DEFAULT_POSSESSION_V3_BALL_MAX_GAP = 3



def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return None if value in (None, "") else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value in (None, "") else value.lower() in {"1", "true", "yes"}


def _finite_v3(value: object, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class PipelineConfig:
    """Every tunable the pipeline has. Build with `from_env()` or directly."""

    # --- Detector (Roboflow hosted API) ---
    roboflow_api_key: str
    roboflow_api_url: str = DEFAULT_ROBOFLOW_API_URL
    player_model_id: str = DEFAULT_PLAYER_MODEL_ID
    shot_model_id: str = DEFAULT_SHOT_MODEL_ID
    court_model_id: str | None = None
    court_keypoint_confidence: float = DEFAULT_COURT_KEYPOINT_CONFIDENCE

    # --- SAM 3.1 V2 (kept separate from legacy SAM2 settings) ---
    sam31_prompt: str = DEFAULT_SAM31_PROMPT
    sam31_frame_start: int = DEFAULT_SAM31_FRAME_START
    sam31_frame_end: int = DEFAULT_SAM31_FRAME_END
    sam31_retention_frames: int = DEFAULT_SAM31_RETENTION_FRAMES
    sam31_output_prefix: str | None = DEFAULT_SAM31_OUTPUT_PREFIX
    sam31_acceptance_required: bool = DEFAULT_SAM31_ACCEPTANCE_REQUIRED

    # Conservative offline SAM3.1 temporal cleanup. Disabled preserves the
    # existing V2 row contract exactly; enable for V2 sanitizer runs.
    bbox_sanitizer_enabled: bool = DEFAULT_BBOX_SANITIZER_ENABLED
    bbox_history_size: int = DEFAULT_BBOX_HISTORY_SIZE
    max_interpolation_gap: int = DEFAULT_BBOX_MAX_INTERPOLATION_GAP
    bbox_signal_threshold: float = DEFAULT_BBOX_SIGNAL_THRESHOLD
    possession_transition_max_gap: int = DEFAULT_POSSESSION_TRANSITION_MAX_GAP
    possession_engine: str = DEFAULT_POSSESSION_ENGINE
    possession_v3_grace_seconds: float = DEFAULT_POSSESSION_V3_GRACE_SECONDS
    possession_v3_hysteresis_margin: float = DEFAULT_POSSESSION_V3_HYSTERESIS_MARGIN
    possession_v3_persistence_seconds: float = DEFAULT_POSSESSION_V3_PERSISTENCE_SECONDS
    possession_v3_event_window_seconds: float = DEFAULT_POSSESSION_V3_EVENT_WINDOW_SECONDS
    possession_v3_calibration_profile: str = DEFAULT_POSSESSION_V3_CALIBRATION_PROFILE
    possession_v3_input_contract_version: int = DEFAULT_POSSESSION_V3_INPUT_CONTRACT_VERSION
    possession_v3_ball_max_gap: int = DEFAULT_POSSESSION_V3_BALL_MAX_GAP

    @property
    def possession_v3_enabled(self) -> bool:
        return self.possession_engine == "v3"

    # --- Sampling ---
    # COST: the biggest lever there is. SAM2's image encoder runs on every
    # processed frame at roughly 4 frames/sec on an L4, so runtime and cost are
    # proportional to how many frames we process — not to how long the video is.
    # Source footage is 60fps; at 15fps we process one frame in four, for a 4x
    # saving. Player positions 66ms apart are more than enough for scouting
    # stats. Set to None to process every frame (4x the cost on 60fps video).
    target_fps: float | None = DEFAULT_TARGET_FPS
    ball_target_fps: float | None = DEFAULT_BALL_TARGET_FPS
    # TrackNet normally consumes every decoded source frame. Set false only for
    # the legacy ball_target_fps compatibility mode.
    ball_all_frames: bool = DEFAULT_BALL_ALL_FRAMES
    shot_target_fps: float | None = None

    # Run the detector every N *processed* frames. Note this is in processed
    # frames, so its meaning in wall-clock depends on target_fps: at 15fps a
    # stride of 15 means one detector call per second of footage.
    # Detector calls are still much cheaper than SAM2, but this is not a free
    # accuracy knob: lower stride means more re-seeding events, which can help or
    # hurt identity depending on the clip.
    frame_stride: int = DEFAULT_FRAME_STRIDE
    shot_frame_stride: int = DEFAULT_SHOT_FRAME_STRIDE

    # Cap on processed frames, not source frames — it bounds what you spend.
    max_frames: int | None = None
    confidence: float = 0.0
    allowed_classes: tuple[str, ...] = ("player", "number")

    # --- SAM2 ---
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID
    sam2_match_iou: float = DEFAULT_MATCH_IOU

    # Optional local temporal ball detector. Disabled by default so hosted
    # ball detection and Modal remain unchanged.
    tracknet_enabled: bool = False
    tracknet_checkpoint: str | None = None
    tracknet_seed_radius: float = DEFAULT_TRACKNET_SEED_RADIUS
    tracknet_heatmap_threshold: float = DEFAULT_TRACKNET_HEATMAP_THRESHOLD
    tracknet_fallback_budget: int = DEFAULT_TRACKNET_FALLBACK_BUDGET

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

    # Write per-sample crops, masks, and accept/reject metadata for team ID plus
    # detector-frame jersey-number crops/metadata. Helpful for debugging, noisy
    # for routine runs.
    team_debug_enabled: bool = True

    # Jersey-number single-frame predictor backend. Phase 2 only supports the
    # seam-first dummy backend plus the hosted Roboflow OCR model.
    number_predictor_backend: str = DEFAULT_NUMBER_PREDICTOR_BACKEND
    number_predictor_roboflow_model_id: str = DEFAULT_NUMBER_PREDICTOR_ROBOFLOW_MODEL_ID
    number_predictor_score_threshold: float = DEFAULT_NUMBER_PREDICTOR_SCORE_THRESHOLD
    number_evidence_enabled: bool = True

    # COST: the detector call for segment N+1 is fired in a background thread
    # while SAM2 is still working on segment N, so the ~0.3s HTTP round-trip
    # overlaps with GPU compute instead of leaving the GPU idle and billing.
    # One worker is enough: there is only ever one call in flight.
    prefetch_detections: bool = True

    def __post_init__(self) -> None:
        if self.possession_engine not in {"legacy", "v3"}:
            raise ValueError("Unknown possession_engine: expected 'legacy' or 'v3'")
        grace = _finite_v3(self.possession_v3_grace_seconds, "possession_v3_grace_seconds")
        hysteresis = _finite_v3(self.possession_v3_hysteresis_margin, "possession_v3_hysteresis_margin")
        persistence = _finite_v3(self.possession_v3_persistence_seconds, "possession_v3_persistence_seconds")
        event_window = _finite_v3(self.possession_v3_event_window_seconds, "possession_v3_event_window_seconds")
        if not 0 < grace <= 0.6:
            raise ValueError("possession_v3_grace_seconds must be between 0 and 0.6")
        if hysteresis < 0:
            raise ValueError("possession_v3_hysteresis_margin must be non-negative")
        if persistence < 0:
            raise ValueError("possession_v3_persistence_seconds must be non-negative")
        if event_window <= 0:
            raise ValueError("possession_v3_event_window_seconds must be positive")
        if type(self.possession_v3_calibration_profile) is not str or not self.possession_v3_calibration_profile.strip():
            raise ValueError("possession_v3_calibration_profile must not be empty")
        if type(self.possession_v3_input_contract_version) is not int or self.possession_v3_input_contract_version < 1:
            raise ValueError("possession_v3_input_contract_version must be at least 1")
        if type(self.possession_v3_ball_max_gap) is not int or self.possession_v3_ball_max_gap < 0:
            raise ValueError("possession_v3_ball_max_gap must be non-negative")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if self.shot_frame_stride < 1:
            raise ValueError("shot_frame_stride must be at least 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be at least 1 when set")
        if self.ball_target_fps is not None and self.ball_target_fps <= 0:
            raise ValueError("ball_target_fps must be positive when set")
        if not 0.0 <= self.sam2_match_iou <= 1.0:
            raise ValueError("sam2_match_iou must be between 0 and 1")
        if self.tracknet_seed_radius <= 0:
            raise ValueError("tracknet_seed_radius must be greater than 0")
        if not 0.0 <= self.tracknet_heatmap_threshold <= 1.0:
            raise ValueError("tracknet_heatmap_threshold must be between 0 and 1")
        if self.tracknet_fallback_budget < 0:
            raise ValueError("tracknet_fallback_budget must be at least 0")
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
        if self.number_predictor_backend not in {"dummy", "roboflow-hosted"}:
            raise ValueError(
                f"Unknown number_predictor_backend: {self.number_predictor_backend!r} (dummy, roboflow-hosted)"
            )
        if not 0.0 <= self.number_predictor_score_threshold <= 1.0:
            raise ValueError("number_predictor_score_threshold must be between 0 and 1")
        if not 0.0 <= self.court_keypoint_confidence <= 1.0:
            raise ValueError("court_keypoint_confidence must be between 0 and 1")
        if self.sam31_frame_start < 0:
            raise ValueError("sam31_frame_start must be at least 0")
        if self.sam31_frame_end < self.sam31_frame_start:
            raise ValueError("sam31_frame_end must not precede sam31_frame_start")
        if self.sam31_retention_frames < 0:
            raise ValueError("sam31_retention_frames must be at least 0")
        if self.bbox_history_size < 2:
            raise ValueError("bbox_history_size must be at least 2")
        if self.max_interpolation_gap < 1:
            raise ValueError("max_interpolation_gap must be at least 1")
        if self.bbox_signal_threshold <= 0:
            raise ValueError("bbox_signal_threshold must be positive")
        if self.possession_transition_max_gap < 1:
            raise ValueError("possession_transition_max_gap must be at least 1")
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
            "shot_model_id": os.getenv("SHOT_MODEL_ID", DEFAULT_SHOT_MODEL_ID),
            "court_model_id": os.getenv("COURT_MODEL_ID", DEFAULT_COURT_MODEL_ID),
            "sam2_model_id": os.getenv("SAM2_MODEL_ID", DEFAULT_SAM2_MODEL_ID),
            "tracknet_enabled": os.getenv("TRACKNET_ENABLED", "0").lower() in {"1", "true", "yes"},
            "tracknet_checkpoint": os.getenv("TRACKNET_CHECKPOINT") or None,
            "tracknet_seed_radius": _env_float("TRACKNET_SEED_RADIUS", DEFAULT_TRACKNET_SEED_RADIUS),
            "tracknet_heatmap_threshold": _env_float(
                "TRACKNET_HEATMAP_THRESHOLD", DEFAULT_TRACKNET_HEATMAP_THRESHOLD
            ),
            "tracknet_fallback_budget": _env_int(
                "TRACKNET_FALLBACK_BUDGET", DEFAULT_TRACKNET_FALLBACK_BUDGET
            ),
            "shot_target_fps": _env_optional_float("SHOT_TARGET_FPS"),
            "ball_target_fps": _env_optional_float("BALL_TARGET_FPS"),
            "ball_all_frames": os.getenv("BALL_ALL_FRAMES", "1").lower() in {"1", "true", "yes"},
            "shot_frame_stride": _env_int("SHOT_FRAME_STRIDE", DEFAULT_SHOT_FRAME_STRIDE),
            "court_keypoint_confidence": _env_float(
                "COURT_KEYPOINT_CONFIDENCE", DEFAULT_COURT_KEYPOINT_CONFIDENCE
            ),
            "sam31_prompt": os.getenv("SAM31_PROMPT", DEFAULT_SAM31_PROMPT),
            "sam31_frame_start": _env_int("SAM31_FRAME_START", DEFAULT_SAM31_FRAME_START),
            "sam31_frame_end": _env_int("SAM31_FRAME_END", DEFAULT_SAM31_FRAME_END),
            "sam31_retention_frames": _env_int(
                "SAM31_RETENTION_FRAMES", DEFAULT_SAM31_RETENTION_FRAMES
            ),
            "sam31_output_prefix": os.getenv("SAM31_OUTPUT_PREFIX") or DEFAULT_SAM31_OUTPUT_PREFIX,
            "sam31_acceptance_required": _env_bool(
                "SAM31_ACCEPTANCE_REQUIRED", DEFAULT_SAM31_ACCEPTANCE_REQUIRED
            ),
            "bbox_sanitizer_enabled": _env_bool(
                "BBOX_SANITIZER_ENABLED", DEFAULT_BBOX_SANITIZER_ENABLED
            ),
            "bbox_history_size": _env_int("BBOX_HISTORY_SIZE", DEFAULT_BBOX_HISTORY_SIZE),
            "max_interpolation_gap": _env_int(
                "BBOX_MAX_INTERPOLATION_GAP", DEFAULT_BBOX_MAX_INTERPOLATION_GAP
            ),
            "bbox_signal_threshold": _env_float(
                "BBOX_SIGNAL_THRESHOLD", DEFAULT_BBOX_SIGNAL_THRESHOLD
            ),
            "possession_transition_max_gap": _env_int(
                "POSSESSION_TRANSITION_MAX_GAP", DEFAULT_POSSESSION_TRANSITION_MAX_GAP
            ),
            "possession_engine": os.getenv("POSSESSION_ENGINE", DEFAULT_POSSESSION_ENGINE),
            "possession_v3_grace_seconds": _env_float(
                "POSSESSION_V3_GRACE_SECONDS", DEFAULT_POSSESSION_V3_GRACE_SECONDS
            ),
            "possession_v3_hysteresis_margin": _env_float(
                "POSSESSION_V3_HYSTERESIS_MARGIN", DEFAULT_POSSESSION_V3_HYSTERESIS_MARGIN
            ),
            "possession_v3_persistence_seconds": _env_float(
                "POSSESSION_V3_PERSISTENCE_SECONDS", DEFAULT_POSSESSION_V3_PERSISTENCE_SECONDS
            ),
            "possession_v3_event_window_seconds": _env_float(
                "POSSESSION_V3_EVENT_WINDOW_SECONDS", DEFAULT_POSSESSION_V3_EVENT_WINDOW_SECONDS
            ),
            "possession_v3_calibration_profile": os.getenv(
                "POSSESSION_V3_CALIBRATION_PROFILE", DEFAULT_POSSESSION_V3_CALIBRATION_PROFILE
            ),
            "possession_v3_input_contract_version": _env_int(
                "POSSESSION_V3_INPUT_CONTRACT_VERSION", DEFAULT_POSSESSION_V3_INPUT_CONTRACT_VERSION
            ),
            "possession_v3_ball_max_gap": _env_int(
                "POSSESSION_V3_BALL_MAX_GAP", DEFAULT_POSSESSION_V3_BALL_MAX_GAP
            ),
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
            "number_predictor_backend": os.getenv(
                "NUMBER_PREDICTOR_BACKEND", DEFAULT_NUMBER_PREDICTOR_BACKEND
            ),
            "number_predictor_roboflow_model_id": os.getenv(
                "NUMBER_PREDICTOR_ROBOFLOW_MODEL_ID",
                DEFAULT_NUMBER_PREDICTOR_ROBOFLOW_MODEL_ID,
            ),
            "number_predictor_score_threshold": _env_float(
                "NUMBER_PREDICTOR_SCORE_THRESHOLD",
                DEFAULT_NUMBER_PREDICTOR_SCORE_THRESHOLD,
            ),
            "number_evidence_enabled": os.getenv("NUMBER_EVIDENCE_ENABLED", "1").lower()
            in {"1", "true", "yes"},
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)  # type: ignore[arg-type]
