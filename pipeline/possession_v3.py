"""Versioned V3 possession contracts; resolution is deferred to later units."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.config import PipelineConfig

V3_CONTRACT_VERSION = 1
V3_INPUT_CONTRACT_VERSION = 1
DEFAULT_CANDIDATE_AMBIGUITY_MARGIN = 0.1
MIN_CANDIDATE_SCORE = 0.5
MASK_SCORE_WEIGHT = 0.35
BBOX_SCORE_WEIGHT = 0.35
PLAYER_CONFIDENCE_WEIGHT = 0.15
BALL_QUALITY_WEIGHT = 0.10
PREVIOUS_HOLDER_WEIGHT = 0.10
class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value
class PossessionState(_ValueEnum):
    NO_POSSESSION = "NO_POSSESSION"
    CANDIDATE = "CANDIDATE"
    OBSERVED = "OBSERVED"
    GRACE = "GRACE"
    CONTESTED = "CONTESTED"
    UNKNOWN = "UNKNOWN"
    INFERRED = "INFERRED"
    TRANSITIONING = "TRANSITIONING"
    LOST_NO_DETECTION = "LOST_NO_DETECTION"
    UNKNOWN_RECOVERY = "UNKNOWN_RECOVERY"
class EvidenceKind(_ValueEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    CONTESTED = "contested"
    UNKNOWN = "unknown"
class EventType(_ValueEnum):
    TRANSFER = "possession_transfer"
    PASS = "PASS"
    STEAL = "STEAL"
    LOSS_OF_CONTROL = "LOSS_OF_CONTROL"
    TURNOVER = "TURNOVER"
    UNKNOWN = "UNKNOWN"
    UNKNOWN_RECOVERY = "UNKNOWN_RECOVERY"
class ClearReason(_ValueEnum):
    GRACE_EXPIRED = "GRACE_EXPIRED"
    LOOSE_BALL_CONFIRMED = "LOOSE_BALL_CONFIRMED"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    END_OF_SEGMENT = "END_OF_SEGMENT"
    EXPLICIT_RESET = "EXPLICIT_RESET"
    LOST_NO_DETECTION = "LOST_NO_DETECTION"
def _frame(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
def _schema(value: int) -> int:
    if type(value) is not int or value != V3_CONTRACT_VERSION:
        raise ValueError(f"schema_version must be {V3_CONTRACT_VERSION}")
    return value
def _score(value: float | None, name: str) -> float | None:
    if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return None if value is None else float(value)
def _required_score(value, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return _score(value, name)
def _enum(value, kind, name):
    try:
        return value if isinstance(value, kind) else kind(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}: {value!r}") from error
def _text_list(value, name):
    if not isinstance(value, (list, tuple)) or any(type(x) is not str or not x.strip() for x in value):
        raise ValueError(f"{name} must contain non-empty strings")
def _identity(stable, raw, team):
    if stable is not None and (type(stable) is not str or not stable.strip()):
        raise ValueError("stable identity must be a non-empty string")
    if raw is not None and (type(raw) is not int or raw < 1):
        raise ValueError("raw identity must be a positive integer")
    if team is not None and (type(team) is not int or team < 0):
        raise ValueError("team_id must be a non-negative integer")
def _holder(value):
    if value is not None and (type(value) is not str or not value.strip()):
        raise ValueError("holder identity must be a non-empty string when set")
def _required_text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value
def _optional_nonnegative_frame(value, name):
    if value is not None:
        _frame(value, name)
def _thresholds(value):
    if not isinstance(value, dict) or any(type(k) is not str or not k.strip() or type(v) not in {int, float} or not math.isfinite(v) for k, v in value.items()):
        raise ValueError("thresholds/contributions must map names to finite numbers")
@dataclass(frozen=True)
class Provenance:
    source_records: list[str] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.source_records, (list, tuple)) or any(type(x) is not str or not x.strip() for x in self.source_records):
            raise ValueError("source_records must contain non-empty strings")
        if not isinstance(self.frame_indices, (list, tuple)) or any(type(x) is not int or x < 0 for x in self.frame_indices):
            raise ValueError("frame_indices must contain non-negative integers")
        if not isinstance(self.reason_codes, (list, tuple)) or any(type(x) is not str or not x.strip() for x in self.reason_codes):
            raise ValueError("reason_codes must contain non-empty strings")
        object.__setattr__(self, "source_records", list(self.source_records))
        object.__setattr__(self, "frame_indices", list(self.frame_indices))
        object.__setattr__(self, "reason_codes", list(self.reason_codes))
@dataclass(frozen=True)
class CandidateEvidence:
    stable_player_id: str | None
    raw_track_id: int | None
    team_id: int | None
    score: float
    contributions: dict[str, float] = field(default_factory=dict)
    rejected_reasons: list[str] = field(default_factory=list)
    fallback_identity: str | None = None

    def __post_init__(self) -> None:
        _required_score(self.score, "score")
        _identity(self.stable_player_id, self.raw_track_id, self.team_id)
        _thresholds(self.contributions)
        _text_list(self.rejected_reasons, "rejected_reasons")
        if self.fallback_identity is not None and (type(self.fallback_identity) is not str or not self.fallback_identity.strip()):
            raise ValueError("fallback_identity must be a non-empty string when set")


@dataclass(frozen=True)
class V4CandidateEvidence(CandidateEvidence):
    """V4-only candidate sidecar; the V3 contract remains unchanged."""
    spatial_score: float | None = None
    mask_score: float | None = None
    bbox_score: float | None = None
    mask_bbox_agreement: float | None = None
    source: str = "v4"
    normalized_distance: float | None = None
    relative_motion: float = 0.0
    ball_quality: float = 0.0
    continuity: float = 0.0
    holder_control_score: float = 0.0
    holder_release_score: float = 0.0
    challenger_acquisition_score: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for value, name in ((self.spatial_score, "spatial_score"), (self.mask_score, "mask_score"),
                            (self.bbox_score, "bbox_score"), (self.mask_bbox_agreement, "mask_bbox_agreement")):
            _score(value, name)
        if type(self.source) is not str or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        for value, name in ((self.normalized_distance, "normalized_distance"),
                            (self.relative_motion, "relative_motion"), (self.ball_quality, "ball_quality"),
                            (self.continuity, "continuity"), (self.holder_control_score, "holder_control_score"),
                            (self.holder_release_score, "holder_release_score"),
                            (self.challenger_acquisition_score, "challenger_acquisition_score")):
            _score(value, name)
@dataclass(frozen=True)
class PossessionContext:
    """Serializable roles and temporal state owned by the future resolver."""
    effective_holder_id: str | None = None
    effective_holder_team: int | None = None
    observed_candidate_id: str | None = None
    challenger_id: str | None = None
    challenger_start_frame: int | None = None
    challenger_elapsed_seconds: float = 0.0
    challenger_confirmation_frame: int | None = None
    grace_start_frame: int | None = None
    grace_remaining_seconds: float = 0.0
    last_observed_holder_frame: int | None = None
    last_observed_holder_score: float | None = None
    recovery_source_id: str | None = None
    recovery_source_team: int | None = None
    recovery_reason: str | None = None
    recovery_remaining_seconds: float = 0.0
    control_evidence: float | None = None
    release_evidence: float | None = None
    acquisition_evidence: float | None = None
    ball_in_transit: bool = False
    evidence_window_start: int | None = None
    evidence_window_end: int | None = None

    def __post_init__(self) -> None:
        _holder(self.effective_holder_id)
        if self.effective_holder_team is not None and (type(self.effective_holder_team) is not int or self.effective_holder_team < 0):
            raise ValueError("effective_holder_team must be a non-negative integer when set")
        _holder(self.observed_candidate_id)
        _holder(self.challenger_id)
        _optional_nonnegative_frame(self.challenger_start_frame, "challenger_start_frame")
        _optional_nonnegative_frame(self.challenger_confirmation_frame, "challenger_confirmation_frame")
        _optional_nonnegative_frame(self.grace_start_frame, "grace_start_frame")
        if self.challenger_start_frame is None and self.challenger_confirmation_frame is not None:
            raise ValueError("challenger confirmation requires challenger start")
        if (self.challenger_start_frame is not None and self.challenger_confirmation_frame is not None
                and self.challenger_confirmation_frame < self.challenger_start_frame):
            raise ValueError("challenger confirmation must not precede challenger start")
        if self.challenger_id is None and (self.challenger_start_frame is not None or self.challenger_confirmation_frame is not None):
            raise ValueError("challenger frames require challenger_id")
        if (type(self.challenger_elapsed_seconds) not in {int, float}
                or not math.isfinite(self.challenger_elapsed_seconds)
                or self.challenger_elapsed_seconds < 0):
            raise ValueError("challenger_elapsed_seconds must be finite and non-negative")
        if self.challenger_id is None and self.challenger_elapsed_seconds != 0:
            raise ValueError("challenger elapsed time requires challenger_id")
        if type(self.grace_remaining_seconds) not in {int, float} or not math.isfinite(self.grace_remaining_seconds) or self.grace_remaining_seconds < 0:
            raise ValueError("grace_remaining_seconds must be finite and non-negative")
        _optional_nonnegative_frame(self.last_observed_holder_frame, "last_observed_holder_frame")
        _score(self.last_observed_holder_score, "last_observed_holder_score")
        _holder(self.recovery_source_id)
        if self.recovery_source_team is not None and (type(self.recovery_source_team) is not int or self.recovery_source_team < 0):
            raise ValueError("recovery_source_team must be a non-negative integer when set")
        if self.recovery_reason is not None:
            _required_text(self.recovery_reason, "recovery_reason")
        if (type(self.recovery_remaining_seconds) not in {int, float}
                or not math.isfinite(self.recovery_remaining_seconds)
                or self.recovery_remaining_seconds < 0):
            raise ValueError("recovery_remaining_seconds must be finite and non-negative")
@dataclass(frozen=True)
class PossessionAudit:
    last_holder_id: str | None = None
    last_holder_team: int | None = None
    clear_reason: ClearReason | None = None
    switch_confirmed: bool = False
    observation_quality: str = "unknown"
    observation_source: str = "unknown"
    authority: str = "unknown"
    grace_eligible: bool = False
    grace_elapsed_seconds: float = 0.0
    previous_holder_id: str | None = None
    previous_holder_team: int | None = None
    new_holder_id: str | None = None
    new_holder_team: int | None = None
    previous_holder_score: float | None = None
    new_holder_score: float | None = None
    challenger_start_frame: int | None = None
    challenger_confirmation_frame: int | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    reverted_switch_duration_seconds: float | None = None
    provenance: Provenance = field(default_factory=Provenance)
    recovery_source_id: str | None = None
    recovery_source_team: int | None = None
    recovery_reason: str | None = None
    recovery_window_seconds: float = 0.0
    v4_schema_version: int | None = None
    feature_flag: str | None = None
    control_score: float | None = None
    release_score: float | None = None
    acquisition_score: float | None = None
    gate_decision: str | None = None
    gate_reason_codes: list[str] = field(default_factory=list)
    spatial_contributions: dict[str, float] = field(default_factory=dict)
    evidence_sources: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _holder(self.last_holder_id)
        if self.last_holder_team is not None and (type(self.last_holder_team) is not int or self.last_holder_team < 0):
            raise ValueError("last_holder_team must be a non-negative integer when set")
        if self.clear_reason is not None:
            object.__setattr__(self, "clear_reason", _enum(self.clear_reason, ClearReason, "clear_reason"))
        if type(self.switch_confirmed) is not bool:
            raise ValueError("switch_confirmed must be boolean")
        quality = _required_text(self.observation_quality, "observation_quality")
        if quality not in {kind.value for kind in EvidenceKind}:
            raise ValueError(f"invalid observation_quality: {quality!r}")
        _required_text(self.observation_source, "observation_source")
        _required_text(self.authority, "authority")
        if type(self.grace_eligible) is not bool:
            raise ValueError("grace_eligible must be boolean")
        if (type(self.grace_elapsed_seconds) not in {int, float}
                or not math.isfinite(self.grace_elapsed_seconds)
                or self.grace_elapsed_seconds < 0):
            raise ValueError("grace_elapsed_seconds must be finite and non-negative")
        _score(self.previous_holder_score, "previous_holder_score")
        _score(self.new_holder_score, "new_holder_score")
        for value, name in ((self.challenger_start_frame, "challenger_start_frame"),
                            (self.challenger_confirmation_frame, "challenger_confirmation_frame"),
                            (self.frame_start, "frame_start"), (self.frame_end, "frame_end")):
            _optional_nonnegative_frame(value, name)
        for value, name in ((self.timestamp_start, "timestamp_start"), (self.timestamp_end, "timestamp_end")):
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if (self.reverted_switch_duration_seconds is not None and
                (type(self.reverted_switch_duration_seconds) not in {int, float} or
                 not math.isfinite(self.reverted_switch_duration_seconds) or
                 self.reverted_switch_duration_seconds < 0)):
            raise ValueError("reverted_switch_duration_seconds must be finite and non-negative")
        if self.frame_start is not None and self.frame_end is not None and self.frame_end < self.frame_start:
            raise ValueError("frame_end must not precede frame_start")
        if self.timestamp_start is not None and self.timestamp_end is not None and self.timestamp_end < self.timestamp_start:
            raise ValueError("timestamp_end must not precede timestamp_start")
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be Provenance")
        _holder(self.recovery_source_id)
        if self.recovery_source_team is not None and (type(self.recovery_source_team) is not int or self.recovery_source_team < 0):
            raise ValueError("recovery_source_team must be a non-negative integer when set")
        if self.recovery_reason is not None:
            _required_text(self.recovery_reason, "recovery_reason")
        if (type(self.recovery_window_seconds) not in {int, float}
                or not math.isfinite(self.recovery_window_seconds)
                or self.recovery_window_seconds < 0):
            raise ValueError("recovery_window_seconds must be finite and non-negative")
@dataclass(frozen=True)
class PossessionObservation:
    frame_index: int
    state: PossessionState
    evidence_kind: EvidenceKind
    schema_version: int = V3_CONTRACT_VERSION
    stable_player_id: str | None = None
    raw_track_id: int | None = None
    team_id: int | None = None
    score: float | None = None
    score_margin: float | None = None
    candidates: list[CandidateEvidence] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    profile: str = "default"
    thresholds: dict[str, float] = field(default_factory=dict)
    timestamp_seconds: float | None = None
    context: PossessionContext = field(default_factory=PossessionContext)
    audit: PossessionAudit = field(default_factory=PossessionAudit)

    def __post_init__(self) -> None:
        _frame(self.frame_index, "frame_index")
        _schema(self.schema_version)
        object.__setattr__(self, "state", _enum(self.state, PossessionState, "state"))
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be Provenance")
        if not isinstance(self.context, PossessionContext):
            raise ValueError("context must be PossessionContext")
        if not isinstance(self.audit, PossessionAudit):
            raise ValueError("audit must be PossessionAudit")
        object.__setattr__(self, "evidence_kind", _enum(self.evidence_kind, EvidenceKind, "evidence_kind"))
        _identity(self.stable_player_id, self.raw_track_id, self.team_id)
        _score(self.score, "score")
        _score(self.score_margin, "score_margin")
        if type(self.profile) is not str or not self.profile.strip():
            raise ValueError("profile must be a non-empty string")
        _thresholds(self.thresholds)
        if self.timestamp_seconds is not None and (type(self.timestamp_seconds) not in {int, float} or not math.isfinite(self.timestamp_seconds)):
            raise ValueError("timestamp_seconds must be finite when set")
        if not isinstance(self.candidates, (list, tuple)) or any(not isinstance(x, CandidateEvidence) for x in self.candidates):
            raise ValueError("candidates must contain CandidateEvidence")
@dataclass(frozen=True)
class BallTransition:
    transition_id: str
    source_holder: str | None
    destination_holder: str | None
    primitive_types: list[str]
    provenance: Provenance
    schema_version: int = V3_CONTRACT_VERSION
    state: PossessionState = PossessionState.UNKNOWN
    frame_start: int | None = None
    frame_end: int | None = None
    confidence: float | None = None
    profile: str = "default"
    thresholds: dict[str, float] = field(default_factory=dict)
    source_team_id: int | None = None
    destination_team_id: int | None = None
    duration_seconds: float | None = None
    loose_seconds: float = 0.0
    proximity_distance: float | None = None
    score: float | None = None
    score_margin: float | None = None
    release_evidence: bool = False
    reception_evidence: bool = False
    trajectory_evidence: bool = False
    loss_of_control_evidence: bool = False
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    recovery_source_id: str | None = None
    recovery_reason: str | None = None
    recovery_window_seconds: float = 0.0

    def __post_init__(self) -> None:
        if type(self.transition_id) is not str or not self.transition_id.strip():
            raise ValueError("transition_id and primitive_types must be non-empty strings")
        _text_list(self.primitive_types, "primitive_types")
        _schema(self.schema_version)
        object.__setattr__(self, "state", _enum(self.state, PossessionState, "state"))
        _holder(self.source_holder)
        _holder(self.destination_holder)
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be Provenance")
        if self.frame_start is not None: _frame(self.frame_start, "frame_start")
        if self.frame_end is not None: _frame(self.frame_end, "frame_end")
        if self.frame_start is not None and self.frame_end is not None and self.frame_end < self.frame_start:
            raise ValueError("frame_end must not precede frame_start")
        _score(self.confidence, "confidence")
        if type(self.profile) is not str or not self.profile.strip():
            raise ValueError("profile must be a non-empty string")
        _thresholds(self.thresholds)
        _identity(self.source_holder, None, self.source_team_id)
        _identity(self.destination_holder, None, self.destination_team_id)
        for value, name in ((self.duration_seconds, "duration_seconds"), (self.loose_seconds, "loose_seconds"), (self.proximity_distance, "proximity_distance")):
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        _score(self.score, "score")
        _score(self.score_margin, "score_margin")
        for value, name in ((self.timestamp_start, "timestamp_start"), (self.timestamp_end, "timestamp_end")):
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.timestamp_start is not None and self.timestamp_end is not None and self.timestamp_end < self.timestamp_start:
            raise ValueError("timestamp_end must not precede timestamp_start")
        if type(self.loss_of_control_evidence) is not bool:
            raise ValueError("loss_of_control_evidence must be boolean")
        _holder(self.recovery_source_id)
        if self.recovery_reason is not None:
            _required_text(self.recovery_reason, "recovery_reason")
        if (type(self.recovery_window_seconds) not in {int, float}
                or not math.isfinite(self.recovery_window_seconds)
                or self.recovery_window_seconds < 0):
            raise ValueError("recovery_window_seconds must be finite and non-negative")
        if ((self.recovery_source_id is not None or self.state is PossessionState.UNKNOWN_RECOVERY)
                and self.recovery_window_seconds <= 0):
            raise ValueError("recovery transitions require positive recovery time")
        object.__setattr__(self, "primitive_types", list(self.primitive_types))
@dataclass(frozen=True)
class PossessionEvent:
    event_type: EventType
    frame_index: int
    provenance: Provenance
    schema_version: int = V3_CONTRACT_VERSION
    transition_id: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    source_holder: str | None = None
    destination_holder: str | None = None
    source_team_id: int | None = None
    destination_team_id: int | None = None
    score: float | None = None
    score_margin: float | None = None
    timestamp_seconds: float | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None

    def __post_init__(self) -> None:
        _frame(self.frame_index, "frame_index")
        _schema(self.schema_version)
        object.__setattr__(self, "event_type", _enum(self.event_type, EventType, "event_type"))
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be Provenance")
        if self.transition_id is not None and (type(self.transition_id) is not str or not self.transition_id.strip()):
            raise ValueError("transition_id must be a non-empty string when set")
        _text_list(self.reason_codes, "reason_codes")
        _identity(self.source_holder, None, self.source_team_id)
        _identity(self.destination_holder, None, self.destination_team_id)
        _score(self.score, "score")
        _score(self.score_margin, "score_margin")
        for value, name in ((self.timestamp_seconds, "timestamp_seconds"), (self.timestamp_start, "timestamp_start"), (self.timestamp_end, "timestamp_end")):
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.timestamp_start is not None and self.timestamp_end is not None and self.timestamp_end < self.timestamp_start:
            raise ValueError("timestamp_end must not precede timestamp_start")
        if self.frame_start is not None: _frame(self.frame_start, "frame_start")
        if self.frame_end is not None: _frame(self.frame_end, "frame_end")
        if self.frame_start is not None and self.frame_end is not None and self.frame_end < self.frame_start:
            raise ValueError("frame_end must not precede frame_start")
def engine_name(config: PipelineConfig) -> str:
    """Return the validated selector; legacy remains the default."""
    return config.possession_engine if config.possession_v3_enabled else "legacy"


# Work Unit 2 deliberately keeps the input boundary duck-typed: upstream
# sanitizers already own their row contracts and V3 must not rewrite them.
def _point(ball):
    if not isinstance(ball, dict):
        return None
    center = ball.get("center_xy", ball.get("center", ball.get("ball_center")))
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        box = ball.get("xyxy")
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) if (
            isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(value) in {int, float} and math.isfinite(float(value)) for value in box)
        ) else None
    x = ball.get("x", ball.get("center_x", center[0] if isinstance(center, (list, tuple)) and len(center) >= 2 else None))
    y = ball.get("y", ball.get("center_y", center[1] if isinstance(center, (list, tuple)) and len(center) >= 2 else None))
    return (float(x), float(y)) if (
        type(x) in {int, float} and type(y) in {int, float}
        and math.isfinite(float(x)) and math.isfinite(float(y))
    ) else None


def _inside_box(point, box, margin=0.0):
    return (isinstance(box, (list, tuple)) and len(box) == 4 and
            all(isinstance(v, (int, float)) for v in box) and
            box[0] - margin <= point[0] <= box[2] + margin and
            box[1] - margin <= point[1] <= box[3] + margin)


def _identity_key(row):
    stable = row.get("stable_player_id")
    raw = row.get("raw_track_id", row.get("player_track_id"))
    if isinstance(stable, str) and stable.strip():
        return stable, raw if type(raw) is int and raw > 0 else None
    return None, raw if type(raw) is int and raw > 0 else None


def score_candidates(players, ball, *, previous_holder=None, ambiguity_margin=.1):
    """Return deterministic, explainable CandidateEvidence for one ball row."""
    point = _point(ball)
    quality_value = ball.get("confidence", ball.get("score", 0.0)) if isinstance(ball, dict) else 0.0
    quality = float(quality_value) if type(quality_value) in {int, float} and math.isfinite(float(quality_value)) else 0.0
    quality = max(0.0, min(1.0, quality))
    output = []
    for row in players or []:
        stable, raw = _identity_key(row if isinstance(row, dict) else {})
        team = row.get("team_id") if isinstance(row, dict) else None
        box = row.get("xyxy", row.get("bbox", row.get("box"))) if isinstance(row, dict) else None
        reasons = []
        if stable is None and raw is None: reasons.append("missing_authoritative_identity")
        if type(team) is not int or team < 0: reasons.append("invalid_team")
        if point is None: reasons.append("invalid_ball_geometry")
        clean_value = row.get("clean_mask", row.get("mask_clean", False)) if isinstance(row, dict) else False
        clean = type(clean_value) is bool and clean_value
        mask_contains = row.get("mask_contains") if isinstance(row, dict) else None
        mask_hit = False
        mask_available = False
        mask_distance = None
        if callable(mask_contains) and point is not None:
            try:
                mask_result = mask_contains(*point)
            except Exception:
                mask_result = None
            if type(mask_result) is bool:
                mask_hit, mask_available = mask_result, bool(row.get("mask_valid", True))
                distance_fn = row.get("mask_distance")
                if mask_available and callable(distance_fn):
                    try:
                        value = distance_fn(*point)
                        if type(value) in {int, float} and math.isfinite(float(value)):
                            mask_distance = max(0.0, float(value))
                    except Exception:
                        mask_distance = None
        elif type(mask_contains) is bool:
            mask_hit, mask_available = mask_contains, point is not None and clean
        height = (box[3] - box[1]) if isinstance(box, (list, tuple)) and len(box) == 4 else 0
        speed = float(ball.get("speed", 0.0)) if isinstance(ball, dict) and isinstance(ball.get("speed", 0), (int, float)) else 0.0
        margin = max(2.0, height * .15 + speed * .02)
        box_hit = point is not None and _inside_box(point, box, margin)
        if mask_available and mask_hit: geometry, mask = 1.0, 1.0
        elif mask_available and mask_distance is not None:
            # A valid mask remains authoritative just outside its foreground.
            proximity = max(0.0, 1.0 - mask_distance / max(2.0, margin))
            geometry, mask = proximity, proximity
        elif box_hit: geometry, mask = .7, 0.0
        else: geometry, mask = 0.0, 0.0
        if not geometry: reasons.append("outside_dynamic_player_region")
        confidence_value = row.get("confidence", row.get("player_confidence")) if isinstance(row, dict) else None
        confidence_available = confidence_value is not None
        confidence_valid = (not confidence_available or
                            (type(confidence_value) in {int, float} and
                             math.isfinite(float(confidence_value)) and
                             0 <= float(confidence_value) <= 1))
        confidence = float(confidence_value) if confidence_available and confidence_valid else 0.0
        confidence = max(0.0, min(1.0, confidence))
        if not confidence_valid:
            if type(confidence_value) in {int, float} and not math.isfinite(float(confidence_value)):
                reasons.append("non_finite_confidence")
            elif type(confidence_value) in {int, float}:
                reasons.append("out_of_range_confidence")
            else:
                reasons.append("invalid_confidence")
        prior = 1.0 if stable == previous_holder else 0.0
        active = {
            "mask": mask_available,
            "bbox": (not mask_available and point is not None and
                     isinstance(box, (list, tuple)) and len(box) == 4),
            "player_confidence": confidence_available,
            "ball_quality": point is not None,
            "previous_holder": previous_holder is not None,
        }
        weights = {"mask": MASK_SCORE_WEIGHT, "bbox": BBOX_SCORE_WEIGHT,
                   "player_confidence": PLAYER_CONFIDENCE_WEIGHT,
                   "ball_quality": BALL_QUALITY_WEIGHT,
                   "previous_holder": PREVIOUS_HOLDER_WEIGHT}
        denominator = sum(weight for name, weight in weights.items() if active[name])
        raw_contributions = {"mask": mask * MASK_SCORE_WEIGHT, "bbox": geometry * BBOX_SCORE_WEIGHT,
                             "player_confidence": confidence * PLAYER_CONFIDENCE_WEIGHT,
                             "ball_quality": quality * BALL_QUALITY_WEIGHT,
                             "previous_holder": prior * PREVIOUS_HOLDER_WEIGHT}
        contributions = {name: (value / denominator if active[name] and denominator else 0.0)
                         for name, value in raw_contributions.items()}
        score = sum(contributions.values()) if not reasons or reasons == ["outside_dynamic_player_region"] else 0.0
        if mask_available:
            # Equal mask evidence must remain equal; confidence, prior, and ball
            # quality cannot manufacture a winner over an indistinguishable mask.
            score = mask
        if reasons and reasons != ["outside_dynamic_player_region"]: score = 0.0
        fallback = f"raw:{raw}" if stable is None and raw is not None else None
        output.append(CandidateEvidence(stable, raw, team if type(team) is int and team >= 0 else None,
                                        min(1.0, score), contributions, reasons, fallback))
    output = sorted(output, key=lambda c: (-c.score, c.stable_player_id or "", c.raw_track_id or 0))
    if len(output) > 1 and output[0].score - output[1].score < max(0.0, ambiguity_margin):
        output = [replace(candidate, rejected_reasons=candidate.rejected_reasons + ["ambiguous_margin"])
                  for candidate in output]
    return output


def score_candidates_v4(players, ball, *, previous_holder=None, ambiguity_margin=.1,
                        disagreement_score=.25, previous_ball=None, previous_players=None):
    """Fuse mask and box evidence into one conservative spatial vote."""
    point = _point(ball)
    output = []
    for row in players or []:
        row = row if isinstance(row, dict) else {}
        stable, raw = _identity_key(row)
        team = row.get("team_id")
        box = row.get("xyxy", row.get("bbox", row.get("box")))
        reasons = []
        if stable is None and raw is None:
            reasons.append("invalid_identity")
        if type(team) is not int or team < 0:
            reasons.append("invalid_team")
        if point is None:
            reasons.append("invalid_ball_geometry")
        mask_available = row.get("mask_valid") is True
        mask_value = row.get("mask_contains")
        if callable(mask_value) and point is not None:
            try:
                mask_value = mask_value(*point)
            except Exception:
                mask_value = None
        if mask_available and type(mask_value) is not bool:
            mask_available = False
            reasons.append("invalid_mask")
        mask_score = 1.0 if mask_available and mask_value else 0.0 if mask_available else None
        valid_box = (isinstance(box, (list, tuple)) and len(box) == 4 and
                     all(type(value) in {int, float} and math.isfinite(float(value)) for value in box) and
                     box[2] >= box[0] and box[3] >= box[1])
        if not valid_box:
            reasons.append("invalid_bbox")
        bbox_score = 1.0 if valid_box and point is not None and _inside_box(point, box) else 0.0 if valid_box else None
        agreement = None
        if mask_score is None:
            spatial, source = 0.0, "missing-mask"
            reasons.append("missing_mask")
        elif bbox_score is None:
            spatial, source = 0.0, "missing-bbox"
            reasons.append("missing_bbox")
        elif mask_score == bbox_score:
            spatial, agreement, source = mask_score, 1.0, "mask+bbox"
        else:
            spatial, agreement, source = max(0.0, min(1.0, float(disagreement_score))), 0.0, "mask+bbox-disagreement"
            reasons.append("mask_bbox_disagreement")
        if spatial == 0.0 and not reasons:
            reasons.append("outside_spatial_region")
        quality_value = ball.get("confidence", ball.get("score", 0.0)) if isinstance(ball, dict) else 0.0
        quality_valid = (type(quality_value) in {int, float} and math.isfinite(float(quality_value))
                         and 0 <= float(quality_value) <= 1)
        quality = float(quality_value) if quality_valid else 0.0
        quality = max(0.0, min(1.0, quality))
        if isinstance(ball, dict) and ("confidence" in ball or "score" in ball) and not quality_valid:
            reasons.append("invalid_ball_confidence")
        confidence_value = row.get("confidence", row.get("player_confidence"))
        if confidence_value is not None and (type(confidence_value) not in {int, float}
                                              or not math.isfinite(float(confidence_value))
                                              or not 0 <= float(confidence_value) <= 1):
            reasons.append("invalid_confidence")
        distance = None
        if valid_box and point is not None:
            px, py = (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)
            size = max(1.0, float(box[2] - box[0]), float(box[3] - box[1]))
            distance = min(1.0, max(0.0, math.hypot(point[0] - px, point[1] - py) / size))
        proximity = 0.0 if distance is None else max(0.0, 1.0 - min(1.0, distance))
        def bounded(name, default=0.0):
            value = row.get(name, default)
            return float(value) if type(value) in {int, float} and math.isfinite(float(value)) and 0 <= value <= 1 else default
        approach = bounded("approach_score", bounded("relative_motion"))
        previous_row = next((item for item in (previous_players or [])
                             if isinstance(item, dict) and
                             ((_identity_key(item)[0] == stable and stable is not None) or
                              (_identity_key(item)[1] == raw and raw is not None))), None)
        previous_point, previous_player_point = _point(previous_ball), None
        previous_box = (previous_row.get("xyxy", previous_row.get("bbox", previous_row.get("box")))
                        if previous_row else None)
        if isinstance(previous_box, (list, tuple)) and len(previous_box) == 4 and all(
                type(value) in {int, float} and math.isfinite(float(value)) for value in previous_box):
            previous_player_point = ((float(previous_box[0] + previous_box[2]) / 2),
                                     (float(previous_box[1] + previous_box[3]) / 2))
        if previous_point is not None and previous_player_point is not None and valid_box and point is not None:
            current_player_point = ((float(box[0] + box[2]) / 2), (float(box[1] + box[3]) / 2))
            current_distance = math.hypot(point[0] - current_player_point[0], point[1] - current_player_point[1])
            previous_distance = math.hypot(previous_point[0] - previous_player_point[0],
                                           previous_point[1] - previous_player_point[1])
            size = max(1.0, float(box[2] - box[0]), float(box[3] - box[1]))
            approach = max(0.0, min(1.0, (previous_distance - current_distance) / size))
        continuity = bounded("continuity", 1.0 if stable == previous_holder else 0.0)
        control = max(0.0, min(1.0, .55 * spatial + .25 * proximity + .1 * quality + .1 * continuity))
        release = max(0.0, min(1.0, 1.0 - control))
        acquisition_base = .35 * spatial + .2 * proximity + .1 * quality + .1 * continuity
        acquisition = max(0.0, min(1.0, .5 * acquisition_base + .5 * approach))
        contributions = {"spatial_score": spatial, "spatial_fusion": spatial, "normalized_distance": proximity,
                         "relative_motion": approach, "ball_quality": quality, "continuity": continuity}
        if distance is None:
            reasons.append("invalid_normalized_distance")
        invalid_evidence = bool({"invalid_identity", "invalid_team", "invalid_ball_geometry",
                                 "invalid_mask", "invalid_bbox", "invalid_confidence",
                                 "invalid_ball_confidence", "invalid_normalized_distance"}.intersection(reasons))
        if invalid_evidence:
            spatial, source, approach = 0.0, "invalid-evidence", 0.0
            control = release = acquisition = 0.0
            contributions["spatial_score"] = contributions["spatial_fusion"] = 0.0
            contributions["relative_motion"] = 0.0
            reasons.append("invalid_evidence")
        fallback = f"raw:{raw}" if stable is None and raw is not None else None
        output.append(V4CandidateEvidence(
            stable, raw, team if type(team) is int and team >= 0 else None,
            spatial, contributions, reasons, fallback,
            spatial, mask_score, bbox_score, agreement, source, distance, approach, quality,
            continuity, control, release, acquisition,
        ))
    output.sort(key=lambda c: (-c.score, c.stable_player_id or "", c.raw_track_id or 0))
    if len(output) > 1 and output[0].score - output[1].score < max(0.0, ambiguity_margin):
        output = [replace(candidate, rejected_reasons=candidate.rejected_reasons + ["ambiguous_margin"])
                  for candidate in output]
    return output


class PossessionResolver:
    """Small deterministic temporal resolver; no ball/event semantics live here."""
    def __init__(self, *, real_fps, grace_seconds=.6, recovery_window_seconds=.8, hysteresis_margin=.1,
                 ambiguity_margin=None, persistence_seconds=0.0, keep_threshold=MIN_CANDIDATE_SCORE, profile="default",
                 v4=False, holder_control_score=.55, holder_release_score=.35,
                 challenger_acquisition_score=.6, release_persistence_seconds=.2,
                 acquisition_persistence_seconds=.2):
        if not isinstance(real_fps, (int, float)) or real_fps <= 0: raise ValueError("real_fps must be positive")
        self.fps = float(real_fps); self.grace = min(.6, max(0.0, grace_seconds))
        if type(recovery_window_seconds) not in {int, float} or not math.isfinite(float(recovery_window_seconds)) or recovery_window_seconds < 0:
            raise ValueError("recovery_window_seconds must be finite and non-negative")
        self.recovery_window = float(recovery_window_seconds)
        self.hysteresis = max(0.0, hysteresis_margin)
        self.ambiguity_margin = self.hysteresis if ambiguity_margin is None else max(0.0, ambiguity_margin)
        self.persistence = max(0.0, persistence_seconds)
        if (type(keep_threshold) not in {int, float} or not math.isfinite(float(keep_threshold))
                or not 0 <= float(keep_threshold) <= 1):
            raise ValueError("keep_threshold must be finite and between 0 and 1")
        self.keep_threshold = float(keep_threshold)
        self.profile = profile; self.holder = None; self.last_frame = None; self.last_time = None
        self.v4 = bool(v4)
        for value, name in ((holder_control_score, "holder_control_score"),
                            (holder_release_score, "holder_release_score"),
                            (challenger_acquisition_score, "challenger_acquisition_score")):
            if type(value) not in {int, float} or not math.isfinite(float(value)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        self.holder_control_score = float(holder_control_score)
        self.holder_release_score = float(holder_release_score)
        self.challenger_acquisition_score = float(challenger_acquisition_score)
        self.release_persistence = max(0.0, float(release_persistence_seconds))
        self.acquisition_persistence = max(0.0, float(acquisition_persistence_seconds))
        self.release_pending = None
        self.pending = None; self.gap_seconds = 0.0
        self.holder_stable = self.holder_raw = self.holder_team = self.holder_fallback = None
        self.last_observed_frame = self.last_observed_score = None
        self.last_observed_timestamp = None
        self.history = []; self.last_timestamp = None
        self._audit_for_last_transition = PossessionAudit()
        self.recovery_source_id = self.recovery_source_team = self.recovery_reason = None
        self.recovery_start_frame = self.recovery_start_timestamp = None

    def _recovery_elapsed(self, frame, timestamp):
        if self.recovery_source_id is None:
            return 0.0
        if timestamp is not None and self.recovery_start_timestamp is not None:
            return max(0.0, float(timestamp) - float(self.recovery_start_timestamp))
        if self.recovery_start_frame is not None:
            return max(0.0, (frame - self.recovery_start_frame) / self.fps)
        return 0.0

    def _observation(self, frame, state, candidate=None, candidates=(), reason=(), observed=None,
                     challenger=None, clear_reason=None, switch_confirmed=False):
        effective = self.holder_stable or self.holder_fallback or (f"raw:{self.holder_raw}" if self.holder_raw else None)
        direct = observed or candidate
        stable = self.holder_stable if state not in {PossessionState.OBSERVED} else (direct.stable_player_id if direct else None)
        raw = self.holder_raw if state not in {PossessionState.OBSERVED} else (direct.raw_track_id if direct else None)
        team = self.holder_team if state not in {PossessionState.OBSERVED} else (direct.team_id if direct else None)
        if state is PossessionState.LOST_NO_DETECTION:
            stable = raw = team = None
        challenger_id = self._key(challenger) if challenger else None
        pending = self.pending if challenger_id is not None and self.pending and self.pending[0] == challenger_id else None
        observed_id = self._key(direct) if direct else None
        prov = Provenance(frame_indices=[frame], reason_codes=list(reason))
        quality = (EvidenceKind.OBSERVED if state is PossessionState.OBSERVED else
                   EvidenceKind.INFERRED if state is PossessionState.INFERRED else
                   EvidenceKind.CONTESTED if state is PossessionState.CONTESTED else EvidenceKind.UNKNOWN)
        evidence = direct or (candidates[0] if candidates else None)
        v4_evidence = evidence if isinstance(evidence, V4CandidateEvidence) else None
        context = PossessionContext(effective_holder_id=None if state is PossessionState.LOST_NO_DETECTION else effective,
                                    effective_holder_team=None if state is PossessionState.LOST_NO_DETECTION else self.holder_team,
                                    observed_candidate_id=observed_id, challenger_id=challenger_id,
                                    challenger_start_frame=pending[2] if pending else None,
                                     challenger_elapsed_seconds=pending[1] if pending else 0.0,
                                    grace_start_frame=self.last_frame - int(self.gap_seconds * self.fps) if self.gap_seconds else None,
                                    grace_remaining_seconds=max(0.0, self.grace - self.gap_seconds),
                                     last_observed_holder_frame=self.last_observed_frame,
                                     last_observed_holder_score=self.last_observed_score,
                                     recovery_source_id=self.recovery_source_id,
                                     recovery_source_team=self.recovery_source_team,
                                     recovery_reason=self.recovery_reason,
                                      recovery_remaining_seconds=max(0.0, self.recovery_window - self._recovery_elapsed(frame, self.last_timestamp)),
                                      control_evidence=(v4_evidence.holder_control_score if v4_evidence else None),
                                      release_evidence=(v4_evidence.holder_release_score if v4_evidence else None),
                                      acquisition_evidence=(v4_evidence.challenger_acquisition_score if v4_evidence else None),
                                      evidence_window_start=(pending[2] if pending else None),
                                      evidence_window_end=frame)
        authority = "stable" if (direct and direct.stable_player_id) else "raw" if direct else "unknown"
        gate_reasons = list(reason)
        audit = PossessionAudit(last_holder_id=(effective if state is PossessionState.LOST_NO_DETECTION else self.recovery_source_id),
                                 last_holder_team=(self.holder_team if state is PossessionState.LOST_NO_DETECTION else self.recovery_source_team),
                                 clear_reason=clear_reason or self._audit_for_last_transition.clear_reason,
                                 switch_confirmed=switch_confirmed or self._audit_for_last_transition.new_holder_id is not None,
                                observation_quality=quality.value, observation_source="resolver",
                                 authority=authority, grace_eligible=bool(effective and self.gap_seconds < self.grace),
                                  grace_elapsed_seconds=self.gap_seconds,
                                  recovery_source_id=self.recovery_source_id,
                                  recovery_source_team=self.recovery_source_team,
                                  recovery_reason=self.recovery_reason,
                                  recovery_window_seconds=self.recovery_window,
                                  v4_schema_version=1 if self.v4 else None,
                                  feature_flag="v4" if self.v4 else None,
                                  control_score=v4_evidence.holder_control_score if v4_evidence else None,
                                  release_score=v4_evidence.holder_release_score if v4_evidence else None,
                                  acquisition_score=v4_evidence.challenger_acquisition_score if v4_evidence else None,
                                  gate_decision="allow" if switch_confirmed else ("hold" if self.v4 else None),
                                  gate_reason_codes=gate_reasons,
                                  spatial_contributions=(dict(v4_evidence.contributions) if v4_evidence else {}),
                                  evidence_sources=([v4_evidence.source] if v4_evidence else []))
        audit = replace(audit, **{name: getattr(self._audit_for_last_transition, name)
                                  for name in ("previous_holder_id", "previous_holder_team", "new_holder_id",
                                               "new_holder_team", "previous_holder_score", "new_holder_score",
                                               "challenger_start_frame", "challenger_confirmation_frame",
                                               "frame_start", "frame_end", "timestamp_start", "timestamp_end",
                                               "provenance")})
        obs = PossessionObservation(frame, state, quality, stable_player_id=stable, raw_track_id=raw,
                                     team_id=team, score=(candidate or direct).score if (candidate or direct) else None,
                                    score_margin=(candidates[0].score - candidates[1].score if len(candidates) > 1 else None),
                                    candidates=list(candidates), provenance=prov, profile=self.profile,
                                    timestamp_seconds=self.last_timestamp,
                                     thresholds={"hysteresis_margin": self.hysteresis, "grace_seconds": self.grace,
                                                 "keep_threshold": self.keep_threshold},
                                     context=context, audit=audit)
        self.history.append(obs); return obs

    def _key(self, candidate):
        return candidate.stable_player_id or candidate.fallback_identity or (f"raw:{candidate.raw_track_id}" if candidate.raw_track_id else None)

    def _challenger(self, candidates):
        return next((candidate for candidate in candidates if self._key(candidate) != self.holder), None)

    def _v4_ready(self, candidate, candidates=()):
        prior = next((item for item in candidates if self._key(item) == self.holder), None)
        release_ready = (isinstance(prior, V4CandidateEvidence)
                         and prior.holder_release_score >= self.holder_release_score)
        return (self.v4 and isinstance(candidate, V4CandidateEvidence) and release_ready
                and candidate.challenger_acquisition_score >= self.challenger_acquisition_score
                and candidate.relative_motion > 0)

    def _v4_gate_reasons(self, candidate, candidates=()):
        """Explain why a V4 challenger cannot replace the effective holder."""
        prior = next((item for item in candidates if self._key(item) == self.holder), None)
        reasons = []
        if not isinstance(prior, V4CandidateEvidence) or prior.holder_release_score < self.holder_release_score:
            reasons.append("release_evidence_required")
        if (not isinstance(candidate, V4CandidateEvidence)
                or candidate.challenger_acquisition_score < self.challenger_acquisition_score
                or candidate.relative_motion <= 0):
            reasons.append("acquisition_evidence_required")
        return tuple(reasons)

    def _v4_invalid_candidate(self, candidate):
        if self.v4 and isinstance(candidate, CandidateEvidence) and not isinstance(candidate, V4CandidateEvidence):
            return True
        return (self.v4 and isinstance(candidate, V4CandidateEvidence)
                and (candidate.team_id is None
                     or self._key(candidate) is None
                     or bool({"invalid_bbox", "invalid_identity", "invalid_team",
                              "invalid_ball_geometry", "invalid_mask", "invalid_confidence",
                              "invalid_ball_confidence", "invalid_normalized_distance",
                              "invalid_evidence"}.intersection(
                                  candidate.rejected_reasons))))

    def resolve(self, frame_index, candidates, *, missing=False, invalid=False, ball_lost=False,
                authoritative_loss=False, duplicate=False, timestamp=None):
        self._audit_for_last_transition = PossessionAudit()
        candidates = sorted(candidates or [], key=lambda c: (-c.score, c.stable_player_id or "", c.raw_track_id or 0))
        timestamp_valid = (timestamp is None or
                           (type(timestamp) in {int, float} and math.isfinite(float(timestamp))
                            and float(timestamp) >= 0))
        timestamp_reversed = (timestamp_valid and timestamp is not None and self.last_time is not None
                              and timestamp < self.last_time)
        timestamp_invalid = not timestamp_valid or timestamp_reversed
        if self.last_frame is None:
            elapsed = 0.0
        elif timestamp_invalid:
            elapsed = 0.0
        elif timestamp is not None and self.last_time is not None:
            elapsed = timestamp - self.last_time
        else:
            elapsed = (frame_index - self.last_frame) / self.fps
        self.last_frame = frame_index
        if not timestamp_invalid:
            if timestamp is not None:
                self.last_time = timestamp
            self.last_timestamp = timestamp
        if self.recovery_source_id is not None and self._recovery_elapsed(frame_index, timestamp) >= self.recovery_window:
            self.recovery_source_id = self.recovery_source_team = self.recovery_reason = None
            self.recovery_start_frame = self.recovery_start_timestamp = None
        invalid_input = invalid or authoritative_loss or duplicate or missing or timestamp_invalid
        if invalid_input:
            last = self.holder_stable or self.holder_fallback or (f"raw:{self.holder_raw}" if self.holder_raw else None)
            team = self.holder_team
            if self.v4 and self.holder:
                return self._observation(frame_index, PossessionState.UNKNOWN, candidates=candidates,
                                         reason=("invalid_timestamp" if not timestamp_valid else
                                                 "reverse_timestamp" if timestamp_reversed else
                                                 "unknown_evidence",))
            previous = self.holder
            self.clear_effective_holder(ClearReason.EXPLICIT_RESET, frame_index=frame_index, timestamp=timestamp,
                                         frame_start=self.last_observed_frame,
                                         timestamp_start=self.last_observed_timestamp)
            reason = ("invalid_timestamp" if not timestamp_valid else "reverse_timestamp" if timestamp_reversed
                      else "missing_authoritative_frame" if missing else "invalid_evidence")
            observation = self._observation(frame_index, PossessionState.UNKNOWN, candidates=candidates,
                                            reason=(reason,))
            observation = replace(observation, audit=replace(observation.audit, last_holder_id=last,
                                                              last_holder_team=team))
            self.history[-1] = observation
            return observation
        if ball_lost or not candidates:
            if self.holder:
                self.gap_seconds += elapsed
                if self.gap_seconds < self.grace:
                    self.pending = None
                    return self._observation(frame_index, PossessionState.INFERRED,
                                             reason=("ball_lost" if ball_lost else "unknown",))
                return self._lost_no_detection(frame_index, candidates, "grace_expired")
            return self._observation(frame_index, PossessionState.NO_POSSESSION, candidates=candidates,
                                     reason=("no_detection" if ball_lost else "evidence_loss",))
        if candidates:
            top = candidates[0]
            if self.v4 and self._v4_invalid_candidate(top):
                self.pending = None
                if self.holder:
                    self.gap_seconds += elapsed
                    if self.gap_seconds >= self.grace:
                        return self._lost_no_detection(frame_index, candidates, "grace_expired")
                return self._observation(frame_index, PossessionState.CONTESTED,
                                         candidates=candidates, observed=top,
                                         challenger=self._challenger(candidates) if self.v4 else None,
                                         reason=("legacy_candidate_without_v4_evidence"
                                                  if not isinstance(top, V4CandidateEvidence)
                                                  else "invalid_candidate_evidence",))
            if {"ambiguous_margin", "mask_bbox_disagreement"}.intersection(top.rejected_reasons):
                self.pending = None
                if self.holder:
                    self.gap_seconds += elapsed
                    if self.gap_seconds >= self.grace:
                        return self._lost_no_detection(frame_index, candidates, "grace_expired")
                return self._observation(frame_index, PossessionState.CONTESTED, candidates=candidates,
                                         observed=top, challenger=self._challenger(candidates) if self.v4 else None,
                                         reason=("ambiguous_spatial_evidence",))
            if self._key(top) == self.holder and top.score >= self.keep_threshold:
                self.release_pending = None
                if self.v4 and isinstance(top, V4CandidateEvidence) and top.holder_control_score < self.holder_control_score:
                    self.release_pending = (frame_index, timestamp)
                self.gap_seconds = 0.0
                self.pending = None
                self.last_observed_frame, self.last_observed_score = frame_index, top.score
                self.last_observed_timestamp = timestamp
                return self._observation(frame_index, PossessionState.OBSERVED, top, candidates)
            top = candidates[0]
            if top.score <= 0:
                if self._v4_invalid_candidate(top) and self.holder:
                    self.gap_seconds += elapsed
                    if self.gap_seconds >= self.grace:
                        return self._lost_no_detection(frame_index, candidates, "grace_expired")
                    return self._observation(frame_index, PossessionState.CONTESTED,
                                             candidates=candidates, observed=top,
                                             reason=("invalid_candidate_evidence",))
                if self.v4 and self._key(top) == self.holder and isinstance(top, V4CandidateEvidence):
                    started = self.release_pending or (frame_index, timestamp)
                    release_elapsed = ((timestamp - started[1]) if timestamp is not None and started[1] is not None
                                       else (frame_index - started[0]) / self.fps)
                    if release_elapsed < self.release_persistence:
                        self.release_pending = started
                        return self._observation(frame_index, PossessionState.TRANSITIONING,
                                                 candidates=candidates, observed=top,
                                                 reason=("release_persistence",))
                self.clear_effective_holder(ClearReason.LOOSE_BALL_CONFIRMED, frame_index=frame_index,
                                            timestamp=timestamp, frame_start=self.last_observed_frame,
                                            timestamp_start=self.last_observed_timestamp)
                return self._observation(frame_index, PossessionState.NO_POSSESSION, candidates=candidates, reason=("no_valid_candidate",))
            if len(candidates) > 1 and candidates[0].score - candidates[1].score < self.ambiguity_margin:
                if self.holder:
                    self.gap_seconds += elapsed
                    if self.gap_seconds >= self.grace:
                        return self._lost_no_detection(frame_index, candidates, "grace_expired")
                    return self._observation(frame_index, PossessionState.CONTESTED, candidates=candidates,
                                             observed=top, challenger=self._challenger(candidates) if self.v4 else None,
                                             reason=("insufficient_margin",))
                return self._observation(frame_index, PossessionState.CONTESTED, candidates=candidates,
                                         observed=top, challenger=self._challenger(candidates) if self.v4 else None,
                                         reason=("insufficient_margin",))
            if top.score < self.keep_threshold and self.holder is None:
                return self._observation(frame_index, PossessionState.CANDIDATE, top, candidates, ("weak_candidate",))
            if self.holder is None:
                self.confirm_switch(top, frame_index=frame_index, timestamp=timestamp)
                self.last_observed_frame, self.last_observed_score = frame_index, top.score
                self.last_observed_timestamp = timestamp
                return self._observation(frame_index, PossessionState.OBSERVED, top, candidates)
            if self.holder_stable is not None and top.stable_player_id is None:
                self.pending = None
                self.gap_seconds += elapsed
                if self.gap_seconds >= self.grace:
                    return self._lost_no_detection(frame_index, candidates, "grace_expired")
                return self._observation(frame_index, PossessionState.CONTESTED, candidates=candidates,
                                         observed=top, reason=("raw_fallback_cannot_replace_stable",))
            if top.score < self.keep_threshold:
                self.pending = None
                self.gap_seconds += elapsed
                if self.gap_seconds >= self.grace:
                    return self._lost_no_detection(frame_index, candidates, "grace_expired")
                return self._observation(frame_index, PossessionState.TRANSITIONING,
                                         candidates=candidates, observed=top,
                                         reason=("below_keep_threshold",))
            if top.score - next((c.score for c in candidates if self._key(c) == self.holder), 0.0) >= self.hysteresis:
                if self.v4 and not self._v4_ready(top, candidates):
                    self.pending = None
                    return self._observation(frame_index, PossessionState.TRANSITIONING,
                                             candidates=candidates, observed=top, challenger=top,
                                             reason=self._v4_gate_reasons(top, candidates))
                key = self._key(top)
                required_persistence = self.acquisition_persistence if self.v4 else self.persistence
                accumulated = self.pending[1] + elapsed if self.pending and self.pending[0] == key else elapsed
                if self.pending and self.pending[0] == key and accumulated >= required_persistence:
                    self.confirm_switch(top, frame_index=frame_index, timestamp=timestamp,
                                        candidates=candidates, persistence_elapsed=accumulated)
                    self.gap_seconds = 0.0
                    self.last_observed_frame, self.last_observed_score = frame_index, top.score
                    self.last_observed_timestamp = timestamp
                    return self._observation(frame_index, PossessionState.OBSERVED, top, candidates,
                                             ("stable_switch",), switch_confirmed=True)
                start_frame = self.pending[2] if self.pending and self.pending[0] == key else frame_index
                start_timestamp = self.pending[3] if self.pending and self.pending[0] == key else timestamp
                self.pending = (key, accumulated, start_frame, start_timestamp)
                self.gap_seconds += elapsed
                if self.gap_seconds >= self.grace:
                    return self._lost_no_detection(frame_index, candidates, "grace_expired")
                return self._observation(frame_index, PossessionState.TRANSITIONING, candidates=candidates,
                                         observed=top, challenger=top, reason=("switch_persistence",))
            self.pending = None
            self.gap_seconds += elapsed
            if self.gap_seconds >= self.grace:
                return self._lost_no_detection(frame_index, candidates, "grace_expired")
            return self._observation(frame_index, PossessionState.UNKNOWN, candidates=candidates,
                                     observed=top, reason=("challenger_rejected",))

    def confirm_switch(self, candidate, *, frame_index=None, timestamp=None, candidates=(),
                       persistence_elapsed=None):
        """The sole mutation path for an effective holder change."""
        if self._v4_invalid_candidate(candidate):
            return self.holder
        if not isinstance(candidate, CandidateEvidence) or self._key(candidate) is None:
            raise ValueError("confirmed switch requires an identified candidate")
        if self.holder_stable is not None and candidate.stable_player_id is None:
            raise ValueError("confirmed switch requires stable identity")
        if candidate.score < self.keep_threshold:
            raise ValueError("confirmed switch requires keep_threshold score")
        if self.v4 and self.holder is not None:
            gate_reasons = self._v4_gate_reasons(candidate, candidates)
            if gate_reasons:
                raise ValueError("V4 switch gate rejected: " + ", ".join(gate_reasons))
            if persistence_elapsed is None or persistence_elapsed < self.acquisition_persistence:
                raise ValueError("V4 switch gate requires temporal persistence")
        previous = self.holder
        previous_team = self.holder_team
        start_frame = self.pending[2] if self.pending and self.pending[0] == self._key(candidate) else frame_index
        start_time = self.pending[3] if self.pending and self.pending[0] == self._key(candidate) else timestamp
        self.holder = self._key(candidate)
        self.holder_stable, self.holder_raw = candidate.stable_player_id, candidate.raw_track_id
        self.holder_team, self.holder_fallback = candidate.team_id, candidate.fallback_identity
        self.pending = None
        self._audit_for_last_transition = PossessionAudit(
            previous_holder_id=previous, previous_holder_team=previous_team,
            new_holder_id=self.holder, new_holder_team=self.holder_team,
            previous_holder_score=next((item.score for item in candidates
                                        if self._key(item) == previous), self.last_observed_score),
            new_holder_score=candidate.score,
            challenger_start_frame=start_frame,
            challenger_confirmation_frame=frame_index,
            frame_start=start_frame, frame_end=frame_index,
            timestamp_start=start_time, timestamp_end=timestamp,
            provenance=Provenance(frame_indices=[x for x in (start_frame, frame_index) if x is not None],
                                  reason_codes=["confirmed_switch"]),
        )
        return self.holder

    def clear_effective_holder(self, reason, *, frame_index=None, timestamp=None,
                               frame_start=None, timestamp_start=None, recovery_reason=None,
                               recovery_eligible=False):
        """Remove an effective holder; recovery is seeded only by _lost_no_detection."""
        reason = _enum(reason, ClearReason, "clear reason")
        if reason not in {ClearReason.GRACE_EXPIRED, ClearReason.LOOSE_BALL_CONFIRMED,
                          ClearReason.OUT_OF_BOUNDS, ClearReason.END_OF_SEGMENT,
                          ClearReason.EXPLICIT_RESET}:
            raise ValueError(f"unsupported clear reason: {reason.value}")
        previous, team = self.holder, self.holder_team
        frame_start = self.last_frame if frame_start is None else frame_start
        timestamp_start = self.last_timestamp if timestamp_start is None else timestamp_start
        valid_end = (timestamp if type(timestamp) in {int, float} and math.isfinite(float(timestamp))
                     and (timestamp_start is None or timestamp >= timestamp_start) else None)
        self._audit_for_last_transition = PossessionAudit(
            previous_holder_id=previous, previous_holder_team=team,
            frame_start=frame_start, frame_end=frame_index,
            timestamp_start=timestamp_start, timestamp_end=valid_end,
            clear_reason=reason,
            provenance=Provenance(frame_indices=[x for x in (frame_start, frame_index) if x is not None],
                                   reason_codes=[reason.value]),
        )
        # Compatibility keywords are intentionally ignored: public clear calls
        # cannot manufacture recovery provenance or eligibility.
        del recovery_reason, recovery_eligible, previous, team
        self.recovery_source_id = self.recovery_source_team = self.recovery_reason = None
        self.recovery_start_frame = self.recovery_start_timestamp = None
        self.holder = None; self.pending = None; self.gap_seconds = 0.0
        self.holder_stable = self.holder_raw = self.holder_team = self.holder_fallback = None

    def _lost_no_detection(self, frame, candidates, reason):
        prior, team = self.holder, self.holder_team
        elapsed = self.gap_seconds
        self.clear_effective_holder(ClearReason.GRACE_EXPIRED, frame_index=frame,
                                     timestamp=self.last_timestamp,
                                     frame_start=self.last_observed_frame,
                                     timestamp_start=self.last_observed_timestamp)
        if prior is not None and self.recovery_window > 0:
            self.recovery_source_id = prior
            self.recovery_source_team = team
            self.recovery_reason = "lost_no_detection"
            self.recovery_start_frame = frame
            self.recovery_start_timestamp = self.last_timestamp
        observation = self._observation(frame, PossessionState.LOST_NO_DETECTION,
                                        candidates=candidates, reason=(reason,),
                                        clear_reason=ClearReason.LOST_NO_DETECTION)
        observation = replace(observation, audit=replace(observation.audit, last_holder_id=prior,
                                                          last_holder_team=team,
                                                          grace_elapsed_seconds=elapsed))
        self.history[-1] = observation
        return observation

    def retroactive_confirm(self, start_frame, confirmation_frame):
        if type(start_frame) is not int or type(confirmation_frame) is not int or start_frame >= confirmation_frame:
            raise ValueError("retroactive confirmation bounds must be ordered frame integers")
        by_frame = {obs.frame_index: (i, obs) for i, obs in enumerate(self.history)}
        if start_frame not in by_frame or confirmation_frame not in by_frame:
            raise ValueError("retroactive confirmation endpoints must be observed")
        eligible = [by_frame.get(frame) for frame in range(start_frame, confirmation_frame)]
        if any(item is None or item[1].state is not PossessionState.INFERRED for item in eligible):
            raise ValueError("retroactive interval must be contiguous inferred evidence")
        confirmation_index, confirmation = by_frame[confirmation_frame]
        if confirmation.state is not PossessionState.OBSERVED or confirmation.stable_player_id is None:
            raise ValueError("confirmation must be a unique observed stable holder")
        if len(confirmation.candidates) > 1 and (confirmation.score_margin is None or confirmation.score_margin < self.hysteresis):
            raise ValueError("confirmation must be unique")
        key = confirmation.stable_player_id or confirmation.raw_track_id
        if any((obs.stable_player_id or obs.raw_track_id) != key for _, obs in eligible):
            raise ValueError("inferred interval identity is inconsistent")
        revised = []
        for i, obs in eligible:
            if start_frame <= obs.frame_index < confirmation_frame and obs.state is PossessionState.INFERRED:
                revised.append(replace(obs, provenance=replace(obs.provenance,
                    reason_codes=obs.provenance.reason_codes + ["retroactive_confirmation"],
                    frame_indices=obs.provenance.frame_indices + [confirmation_frame])))
                self.history[i] = revised[-1]
        return revised


def _ball_center(row):
    center = row.get("center_xy", row.get("center")) if isinstance(row, dict) else None
    return tuple(float(v) for v in center) if isinstance(center, (list, tuple)) and len(center) == 2 and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in center) else None


def _ball_source(row):
    state = row.get("track_state") if isinstance(row, dict) else None
    return state if state in {"detected", "fallback", "lost"} else None


def _bounded_number(value):
    return type(value) in {int, float} and math.isfinite(float(value)) and 0 <= float(value) <= 1


def _provenance(row, frame):
    records = row.get("source_records")
    frames = row.get("source_frame_indices")
    if records is None:
        records = [f"ball_tracks.jsonl#frame={frame}"]
    elif not isinstance(records, (list, tuple)) or not records or any(type(value) is not str or not value.strip() for value in records):
        return None
    if frames is None:
        frames = [frame]
    elif not isinstance(frames, (list, tuple)) or not frames or any(type(value) is not int or value < 0 for value in frames):
        return None
    return list(records), list(frames)


def normalize_ball_rows(rows, *, real_fps, max_gap_frames=None, config=None):
    """Copy closed TrackNet rows and fill only bounded, valid coordinate gaps."""
    if max_gap_frames is None and config is not None:
        max_gap_frames = config.possession_v3_ball_max_gap
    if type(real_fps) not in {int, float} or not math.isfinite(float(real_fps)) or real_fps <= 0 or type(max_gap_frames) is not int or max_gap_frames < 0:
        raise ValueError("real_fps must be positive and max_gap_frames must be non-negative")
    raw_rows = [dict(row) for row in rows if isinstance(row, dict) and type(row.get("frame_index")) is int and row["frame_index"] >= 0]
    previous_time = None
    for previous, current in zip(raw_rows, raw_rows[1:]):
        if current["frame_index"] < previous["frame_index"]:
            raise ValueError("ball frame indices must be monotonic")
    for row in raw_rows:
        timestamp = row.get("timestamp_seconds")
        if timestamp is not None and (type(timestamp) not in {int, float} or not math.isfinite(float(timestamp))):
            raise ValueError("timestamp_seconds must be finite when present")
        if timestamp is not None and previous_time is not None and timestamp < previous_time:
            raise ValueError("ball timestamps must be monotonic")
        if timestamp is not None:
            previous_time = timestamp
    ordered = sorted(raw_rows, key=lambda row: row["frame_index"])
    if len({row["frame_index"] for row in ordered}) != len(ordered):
        raise ValueError("duplicate ball frame indices")
    output = []
    for row in ordered:
        source = _ball_source(row)
        point = _ball_center(row)
        confidence = row.get("confidence")
        valid_quality = source == "lost" or _bounded_number(confidence)
        provenance = _provenance(row, row["frame_index"])
        if source and point is not None and valid_quality and provenance is not None:
            copy = dict(row)
            records, frames = provenance
            copy.update(ball_source=source, quality=float(confidence) if _bounded_number(confidence) else 0.0,
                        source_frame_indices=frames, source_records=records)
            output.append(copy)
        else:
            output.append(dict(row, ball_source="lost", quality=0.0))
    filled = []
    for left, right in zip(output, output[1:]):
        filled.append(left)
        gap = right["frame_index"] - left["frame_index"] - 1
        if gap < 1 or gap > max_gap_frames or left["ball_source"] not in {"detected", "fallback"} or right["ball_source"] not in {"detected", "fallback"}:
            continue
        lt = left.get("timestamp_seconds")
        rt = right.get("timestamp_seconds")
        duration = (float(rt) - float(lt)) if isinstance(lt, (int, float)) and isinstance(rt, (int, float)) else (gap + 1) / float(real_fps)
        if duration > (max_gap_frames + 1) / float(real_fps) + 1e-9:
            continue
        a, b = _ball_center(left), _ball_center(right)
        for frame in range(1, gap + 1):
            ratio = frame / (gap + 1)
            point = [a[i] + (b[i] - a[i]) * ratio for i in range(2)]
            records = left["source_records"] + right["source_records"]
            frames = left["source_frame_indices"] + right["source_frame_indices"]
            filled.append({"frame_index": left["frame_index"] + frame,
                           "timestamp_seconds": (float(lt) + duration * ratio) if isinstance(lt, (int, float)) else None,
                           "track_state": "inferred", "ball_source": "interpolated", "quality": min(left["quality"], right["quality"]) * .5,
                           "center_xy": point, "xyxy": None, "track_id": None,
                           "source_frame_indices": list(dict.fromkeys(frames)),
                           "source_records": list(dict.fromkeys(records))})
    if output:
        filled.append(output[-1])
    return sorted(filled, key=lambda row: row["frame_index"])


def _holder_by_frame(states):
    result = {}
    for state in states or []:
        frame = state.frame_index if isinstance(state, PossessionObservation) else state.get("frame_index")
        if isinstance(state, PossessionObservation):
            holder = state.context.effective_holder_id
        else:
            context = state.get("context", {})
            holder = context.get("effective_holder_id") if isinstance(context, dict) else None
            if holder is None:
                holder = state.get("stable_player_id")
        result[frame] = holder
    return result


def _transition_context_by_frame(states):
    """Keep effective continuity while exposing observed candidate evidence."""
    result = {}
    for state in states or []:
        frame = state.frame_index if isinstance(state, PossessionObservation) else state.get("frame_index")
        if isinstance(state, PossessionObservation):
            context, quality, team, audit = state.context, state.state, state.team_id, state.audit
            effective, observed = context.effective_holder_id, context.observed_candidate_id
            effective_team = context.effective_holder_team
            recovery_source = context.recovery_source_id
            recovery_team = context.recovery_source_team
            recovery_reason = context.recovery_reason
            recovery_remaining = context.recovery_remaining_seconds
        else:
            context = state.get("context") or {}
            effective = context.get("effective_holder_id") or state.get("stable_player_id")
            observed = context.get("observed_candidate_id")
            quality = state.get("state")
            team = state.get("team_id")
            effective_team = context.get("effective_holder_team")
            audit = state.get("audit") or {}
            recovery_source = (context.get("recovery_source_id")
                               if context.get("recovery_source_id") is not None
                               else audit.get("recovery_source_id"))
            recovery_team = (context.get("recovery_source_team")
                             if context.get("recovery_source_team") is not None
                             else audit.get("recovery_source_team"))
            recovery_reason = (context.get("recovery_reason")
                               if context.get("recovery_reason") is not None
                               else audit.get("recovery_reason"))
            recovery_remaining = context.get("recovery_remaining_seconds", 0.0)
        if isinstance(quality, PossessionState):
            quality = quality.value
        audit_provenance = audit.provenance if isinstance(audit, PossessionAudit) else audit.get("provenance", {})
        reason_codes = audit_provenance.reason_codes if isinstance(audit_provenance, Provenance) else (audit_provenance or {}).get("reason_codes", [])
        result[frame] = {"effective": effective, "observed": observed, "state": quality,
                         "team": team, "effective_team": effective_team,
                         "transient": "TRANSIENT_SWITCH" in reason_codes or "reverted_switch" in reason_codes,
                         "recovery_source": recovery_source, "recovery_team": recovery_team,
                         "recovery_reason": recovery_reason,
                         "recovery_remaining": recovery_remaining}
    return result


def _team_by_frame(states):
    result = {}
    for state in states or []:
        frame = state.frame_index if isinstance(state, PossessionObservation) else state.get("frame_index")
        team = state.team_id if isinstance(state, PossessionObservation) else state.get("team_id")
        result[frame] = team
    return result


def debounce_reverted_switches(observations, *, real_fps, window_seconds=.5):
    """Collapse short, fully resolved A→B→A intervals without hiding evidence."""
    if type(real_fps) not in {int, float} or not math.isfinite(float(real_fps)) or real_fps <= 0:
        raise ValueError("real_fps must be positive")
    if type(window_seconds) not in {int, float} or not math.isfinite(float(window_seconds)) or window_seconds < 0:
        raise ValueError("window_seconds must be finite and non-negative")
    rows = list(observations)
    if not rows or not all(isinstance(row, (PossessionObservation, dict)) for row in rows):
        return rows
    rows.sort(key=lambda row: row.frame_index if isinstance(row, PossessionObservation) else row["frame_index"])

    def holder(row):
        if isinstance(row, PossessionObservation):
            return row.context.effective_holder_id
        context = row.get("context", {})
        return context.get("effective_holder_id") if isinstance(context, dict) else None

    def timestamp(row):
        value = row.timestamp_seconds if isinstance(row, PossessionObservation) else row.get("timestamp_seconds")
        return float(value) if type(value) in {int, float} and math.isfinite(float(value)) and value >= 0 else None

    def frame(row):
        return row.frame_index if isinstance(row, PossessionObservation) else row["frame_index"]

    def team(row):
        if isinstance(row, PossessionObservation):
            return row.context.effective_holder_team
        context = row.get("context", {})
        return context.get("effective_holder_team") if isinstance(context, dict) else None

    def duration(start, end):
        left, right = timestamp(start), timestamp(end)
        return right - left if left is not None and right is not None else (frame(end) - frame(start)) / float(real_fps)

    def mark(row, effective_holder, effective_team, start, end, elapsed):
        source_frames = [frame(start), frame(end)]
        if isinstance(row, PossessionObservation):
            context = replace(row.context, effective_holder_id=effective_holder, effective_holder_team=effective_team)
            provenance = replace(row.audit.provenance,
                                 frame_indices=list(dict.fromkeys(row.audit.provenance.frame_indices + source_frames)),
                                 reason_codes=list(dict.fromkeys(row.audit.provenance.reason_codes + ["TRANSIENT_SWITCH", "reverted_switch"])))
            audit = replace(row.audit, frame_start=frame(start), frame_end=frame(end),
                            timestamp_start=timestamp(start), timestamp_end=timestamp(end),
                            reverted_switch_duration_seconds=elapsed, provenance=provenance)
            return replace(row, context=context, audit=audit)
        copy = dict(row)
        context = dict(copy.get("context") or {})
        context["effective_holder_id"] = effective_holder
        context["effective_holder_team"] = effective_team
        copy["context"] = context
        audit = dict(copy.get("audit") or {})
        provenance = dict(audit.get("provenance") or {})
        provenance["frame_indices"] = list(dict.fromkeys(list(provenance.get("frame_indices", [])) + source_frames))
        provenance["reason_codes"] = list(dict.fromkeys(list(provenance.get("reason_codes", [])) + ["TRANSIENT_SWITCH", "reverted_switch"]))
        audit.update(frame_start=frame(start), frame_end=frame(end), timestamp_start=timestamp(start),
                    timestamp_end=timestamp(end), reverted_switch_duration_seconds=elapsed, provenance=provenance)
        copy["audit"] = audit
        return copy

    output = list(rows)
    i = 0
    while i < len(rows):
        first = holder(rows[i])
        if first is None:
            i += 1
            continue
        j = i + 1
        while j < len(rows) and holder(rows[j]) == first:
            j += 1
        if j >= len(rows) or holder(rows[j]) is None:
            i = j
            continue
        challenger = holder(rows[j])
        k = j + 1
        while k < len(rows) and holder(rows[k]) == challenger:
            k += 1
        if k < len(rows) and holder(rows[k]) == first:
            elapsed = duration(rows[j], rows[k])
            if elapsed <= float(window_seconds) + 1e-9:
                for index in range(j, k):
                    output[index] = mark(output[index], first, team(rows[i]), rows[j], rows[k], elapsed)
                i = k
                continue
        i = j
    return output


def _reject_duplicate_ball_frames(rows):
    frames = [row.get("frame_index") for row in rows]
    if len(frames) != len(set(frames)):
        raise ValueError("duplicate ball frame indices")
    last_time = None
    for row in rows:
        timestamp = row.get("timestamp_seconds")
        if timestamp is not None and last_time is not None and timestamp < last_time:
            raise ValueError("ball rows must be monotonic")
        if timestamp is not None:
            last_time = timestamp
    if any(b < a for a, b in zip(frames, frames[1:])):
        raise ValueError("ball rows must be monotonic")


def _timestamp_range(rows):
    values = [row["timestamp_seconds"] for row in rows if type(row.get("timestamp_seconds")) in {int, float} and math.isfinite(float(row["timestamp_seconds"]))]
    return (min(values), max(values)) if values else (None, None)


def derive_ball_primitives(ball_rows, possession_states, *, real_fps, loose_seconds=.3, release_persistence_frames=2, reception_persistence_frames=2):
    """Derive only conservative, provenance-bearing primitive ball events."""
    if type(real_fps) not in {int, float} or not math.isfinite(float(real_fps)) or real_fps <= 0:
        raise ValueError("real_fps must be positive")
    if type(loose_seconds) not in {int, float} or not math.isfinite(float(loose_seconds)) or loose_seconds < 0:
        raise ValueError("loose_seconds must be finite and non-negative")
    for value, name in ((release_persistence_frames, "release_persistence_frames"), (reception_persistence_frames, "reception_persistence_frames")):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    ball_rows = list(ball_rows)
    _reject_duplicate_ball_frames(ball_rows)
    rows = sorted((row for row in ball_rows if type(row.get("frame_index")) is int and row["frame_index"] >= 0
                   and row.get("ball_source") in {"detected", "fallback", "interpolated"}
                   and _ball_center(row) is not None and _bounded_number(row.get("quality"))
                   and _provenance(row, row["frame_index"]) is not None
                   and "source_records" in row and "source_frame_indices" in row
                   and (row.get("timestamp_seconds") is None or (type(row.get("timestamp_seconds")) in {int, float} and math.isfinite(float(row["timestamp_seconds"]))))), key=lambda row: row["frame_index"])
    holders = _holder_by_frame(possession_states)
    teams = _team_by_frame(possession_states)
    context_by_frame = _transition_context_by_frame(possession_states)
    result = []
    i = 0
    while i < len(rows):
        frame = rows[i]["frame_index"]
        holder = holders.get(frame)
        if holder is not None:
            j = i + 1
            while j < len(rows) and holders.get(rows[j]["frame_index"]) is None:
                j += 1
            destination = holders.get(rows[j]["frame_index"]) if j < len(rows) else None
            destination_end = j
            while (destination is not None and destination_end + 1 < len(rows)
                   and rows[destination_end + 1]["frame_index"] == rows[destination_end]["frame_index"] + 1
                   and holders.get(rows[destination_end + 1]["frame_index"]) == destination):
                destination_end += 1
            if (j < len(rows) and destination not in {None, holder}
                    and len(rows[i + 1:j]) >= release_persistence_frames
                    and destination_end - j + 1 >= reception_persistence_frames
                    and not any(context_by_frame.get(rows[k]["frame_index"], {}).get("recovery_source") == holder
                                for k in range(i + 1, j))):
                gap = rows[i + 1:j]
                if gap and all(rows[k]["frame_index"] == rows[k - 1]["frame_index"] + 1 for k in range(i + 2, j)):
                    destination = holders[rows[j]["frame_index"]]
                    points = [_ball_center(row) for row in [rows[i], *gap, rows[j]]]
                    points = [point for point in points if point is not None]
                    moved = len(points) > 1 and any(points[n] != points[n - 1] for n in range(1, len(points)))
                    refs = sorted({ref for row in [rows[i], *gap, rows[j]] for ref in row.get("source_records", [])})
                    base = {"frame_index": gap[0]["frame_index"], "source_frame_indices": sorted({row["frame_index"] for row in [rows[i], *gap, rows[j]]}), "provenance": {"source_records": refs, "frame_indices": sorted({row["frame_index"] for row in [rows[i], *gap, rows[j]]}), "reason_codes": []}}
                    start, end = _timestamp_range([rows[i], *gap])
                    result.append(dict(base, primitive_type="BALL_RELEASE", reason_codes=["holder_separation"], timestamp_start=start, timestamp_end=end))
                    if moved:
                        start, end = _timestamp_range(gap)
                        result.append(dict(base, primitive_type="BALL_IN_TRANSIT", frame_index=gap[0]["frame_index"], reason_codes=["trajectory"], timestamp_start=start, timestamp_end=end, provenance=dict(base["provenance"], frame_indices=[row["frame_index"] for row in gap])))
                        start, end = _timestamp_range(rows[j:destination_end + 1])
                        result.append(dict(base, primitive_type="BALL_RECEPTION", frame_index=rows[destination_end]["frame_index"], reason_codes=["unique_reception"], timestamp_start=start, timestamp_end=end, provenance=dict(base["provenance"], frame_indices=[rows[destination_end]["frame_index"]])))
                i = destination_end
                continue
        i += 1
    # Grace deliberately keeps the effective holder. A persistent observed
    # candidate still supplies release evidence, but only OBSERVED rows can
    # become the reception.
    for destination_start, row in enumerate(rows):
        frame = row["frame_index"]
        destination_context = context_by_frame.get(frame, {})
        destination = destination_context.get("observed")
        if destination is None or destination_context.get("state") != PossessionState.OBSERVED.value:
            continue
        if destination_start > 0:
            previous = context_by_frame.get(rows[destination_start - 1]["frame_index"], {})
            if (previous.get("observed") == destination
                    and previous.get("state") == PossessionState.OBSERVED.value
                    and previous.get("effective") == destination_context.get("effective")):
                continue
        source_index = destination_start - 1
        if source_index < 0:
            continue
        source_context = context_by_frame.get(rows[source_index]["frame_index"], {})
        source = source_context.get("effective")
        recovery = False
        source_team = source_context.get("effective_team")
        if source is None and destination_context.get("recovery_source") is not None:
            if destination_context.get("recovery_remaining", 0.0) <= 0:
                continue
            source = destination_context.get("recovery_source")
            source_team = destination_context.get("recovery_team")
            recovery = destination_context.get("recovery_remaining", 0.0) > 0
            source_index = next((index for index in range(destination_start - 1, -1, -1)
                                 if context_by_frame.get(rows[index]["frame_index"], {}).get("effective") == source), -1)
            if source_index < 0:
                continue
        if source is None or source == destination or destination_context.get("transient") or (recovery and source_team is None):
            continue
        candidate_start = destination_start
        while candidate_start > (source_index + 1 if recovery else 0):
            previous = context_by_frame.get(rows[candidate_start - 1]["frame_index"], {})
            if recovery:
                if (previous.get("observed") != destination or previous.get("effective") is not None
                        or previous.get("recovery_source") != source):
                    break
            elif previous.get("observed") != destination or previous.get("effective") != source:
                break
            candidate_start -= 1
        # Candidate-only grace/recovery frames are the release persistence window.
        if (destination_start - candidate_start if not recovery else destination_start - source_index - 1) < release_persistence_frames:
            continue
        destination_end = destination_start
        while (destination_end + 1 < len(rows)
               and rows[destination_end + 1]["frame_index"] == rows[destination_end]["frame_index"] + 1):
            next_context = context_by_frame.get(rows[destination_end + 1]["frame_index"], {})
            if (next_context.get("observed") != destination
                    or next_context.get("state") != PossessionState.OBSERVED.value
                    or next_context.get("transient")):
                break
            destination_end += 1
        if destination_end - destination_start + 1 < reception_persistence_frames:
            continue
        gap = rows[(source_index + 1 if recovery else candidate_start):destination_start]
        if not gap or any(rows[k]["frame_index"] != rows[k - 1]["frame_index"] + 1
                          for k in range(candidate_start + 1, destination_start)):
            continue
        points = [_ball_center(rows[source_index]), *(_ball_center(item) for item in gap), _ball_center(rows[destination_start])]
        points = [point for point in points if point is not None]
        if len(points) < 2 or not any(points[n] != points[n - 1] for n in range(1, len(points))):
            continue
        evidence_rows = [rows[source_index], *gap, rows[destination_start]]
        base = {"frame_index": gap[0]["frame_index"],
                "source_frame_indices": sorted({item["frame_index"] for item in evidence_rows}),
                "provenance": {"source_records": sorted({ref for item in evidence_rows for ref in item.get("source_records", [])}),
                                "frame_indices": sorted({item["frame_index"] for item in evidence_rows}),
                                "reason_codes": ["recovery_source" if recovery else "observed_candidate_grace"]}}
        start, end = _timestamp_range([rows[source_index], *gap])
        result.append(dict(base, primitive_type="BALL_RELEASE", reason_codes=["recovery_source" if recovery else "holder_separation"], timestamp_start=start, timestamp_end=end))
        start, end = _timestamp_range(gap)
        result.append(dict(base, primitive_type="BALL_IN_TRANSIT", reason_codes=["trajectory"], timestamp_start=start, timestamp_end=end,
                           provenance=dict(base["provenance"], frame_indices=[item["frame_index"] for item in gap])))
        start, end = _timestamp_range(rows[destination_start:destination_end + 1])
        result.append(dict(base, primitive_type="BALL_RECEPTION", frame_index=rows[destination_end]["frame_index"], reason_codes=["unique_reception"],
                           timestamp_start=start, timestamp_end=end,
                           provenance=dict(base["provenance"], frame_indices=[rows[destination_end]["frame_index"]])))
    # A loose interval is emitted once, only after its real-FPS duration is met.
    empty = [row for row in rows if holders.get(row["frame_index"]) is None]
    start = 0
    while start < len(empty):
        end = start
        while end + 1 < len(empty) and empty[end + 1]["frame_index"] == empty[end]["frame_index"] + 1:
            end += 1
        if (empty[end]["frame_index"] - empty[start]["frame_index"]) / float(real_fps) >= loose_seconds:
            refs = sorted({ref for row in empty[start:end + 1] for ref in row.get("source_records", [])})
            timestamp_start, timestamp_end = _timestamp_range(empty[start:end + 1])
            result.append({"primitive_type": "LOOSE_BALL", "frame_index": empty[start]["frame_index"] + int(loose_seconds * real_fps), "timestamp_start": timestamp_start, "timestamp_end": timestamp_end,
                           "source_frame_indices": [row["frame_index"] for row in empty[start:end + 1]],
                           "provenance": {"source_records": refs, "frame_indices": [row["frame_index"] for row in empty[start:end + 1]], "reason_codes": ["unpossessed_duration"]},
                           "reason_codes": ["unpossessed_duration"]})
        start = end + 1
    order = {"BALL_RELEASE": 0, "BALL_IN_TRANSIT": 1, "BALL_RECEPTION": 2, "LOOSE_BALL": 3}
    return sorted(result, key=lambda item: (item["frame_index"], order[item["primitive_type"]]))


def derive_ball_transitions(ball_rows, possession_states, *, real_fps, loose_seconds=.3, release_persistence_frames=2, reception_persistence_frames=2, event_window_seconds=None):
    ball_rows = list(ball_rows)
    _reject_duplicate_ball_frames(ball_rows)
    if event_window_seconds is not None and (type(event_window_seconds) not in {int, float} or not math.isfinite(float(event_window_seconds)) or event_window_seconds <= 0):
        raise ValueError("event_window_seconds must be positive and finite")
    primitives = derive_ball_primitives(ball_rows, possession_states, real_fps=real_fps, loose_seconds=loose_seconds, release_persistence_frames=release_persistence_frames, reception_persistence_frames=reception_persistence_frames)
    holders = _holder_by_frame(possession_states)
    teams = _team_by_frame(possession_states)
    context_by_frame = _transition_context_by_frame(possession_states)

    def incomplete_recovery():
        ordered = sorted((row for row in ball_rows if type(row.get("frame_index")) is int), key=lambda row: row["frame_index"])
        for destination_row in ordered:
            context = context_by_frame.get(destination_row["frame_index"], {})
            destination = context.get("observed")
            source = context.get("recovery_source")
            if (context.get("state") != PossessionState.OBSERVED.value or not source or not destination
                    or source == destination or context.get("recovery_remaining", 0.0) <= 0):
                continue
            source_index = next((index for index in range(len(ordered) - 1, -1, -1)
                                 if ordered[index]["frame_index"] < destination_row["frame_index"]
                                 and context_by_frame.get(ordered[index]["frame_index"], {}).get("effective") == source), None)
            if source_index is None:
                continue
            segment_rows = ordered[source_index:ordered.index(destination_row) + 1]
            refs = sorted({ref for row in segment_rows for ref in row.get("source_records", [])})
            frames = [row["frame_index"] for row in segment_rows]
            start_time = segment_rows[0].get("timestamp_seconds")
            end_time = segment_rows[-1].get("timestamp_seconds")
            duration = ((float(end_time) - float(start_time)) if start_time is not None and end_time is not None
                        else (frames[-1] - frames[0]) / float(real_fps))
            if event_window_seconds is not None and duration > event_window_seconds:
                continue
            primitive_types = [item["primitive_type"] for item in primitives
                               if frames[0] <= item["frame_index"] <= frames[-1]]
            return BallTransition(
                f"ball-recovery-{destination_row['frame_index']}", source, destination,
                primitive_types or ["RECOVERY_PENDING"],
                Provenance(refs, frames, ["raw_ball_provenance", "recovery_source"]),
                state=PossessionState.UNKNOWN_RECOVERY,
                frame_start=frames[0], frame_end=frames[-1], source_team_id=context.get("recovery_team"),
                destination_team_id=context.get("team"), duration_seconds=duration,
                timestamp_start=start_time, timestamp_end=end_time,
                release_evidence="BALL_RELEASE" in primitive_types,
                reception_evidence="BALL_RECEPTION" in primitive_types,
                trajectory_evidence="BALL_IN_TRANSIT" in primitive_types,
                recovery_source_id=source, recovery_reason=context.get("recovery_reason"),
                recovery_window_seconds=context.get("recovery_remaining", 0.0))
        return None

    if not primitives:
        pending = incomplete_recovery()
        return [pending] if pending is not None else []
    grouped = [item for item in primitives if item["primitive_type"] != "LOOSE_BALL"]
    if not {item["primitive_type"] for item in grouped}.issuperset({"BALL_RELEASE", "BALL_IN_TRANSIT", "BALL_RECEPTION"}):
        pending = incomplete_recovery()
        return [pending] if pending is not None else []
    transitions = []
    releases = [index for index, item in enumerate(grouped) if item["primitive_type"] == "BALL_RELEASE"]
    for release_index in releases:
        release = grouped[release_index]
        reception = next((item for item in grouped[release_index + 1:] if item["primitive_type"] == "BALL_RECEPTION"), None)
        if reception is None:
            continue
        end_index = grouped.index(reception, release_index + 1) + 1
        segment = grouped[release_index:end_index]
        start_context = context_by_frame.get(release["frame_index"] - 1, {})
        end_context = context_by_frame.get(reception["frame_index"], {})
        start_holder = start_context.get("effective") or start_context.get("observed") or holders.get(release["frame_index"] - 1)
        if start_holder is None:
            start_holder = end_context.get("recovery_source")
            if start_holder is not None:
                start_context = {**start_context, "effective_team": end_context.get("recovery_team"),
                                 "recovery_source": start_holder, "recovery_reason": end_context.get("recovery_reason"),
                                 "recovery_remaining": end_context.get("recovery_remaining", 0.0)}
        end_holder = (end_context.get("observed")
                      if end_context.get("state") == PossessionState.OBSERVED.value
                      else end_context.get("effective") or holders.get(reception["frame_index"]))
        start_row = next(row for row in ball_rows if row.get("frame_index") == release["frame_index"])
        end_row = next(row for row in ball_rows if row.get("frame_index") == reception["frame_index"])
        start_time, end_time = start_row.get("timestamp_seconds"), end_row.get("timestamp_seconds")
        duration = (float(end_time) - float(start_time)) if start_time is not None and end_time is not None else (reception["frame_index"] - release["frame_index"]) / float(real_fps)
        if event_window_seconds is not None and duration > event_window_seconds:
            continue
        refs = sorted({ref for item in segment for ref in item["provenance"]["source_records"]})
        recovery_source = (start_holder if any("recovery_source" in item.get("reason_codes", [])
                                               or "recovery_source" in item.get("provenance", {}).get("reason_codes", [])
                                               for item in segment) else None)
        transitions.append(BallTransition(f"ball-transition-{release['frame_index']}", start_holder, end_holder,
                             [item["primitive_type"] for item in segment],
                             Provenance(refs, sorted({frame for item in segment for frame in item["provenance"]["frame_indices"]}), ["raw_ball_provenance", "primitive_geometry"]),
                             frame_start=release["frame_index"], frame_end=reception["frame_index"], confidence=.5,
                              source_team_id=(start_context.get("effective_team")
                                              if start_context.get("effective_team") is not None
                                              else teams.get(release["frame_index"] - 1)),
                              destination_team_id=(end_context.get("team")
                                                  if end_context.get("team") is not None
                                                  else teams.get(reception["frame_index"])),
                              duration_seconds=duration, timestamp_start=start_time, timestamp_end=end_time,
                              release_evidence=True, reception_evidence=True, trajectory_evidence=True, score_margin=None,
                              recovery_source_id=recovery_source,
                              recovery_reason=end_context.get("recovery_reason") if recovery_source else None,
                              recovery_window_seconds=end_context.get("recovery_remaining", 0.0) if recovery_source else 0.0))
    return transitions


def classify_transition(transition, *, rapid_seconds=.3, proximity_threshold=80.0,
                        loose_threshold=.3, minimum_margin=0.1):
    """Classify one transition without upgrading ambiguous evidence to fact."""
    if not isinstance(transition, BallTransition):
        raise TypeError("transition must be BallTransition")
    if "TRANSIENT_SWITCH" in transition.provenance.reason_codes or "reverted_switch" in transition.provenance.reason_codes:
        return []
    frame = transition.frame_end if transition.frame_end is not None else transition.frame_start
    frame = frame if frame is not None else (transition.provenance.frame_indices[-1] if transition.provenance.frame_indices else 0)
    common = dict(transition_id=transition.transition_id, provenance=transition.provenance,
                  source_holder=transition.source_holder, destination_holder=transition.destination_holder,
                  source_team_id=transition.source_team_id, destination_team_id=transition.destination_team_id,
                  score=transition.score, score_margin=transition.score_margin,
                  frame_start=transition.frame_start, frame_end=transition.frame_end,
                  timestamp_start=transition.timestamp_start, timestamp_end=transition.timestamp_end)
    def event(kind, reasons):
        return PossessionEvent(kind, frame, transition.provenance, transition_id=transition.transition_id,
                                 reason_codes=list(reasons), **{key: value for key, value in common.items() if key != "transition_id" and key != "provenance"})

    if transition.state is PossessionState.UNKNOWN_RECOVERY:
        return [event(EventType.UNKNOWN_RECOVERY, ("UNKNOWN_RECOVERY", transition.recovery_reason or "recovery_evidence_insufficient"))]

    aliases = {"RELEASE": "BALL_RELEASE", "TRANSIT": "BALL_IN_TRANSIT",
               "RECEPTION": "BALL_RECEPTION", "LOOSE": "LOOSE_BALL"}
    primitives = {aliases.get(str(value).upper(), str(value).upper()) for value in transition.primitive_types}
    complete = ({"BALL_RELEASE", "BALL_IN_TRANSIT", "BALL_RECEPTION"} <= primitives and
                transition.release_evidence and transition.reception_evidence and transition.trajectory_evidence)
    if "OUT_OF_BOUNDS" in primitives:
        return [event(EventType.UNKNOWN, ("out_of_bounds_deferred",))]
    if transition.source_holder is None or transition.source_team_id is None:
        return [event(EventType.UNKNOWN, ("missing_source_identity_or_team",))]
    if not complete:
        if transition.destination_holder is None and transition.release_evidence:
            if transition.loss_of_control_evidence:
                return [event(EventType.LOSS_OF_CONTROL, ("release_without_unique_reception",))]
            return [event(EventType.UNKNOWN, ("release_without_explicit_loss",))]
        return [event(EventType.UNKNOWN, ("incomplete_transition_evidence",))]
    if transition.destination_holder is None:
        if transition.loss_of_control_evidence:
            return [event(EventType.LOSS_OF_CONTROL, ("missing_destination_reception",))]
        return [event(EventType.UNKNOWN, ("missing_destination_reception",))]
    if transition.destination_team_id is None:
        return [event(EventType.UNKNOWN, ("missing_destination_team",))]
    if transition.source_holder == transition.destination_holder:
        return [event(EventType.UNKNOWN, ("unchanged_holder",))]
    same_team = transition.source_team_id == transition.destination_team_id
    if same_team:
        reasons = ("same_team_possession_transfer",)
        transfer = event(EventType.TRANSFER, reasons + (("loose_ball_recovery",) if transition.loose_seconds > loose_threshold or "LOOSE_BALL" in primitives else ()))
        if transition.loose_seconds <= loose_threshold and "LOOSE_BALL" not in primitives:
            return [transfer, event(EventType.PASS, ("release_trajectory_reception",))]
        return [transfer]
    rapid = transition.duration_seconds is not None and transition.duration_seconds <= rapid_seconds
    local = transition.proximity_distance is not None and transition.proximity_distance <= proximity_threshold
    margin_ok = transition.score_margin is not None and transition.score_margin >= minimum_margin
    transfer = event(EventType.TRANSFER, ("opponent_possession_change",))
    if rapid and local and margin_ok and transition.loose_seconds <= loose_threshold:
        return [transfer, event(EventType.STEAL, ("rapid_local_opponent_transfer",))]
    confirmed_loss = transition.loss_of_control_evidence or "LOSS_OF_CONTROL" in primitives or "TURNOVER" in primitives
    if confirmed_loss:
        return [transfer, event(EventType.TURNOVER, ("confirmed_loss_opponent_recovery",))]
    return [transfer, event(EventType.UNKNOWN, ("opponent_change_without_turnover_evidence",))]


extract_ball_primitives = derive_ball_primitives
