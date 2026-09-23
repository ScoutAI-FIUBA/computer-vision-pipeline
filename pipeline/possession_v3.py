"""Versioned V3 possession contracts; resolution is deferred to later units."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.config import PipelineConfig

V3_CONTRACT_VERSION = 1
V3_INPUT_CONTRACT_VERSION = 1
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

    def __post_init__(self) -> None:
        _required_score(self.score, "score")
        _identity(self.stable_player_id, self.raw_track_id, self.team_id)
        _thresholds(self.contributions)
        _text_list(self.rejected_reasons, "rejected_reasons")
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

    def __post_init__(self) -> None:
        _frame(self.frame_index, "frame_index")
        _schema(self.schema_version)
        object.__setattr__(self, "state", _enum(self.state, PossessionState, "state"))
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be Provenance")
        object.__setattr__(self, "evidence_kind", _enum(self.evidence_kind, EvidenceKind, "evidence_kind"))
        _identity(self.stable_player_id, self.raw_track_id, self.team_id)
        _score(self.score, "score")
        _score(self.score_margin, "score_margin")
        if type(self.profile) is not str or not self.profile.strip():
            raise ValueError("profile must be a non-empty string")
        _thresholds(self.thresholds)
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
        object.__setattr__(self, "primitive_types", list(self.primitive_types))
@dataclass(frozen=True)
class PossessionEvent:
    event_type: EventType
    frame_index: int
    provenance: Provenance
    schema_version: int = V3_CONTRACT_VERSION
    transition_id: str | None = None
    reason_codes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _frame(self.frame_index, "frame_index")
        _schema(self.schema_version)
        object.__setattr__(self, "event_type", _enum(self.event_type, EventType, "event_type"))
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be Provenance")
        if self.transition_id is not None and (type(self.transition_id) is not str or not self.transition_id.strip()):
            raise ValueError("transition_id must be a non-empty string when set")
        _text_list(self.reason_codes, "reason_codes")
def engine_name(config: PipelineConfig) -> str:
    """Return the validated selector; legacy remains the default."""
    return "v3" if config.possession_v3_enabled else "legacy"
