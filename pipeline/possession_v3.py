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
    fallback_identity: str | None = None

    def __post_init__(self) -> None:
        _required_score(self.score, "score")
        _identity(self.stable_player_id, self.raw_track_id, self.team_id)
        _thresholds(self.contributions)
        _text_list(self.rejected_reasons, "rejected_reasons")
        if self.fallback_identity is not None and (type(self.fallback_identity) is not str or not self.fallback_identity.strip()):
            raise ValueError("fallback_identity must be a non-empty string when set")
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


# Work Unit 2 deliberately keeps the input boundary duck-typed: upstream
# sanitizers already own their row contracts and V3 must not rewrite them.
def _point(ball):
    if not isinstance(ball, dict):
        return None
    center = ball.get("center", ball.get("ball_center"))
    x = ball.get("x", ball.get("center_x", center[0] if isinstance(center, (list, tuple)) and len(center) >= 2 else None))
    y = ball.get("y", ball.get("center_y", center[1] if isinstance(center, (list, tuple)) and len(center) >= 2 else None))
    return (float(x), float(y)) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None


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
    quality = float(ball.get("confidence", ball.get("score", 0.0))) if isinstance(ball, dict) else 0.0
    quality = max(0.0, min(1.0, quality))
    output = []
    for row in players or []:
        stable, raw = _identity_key(row if isinstance(row, dict) else {})
        team = row.get("team_id") if isinstance(row, dict) else None
        box = row.get("bbox", row.get("box")) if isinstance(row, dict) else None
        reasons = []
        if stable is None and raw is None: reasons.append("missing_authoritative_identity")
        if type(team) is not int or team < 0: reasons.append("invalid_team")
        if point is None: reasons.append("invalid_ball_geometry")
        clean = bool(row.get("clean_mask", row.get("mask_clean", False))) if isinstance(row, dict) else False
        mask_hit = bool(row.get("mask_contains", False)) if isinstance(row, dict) else False
        if callable(row.get("mask_contains")) and point is not None: mask_hit = bool(row["mask_contains"](*point))
        height = (box[3] - box[1]) if isinstance(box, (list, tuple)) and len(box) == 4 else 0
        speed = float(ball.get("speed", 0.0)) if isinstance(ball, dict) and isinstance(ball.get("speed", 0), (int, float)) else 0.0
        margin = max(2.0, height * .15 + speed * .02)
        box_hit = point is not None and _inside_box(point, box, margin)
        if clean and mask_hit: geometry, mask = 1.0, 1.0
        elif box_hit: geometry, mask = .7, 0.0
        else: geometry, mask = 0.0, 0.0
        if not geometry: reasons.append("outside_dynamic_player_region")
        confidence = float(row.get("confidence", row.get("player_confidence", 0.0))) if isinstance(row, dict) else 0.0
        confidence = max(0.0, min(1.0, confidence))
        prior = 1.0 if stable == previous_holder else 0.0
        contributions = {"mask": mask * .35, "bbox": geometry * .35, "player_confidence": confidence * .15,
                         "ball_quality": quality * .10, "previous_holder": prior * .10}
        score = sum(contributions.values()) if not reasons or reasons == ["outside_dynamic_player_region"] else 0.0
        if reasons and reasons != ["outside_dynamic_player_region"]: score = 0.0
        fallback = f"raw:{raw}" if stable is None and raw is not None else None
        output.append(CandidateEvidence(stable, raw, team if type(team) is int and team >= 0 else None,
                                        min(1.0, score), contributions, reasons, fallback))
    return sorted(output, key=lambda c: (-c.score, c.stable_player_id or "", c.raw_track_id or 0))


class PossessionResolver:
    """Small deterministic temporal resolver; no ball/event semantics live here."""
    def __init__(self, *, real_fps, grace_seconds=.6, hysteresis_margin=.1, persistence_seconds=0.0,
                 profile="default"):
        if not isinstance(real_fps, (int, float)) or real_fps <= 0: raise ValueError("real_fps must be positive")
        self.fps = float(real_fps); self.grace = min(.6, max(0.0, grace_seconds))
        self.hysteresis = max(0.0, hysteresis_margin); self.persistence = max(0.0, persistence_seconds)
        self.profile = profile; self.holder = None; self.last_frame = None; self.last_time = None
        self.pending = None; self.gap_seconds = 0.0
        self.holder_stable = self.holder_raw = self.holder_team = self.holder_fallback = None
        self.history = []

    def _observation(self, frame, state, candidate=None, candidates=(), reason=()):
        stable = candidate.stable_player_id if candidate else self.holder_stable
        raw = candidate.raw_track_id if candidate else self.holder_raw
        team = candidate.team_id if candidate else self.holder_team
        prov = Provenance(frame_indices=[frame], reason_codes=list(reason))
        obs = PossessionObservation(frame, state, EvidenceKind.OBSERVED if state is PossessionState.OBSERVED else
                                    EvidenceKind.INFERRED if state is PossessionState.INFERRED else
                                    EvidenceKind.CONTESTED if state is PossessionState.CONTESTED else EvidenceKind.UNKNOWN,
                                    stable_player_id=stable, raw_track_id=raw,
                                    team_id=team, score=candidate.score if candidate else None,
                                    score_margin=(candidates[0].score - candidates[1].score if len(candidates) > 1 else None),
                                    candidates=list(candidates), provenance=prov, profile=self.profile,
                                    thresholds={"hysteresis_margin": self.hysteresis, "grace_seconds": self.grace})
        self.history.append(obs); return obs

    def _key(self, candidate):
        return candidate.stable_player_id or candidate.fallback_identity or (f"raw:{candidate.raw_track_id}" if candidate.raw_track_id else None)

    def resolve(self, frame_index, candidates, *, missing=False, invalid=False, ball_lost=False,
                authoritative_loss=False, duplicate=False, timestamp=None):
        candidates = sorted(candidates or [], key=lambda c: (-c.score, c.stable_player_id or "", c.raw_track_id or 0))
        elapsed = 0.0 if self.last_frame is None else (timestamp - self.last_time if timestamp is not None and self.last_time is not None else (frame_index - self.last_frame) / self.fps)
        self.last_frame, self.last_time = frame_index, timestamp
        invalid_input = invalid or authoritative_loss or duplicate or missing
        if candidates and not invalid_input:
            self.gap_seconds = 0.0
            top = candidates[0]
            if top.score <= 0:
                self.holder = None; self.pending = None; self.gap_seconds = 0.0
                self.holder_stable = self.holder_raw = self.holder_team = self.holder_fallback = None
                return self._observation(frame_index, PossessionState.NO_POSSESSION, candidates=candidates, reason=("no_valid_candidate",))
            if len(candidates) > 1 and candidates[0].score - candidates[1].score < self.hysteresis:
                return self._observation(frame_index, PossessionState.CONTESTED, candidates=candidates, reason=("insufficient_margin",))
            if top.score < .5 and self.holder is None:
                return self._observation(frame_index, PossessionState.CANDIDATE, top, candidates, ("weak_candidate",))
            if self.holder is None:
                self.holder = self._key(top); self.holder_stable, self.holder_raw = top.stable_player_id, top.raw_track_id
                self.holder_team, self.holder_fallback = top.team_id, top.fallback_identity; self.pending = None
                return self._observation(frame_index, PossessionState.OBSERVED, top, candidates)
            if self._key(top) == self.holder:
                self.pending = None
                return self._observation(frame_index, PossessionState.OBSERVED, top, candidates)
            if self.holder_stable is not None and top.stable_player_id is None:
                self.pending = None
                return self._observation(frame_index, PossessionState.CONTESTED,
                                         candidates=candidates, reason=("raw_fallback_cannot_replace_stable",))
            if top.score - next((c.score for c in candidates if self._key(c) == self.holder), 0.0) >= self.hysteresis:
                if self.pending and self.pending[0] == self._key(top) and elapsed + self.pending[1] >= self.persistence:
                    self.holder = self._key(top); self.holder_stable, self.holder_raw = top.stable_player_id, top.raw_track_id
                    self.holder_team, self.holder_fallback = top.team_id, top.fallback_identity; self.pending = None
                    return self._observation(frame_index, PossessionState.OBSERVED, top, candidates, ("stable_switch",))
                self.pending = (self._key(top), elapsed)
                return self._observation(frame_index, PossessionState.TRANSITIONING, candidates=candidates, reason=("switch_persistence",))
        if self.holder and not invalid_input and ball_lost:
            self.gap_seconds += elapsed
        if self.holder and not invalid_input and ball_lost and self.gap_seconds < self.grace:
            return self._observation(frame_index, PossessionState.INFERRED, reason=("bounded_grace",))
        prior = self.holder
        self.holder = None; self.pending = None; self.gap_seconds = 0.0
        self.holder_stable = self.holder_raw = self.holder_team = self.holder_fallback = None
        state = PossessionState.UNKNOWN if missing or invalid_input else PossessionState.NO_POSSESSION
        reason = ("missing_authoritative_frame" if missing else "invalid_evidence" if invalid_input else "evidence_loss",)
        return self._observation(frame_index, state, candidates=candidates, reason=reason)

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
