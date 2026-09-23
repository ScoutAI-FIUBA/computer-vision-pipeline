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
        holder = state.stable_player_id if isinstance(state, PossessionObservation) else state.get("stable_player_id")
        result[frame] = holder
    return result


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
                    and destination_end - j + 1 >= reception_persistence_frames):
                gap = rows[i + 1:j]
                if gap and all(rows[k]["frame_index"] == rows[k - 1]["frame_index"] + 1 for k in range(i + 2, j)):
                    destination = holders[rows[j]["frame_index"]]
                    points = [_ball_center(row) for row in [rows[i], *gap, rows[j]]]
                    points = [point for point in points if point is not None]
                    moved = len(points) > 1 and any(points[n] != points[n - 1] for n in range(1, len(points)))
                    refs = sorted({ref for row in [rows[i], *gap, rows[j]] for ref in row.get("source_records", [])})
                    base = {"frame_index": gap[0]["frame_index"], "source_frame_indices": sorted({row["frame_index"] for row in [rows[i], *gap, rows[j]]}), "provenance": {"source_records": refs, "frame_indices": sorted({row["frame_index"] for row in [rows[i], *gap, rows[j]]}), "reason_codes": []}}
                    result.append(dict(base, primitive_type="BALL_RELEASE", reason_codes=["holder_separation"]))
                    if moved:
                        result.append(dict(base, primitive_type="BALL_IN_TRANSIT", frame_index=gap[0]["frame_index"], reason_codes=["trajectory"] , provenance=dict(base["provenance"], frame_indices=[row["frame_index"] for row in gap])))
                        result.append(dict(base, primitive_type="BALL_RECEPTION", frame_index=rows[destination_end]["frame_index"], reason_codes=["unique_reception"], provenance=dict(base["provenance"], frame_indices=[rows[destination_end]["frame_index"]])))
                i = destination_end
                continue
        i += 1
    # A loose interval is emitted once, only after its real-FPS duration is met.
    empty = [row for row in rows if holders.get(row["frame_index"]) is None]
    start = 0
    while start < len(empty):
        end = start
        while end + 1 < len(empty) and empty[end + 1]["frame_index"] == empty[end]["frame_index"] + 1:
            end += 1
        if (empty[end]["frame_index"] - empty[start]["frame_index"]) / float(real_fps) >= loose_seconds:
            refs = sorted({ref for row in empty[start:end + 1] for ref in row.get("source_records", [])})
            result.append({"primitive_type": "LOOSE_BALL", "frame_index": empty[start]["frame_index"] + int(loose_seconds * real_fps),
                           "source_frame_indices": [row["frame_index"] for row in empty[start:end + 1]],
                           "provenance": {"source_records": refs, "frame_indices": [row["frame_index"] for row in empty[start:end + 1]], "reason_codes": ["unpossessed_duration"]},
                           "reason_codes": ["unpossessed_duration"]})
        start = end + 1
    order = {"BALL_RELEASE": 0, "BALL_IN_TRANSIT": 1, "BALL_RECEPTION": 2, "LOOSE_BALL": 3}
    return sorted(result, key=lambda item: (item["frame_index"], order[item["primitive_type"]]))


def derive_ball_transitions(ball_rows, possession_states, *, real_fps, loose_seconds=.3, release_persistence_frames=2, reception_persistence_frames=2):
    ball_rows = list(ball_rows)
    _reject_duplicate_ball_frames(ball_rows)
    primitives = derive_ball_primitives(ball_rows, possession_states, real_fps=real_fps, loose_seconds=loose_seconds, release_persistence_frames=release_persistence_frames, reception_persistence_frames=reception_persistence_frames)
    if not primitives:
        return []
    grouped = [item for item in primitives if item["primitive_type"] != "LOOSE_BALL"]
    if not {item["primitive_type"] for item in grouped}.issuperset({"BALL_RELEASE", "BALL_IN_TRANSIT", "BALL_RECEPTION"}):
        return []
    holders = _holder_by_frame(possession_states)
    release = next(item for item in grouped if item["primitive_type"] == "BALL_RELEASE")
    reception = next(item for item in grouped if item["primitive_type"] == "BALL_RECEPTION")
    refs = sorted({ref for item in grouped for ref in item["provenance"]["source_records"]})
    return [BallTransition(f"ball-transition-{release['frame_index']}", holders.get(release["frame_index"] - 1), holders.get(reception["frame_index"]),
                          [item["primitive_type"] for item in grouped],
                          Provenance(refs, sorted({frame for item in grouped for frame in item["provenance"]["frame_indices"]}), ["raw_ball_provenance", "primitive_geometry"]),
                          frame_start=release["frame_index"], frame_end=reception["frame_index"], confidence=.5)]


extract_ball_primitives = derive_ball_primitives
