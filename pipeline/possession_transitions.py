"""Deterministic event transitions over confirmed possession rows."""

from __future__ import annotations

from collections.abc import Iterable


POSSESSION_TRANSITIONS_SCHEMA_VERSION = 1
TRANSITION_EVENT_TYPES = {"pass", "steal", "turnover", "lost_possession"}


def _source_rows(row: dict) -> list[str]:
    values = row.get("source_records")
    return [str(value) for value in values] if isinstance(values, list) else []


def _authoritative(row: dict) -> bool:
    return (
        row.get("status") == "confirmed"
        and row.get("player_track_id") is not None
        and bool(row.get("evidence_frame_indices"))
        and bool(_source_rows(row))
    )


def _identity(row: dict) -> tuple[object, object]:
    stable_id = row.get("stable_player_id")
    if stable_id not in (None, ""):
        return "stable", stable_id
    return "raw", row.get("player_track_id")


def _confirmed_rows(rows: list[dict]) -> tuple[list[dict], set[int]]:
    by_frame: dict[int, list[dict]] = {}
    for row in rows:
        if _authoritative(row):
            by_frame.setdefault(int(row["frame_index"]), []).append(row)
    duplicate_frames = {frame for frame, values in by_frame.items() if len(values) > 1}
    return [by_frame[frame][0] for frame in sorted(by_frame) if frame not in duplicate_frames], duplicate_frames


def mark_terminal_loss_rows(possession_rows: Iterable[dict], *, max_frame_gap: int) -> list[dict]:
    """Mark produced terminal evidence after a bounded holder gap."""
    rows = [dict(row) for row in possession_rows if isinstance(row, dict)]
    if type(max_frame_gap) is not int or max_frame_gap < 1:
        raise ValueError("max_frame_gap must be a positive integer")
    confirmed, duplicate_frames = _confirmed_rows(rows)
    if not confirmed:
        return rows
    previous = confirmed[-1]
    after = [row for row in rows if row.get("frame_index", -1) > previous["frame_index"]]
    later_confirmed, _ = _confirmed_rows(after)
    loss_candidates = [row for row in after if not _authoritative(row)]
    if after and not later_confirmed and not (duplicate_frames & {int(row["frame_index"]) for row in after}):
        gap = int(after[-1]["frame_index"]) - int(previous["frame_index"])
        if gap > max_frame_gap and loss_candidates:
            terminal = dict(loss_candidates[-1])
            terminal["terminal_loss"] = True
            terminal["rejection_reason"] = "terminal_loss"
            terminal_frame = terminal["frame_index"]
            source_terminal = loss_candidates[-1]
            rows = [terminal if row is source_terminal else row for row in rows]
    return rows


def _team(row: dict) -> int | None:
    value = row.get("team_id")
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _event(
    *,
    event_type: str,
    status: str,
    previous: dict,
    following: dict | None,
    frame_gap: int | None,
    reason: str,
) -> dict:
    source_records = _source_rows(previous)
    evidence = list(previous.get("evidence_frame_indices") or [])
    if following is not None:
        source_records.extend(record for record in _source_rows(following) if record not in source_records)
        evidence.extend(value for value in following.get("evidence_frame_indices") or [] if value not in evidence)
    return {
        "schema_version": POSSESSION_TRANSITIONS_SCHEMA_VERSION,
        "event_type": event_type,
        "status": status,
        "frame_index": int(previous["frame_index"]),
        "previous_frame_index": int(previous["frame_index"]),
        "next_frame_index": None if following is None else int(following["frame_index"]),
        "frame_gap": frame_gap,
        "previous_player_track_id": previous.get("player_track_id"),
        "previous_stable_player_id": previous.get("stable_player_id"),
        "previous_team_id": _team(previous),
        "next_player_track_id": None if following is None else following.get("player_track_id"),
        "next_stable_player_id": None if following is None else following.get("stable_player_id"),
        "next_team_id": None if following is None else _team(following),
        "evidence_frame_indices": evidence,
        "source_records": source_records,
        "reason": reason,
    }


def resolve_possession_transitions(
    possession_rows: Iterable[dict], *, max_frame_gap: int = 15
) -> list[dict]:
    """Resolve only evidence-backed holder transitions.

    Non-confirmed rows are uncertainty evidence, never holders. A terminal loss
    is confirmed only when the source row explicitly carries terminal-loss
    metadata; otherwise the derived loss remains uncertain.
    """
    if type(max_frame_gap) is not int or max_frame_gap < 1:
        raise ValueError("max_frame_gap must be a positive integer")
    rows = sorted(
        (row for row in possession_rows if isinstance(row, dict) and isinstance(row.get("frame_index"), int)),
        key=lambda row: row["frame_index"],
    )
    confirmed, duplicate_frames = _confirmed_rows(rows)
    output: list[dict] = []
    for previous, following in zip(confirmed, confirmed[1:]):
        gap = int(following["frame_index"]) - int(previous["frame_index"])
        if gap <= 0:
            continue
        if any(previous["frame_index"] < frame < following["frame_index"] for frame in duplicate_frames):
            continue
        if _identity(previous) == _identity(following):
            continue
        if gap > max_frame_gap:
            output.append(_event(
                event_type="lost_possession", status="uncertain", previous=previous,
                following=following, frame_gap=gap, reason="confirmed_holder_gap_exceeded",
            ))
            continue
        previous_team, following_team = _team(previous), _team(following)
        if previous_team is None or following_team is None:
            continue
        output.append(_event(
            event_type="pass" if previous_team == following_team else "steal",
            status="confirmed", previous=previous, following=following, frame_gap=gap,
            reason="same_team_confirmed_transfer" if previous_team == following_team else "opponent_confirmed_transfer",
        ))

    if confirmed:
        previous = confirmed[-1]
        after = [row for row in rows if row["frame_index"] > previous["frame_index"]]
        after_confirmed, _ = _confirmed_rows(after)
        after_has_duplicate = any(row["frame_index"] in duplicate_frames for row in after)
        if after and not after_has_duplicate:
            terminal = after[-1]
            gap = int(terminal["frame_index"]) - int(previous["frame_index"])
            if gap <= max_frame_gap or not after_confirmed:
                explicit_terminal = terminal.get("terminal_loss") is True or terminal.get("rejection_reason") in {
                    "terminal_loss", "confirmed_terminal_loss"
                }
                output.append(_event(
                    event_type="turnover" if explicit_terminal else "lost_possession",
                    status="confirmed" if explicit_terminal else "uncertain",
                    previous=previous, following=terminal, frame_gap=gap,
                    reason="explicit_terminal_loss" if explicit_terminal else "no_confirmed_holder_within_gap",
                ))
    return output
