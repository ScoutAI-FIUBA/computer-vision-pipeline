#!/usr/bin/env python3
"""Render SAM3.1 raw tracking artifacts over the original local video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

MAX_DRAW_COORDINATE = int(np.iinfo(np.int32).max) - 1
TEAM_COLORS = {
    0: (0, 90, 255),
    1: (255, 140, 0),
}
NEUTRAL_COLOR = (160, 255, 160)
POSSESSION_STATUSES = {"confirmed", "unknown", "lost", "unassigned"}
V3_STATES = {"OBSERVED", "INFERRED", "GRACE", "CONTESTED", "NO_POSSESSION", "UNKNOWN", "CANDIDATE", "TRANSITIONING"}


def load_observations(tracks_jsonl: Path) -> dict[int, list[dict[str, Any]]]:
    """Group JSONL observations by source frame index."""
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    manifest_path = tracks_jsonl.parent / "manifest.json"
    rejection_state = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        acceptance = manifest.get("acceptance", {})
        if acceptance.get("tracking_identity") == "rejected":
            rejection_state = "rejected"
    with tracks_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if rejection_state is not None:
                    row.setdefault("rejection_state", rejection_state)
                by_frame[int(row["frame_index"])].append(row)
    return dict(by_frame)


def load_possession_events(possession_jsonl: Path) -> dict[int, dict[str, Any]]:
    """Load authoritative possession records keyed by their exact source frame."""
    by_frame: dict[int, dict[str, Any]] = {}
    with possession_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame_index = row.get("frame_index")
            if not isinstance(frame_index, int) or isinstance(frame_index, bool):
                continue
            if frame_index in by_frame:
                raise ValueError(f"duplicate possession frame_index {frame_index}")
            by_frame[frame_index] = row
    return by_frame


def load_v3_jsonl(path: Path | None) -> dict[int, list[dict[str, Any]]]:
    """Load optional V3 rows without making legacy debug input mandatory."""
    if path is None or not path.is_file():
        return {}
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        frame = row.get("frame_index", row.get("frame_start"))
        if type(frame) is int:
            rows[frame].append(row)
    return dict(rows)


def color_for_track(raw_track_id: int) -> tuple[int, int, int]:
    """Return a stable, visible BGR color for a raw SAM track ID."""
    digest = hashlib.sha256(str(int(raw_track_id)).encode()).digest()
    return tuple(80 + byte % 176 for byte in digest[:3])


def color_for_team(team_id: Any) -> tuple[int, int, int]:
    """Return the deterministic team color, or neutral for unknown IDs."""
    return TEAM_COLORS.get(team_id, NEUTRAL_COLOR) if type(team_id) is int else NEUTRAL_COLOR


def _mask_path(mask_ref: Any, masks_dir: Path) -> Path | None:
    if not isinstance(mask_ref, str) or not mask_ref:
        return None
    root = masks_dir.resolve()
    package_root = root.parent if root.name == "masks" else root
    reference_path = Path(mask_ref)
    references = [reference_path]
    if reference_path.parts[:1] in (("masks",), ("masks_clean",)):
        # render_debug_video receives the raw masks/ directory, while refs are
        # rooted at the artifact package. Check that package root as well.
        references.append(Path("..") / reference_path)
        references.append(Path(*reference_path.parts[1:]))
    for reference in references:
        candidate = (root / reference).resolve()
        if (candidate == root or root in candidate.parents) and candidate.is_file():
            return candidate
        if candidate != package_root and package_root in candidate.parents and candidate.is_file():
            return candidate
    return None


def _load_mask(row: dict[str, Any], masks_dir: Path, shape: tuple[int, int]) -> np.ndarray | None:
    for reference in (row.get("clean_mask_ref"), row.get("mask_ref")):
        path = _mask_path(reference, masks_dir)
        if path is None or not path.is_file():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is not None and mask.ndim == 2 and mask.shape == shape:
            return mask > 0
    return None


def _bbox_xyxy_in_pixels(raw_bbox: Any, shape: tuple[int, int]) -> list[int]:
    """Convert an all-[0, 1] bbox to pixels; otherwise preserve pixel input."""
    values = [float(value) for value in raw_bbox]
    if len(values) != 4:
        raise ValueError("bbox_xyxy must contain four values")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox_xyxy values must be finite")
    if any(abs(value) > MAX_DRAW_COORDINATE for value in values):
        raise ValueError("bbox_xyxy values exceed OpenCV coordinate limits")
    if all(0.0 <= value <= 1.0 for value in values):
        height, width = shape
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    return [int(round(value)) for value in values]


def validate_artifacts(tracks_jsonl: Path, masks_dir: Path, *, allow_incomplete: bool = False) -> None:
    """Reject unavailable or explicitly incomplete returned packages."""
    if not tracks_jsonl.is_file():
        raise ValueError(f"required artifact is missing: {tracks_jsonl}")
    if not masks_dir.is_dir():
        raise ValueError(f"required artifact directory is missing: {masks_dir}")

    manifest_path = tracks_jsonl.parent / "manifest.json"
    if not manifest_path.is_file():
        if allow_incomplete:
            return
        raise ValueError(f"artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read artifact manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("invalid artifact manifest structure: expected an object")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("invalid artifact manifest structure: artifacts must be an object")
    if not isinstance(artifacts.get("tracks"), dict) or not isinstance(artifacts.get("masks"), dict):
        raise ValueError("invalid artifact manifest structure: tracks and masks are required")
    for name in ("tracks", "masks"):
        if not isinstance(artifacts[name].get("key"), str):
            raise ValueError(f"invalid artifact manifest structure: artifacts.{name}.key must be a string")

    for name in ("complete", "truncated"):
        if type(manifest.get(name)) is not bool:
            raise ValueError(f"invalid artifact manifest structure: {name} must be a boolean")
    transfer_errors = manifest.get("transfer_errors")
    if not isinstance(transfer_errors, list) or any(not isinstance(error, str) for error in transfer_errors):
        raise ValueError("invalid artifact manifest structure: transfer_errors must be a list of strings")

    reasons: list[str] = []
    if not manifest["complete"]:
        reasons.append("incomplete")
    if manifest["truncated"]:
        reasons.append("truncated")
    if transfer_errors:
        reasons.append("transfer errors")
    missing = artifacts["masks"].get("missing", [])
    if not isinstance(missing, list) or any(not isinstance(item, str) for item in missing):
        raise ValueError("invalid artifact manifest structure: masks.missing must be a list of strings")
    if missing:
        reasons.append("missing required artifacts")
    if reasons and not allow_incomplete:
        detail = ", ".join(reasons)
        raise ValueError(f"artifact package is {detail}; pass --allow-incomplete to override")


def _possession_label(event: Mapping[str, Any]) -> str:
    status = str(event.get("status", "unknown")).strip().lower()
    if status not in POSSESSION_STATUSES:
        status = "unknown"
    if status != "confirmed":
        return f"POSSESSION: {status.upper()}"
    player_id = event.get("stable_player_id")
    holder = player_id.strip() if isinstance(player_id, str) and player_id.strip() else "UNKNOWN"
    team_id = event.get("team_id")
    team = str(team_id) if team_id is not None else "UNKNOWN"
    return f"POSSESSION: {holder} TEAM: {team}"


def render_possession_overlay(
    frame: np.ndarray,
    event: Mapping[str, Any],
    stats_rows: Iterable[dict[str, Any]] = (),
) -> np.ndarray:
    """Draw an event label and, when available, its exact authoritative track."""
    rendered = frame.copy()
    label = _possession_label(event)
    color = color_for_team(event.get("team_id"))
    cv2.rectangle(rendered, (0, 0), (min(rendered.shape[1] - 1, 420), 28), (0, 0, 0), -1)
    cv2.putText(rendered, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    if str(event.get("status", "")).strip().lower() != "confirmed":
        return rendered
    track_id = event.get("player_track_id")
    if not isinstance(track_id, int) or isinstance(track_id, bool):
        return rendered
    for row in stats_rows:
        row_track_id = row.get("track_id", row.get("raw_track_id"))
        if row_track_id != track_id:
            continue
        try:
            box = _bbox_xyxy_in_pixels(row.get("bbox_xyxy", row.get("xyxy")), rendered.shape[:2])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 5)
        break
    return rendered


def render_v3_overlay(frame: np.ndarray, observation: Mapping[str, Any] | None = None,
                      transition: Mapping[str, Any] | None = None,
                      events: Iterable[Mapping[str, Any]] = ()) -> np.ndarray:
    """Render V3 diagnostics as labels; transition windows remain window facts."""
    rendered = frame.copy()
    labels: list[str] = []
    if observation:
        state = str(observation.get("state", "NO_POSSESSION"))
        state = state if state in V3_STATES else "NO_POSSESSION"
        holder = observation.get("stable_player_id") or observation.get("raw_track_id") or "NONE"
        labels.append(f"HOLDER: {holder} STATE: {state}")
        if observation.get("score") is not None or observation.get("score_margin") is not None:
            labels.append(f"SCORE: {observation.get('score', 'NA')} MARGIN: {observation.get('score_margin', 'NA')}")
        candidates = observation.get("candidates") or []
        if candidates:
            labels.append("CANDIDATES: " + ",".join(str(c.get("score", "NA")) for c in candidates))
            contributions = [c.get("contributions", {}) for c in candidates if c.get("contributions")]
            rejected = [reason for c in candidates for reason in c.get("rejected_reasons", [])]
            if contributions:
                labels.append("CONTRIBUTIONS: " + str(contributions))
            if rejected:
                labels.append("REJECTED: " + ",".join(map(str, rejected)))
        if observation.get("profile") or observation.get("thresholds"):
            labels.append(f"THRESHOLDS: {observation.get('profile', 'NA')} {observation.get('thresholds', {})}")
        provenance = observation.get("provenance", {})
        if provenance.get("frame_indices"):
            labels.append(f"PROVENANCE: {provenance['frame_indices']}")
        if provenance.get("reason_codes"):
            labels.append("REASONS: " + ",".join(map(str, provenance["reason_codes"])))
    if transition:
        primitives = "/".join(str(value).replace("BALL_", "") for value in transition.get("primitive_types", []))
        labels.append(f"TRANSITION: {transition.get('source_holder') or 'NONE'} -> {transition.get('destination_holder') or 'NONE'} {primitives}")
        labels.append(f"TRANSITION SCORE: {transition.get('score', 'NA')} MARGIN: {transition.get('score_margin', 'NA')}")
        if transition.get("frame_start") is not None or transition.get("frame_end") is not None:
            labels.append(f"WINDOW: {transition.get('frame_start', 'NA')}-{transition.get('frame_end', 'NA')} "
                          f"{transition.get('timestamp_start', 'NA')}-{transition.get('timestamp_end', 'NA')}")
        sources = transition.get("provenance", {}).get("source_records", [])
        if sources:
            labels.append("SOURCE: " + ",".join(map(str, sources)))
    for event in events:
        labels.append(f"EVENT: {event.get('event_type', 'UNKNOWN')}")
        if event.get("frame_start") is not None or event.get("frame_end") is not None:
            labels.append(f"EVENT WINDOW: {event.get('frame_start', 'NA')}-{event.get('frame_end', 'NA')}")
        provenance = event.get("provenance", {})
        if provenance.get("source_records") or provenance.get("frame_indices"):
            labels.append("EVENT PROVENANCE: " + ",".join(map(str, provenance.get("source_records", ()))))
        if event.get("reason_codes"):
            labels.append("EVENT REASONS: " + ",".join(map(str, event["reason_codes"])))
    for offset, label in enumerate(labels):
        cv2.putText(rendered, label, (6, min(rendered.shape[0] - 4, 20 + offset * 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return rendered


def render_frame(
    frame: np.ndarray,
    rows: Iterable[dict[str, Any]],
    masks_dir: Path,
    *,
    possession_event: Mapping[str, Any] | None = None,
    possession_rows: Iterable[dict[str, Any]] = (),
    v3_observation: Mapping[str, Any] | None = None,
    v3_transition: Mapping[str, Any] | None = None,
    v3_events: Iterable[Mapping[str, Any]] = (),
) -> np.ndarray:
    """Draw masks and provenance labels without running identity logic."""
    rendered = frame.copy()
    for row in rows:
        try:
            raw_id = int(row["raw_track_id"])
            box = _bbox_xyxy_in_pixels(row["bbox_xyxy"], rendered.shape[:2])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

        color = color_for_team(row.get("team_id"))
        mask = _load_mask(row, masks_dir, rendered.shape[:2])
        if mask is not None:
            overlay = rendered.copy()
            overlay[mask] = color
            rendered = cv2.addWeighted(overlay, 0.35, rendered, 0.65, 0)

        x1, y1, x2, y2 = box
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            rendered,
            f"RAW: {raw_id}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            2,
            cv2.LINE_AA,
        )
        labels = []
        if row.get("bbox_corrected"):
            labels.append("TEMP_FIX")
        elif row.get("bbox_outlier"):
            labels.append("OUTLIER?")
        if row.get("stable_player_id"):
            labels.append(f"PLAYER: {row['stable_player_id']}")
        if row.get("team_id") is not None:
            labels.append(f"TEAM: {row['team_id']}")
        if row.get("rejection_state"):
            labels.append(f"STATE: {row['rejection_state']}")
        for offset, label in enumerate(labels, start=1):
            cv2.putText(
                rendered,
                label,
                (x1, max(12, y1 - 4 - 16 * offset)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                2,
                cv2.LINE_AA,
            )
    v3_events = list(v3_events)
    if possession_event is not None:
        rendered = render_possession_overlay(rendered, possession_event, possession_rows)
    if v3_observation is not None or v3_transition is not None or v3_events:
        rendered = render_v3_overlay(rendered, v3_observation, v3_transition, v3_events)
    return rendered


def render_debug_video(
    video_path: Path,
    tracks_jsonl: Path,
    masks_dir: Path,
    output_path: Path,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    allow_incomplete: bool = False,
    possession_jsonl: Path | None = None,
    stats_tracks_jsonl: Path | None = None,
    v3_dir: Path | None = None,
) -> dict[str, int | None]:
    """Render the selected inclusive source-frame range to an MP4."""
    if start_frame is not None and start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame is not None and end_frame < 0:
        raise ValueError("end_frame must be non-negative")
    if start_frame is not None and end_frame is not None and end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")

    validate_artifacts(tracks_jsonl, masks_dir, allow_incomplete=allow_incomplete)
    if (possession_jsonl is None) != (stats_tracks_jsonl is None):
        raise ValueError("possession_jsonl and stats_tracks_jsonl must be provided together")
    if possession_jsonl is not None and not possession_jsonl.is_file():
        raise ValueError(f"possession artifact is missing: {possession_jsonl}")
    if stats_tracks_jsonl is not None and not stats_tracks_jsonl.is_file():
        raise ValueError(f"stats tracks artifact is missing: {stats_tracks_jsonl}")
    observations = load_observations(tracks_jsonl)
    possession_events = load_possession_events(possession_jsonl) if possession_jsonl else {}
    stats_observations = load_observations(stats_tracks_jsonl) if stats_tracks_jsonl else {}
    v3_observations = load_v3_jsonl(v3_dir / "possession_v3.jsonl") if v3_dir else {}
    v3_transitions = load_v3_jsonl(v3_dir / "ball_transitions_v3.jsonl") if v3_dir else {}
    v3_events = load_v3_jsonl(v3_dir / "possession_events_v3.jsonl") if v3_dir else {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    first_written: int | None = None
    last_written: int | None = None
    frames_written = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if start_frame is not None and frame_index < start_frame:
                frame_index += 1
                continue
            if end_frame is not None and frame_index > end_frame:
                break
            writer.write(render_frame(
                frame,
                observations.get(frame_index, ()),
                masks_dir,
                possession_event=possession_events.get(frame_index),
                possession_rows=stats_observations.get(frame_index, ()),
                v3_observation=(v3_observations.get(frame_index) or [None])[0],
                v3_transition=(v3_transitions.get(frame_index) or [None])[0],
                v3_events=v3_events.get(frame_index, ()),
            ))
            first_written = frame_index if first_written is None else first_written
            last_written = frame_index
            frames_written += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    return {"frames_written": frames_written, "first_frame": first_written, "last_frame": last_written}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SAM3.1 raw tracks and masks locally.")
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--tracks-jsonl", type=Path)
    parser.add_argument("--masks-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path, help="Directory containing tracks.jsonl and masks/")
    parser.add_argument("--stats-dir", type=Path, help="V2 bundle stats/ directory containing possession and tracks JSONL")
    parser.add_argument("--possession-jsonl", type=Path)
    parser.add_argument("--stats-tracks-jsonl", type=Path)
    parser.add_argument("--v3-dir", type=Path, help="Optional directory containing versioned V3 artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Render despite an incomplete/truncated manifest (unsafe; default rejects it)",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    tracks_jsonl = args.tracks_jsonl or (artifact_dir / "tracks.jsonl" if artifact_dir else None)
    masks_dir = args.masks_dir or (artifact_dir / "masks" if artifact_dir else None)
    if tracks_jsonl is None or masks_dir is None:
        parser.error("provide --artifact-dir or both --tracks-jsonl and --masks-dir")
    explicit_stats_paths = (args.possession_jsonl is not None, args.stats_tracks_jsonl is not None)
    if args.stats_dir and any(explicit_stats_paths):
        if not all(explicit_stats_paths):
            parser.error("--stats-dir cannot be combined with only one explicit stats path; provide both explicit paths or neither")
        parser.error("--stats-dir cannot be combined with explicit stats paths; use one stats source mode")
    if not args.stats_dir and any(explicit_stats_paths) and not all(explicit_stats_paths):
        parser.error("provide both --possession-jsonl and --stats-tracks-jsonl")
    possession_jsonl = args.possession_jsonl or (args.stats_dir / "possession_events.jsonl" if args.stats_dir else None)
    stats_tracks_jsonl = args.stats_tracks_jsonl or (args.stats_dir / "tracks.jsonl" if args.stats_dir else None)
    if (possession_jsonl is None) != (stats_tracks_jsonl is None):
        parser.error("provide --stats-dir or both --possession-jsonl and --stats-tracks-jsonl")
    stats = render_debug_video(
        args.video_path,
        tracks_jsonl,
        masks_dir,
        args.output,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        allow_incomplete=args.allow_incomplete,
        possession_jsonl=possession_jsonl,
        stats_tracks_jsonl=stats_tracks_jsonl,
        v3_dir=args.v3_dir,
    )
    print(f"Wrote {stats['frames_written']} frames to {args.output}")


if __name__ == "__main__":
    main()
