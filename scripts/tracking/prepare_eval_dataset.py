#!/usr/bin/env python3
"""Prepare a review-ready tracking evaluation dataset from local source videos.

Local and free: probes videos, cuts human-chosen windows with OpenCV, writes
contact sheets, per-run annotated-output directories, `evaluation_manifest.json`
and a template `tracking_review.jsonl`. It never runs the pipeline and never
decides semantic labels (scenario/difficulty come from the human; every review
field stays `pending`).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

MANIFEST_NAME = "evaluation_manifest.json"
REVIEW_NAME = "tracking_review.jsonl"
SCHEMA_VERSION = 1
MIN_CLIP_SECONDS = 5.0
SCENARIOS = ("crossing", "partial_occlusion", "full_occlusion", "camera_motion", "other")
DIFFICULTIES = ("easy", "medium", "hard")
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
CLIP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_RUN_LABELS = ("baseline", "candidate")
CLIP_FOURCC = "mp4v"
REQUIRED_CLIP_FIELDS = (
    "clip", "source_video", "path", "scenario", "difficulty", "start_frame", "end_frame",
    "duration_seconds", "fps", "width", "height", "configuration", "source_commit", "review_status",
)
SEGMENT_WINDOW = 150
SEGMENT_OVERLAP = 20
POSSESSION_ENGINE = "v5"
PENDING_FIELDS = (
    "scenario_present", "id_switch", "full_loss", "recovery", "stable_identity_involved",
)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int
    fourcc: str

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class Window:
    clip: str
    start_frame: int
    end_frame: int  # exclusive
    scenario: str
    difficulty: str
    source: str | None = None


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        info = VideoInfo(
            path=path,
            fps=fps,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fourcc=_fourcc_to_str(capture.get(cv2.CAP_PROP_FOURCC)),
        )
    finally:
        capture.release()
    if info.fps <= 0 or info.frame_count <= 0 or info.width <= 0 or info.height <= 0:
        raise ValueError(f"Video has unusable metadata (fps/frames/size): {path}")
    return info


def discover_videos(sources: list[Path], source_dir: Path | None) -> list[Path]:
    videos = list(sources)
    if source_dir is not None:
        if not source_dir.is_dir():
            raise ValueError(f"Source directory not found: {source_dir}")
        videos.extend(
            sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
        )
    for video in videos:
        if not video.is_file():
            raise ValueError(f"Source video not found: {video}")
    return videos


def parse_window(spec: str) -> Window:
    """Parse `clip:start_frame:end_frame:scenario:difficulty[:source]` (end exclusive)."""
    parts = spec.split(":")
    if len(parts) not in (5, 6):
        raise ValueError(
            f"Invalid --window '{spec}': expected clipname:start_frame:end_frame:scenario:difficulty[:source]"
        )
    clip, start_raw, end_raw, scenario, difficulty = parts[:5]
    source = parts[5] if len(parts) == 6 and parts[5] else None
    if not CLIP_NAME_RE.match(clip):
        raise ValueError(f"Invalid clip name '{clip}': use letters, digits, '.', '_' or '-'")
    try:
        start, end = int(start_raw), int(end_raw)
    except ValueError as error:
        raise ValueError(f"Invalid --window '{spec}': start/end frames must be integers") from error
    if start < 0 or end <= start:
        raise ValueError(f"Invalid window for '{clip}': need 0 <= start_frame < end_frame (got {start}..{end})")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported scenario '{scenario}' for '{clip}'. Supported: {', '.join(SCENARIOS)}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty '{difficulty}' for '{clip}'. Supported: {', '.join(DIFFICULTIES)}")
    return Window(clip, start, end, scenario, difficulty, source)


def validate_window(window: Window, info: VideoInfo) -> None:
    if window.end_frame > info.frame_count:
        raise ValueError(
            f"Window '{window.clip}' ends at frame {window.end_frame} but {info.path.name} "
            f"has only {info.frame_count} frames"
        )
    duration = (window.end_frame - window.start_frame) / info.fps
    if duration < MIN_CLIP_SECONDS:
        raise ValueError(
            f"Window '{window.clip}' is {duration:.2f}s; clips must be at least {MIN_CLIP_SECONDS:g}s "
            f"({window.end_frame - window.start_frame} frames at {info.fps:g} fps)"
        )


def resolve_window_source(window: Window, videos: list[Path]) -> Path:
    if window.source is None:
        if len(videos) == 1:
            return videos[0]
        raise ValueError(
            f"Window '{window.clip}' needs a trailing :source (video stem or name) because "
            f"{len(videos)} source videos were given"
        )
    matches = [v for v in videos if window.source in (v.name, v.stem, str(v))]
    if len(matches) != 1:
        raise ValueError(f"Window '{window.clip}': source '{window.source}' matched {len(matches)} videos")
    return matches[0]


def cut_clip(info: VideoInfo, window: Window, out_path: Path) -> int:
    """Cut [start_frame, end_frame) with cv2.VideoWriter (re-encodes). Returns frames written."""
    capture = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*CLIP_FOURCC), info.fps, (info.width, info.height)
    )
    written = 0
    try:
        if not writer.isOpened():
            raise ValueError(f"Cannot open video writer for {out_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, window.start_frame)
        for _ in range(window.end_frame - window.start_frame):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    expected = window.end_frame - window.start_frame
    if written != expected:
        raise ValueError(f"Clip '{window.clip}': wrote {written} of {expected} frames from {info.path}")
    return written


def make_contact_sheet(
    video_path: Path, out_path: Path, *, count: int = 12, columns: int = 4,
    thumb_width: int = 320, frame_offset: int = 0,
) -> list[int]:
    """Write a grid of evenly spaced frames. Labels are source-frame indices (frame_offset + clip index)."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"Cannot read frames from {video_path}")
        count = max(1, min(count, total))
        indices = sorted({int(round(i * (total - 1) / max(count - 1, 1))) for i in range(count)})
        thumbs = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            height = max(1, int(frame.shape[0] * thumb_width / frame.shape[1]))
            thumb = cv2.resize(frame, (thumb_width, height))
            label = f"frame {frame_offset + index}"
            cv2.putText(thumb, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(thumb, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(thumb)
    finally:
        capture.release()
    if not thumbs:
        raise ValueError(f"No frames could be read from {video_path}")
    columns = max(1, min(columns, len(thumbs)))
    rows = []
    for start in range(0, len(thumbs), columns):
        row = thumbs[start:start + columns]
        row += [row[0] * 0] * (columns - len(row))
        rows.append(cv2.hconcat(row))
    cv2.imwrite(str(out_path), cv2.vconcat(rows))
    return indices


def get_source_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_job_id(clip: str, label: str) -> str:
    # Deterministic; modal_app_v2._job_path only rejects empty ids and path separators, which CLIP_NAME_RE excludes.
    return f"eval-{clip}-{label}"


def build_suggested_run_command(clip_path: Path, job_id: str, frame_count: int) -> str:
    """Copy-pasteable staged Modal command. Never executed by this script (it uses GPU credits)."""
    return " ".join([
        "modal", "run", "modal_app_v2.py", "--staged",
        "--job-id", shlex.quote(job_id),
        "--video-path", shlex.quote(str(clip_path)),
        "--frame-start", "0",
        # Modal's --frame-end is inclusive; the manifest's end_frame is exclusive.
        "--frame-end", str(frame_count - 1),
        "--segmented", "--segment-window", str(SEGMENT_WINDOW), "--segment-overlap", str(SEGMENT_OVERLAP),
        "--possession-engine", POSSESSION_ENGINE,
    ])


def create_run_dirs(dataset_dir: Path, label: str, clips: list[str]) -> Path:
    """Create runs/<label>/<clip>/annotated. Refuses to touch an existing run dir (immutable)."""
    if not CLIP_NAME_RE.match(label):
        raise ValueError(f"Invalid run label '{label}'")
    run_dir = dataset_dir / "runs" / label
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists, refusing to overwrite: {run_dir}")
    for clip in clips:
        (run_dir / clip / "annotated").mkdir(parents=True)
    (run_dir / "RUN_STATUS.json").write_text(
        json.dumps({"label": label, "status": "not_executed", "clips": clips}, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path.name} in {path.parent.name}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} in {path.parent.name} is not a JSON object")
    return payload


def _stats_context(job: Path, job_manifest: dict) -> tuple[str, str]:
    """Best-effort (stats generation id, possession engine); 'unknown' when absent."""
    generation = job_manifest.get("stats_generation")
    if not generation and (job / "stats" / "current.json").is_file():
        generation = _read_json(job / "stats" / "current.json").get("generation")
    engine = "unknown"
    results = job / "stats" / "generations" / str(generation) / "results.json"
    if generation and results.is_file():
        config = _read_json(results).get("config")
        if isinstance(config, dict):
            engine = config.get("possession_engine") or "unknown"
    return generation or "unknown", engine


def inspect_job(job: Path, entry: dict) -> dict:
    """Validate a downloaded Modal job folder against a manifest clip; masks are never read."""
    if not (job / "job.json").is_file():
        raise ValueError(f"{job.name}: no job.json (expected a downloaded Modal job folder)")
    job_manifest = _read_json(job / "job.json")
    current = _read_json(job / "tracking" / "current.json")
    generation = current.get("generation")
    if not generation:
        raise ValueError(f"{job.name}: tracking/current.json has no generation")
    if job_manifest.get("tracking_generation") not in (None, generation):
        raise ValueError(f"{job.name}: job.json and tracking/current.json disagree on the tracking generation")
    gen_dir = job / "tracking" / "generations" / generation
    manifest = _read_json(gen_dir / "manifest.json")
    detail = _read_json(gen_dir / "tracking-manifest.json") if (gen_dir / "tracking-manifest.json").is_file() else {}
    if manifest.get("complete") is not True or detail.get("complete", True) is not True:
        raise ValueError(f"{job.name}: tracking generation {generation[:12]} is not complete")
    if detail.get("truncated"):
        raise ValueError(f"{job.name}: tracking generation {generation[:12]} is truncated")
    bounds = manifest.get("frame_bounds")
    if not (isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(b, int) for b in bounds)):
        raise ValueError(f"{job.name}: tracking manifest has no usable frame_bounds")
    lo, hi = bounds  # inclusive
    start, end = entry["start_frame"], entry["end_frame"]
    if start < lo or end - 1 > hi:
        raise ValueError(
            f"{job.name}: window [{start}, {end}) is outside the tracked frames [{lo}, {hi}] (inclusive)"
        )
    video = detail.get("video") if isinstance(detail.get("video"), dict) else {}
    if video.get("frame_count") not in (None, entry["source_frame_count"]):
        raise ValueError(
            f"{job.name}: job tracked a {video['frame_count']}-frame video but the source clip has "
            f"{entry['source_frame_count']} frames; frame indices would not line up"
        )
    tracks = gen_dir / "tracks.jsonl"
    if not tracks.is_file():
        raise ValueError(f"{job.name}: tracking generation has no tracks.jsonl")
    stats_generation, engine = _stats_context(job, job_manifest)
    config = detail.get("config") if isinstance(detail.get("config"), dict) else {}
    pipeline_config = {
        key: config.get(src, "unknown") for key, src in (
            ("prompt", "prompt"), ("segmented", "sam31_segmented"),
            ("segment_window", "sam31_segment_window"), ("segment_overlap", "sam31_segment_overlap"),
            ("segment_seam_min_score", "sam31_seam_min_score"),
            ("segment_seam_ambiguity_margin", "sam31_seam_ambiguity_margin"),
            ("sam31_max_num_objects", "sam31_max_num_objects"),
        )
    }
    pipeline_config["possession_engine"] = engine
    pipeline_config["tracked_frame_bounds_inclusive"] = [lo, hi]
    return {
        "tracks_path": tracks,
        "job_id": job_manifest.get("job_id") or job.name,
        "tracking_generation": generation,
        "stats_generation": stats_generation,
        "input_fingerprint": manifest.get("input_fingerprint") or job_manifest.get("input_fingerprint") or "unknown",
        "pipeline_config": pipeline_config,
    }


def slice_tracks(tracks_path: Path, start: int, end: int, out_path: Path) -> tuple[int, int]:
    """Write rows with start <= frame_index < end, re-indexed to the clip. Returns (rows, distinct frames)."""
    rows, frames = 0, set()
    with tracks_path.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                frame = row["frame_index"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"{tracks_path.parent.name}/tracks.jsonl line {number}: bad row ({error})") from error
            if not isinstance(frame, int) or isinstance(frame, bool):
                raise ValueError(f"{tracks_path.parent.name}/tracks.jsonl line {number}: frame_index must be an int")
            if start <= frame < end:
                row["frame_index"] = frame - start
                dst.write(json.dumps(row) + "\n")
                rows += 1
                frames.add(frame)
    return rows, len(frames)


def attach_runs(
    dataset_dir: Path, attachments: dict[str, Path], *, label: str = "baseline", repo_root: Path,
) -> dict[str, dict]:
    """Attach existing downloaded job outputs as the `label` run (free; no pipeline execution)."""
    manifest = _read_json(dataset_dir / MANIFEST_NAME)
    entries = {e["clip"]: e for e in manifest.get("clips", [])}
    if label not in manifest.get("run_labels", []):
        raise ValueError(f"Run label '{label}' is not in this dataset ({manifest.get('run_labels')})")
    unknown = set(attachments) - set(entries)
    if unknown:
        raise ValueError(f"--attach-run refers to unknown clips: {', '.join(sorted(unknown))}")
    run_dir = dataset_dir / "runs" / label
    plans = {}
    for clip, job in attachments.items():  # validate everything before writing anything
        target = run_dir / clip
        for name in ("tracks.jsonl", "run_metadata.json"):
            if (target / name).exists():
                raise FileExistsError(f"Run output already exists, refusing to overwrite: {target / name}")
        if not job.is_dir():
            raise ValueError(f"{clip}: job folder not found: {job.name}")
        plans[clip] = inspect_job(job, entries[clip])
    commit = get_source_commit(repo_root)
    results = {}
    for clip, plan in plans.items():
        entry, target = entries[clip], run_dir / clip
        target.mkdir(parents=True, exist_ok=True)
        rows, frames = slice_tracks(plan.pop("tracks_path"), entry["start_frame"], entry["end_frame"],
                                    target / "tracks.jsonl")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "clip": clip,
            "label": label,
            "kind": "attached_existing_run",
            "source_job": attachments[clip].name,  # basename only; no local paths in committed provenance
            **plan,
            "source_commit": "unknown",  # job folders do not record the code revision
            "tooling_commit": commit,
            "window": {"start_frame": entry["start_frame"], "end_frame": entry["end_frame"]},
            "frame_offset": entry["start_frame"],
            "frame_index_note": "frame_index in tracks.jsonl = source frame - frame_offset; "
                                "timestamp_seconds is unchanged (source-relative)",
            "rows": rows,
            "frames_with_rows": frames,
            "clip_frame_count": entry["frame_count"],
            "warning": (
                "Tracks come from tracking the FULL source clip, so segment seams and identities differ from "
                "a run on the cut window. This is a baseline for the window, not a window-specific rerun."
            ),
        }
        (target / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        results[clip] = metadata
    status_path = run_dir / "RUN_STATUS.json"
    status = _read_json(status_path) if status_path.is_file() else {"label": label, "clips": sorted(entries)}
    status.setdefault("attached", {}).update({c: m["job_id"] for c, m in results.items()})
    status["status"] = "attached" if set(status["attached"]) >= set(status.get("clips", [])) else "partially_attached"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return results


def build_review_template(manifest: dict) -> list[dict]:
    """One `clip_review` record per clip. Every semantic field is 'pending' for a human."""
    records = []
    for entry in manifest["clips"]:
        record = {
            "record_type": "clip_review",
            "clip": entry["clip"],
            "declared_scenario": entry["scenario"],
        }
        record.update({field: "pending" for field in PENDING_FIELDS})
        record["stable_player_id"] = None
        record["notes"] = ""
        records.append(record)
    return records


def validate_manifest(manifest: object) -> list[str]:
    """Return a list of problems (empty when valid)."""
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    errors: list[str] = []
    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        return ["manifest.clips must be a non-empty list"]
    seen: set[str] = set()
    for i, entry in enumerate(clips):
        where = f"clips[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{where}: must be an object")
            continue
        missing = [f for f in REQUIRED_CLIP_FIELDS if f not in entry]
        if missing:
            errors.append(f"{where}: missing fields {', '.join(missing)}")
            continue
        where = f"clips[{i}] ({entry['clip']})"
        if entry["clip"] in seen:
            errors.append(f"{where}: duplicate clip name")
        seen.add(entry["clip"])
        if entry["scenario"] not in SCENARIOS:
            errors.append(f"{where}: unsupported scenario '{entry['scenario']}'")
        if entry["difficulty"] not in DIFFICULTIES:
            errors.append(f"{where}: unsupported difficulty '{entry['difficulty']}'")
        if entry["review_status"] not in ("pending", "confirmed"):
            errors.append(f"{where}: review_status must be 'pending' or 'confirmed'")
        try:
            if not entry["end_frame"] > entry["start_frame"] >= 0:
                errors.append(f"{where}: need 0 <= start_frame < end_frame")
            if entry["duration_seconds"] < MIN_CLIP_SECONDS:
                errors.append(f"{where}: duration_seconds {entry['duration_seconds']} < {MIN_CLIP_SECONDS:g}")
            if not (entry["fps"] > 0 and entry["width"] > 0 and entry["height"] > 0):
                errors.append(f"{where}: fps/width/height must be positive")
        except TypeError:
            errors.append(f"{where}: numeric fields have the wrong type")
        config = entry["configuration"]
        if not isinstance(config, dict) or not {"target_fps", "frame_stride"} <= config.keys():
            errors.append(f"{where}: configuration needs target_fps and frame_stride")
        if not isinstance(entry["source_commit"], str) or not entry["source_commit"]:
            errors.append(f"{where}: source_commit must be a non-empty string")
    return errors


def prepare_dataset(
    *, videos: list[Path], windows: list[Window], output_dir: Path, target_fps: int,
    frame_stride: int, run_labels: tuple[str, ...], notes: dict[str, str],
    sheet_frames: int, repo_root: Path,
) -> dict:
    if not windows:
        raise ValueError("At least one --window is required")
    if (output_dir / MANIFEST_NAME).exists():
        raise FileExistsError(f"Dataset already exists, refusing to overwrite: {output_dir / MANIFEST_NAME}")
    names = [w.clip for w in windows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate clip names in --window")
    unknown = set(notes) - set(names)
    if unknown:
        raise ValueError(f"--note refers to unknown clips: {', '.join(sorted(unknown))}")
    infos = {v: probe_video(v) for v in videos}
    resolved = []
    for window in windows:  # validate everything before writing anything
        source = resolve_window_source(window, videos)
        validate_window(window, infos[source])
        resolved.append((window, infos[source]))

    clips_dir, sheets_dir = output_dir / "clips", output_dir / "contact_sheets"
    clips_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(exist_ok=True)
    commit = get_source_commit(repo_root)
    entries = []
    for window, info in resolved:
        clip_path = clips_dir / f"{window.clip}.mp4"
        frames = cut_clip(info, window, clip_path)
        sheet_path = sheets_dir / f"{window.clip}.jpg"
        sheet_frames_idx = make_contact_sheet(
            clip_path, sheet_path, count=sheet_frames, frame_offset=window.start_frame
        )
        job_ids = {label: build_job_id(window.clip, label) for label in run_labels}
        commands = {
            label: build_suggested_run_command(clip_path.resolve(), job_id, frames)
            for label, job_id in job_ids.items()
        }
        entries.append({
            "clip": window.clip,
            "source_video": str(info.path),
            "path": f"clips/{window.clip}.mp4",
            "scenario": window.scenario,
            "difficulty": window.difficulty,
            "start_frame": window.start_frame,
            "end_frame": window.end_frame,
            "duration_seconds": round(frames / info.fps, 3),
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
            "configuration": {"target_fps": target_fps, "frame_stride": frame_stride},
            "source_commit": commit,
            "review_status": "pending",
            "frame_count": frames,
            "source_frame_count": info.frame_count,
            "source_duration_seconds": round(info.duration_seconds, 3),
            "source_fourcc": info.fourcc,
            "encoding": {
                "reencoded": True,
                "tool": "cv2.VideoWriter",
                "opencv_version": cv2.__version__,
                "fourcc": CLIP_FOURCC,
                "container": "mp4",
            },
            "contact_sheet": f"contact_sheets/{window.clip}.jpg",
            "contact_sheet_frames": [window.start_frame + i for i in sheet_frames_idx],
            "notes": notes.get(window.clip, ""),
            "command_executed": False,
            "modal_frame_end_inclusive": frames - 1,
            "suggested_job_ids": job_ids,
            "suggested_run_commands": commands,
            "suggested_run_command": commands[run_labels[0]] if run_labels else None,
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_labels": list(run_labels),
        "suggested_run_command_note": (
            "NOT EXECUTED by this tool; running it costs GPU credits. One staged command per run label "
            "(suggested_run_commands); the job id is deterministic. Modal's --frame-end is inclusive "
            "(modal_frame_end_inclusive) while end_frame is exclusive. target_fps/frame_stride are recorded "
            "intent: modal_app_v2.py does not take them as flags."
        ),
        "clips": entries,
    }
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Generated manifest is invalid: " + "; ".join(errors))
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    review_path = output_dir / REVIEW_NAME
    review_path.write_text(
        "".join(json.dumps(r) + "\n" for r in build_review_template(manifest)), encoding="utf-8"
    )
    for label in run_labels:
        create_run_dirs(output_dir, label, names)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a tracking evaluation dataset (clips, contact sheets, manifest, review template). "
        "Local only: never runs the pipeline."
    )
    parser.add_argument("--output-dir", type=Path, help="Dataset directory to create.")
    parser.add_argument("--source", type=Path, action="append", default=[], help="Source video (repeatable).")
    parser.add_argument("--source-dir", type=Path, help="Directory of source videos.")
    parser.add_argument(
        "--window", action="append", default=[],
        help="clipname:start_frame:end_frame:scenario:difficulty[:source]; end exclusive, >= 5 s. "
        f"scenario in {', '.join(SCENARIOS)}; difficulty in {', '.join(DIFFICULTIES)}.",
    )
    parser.add_argument("--note", action="append", default=[], help="clipname=free text (repeatable).")
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--sheet-frames", type=int, default=12, help="Frames per contact sheet.")
    parser.add_argument("--run-label", action="append", help="Run dir label (repeatable). Default: baseline, candidate.")
    parser.add_argument(
        "--attach-run", action="append", default=[], metavar="CLIP=JOB_DIR",
        help="Attach an existing downloaded Modal job folder as the baseline run for CLIP (repeatable). "
        "Without --window, attaches to the dataset already in --output-dir.",
    )
    parser.add_argument("--attach-label", default="baseline", help="Run label --attach-run writes to.")
    parser.add_argument("--validate", type=Path, metavar="MANIFEST", help="Validate a manifest and exit.")
    return parser


def print_attached(dataset_dir: Path, label: str, attached: dict[str, dict]) -> None:
    for clip, meta in attached.items():
        print(f"Attached {meta['source_job']} as {label} run for {clip}: {meta['rows']} rows over "
              f"{meta['frames_with_rows']} frames -> {dataset_dir / 'runs' / label / clip}")
    print("warning: attached tracks come from the FULL source clip, not a rerun on the cut window.")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.validate is not None:
        try:
            errors = validate_manifest(json.loads(args.validate.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            errors = [f"cannot read manifest: {error}"]
        for error in errors:
            print(f"INVALID: {error}", file=sys.stderr)
        if not errors:
            print(f"Manifest OK: {args.validate}")
        return 1 if errors else 0
    if args.output_dir is None:
        print("error: --output-dir is required", file=sys.stderr)
        return 2
    repo_root = Path(__file__).resolve().parents[2]
    try:
        attachments = {}
        for item in args.attach_run:
            clip, sep, job = item.partition("=")
            if not sep or not job or clip in attachments:
                raise ValueError(f"Invalid or repeated --attach-run '{item}': expected CLIP=JOB_DIR")
            attachments[clip] = Path(job)
        if attachments and not args.window:
            attached = attach_runs(args.output_dir, attachments, label=args.attach_label, repo_root=repo_root)
            print_attached(args.output_dir, args.attach_label, attached)
            return 0
        notes = {}
        for item in args.note:
            name, sep, text = item.partition("=")
            if not sep:
                raise ValueError(f"Invalid --note '{item}': expected clipname=text")
            notes[name] = text
        manifest = prepare_dataset(
            videos=discover_videos(args.source, args.source_dir),
            windows=[parse_window(w) for w in args.window],
            output_dir=args.output_dir,
            target_fps=args.target_fps,
            frame_stride=args.frame_stride,
            run_labels=tuple(args.run_label or DEFAULT_RUN_LABELS),
            notes=notes,
            sheet_frames=args.sheet_frames,
            repo_root=repo_root,
        )
        attached = (
            attach_runs(args.output_dir, attachments, label=args.attach_label, repo_root=repo_root)
            if attachments else {}
        )
    except (ValueError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Dataset written to {args.output_dir}")
    print(f"Manifest: {args.output_dir / MANIFEST_NAME}")
    print(f"Review template: {args.output_dir / REVIEW_NAME} (all semantic fields are 'pending')")
    if len(manifest["clips"]) < 3:
        print(f"note: only {len(manifest['clips'])} clip(s); the evaluation set should have 3-5.")
    for entry in manifest["clips"]:
        print(f"- {entry['clip']}: contact sheet {args.output_dir / entry['contact_sheet']}")
    print("\nSuggested pipeline commands (NOT EXECUTED; these use GPU credits):")
    for entry in manifest["clips"]:
        print(f"  {entry['suggested_run_command']}")
    if attached:
        print_attached(args.output_dir, args.attach_label, attached)
    return 0


if __name__ == "__main__":
    sys.exit(main())
