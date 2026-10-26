#!/usr/bin/env python3
"""Small, read-only TrackNet versus hosted-detector visual experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.ball import bbox_center
from scripts.ball.candidates import extract_candidates
from scripts.ball.debug import overlay_heatmap
from scripts.ball.tracknet import build_detector, inspect_tracknet_checkpoint, padded_window
from scripts.test_ball_detection import infer_frame, roboflow_client, validated_ball_boxes


DEFAULT_MODEL = Path("models/tracknet-finetune/runs/basketball_v001/checkpoints/TrackNet_best.pt")
DEFAULT_CLIPS = ("test/match20", "train/match01", "train/match03")


def resolve_clip_root(root: Path) -> tuple[Path, Path, Path]:
    """Return (clip root, ordered frame directory, labels csv)."""
    if (root / "frame").is_dir():
        frame_dirs = sorted((root / "frame").glob("*"))
        if len(frame_dirs) != 1:
            raise ValueError(f"{root}: expected one frame/<clip> directory")
        frame_dir = frame_dirs[0]
        labels = root / "csv" / f"{frame_dir.name}_ball.csv"
    elif root.is_dir() and root.name.startswith("clip_"):
        frame_dir = root
        root = root.parent.parent
        labels = root / "csv" / f"{frame_dir.name}_ball.csv"
    else:
        raise ValueError(f"{root}: expected a processed match root or frame/clip directory")
    if not frame_dir.is_dir() or not labels.is_file():
        raise ValueError(f"{root}: missing frame directory or labels CSV")
    return root, frame_dir, labels


def ordered_frame_paths(frame_dir: Path) -> list[Path]:
    paths = sorted(frame_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not paths:
        raise ValueError(f"{frame_dir}: no PNG frames found")
    indices = [int(path.stem) for path in paths]
    if indices != list(range(indices[-1] + 1)):
        raise ValueError(f"{frame_dir}: frames must be contiguous from 0; found a gap")
    return paths


def load_labels(path: Path) -> dict[int, tuple[int, float, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"Frame", "Visibility", "X", "Y"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: required columns are Frame,Visibility,X,Y")
        labels = {}
        for row in reader:
            frame = int(row["Frame"])
            if frame in labels:
                raise ValueError(f"{path}: duplicate label for frame {frame}")
            visibility = int(row["Visibility"])
            if visibility not in (0, 1):
                raise ValueError(f"{path}: Visibility must be 0 or 1")
            labels[frame] = (visibility, float(row["X"]), float(row["Y"]))
    return labels


def validate_frame_labels(paths: list[Path], labels: dict[int, tuple[int, float, float]]) -> None:
    expected = set(range(len(paths)))
    if set(labels) != expected:
        missing = sorted(expected - set(labels))
        extra = sorted(set(labels) - expected)
        raise ValueError(f"frame/label mismatch: missing={missing} extra={extra}")


def select_frame_indices(frame_count: int, cap: int | None) -> list[int]:
    if cap is None or cap >= frame_count:
        return list(range(frame_count))
    if cap < 1:
        raise ValueError("frame cap must be positive")
    return sorted({index * frame_count // cap for index in range(cap)})


def clip_identifier(root: Path) -> str:
    """Use split + match, rather than the internal ``frame`` directory name."""
    return f"{root.parent.name}_{root.name}" if root.parent.name in {"train", "val", "validation", "test"} else root.name


def comparison_frame_records(paths: list[Path], labels: dict[int, tuple[int, float, float]], selected: list[int]) -> list[dict[str, object]]:
    return [
        {"frame_index": index, "path": str(paths[index]), "label": list(labels[index])}
        for index in selected
    ]


def hosted_client_or_raise(get_key, client_factory):
    try:
        key = get_key()
    except Exception as error:
        raise RuntimeError(f"Hosted detector comparison unavailable: cannot load credentials ({error})") from error
    if not key:
        raise RuntimeError("Hosted detector comparison unavailable: missing ROBOFLOW_API_KEY")
    try:
        return client_factory(key)
    except ModuleNotFoundError as error:
        raise RuntimeError(f"Hosted detector comparison unavailable: missing dependency ({error})") from error


def validate_output_video(path: Path, expected_frames: int) -> None:
    import cv2

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Output video is empty or missing: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Output video cannot be opened: {path}")
        actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if actual_frames < expected_frames:
            raise RuntimeError(f"Output video is incomplete: {path} has {actual_frames}/{expected_frames} frames")
    finally:
        capture.release()


def draw_detector_boxes(frame, boxes: list[list[float]], *, label: str):
    import cv2

    result = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = bbox_center(box)
        cv2.circle(result, (int(round(cx)), int(round(cy))), 5, (0, 255, 0), -1)
    cv2.putText(result, label, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return result


def annotate_frame(frame, prediction, candidates, detector_boxes, frame_index: int, label: str, ground_truth):
    import cv2

    annotated = overlay_heatmap(frame, prediction.heatmap)
    for candidate in candidates:
        point = (int(round(candidate.x)), int(round(candidate.y)))
        cv2.circle(annotated, point, 11, (0, 255, 255), 2, cv2.LINE_AA)
    visibility, truth_x, truth_y = ground_truth
    if visibility:
        cv2.drawMarker(annotated, (int(round(truth_x)), int(round(truth_y))), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    annotated = draw_detector_boxes(annotated, detector_boxes, label=label)
    cv2.putText(annotated, f"frame={frame_index} TrackNet seq={prediction.frame_indices}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Not a required mutually exclusive group: Python 3.12 rejects a positional with nargs="*" inside one.
    parser.add_argument("clips", nargs="*", type=Path, help="Two or three processed match roots or frame/clip directories")
    parser.add_argument("--processed-root", type=Path, help="Processed dataset root; use --clip-name or deterministic defaults")
    parser.add_argument("--clip-name", action="append", help="Relative match root under --processed-root; repeat 2 or 3 times")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tracknet-only", action="store_true", help="Explicitly skip the hosted comparison")
    parser.add_argument("--detector-frame-cap", type=int, default=20, help="Maximum hosted calls per clip; default 20")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the bounded plan; no torch or network")
    args = parser.parse_args(argv)
    if bool(args.clips) == bool(args.processed_root):
        parser.error("provide either the clip directories or --processed-root, not both and not neither")
    roots = [args.processed_root / name for name in (args.clip_name or DEFAULT_CLIPS)] if args.processed_root else args.clips
    if len(roots) not in (2, 3):
        parser.error("provide exactly 2 or 3 clips")
    if args.detector_frame_cap < 1:
        parser.error("--detector-frame-cap must be positive")
    args.clip_roots = roots
    return args


def main(argv: list[str] | None = None) -> None:
    import cv2

    args = parse_args(argv)
    model_info = inspect_tracknet_checkpoint(args.model)
    if not model_info.get("exists"):
        raise SystemExit(f"TrackNet checkpoint not found: {args.model}")
    clips = []
    for root in args.clip_roots:
        resolved_root, frame_dir, labels_path = resolve_clip_root(root)
        paths = ordered_frame_paths(frame_dir)
        labels = load_labels(labels_path)
        try:
            validate_frame_labels(paths, labels)
        except ValueError as error:
            raise SystemExit(f"{labels_path}: {error}") from error
        selected = select_frame_indices(len(paths), args.detector_frame_cap)
        clips.append((resolved_root, frame_dir, labels_path, paths, labels, selected))

    print(json.dumps({"checkpoint": model_info, "sequence": {"size": model_info.get("seq_len"), "bg_mode": model_info.get("bg_mode"), "input_size": [512, 288], "padding": "repeat edge; emit every frame"}, "clips": [{"root": str(root), "clip_id": clip_identifier(root), "frames": len(paths), "detector_frames": len(selected), "planned_hosted_calls": 0 if args.tracknet_only else len(selected)} for root, _, _, paths, _, selected in clips], "dry_run": args.dry_run, "tracknet_only": args.tracknet_only}, indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run: no torch inference, credentials, or network calls.")
        return

    try:
        detector = build_detector(args.model, backend="pytorch", allow_stub=False, context_size=int(model_info["seq_len"]), input_size=(512, 288))
    except (RuntimeError, ModuleNotFoundError) as error:
        raise SystemExit(f"TrackNet comparison unavailable: {error}") from error

    hosted_results: dict[tuple[int, int], list[list[float]]] = {}
    if not args.tracknet_only:
        try:
            from scripts.shared.env_config import get_roboflow_api_key

            api_client = hosted_client_or_raise(get_roboflow_api_key, roboflow_client)
            for clip_number, (_, _, _, paths, _, selected) in enumerate(clips):
                for frame_index in selected:
                    frame = cv2.imread(str(paths[frame_index]))
                    if frame is None:
                        raise RuntimeError(f"Cannot read comparison frame: {paths[frame_index]}")
                    result = infer_frame(api_client, frame, f"_{frame_index}.jpg")
                    hosted_results[(clip_number, frame_index)] = validated_ball_boxes(result, ("ball", "basketball"), args.confidence)
        except Exception as error:
            raise SystemExit(str(error)) from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for clip_number, (clip_root, _, labels_path, paths, labels, selected) in enumerate(clips):
        output = args.output_dir / f"{clip_identifier(clip_root)}_tracknet_vs_detector.mp4"
        first = cv2.imread(str(paths[0]))
        if first is None:
            raise SystemExit(f"Cannot read first frame: {paths[0]}")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
        if not writer.isOpened():
            writer.release()
            raise SystemExit(f"Could not open output video writer: {output}")
        frame_indices = list(range(len(paths)))
        for frame_index, path in enumerate(paths):
            frame = cv2.imread(str(path))
            if frame is None:
                writer.release()
                raise SystemExit(f"Cannot read frame {frame_index}: {path}")
            window_indices = padded_window(frame_indices, frame_index, detector.context_size)
            window_frames = [cv2.imread(str(paths[index])) for index in window_indices]
            if any(window_frame is None for window_frame in window_frames):
                writer.release()
                raise SystemExit(f"Cannot read TrackNet window for frame {frame_index}: {window_indices}")
            prediction = detector.predict(window_frames, window_indices)
            candidates = extract_candidates(prediction.heatmap, original_frame_size=(width, height), min_score=0.5, min_distance=12, max_candidates=1)
            boxes = hosted_results.get((clip_number, frame_index), [])
            label = "Current detector" if (clip_number, frame_index) in hosted_results else ("TrackNet-only" if args.tracknet_only else "Current detector: not sampled")
            writer.write(annotate_frame(frame, prediction, candidates, boxes, frame_index, label, labels[frame_index]))
        writer.release()
        validate_output_video(output, len(paths))
        summary = {"clip": str(clip_root), "clip_id": clip_identifier(clip_root), "labels": str(labels_path), "frames": len(paths), "comparison_frames": [] if args.tracknet_only else comparison_frame_records(paths, labels, selected), "tracknet_only": args.tracknet_only, "output_video": str(output), "output_validated": True, "tracknet_checkpoint": str(args.model), "tracknet_sequence": detector.context_size, "visible_labels": sum(value[0] for value in labels.values())}
        (args.output_dir / f"{output.stem}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
    print(json.dumps({"clips": summaries}, indent=2))


if __name__ == "__main__":
    main()
