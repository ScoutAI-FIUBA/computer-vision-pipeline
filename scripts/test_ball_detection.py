#!/usr/bin/env python3
"""Run the player detector model and keep only ball detections."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.ball import DEFAULT_BALL_CLASSES, bbox_center, filter_ball_predictions
from pipeline.tracking import xyxy_iou
from scripts.shared.annotation_helpers import (
    annotate_detections,
    detections_from_inference,
)
from scripts.shared.env_config import ROBOFLOW_API_URL, get_roboflow_api_key

MODEL_ID = "basketball-player-detection-3-ycjdo/16"
CANDIDATE_MODEL_ID = "nba-ball-detection/2"
API_URL = ROBOFLOW_API_URL
EPSILON = 1e-12
DEFAULT_POINT_DISTANCE_PX = 20.0


def roboflow_client(api_key: str):
    from inference_sdk import InferenceHTTPClient

    return InferenceHTTPClient(api_url=API_URL, api_key=api_key)


def infer_frame(client, frame, suffix: str, model_id: str = MODEL_ID) -> dict:
    import cv2

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        if not cv2.imwrite(str(temp_path), frame):
            raise RuntimeError("Failed to write temporary frame image")
        return client.infer(str(temp_path), model_id=model_id)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def open_video(video_path: Path):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def draw_ball_detections(frame, result: dict, min_confidence: float):
    detections = detections_from_inference(result, min_confidence=min_confidence)
    return annotate_detections(frame, detections)


def process_video(
    video_path: Path,
    output_path: Path,
    api_key: str,
    *,
    frame_stride: int,
    max_frames: int | None,
    confidence: float,
    class_names: tuple[str, ...],
    model_id: str = MODEL_ID,
) -> dict:
    import cv2

    capture = open_video(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / frame_stride), (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    frame_index = 0
    roboflow_calls = 0
    frames_with_ball = 0
    total_ball_predictions = 0
    seen_classes: set[str] = set()

    try:
        client = roboflow_client(api_key)
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            raw_result = infer_frame(client, frame, f"_{frame_index}.jpg", model_id)
            roboflow_calls += 1
            validated_ball_boxes(raw_result, class_names, confidence)
            seen_classes.update(
                str(prediction.get("class", "")).strip()
                for prediction in raw_result.get("predictions", [])
                if str(prediction.get("class", "")).strip()
            )

            ball_result = filter_ball_predictions(raw_result, class_names)
            ball_count = len(ball_result.get("predictions", []))
            total_ball_predictions += ball_count
            if ball_count:
                frames_with_ball += 1

            writer.write(draw_ball_detections(frame, ball_result, confidence))
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {
        "model_id": model_id,
        "frame_count": frame_count or None,
        "frames_read": frame_index,
        "roboflow_calls": roboflow_calls,
        "frames_with_ball": frames_with_ball,
        "total_ball_predictions": total_ball_predictions,
        "seen_classes": sorted(seen_classes),
        "output_path": str(output_path),
    }


def load_ground_truth(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: each record must be a JSON object")
        balls = record.get("balls")
        if not isinstance(balls, list):
            raise ValueError(f"{path}:{line_number}: balls must be a list of [x1, y1, x2, y2] boxes")
        validated_balls = [
            validate_box(box, context=f"{path}:{line_number}: balls[{index}]")
            for index, box in enumerate(balls)
        ]
        raw_image = record.get("image")
        if not isinstance(raw_image, str) or not raw_image:
            raise ValueError(f"{path}:{line_number}: image is required")
        image = Path(raw_image)
        image = image if image.is_absolute() else path.parent / image
        if not image.is_file():
            raise ValueError(f"{path}:{line_number}: image not found: {image}")
        clip = record.get("clip", image.parent.name)
        if not isinstance(clip, str):
            raise ValueError(f"{path}:{line_number}: clip must be a string")
        cases.append({"image": str(image), "clip": clip, "balls": validated_balls})
    if not cases:
        raise ValueError(f"{path}: no labeled frames found")
    return cases


def csv_number(value: object, *, context: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{context} must be a finite number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{context} must be a finite number")
    return numeric


def load_tracknet_ground_truth(root: Path) -> list[dict]:
    csv_paths = sorted(root.glob("**/csv/*_ball.csv"))
    if not csv_paths:
        raise ValueError(f"{root}: no csv/*_ball.csv labels found")

    cases = []
    for csv_path in csv_paths:
        clip_name = csv_path.name.removesuffix("_ball.csv")
        game_root = csv_path.parent.parent
        frame_dir = game_root / "frame" / clip_name
        if not frame_dir.is_dir():
            raise ValueError(f"{csv_path}: corresponding frame directory not found: {frame_dir}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"Frame", "Visibility", "X", "Y"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{csv_path}: required columns are Frame, Visibility, X, Y")
            for line_number, row in enumerate(reader, start=2):
                context = f"{csv_path}:{line_number}"
                try:
                    frame_index = int(row["Frame"])
                    visibility = int(row["Visibility"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{context}: Frame and Visibility must be integers") from error
                if frame_index < 0 or visibility not in {0, 1}:
                    raise ValueError(f"{context}: Frame must be non-negative and Visibility must be 0 or 1")
                image = frame_dir / f"{frame_index}.png"
                if not image.is_file():
                    raise ValueError(f"{context}: corresponding frame not found: {image}")
                points = []
                if visibility == 1:
                    points = [[csv_number(row["X"], context=f"{context}: X"), csv_number(row["Y"], context=f"{context}: Y")]]
                cases.append(
                    {
                        "image": str(image),
                        "clip": f"{game_root.name}/{clip_name}",
                        "points": points,
                    }
                )
    if not cases:
        raise ValueError(f"{root}: no corresponding frames/labels found")
    return cases


def capped_cases(cases: list[dict], frame_cap: int | None) -> list[dict]:
    if frame_cap is None or frame_cap >= len(cases):
        return cases
    if frame_cap < 1:
        raise ValueError("--frame-cap must be at least 1")

    selected_indices = [index * len(cases) // frame_cap for index in range(frame_cap)]
    negative_indices = [index for index, case in enumerate(cases) if not case.get("balls") and not case.get("points")]
    positive_indices = [index for index, case in enumerate(cases) if case.get("balls") or case.get("points")]
    if negative_indices and positive_indices:
        if frame_cap < 2:
            raise ValueError("--frame-cap must be at least 2 when positive and negative frames exist")
        if not any(index in negative_indices for index in selected_indices):
            selected_indices[-1] = negative_indices[0]
        if not any(index in positive_indices for index in selected_indices):
            selected_indices[0] = positive_indices[0]
    return [cases[index] for index in sorted(set(selected_indices))]


def number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{context} must be a finite number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{context} must be a finite number")
    return numeric


def validate_box(box: object, *, context: str) -> list[float]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{context} must be [x1, y1, x2, y2]")
    values = [number(value, context=f"{context}[{index}]") for index, value in enumerate(box)]
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"{context} must have x2 > x1 and y2 > y1")
    return values


def prediction_box(prediction: dict, *, context: str = "prediction") -> list[float]:
    x = number(prediction.get("x"), context=f"{context}.x")
    y = number(prediction.get("y"), context=f"{context}.y")
    width = number(prediction.get("width"), context=f"{context}.width")
    height = number(prediction.get("height"), context=f"{context}.height")
    if width <= 0 or height <= 0:
        raise ValueError(f"{context}.width and {context}.height must be positive")
    return [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0]


def validated_ball_boxes(result: object, class_names: tuple[str, ...], confidence: float) -> list[list[float]]:
    if not isinstance(result, dict):
        raise ValueError("hosted response must be an object")
    predictions = result.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("hosted response predictions must be a list")

    allowed = {name.strip().lower() for name in class_names if name.strip()}
    boxes = []
    for index, prediction in enumerate(predictions):
        context = f"prediction[{index}]"
        if not isinstance(prediction, dict):
            raise ValueError(f"{context} must be an object")
        class_name = prediction.get("class")
        if not isinstance(class_name, str):
            raise ValueError(f"{context}.class must be a string")
        score = number(prediction.get("confidence"), context=f"{context}.confidence")
        box = prediction_box(prediction, context=context)
        if class_name.strip().lower() in allowed and score >= confidence:
            boxes.append(box)
    return boxes


def maximum_cardinality_matching(adjacency: list[list[int]]) -> list[tuple[int, int]]:
    prediction_by_truth: dict[int, int] = {}

    def augment(prediction_index: int, seen: set[int]) -> bool:
        for truth_index in adjacency[prediction_index]:
            if truth_index in seen:
                continue
            seen.add(truth_index)
            previous = prediction_by_truth.get(truth_index)
            if previous is None or augment(previous, seen):
                prediction_by_truth[truth_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(adjacency)):
        augment(prediction_index, set())
    return sorted((prediction_index, truth_index) for truth_index, prediction_index in prediction_by_truth.items())


def maximum_cardinality_matches(
    predicted_boxes: list[list[float]], ground_truth: list[list[float]], iou_threshold: float = 0.5
) -> list[tuple[int, int]]:
    return maximum_cardinality_matching(
        [
            sorted(
                (
                    truth_index
                    for truth_index, truth in enumerate(ground_truth)
                    if xyxy_iou(box, truth) >= iou_threshold
                ),
                key=lambda truth_index: (-xyxy_iou(box, ground_truth[truth_index]), truth_index),
            )
            for box in predicted_boxes
        ]
    )


def maximum_point_matches(
    predicted_boxes: list[list[float]], ground_truth: list[list[float]], distance_threshold: float
) -> list[tuple[int, int]]:
    centers = [bbox_center(box) for box in predicted_boxes]
    return maximum_cardinality_matching(
        [
            sorted(
                (
                    truth_index
                    for truth_index, truth in enumerate(ground_truth)
                    if math.dist(center, truth) <= distance_threshold
                ),
                key=lambda truth_index: (math.dist(center, ground_truth[truth_index]), truth_index),
            )
            for center in centers
        ]
    )


def evaluate_results(
    cases: list[dict],
    results: list[dict],
    *,
    confidence: float,
    class_names: tuple[str, ...] = DEFAULT_BALL_CLASSES,
    point_distance_px: float = DEFAULT_POINT_DISTANCE_PX,
) -> dict:
    if not cases:
        raise ValueError("at least one labeled frame is required")
    if len(results) != len(cases):
        raise ValueError("hosted result count must match labeled frame count")
    true_positives = false_positives = false_negatives = visible_frames = 0
    center_errors = []
    clips: dict[str, dict[str, int]] = {}
    label_type = "point" if "points" in cases[0] else "box"
    if any(("points" in case) != (label_type == "point") for case in cases):
        raise ValueError("all labeled frames must use the same ground-truth type")

    for case, result in zip(cases, results, strict=True):
        ground_truth = case["points"] if label_type == "point" else case["balls"]
        predicted_boxes = validated_ball_boxes(result, class_names, confidence)
        matches = (
            maximum_point_matches(predicted_boxes, ground_truth, point_distance_px)
            if label_type == "point"
            else maximum_cardinality_matches(predicted_boxes, ground_truth)
        )

        frame_true_positives = len(matches)
        true_positives += frame_true_positives
        false_positives += len(predicted_boxes) - frame_true_positives
        false_negatives += len(ground_truth) - frame_true_positives
        visible_frames += bool(ground_truth)
        clip = clips.setdefault(str(case["clip"]), {"true_positives": 0, "visible_balls": 0})
        clip["true_positives"] += frame_true_positives
        clip["visible_balls"] += len(ground_truth)
        for prediction_index, truth_index in matches:
            predicted = predicted_boxes[prediction_index]
            expected = ground_truth[truth_index]
            predicted_center = bbox_center(predicted)
            expected_center = expected if label_type == "point" else bbox_center(expected)
            center_errors.append(math.dist(predicted_center, expected_center))

    predicted_count = true_positives + false_positives
    visible_balls = true_positives + false_negatives
    return {
        "frames": len(cases),
        "visible_frames": visible_frames,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "recall": true_positives / visible_balls if visible_balls else 0.0,
        "precision": true_positives / predicted_count if predicted_count else 0.0,
        "false_positives_per_frame": false_positives / len(cases),
        "mean_center_error_px": sum(center_errors) / len(center_errors) if center_errors else None,
        "label_type": label_type,
        "point_distance_threshold_px": point_distance_px if label_type == "point" else None,
        "ap50": None,
        "ap50_note": "Skipped: the acceptance gate uses thresholded recall, precision, and false positives/frame.",
        "clips": {
            clip: {
                **counts,
                "recall": counts["true_positives"] / counts["visible_balls"] if counts["visible_balls"] else 0.0,
            }
            for clip, counts in sorted(clips.items())
        },
    }


def acceptance_result(baseline: dict, candidate: dict) -> dict:
    clip_regressions = {
        clip: baseline_clip["recall"] - candidate["clips"].get(clip, {"recall": 0.0})["recall"]
        for clip, baseline_clip in baseline["clips"].items()
        if baseline_clip["visible_balls"]
    }
    checks = {
        "precision_equal_or_better": candidate["precision"] + EPSILON >= baseline["precision"],
        "recall_improves_by_10_points": candidate["recall"] - baseline["recall"] + EPSILON >= 0.10,
        "false_positives_per_frame_not_increased": candidate["false_positives_per_frame"] <= baseline["false_positives_per_frame"] + EPSILON,
        "no_clip_regresses_over_5_points": all(regression <= 0.05 + EPSILON for regression in clip_regressions.values()),
    }
    return {"passed": all(checks.values()), "checks": checks, "clip_recall_regressions": clip_regressions}


def evaluate_models(
    cases: list[dict],
    source: Path,
    model_ids: list[str],
    api_key: str,
    *,
    confidence: float,
    class_names: tuple[str, ...],
    point_distance_px: float,
) -> dict:
    client = roboflow_client(api_key)
    models = []
    for model_id in model_ids:
        results = [client.infer(case["image"], model_id=model_id) for case in cases]
        models.append(
            {
                "model_id": model_id,
                **evaluate_results(
                    cases,
                    results,
                    confidence=confidence,
                    class_names=class_names,
                    point_distance_px=point_distance_px,
                ),
            }
        )

    baseline = models[0]
    comparisons = [
        {"baseline_model_id": baseline["model_id"], "candidate_model_id": candidate["model_id"], **acceptance_result(baseline, candidate)}
        for candidate in models[1:]
    ]
    return {
        "ground_truth": str(source),
        "confidence": confidence,
        "iou_threshold": 0.5,
        "point_distance_threshold_px": point_distance_px if "points" in cases[0] else None,
        "models": models,
        "comparisons": comparisons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect ball detections or compare hosted models on labeled frames.")
    parser.add_argument("video", type=Path, nargs="?", help="Path to the input video")
    parser.add_argument("--ground-truth", type=Path, help="JSONL labels for frame-based model comparison")
    parser.add_argument("--tracknet-root", type=Path, help="Processed TrackNet game, split, or dataset root")
    parser.add_argument("--model-id", dest="model_ids", action="append", help="Hosted model ID; repeat to compare models")
    parser.add_argument("--json-output", type=Path, help="Evaluation JSON output path")
    parser.add_argument("--frame-cap", type=int, help="Deterministic maximum labeled frames for model evaluation")
    parser.add_argument(
        "--point-distance-px",
        type=float,
        default=DEFAULT_POINT_DISTANCE_PX,
        help=f"Maximum center distance for TrackNet point matches. Default: {DEFAULT_POINT_DISTANCE_PX:g}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the evaluation plan without API calls")
    parser.add_argument("--output", type=Path, help="Path for the annotated sampled video")
    parser.add_argument("--frame-stride", type=int, default=10, help="Infer every N frames. Default: 10")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to read")
    parser.add_argument("--confidence", type=float, default=0.4, help="Minimum detection confidence. Default: 0.4")
    parser.add_argument(
        "--ball-class", dest="ball_classes", action="append", help="Allowed ball class. Repeat for aliases."
    )
    return parser


def resolved_ball_classes(extra_classes: list[str] | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*DEFAULT_BALL_CLASSES, *(extra_classes or []))))


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    evaluation_inputs = sum(value is not None for value in (args.ground_truth, args.tracknet_root))
    if evaluation_inputs > 1:
        parser.error("use only one of --ground-truth or --tracknet-root")
    if evaluation_inputs:
        args.model_ids = args.model_ids or [MODEL_ID, CANDIDATE_MODEL_ID]
        if len(args.model_ids) < 2:
            parser.error("model evaluation requires at least two --model-id values")
    if args.dry_run and not evaluation_inputs:
        parser.error("--dry-run requires --ground-truth or --tracknet-root")
    if args.frame_cap is not None and args.frame_cap < 1:
        parser.error("--frame-cap must be at least 1")
    if not math.isfinite(args.point_distance_px) or args.point_distance_px <= 0:
        parser.error("--point-distance-px must be a finite number greater than 0")
    return args


def main() -> None:
    args = parse_cli_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    evaluation_source = args.ground_truth or args.tracknet_root
    if sum(value is not None for value in (args.video, evaluation_source)) != 1:
        raise SystemExit("Provide one video, --ground-truth, or --tracknet-root")
    class_names = resolved_ball_classes(args.ball_classes)

    if evaluation_source:
        if not evaluation_source.exists():
            raise SystemExit(f"Ground truth not found: {evaluation_source}")
        try:
            cases = (
                load_ground_truth(args.ground_truth)
                if args.ground_truth
                else load_tracknet_ground_truth(args.tracknet_root)
            )
            cases = capped_cases(cases, args.frame_cap)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

        print(f"Models: {', '.join(args.model_ids)} (first is baseline)")
        print(f"Planned labeled frames: {len(cases)}")
        visible_frames = sum(bool(case.get("balls") or case.get("points")) for case in cases)
        print(f"Planned visible/no-ball frames: {visible_frames}/{len(cases) - visible_frames}")
        print(f"Planned inference calls: {len(cases) * len(args.model_ids)}")
        if "points" in cases[0]:
            print(f"Point match threshold: {args.point_distance_px:g}px")
        if args.dry_run:
            print("Dry run: no API calls made")
            return

        api_key = get_roboflow_api_key()
        if not api_key:
            raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")
        try:
            report = evaluate_models(
                cases,
                evaluation_source,
                args.model_ids,
                api_key,
                confidence=args.confidence,
                class_names=class_names,
                point_distance_px=args.point_distance_px,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

        json_output = args.json_output or (
            args.ground_truth.with_suffix(".results.json")
            if args.ground_truth
            else Path("outputs") / f"{args.tracknet_root.name}_ball_seed_results.json"
        )
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for model in report["models"]:
            center_error = model["mean_center_error_px"]
            center_text = f"{center_error:.2f}" if center_error is not None else "n/a"
            print(
                f"{model['model_id']}: recall={model['recall']:.3f} precision={model['precision']:.3f} "
                f"fp/frame={model['false_positives_per_frame']:.3f} center_error_px={center_text}"
            )
        print("AP50: skipped; the acceptance gate uses thresholded recall, precision, and false positives/frame.")
        for comparison in report["comparisons"]:
            print(f"Acceptance {comparison['candidate_model_id']}: {'PASS' if comparison['passed'] else 'FAIL'}")
        print(f"JSON report: {json_output}")
        return

    api_key = get_roboflow_api_key()
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable")
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    args.output = args.output or Path("outputs") / f"{args.video.stem}_ball_detections.mp4"
    model_id = (args.model_ids or [MODEL_ID])[0]

    try:
        stats = process_video(
            args.video,
            args.output,
            api_key,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            confidence=args.confidence,
            class_names=class_names,
            model_id=model_id,
        )
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(f"Model: {stats['model_id']}")
    print(f"Frames read: {stats['frames_read']}/{stats['frame_count'] or 'unknown'}")
    print(f"Roboflow calls: {stats['roboflow_calls']}")
    print(f"Frames with ball: {stats['frames_with_ball']}")
    print(f"Total ball detections: {stats['total_ball_predictions']}")
    print(f"Seen classes: {', '.join(stats['seen_classes']) or 'none'}")
    print(f"Annotated sampled video saved to: {stats['output_path']}")


if __name__ == "__main__":
    main()
