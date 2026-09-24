"""Deterministic, offline V3 calibration reports (no models or Modal)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CalibrationProfile:
    name: str = "default"
    grace_states: tuple[str, ...] = ("GRACE", "INFERRED")
    target_clip_count: str = "10-30"


PROFILES = {"default": CalibrationProfile(), "pilot": CalibrationProfile("pilot")}
DEFAULT_THRESHOLDS = {"coverage": .8, "holder_accuracy": .8, "false_holder_rate": .2,
                      "false_switch_rate": .2, "primitive_precision": .8,
                      "primitive_recall": .8, "primitive_f1": .8, "semantic_precision": .8,
                      "semantic_recall": .8, "semantic_f1": .8}
_HIGHER_IS_BETTER = {"coverage", "holder_accuracy", "primitive_precision", "primitive_recall",
                     "primitive_f1", "semantic_precision", "semantic_recall", "semantic_f1",
                     "grace_recovery_rate", "unknown_recovery_rate"}
_COMPARISON_METRICS = set(DEFAULT_THRESHOLDS) | {"unknown_rate", "contested_rate", "fragmentation"}


def load_clips(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _events(rows: Iterable[tuple[str, Iterable[Mapping]]]) -> tuple[set[tuple], bool]:
    events = set()
    valid = True
    for clip_id, batch in rows:
        if not isinstance(batch, Iterable) or isinstance(batch, (str, bytes, Mapping)):
            valid = False
            continue
        for row in batch:
            event_type = row.get("event_type", row.get("type")) if isinstance(row, Mapping) else None
            frame = row.get("frame_index") if isinstance(row, Mapping) else None
            if not isinstance(event_type, str) or not event_type.strip() or type(frame) is not int or frame < 0:
                valid = False
                continue
            events.add((clip_id, event_type, frame))
    return events, valid


def _prf(predicted: Iterable[tuple[str, Iterable[Mapping]]],
         truth: Iterable[tuple[str, Iterable[Mapping]]]) -> dict[str, float | None]:
    actual, actual_valid = _events(predicted)
    expected, expected_valid = _events(truth)
    if not actual_valid or not expected_valid:
        return {"precision": None, "recall": None, "f1": None}
    if not expected:
        return {"precision": None, "recall": None, "f1": None}
    true_positive = len(actual & expected)
    precision = true_positive / len(actual) if actual else 0.0
    recall = true_positive / len(expected)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def _metric(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _validated_thresholds(thresholds):
    values = dict(DEFAULT_THRESHOLDS)
    values.update(thresholds or {})
    unknown = set(values) - (set(DEFAULT_THRESHOLDS) | {"grace_recovery_rate", "unknown_recovery_rate"})
    if unknown or any(type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1
                      for value in values.values()):
        raise ValueError("thresholds must be known finite values between 0 and 1")
    return values
def _has_comparison_evidence(comparison) -> bool:
    if not isinstance(comparison, Mapping):
        return False
    for name in ("legacy", "v2"):
        metrics = comparison.get(name)
        if not isinstance(metrics, Mapping) or not metrics:
            return False
        if any(key not in _COMPARISON_METRICS for key in metrics):
            return False
        if any(type(value) not in {int, float} or not math.isfinite(float(value)) or not 0 <= value <= 1
               for value in metrics.values()):
            return False
    return True
def evaluate_clips(clips: Iterable[Mapping], *, profile: str = "default", thresholds=None,
                   feature_flag: str = "v3", benchmark_reference: str = "calibration-fixture",
                   configuration: Mapping | None = None, comparison: Mapping | None = None,
                   replay_evidence: bool | None = None, minimum_clips: int = 10,
                   maximum_clips: int = 30, calibration_mode: str = "benchmark") -> dict:
    clips = list(clips)
    if any(not isinstance(clip.get("clip_id"), str) or not clip["clip_id"].strip() for clip in clips):
        raise ValueError("each clip requires a non-empty clip_id")
    if len({clip["clip_id"] for clip in clips}) != len(clips):
        raise ValueError("clip_id values must be unique")
    if profile not in PROFILES:
        raise ValueError(f"unknown calibration profile: {profile}")
    config = PROFILES[profile]
    frames = [frame for clip in clips for frame in sorted(clip.get("frames", []), key=lambda x: x["frame_index"])]
    labeled = [frame for frame in frames if "ground_truth_holder" in frame or "gt_holder" in frame]
    truth = lambda row: row.get("ground_truth_holder", row.get("gt_holder"))
    predicted = lambda row: row.get("predicted_holder", row.get("holder"))
    known = lambda value: value not in (None, "UNKNOWN")
    accuracy = [predicted(row) == truth(row) for row in labeled]
    false_holder = [known(predicted(row)) for row in labeled if truth(row) == "UNKNOWN"]
    switches, switch_denominator = 0, 0
    fragments, truth_fragments = 0, 0
    for clip in clips:
        rows = [row for row in sorted(clip.get("frames", []), key=lambda x: x["frame_index"])
                if "ground_truth_holder" in row or "gt_holder" in row]
        pred_runs = truth_runs = 0
        previous_pred = previous_truth = object()
        for row in rows:
            p, t = predicted(row), truth(row)
            if p != previous_pred: pred_runs += 1
            if t != previous_truth: truth_runs += 1
            previous_pred, previous_truth = p, t
        fragments += max(0, pred_runs - truth_runs)
        truth_fragments += truth_runs
        for before, after in zip(rows, rows[1:]):
            if truth(before) == truth(after) and known(truth(before)):
                switch_denominator += 1
                switches += predicted(before) != predicted(after)
    unknown_predictions = [not known(predicted(row)) for row in labeled]
    recovery = []
    grace_recovery = [predicted(row) == truth(row) for row in labeled
                      if known(truth(row)) and row.get("predicted_state") in config.grace_states]
    for clip in clips:
        rows = [row for row in sorted(clip.get("frames", []), key=lambda x: x["frame_index"])
                if "ground_truth_holder" in row or "gt_holder" in row]
        for before, after in zip(rows, rows[1:]):
            if truth(before) == "UNKNOWN" and known(truth(after)):
                recovery.append(predicted(after) == truth(after))
    primitive = _prf(((clip["clip_id"], clip.get("primitive_events", [])) for clip in clips),
                     ((clip["clip_id"], clip.get("ground_truth_primitive_events", [])) for clip in clips))
    semantic = _prf(((clip["clip_id"], clip.get("semantic_events", [])) for clip in clips),
                    ((clip["clip_id"], clip.get("ground_truth_semantic_events", [])) for clip in clips))
    total_frames = len(frames)
    metrics = {"holder_accuracy": _metric(accuracy), "false_holder_rate": _metric(false_holder),
               "false_switch_rate": switches / switch_denominator if switch_denominator else None,
               "fragmentation": fragments / truth_fragments if truth_fragments else None,
               "unknown_recovery_rate": _metric(recovery), "grace_recovery_rate": _metric(grace_recovery),
               "unknown_rate": _metric(unknown_predictions), "labeled_frames": len(labeled),
               "coverage": len(labeled) / total_frames if total_frames else None,
               "contested_rate": sum(row.get("predicted_state") == "CONTESTED" for row in frames) / total_frames if total_frames else None,
               "primitive_precision": primitive["precision"], "primitive_recall": primitive["recall"],
               "primitive_f1": primitive["f1"], "primitive_event_precision": primitive["precision"],
               "primitive_event_recall": primitive["recall"], "primitive_event_f1": primitive["f1"],
               "semantic_precision": semantic["precision"], "semantic_recall": semantic["recall"],
               "semantic_f1": semantic["f1"], "semantic_event_precision": semantic["precision"],
               "semantic_event_recall": semantic["recall"], "semantic_event_f1": semantic["f1"]}
    event_counts = {}
    for kind in ("primitive", "semantic"):
        predicted_events = [event for clip in clips for event in clip.get(f"{kind}_events", [])]
        truth_events = [event for clip in clips for event in clip.get(f"ground_truth_{kind}_events", [])]
        by_type = lambda events: {name: sum(isinstance(event, Mapping) and event.get("event_type", event.get("type")) == name for event in events)
                                  for name in sorted({event.get("event_type", event.get("type")) for event in events if isinstance(event, Mapping)}, key=str)}
        event_counts[kind] = {"predicted": len(predicted_events), "ground_truth": len(truth_events),
                              "predicted_by_type": by_type(predicted_events), "ground_truth_by_type": by_type(truth_events)}
    if calibration_mode not in {"benchmark", "non_authoritative"}:
        raise ValueError("calibration_mode must be benchmark or non_authoritative")
    if type(minimum_clips) is not int or not 1 <= minimum_clips <= 30 or type(maximum_clips) is not int or not minimum_clips <= maximum_clips <= 30:
        raise ValueError("clip limits must be integers between 1 and 30")
    if calibration_mode == "benchmark" and minimum_clips < 10:
        raise ValueError("benchmark minimum_clips must be at least 10")
    gate_thresholds = _validated_thresholds(thresholds)
    checks = {name: metrics.get(name) is not None and (metrics[name] >= value if name in _HIGHER_IS_BETTER else metrics[name] <= value) for name, value in gate_thresholds.items()}
    classification = {name: sum(clip.get("difficulty", "medium") == name for clip in clips)
                      for name in ("easy", "medium", "hard")}
    checks.update({"deterministic_replay": replay_evidence is True,
                   "clip_count": minimum_clips <= len(clips) <= maximum_clips,
                   "minimum_clip_floor": len(clips) >= 10,
                   "difficulty_coverage": all(classification.values()),
                   "legacy_v2_comparison": _has_comparison_evidence(comparison),
                   "authoritative_mode": calibration_mode == "benchmark"})
    promoted = all(checks.values()) and calibration_mode == "benchmark" and len(clips) >= 10
    gate = {"decision": "PROMOTE" if promoted else "REJECT", "checks": checks,
            "thresholds": gate_thresholds,
            "reason": "all acceptance gates passed" if promoted else "required benchmark evidence missing or failed"}
    return {"schema": "possession-v3-calibration", "schema_version": 1,
            "profile": asdict(config), "clip_count": len(clips),
            "ground_truth_available": bool(labeled), "quality_claim": "descriptive_only",
            "metrics": metrics, "event_counts": event_counts,
            "fixture_classification": classification,
            "legacy_v2_comparison": dict(comparison or {"legacy": None, "v2": None}),
            "promotion_gate": gate,
            "authoritative": promoted,
            "deterministic_replay": replay_evidence is True,
            "traceability": {"profile": asdict(config), "thresholds": gate_thresholds,
                             "feature_flag": feature_flag, "benchmark_reference": benchmark_reference,
                             "fixture_ids": [clip["clip_id"] for clip in clips],
                             "clip_bounds": {"minimum": minimum_clips, "maximum": maximum_clips},
                             "calibration_mode": calibration_mode,
                             "configuration": dict(configuration or {})}}


def write_report(input_path: Path, output_path: Path, *, profile: str = "default", thresholds=None,
                 feature_flag: str = "v3", benchmark_reference: str = "calibration-fixture",
                 configuration: Mapping | None = None, comparison: Mapping | None = None,
                 minimum_clips: int = 10, maximum_clips: int = 30,
                 calibration_mode: str = "benchmark") -> dict:
    clips = load_clips(input_path)
    kwargs = dict(profile=profile, thresholds=thresholds, feature_flag=feature_flag,
                  benchmark_reference=benchmark_reference, configuration=configuration,
                  comparison=comparison, minimum_clips=minimum_clips, maximum_clips=maximum_clips,
                  calibration_mode=calibration_mode)
    first = evaluate_clips(clips, **kwargs)
    second = evaluate_clips(clips, **kwargs)
    kwargs["replay_evidence"] = first == second
    report = evaluate_clips(clips, **kwargs)
    report["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    hash_input = {"clips": clips, "traceability": report["traceability"], "profile": profile}
    report["benchmark_input_sha256"] = hashlib.sha256(
        json.dumps(hash_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", default="default", choices=sorted(PROFILES))
    parser.add_argument("--legacy-comparison", type=Path, required=True,
                        help="JSON object containing legacy comparison metrics")
    parser.add_argument("--v2-comparison", type=Path, required=True,
                        help="JSON object containing V2 comparison metrics")
    args = parser.parse_args()
    comparison = {"legacy": json.loads(args.legacy_comparison.read_text(encoding="utf-8")),
                  "v2": json.loads(args.v2_comparison.read_text(encoding="utf-8"))}
    write_report(args.input, args.output, profile=args.profile, comparison=comparison)


if __name__ == "__main__":
    main()
