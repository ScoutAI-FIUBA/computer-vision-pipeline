"""Offline, read-only A/B evaluation of team embedding backbones."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Keep direct `python scripts/...` execution consistent with the other repo scripts.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.team_embedder import DEFAULT_SIGLIP_MODEL_ID, build_team_embedder
from pipeline.team_id import KMeansTeamClusterer, TeamObservation, identify_teams

DEFAULT_CANDIDATE_MODEL_ID = "Marqo/marqo-fashionSigLIP"


@dataclass(frozen=True)
class CropRecord:
    track_id: int
    frame_index: int
    crop_path: Path
    ground_truth: str | None = None


def load_crop_records(input_path: Path, labels: dict[str, str] | None = None) -> list[CropRecord]:
    """Load selected team-debug crops in stable track/frame order."""

    if input_path.is_dir():
        metadata_path = input_path / "team_debug.json"
        crop_root = input_path / "team_debug"
    else:
        metadata_path = input_path
        crop_root = input_path.parent / "team_debug"
    if not metadata_path.is_file():
        raise ValueError(f"missing team-debug metadata: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid team-debug JSON in {metadata_path}: {exc.msg}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError("team-debug metadata must be a non-empty object keyed by track ID")

    records: list[CropRecord] = []
    try:
        track_keys = sorted(payload, key=lambda value: int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("team-debug track keys must be integer strings") from exc
    for track_key in track_keys:
        track_payload = payload[track_key]
        if not isinstance(track_payload, dict):
            raise ValueError(f"track {track_key} metadata must be an object")
        samples = track_payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"track {track_key} is missing a samples list")
        track_id = int(track_key)
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"track {track_id} sample {sample_index} metadata must be an object")
            if not sample.get("accepted") or not sample.get("selected"):
                continue
            frame_index = sample.get("frame_index")
            if not isinstance(frame_index, int):
                raise ValueError(f"track {track_id} sample {sample_index} is missing integer frame_index")
            prefix = f"{sample_index:03d}_frame_{frame_index:06d}_accepted.png"
            crop_path = crop_root / f"track_{track_id:04d}" / prefix
            if not crop_path.is_file():
                raise ValueError(f"missing crop for track {track_id}, frame {frame_index}: {crop_path}")
            records.append(CropRecord(track_id, frame_index, crop_path, (labels or {}).get(str(track_id))))
    if not records:
        raise ValueError("no accepted selected crops found")
    return sorted(records, key=lambda record: (record.track_id, record.frame_index, str(record.crop_path)))


def _read_crops(records: list[CropRecord]):
    import cv2

    crops = []
    for record in records:
        crop = cv2.imread(str(record.crop_path), cv2.IMREAD_COLOR)
        if crop is None:
            raise ValueError(f"unable to read crop: {record.crop_path}")
        crops.append(crop)
    return crops


def _metrics(records: list[CropRecord], result) -> dict:
    track_ids = sorted({record.track_id for record in records})
    assignments = {item.track_id: item.team_id for item in result.assignments}
    assigned = {track_id: team_id for track_id, team_id in assignments.items() if team_id is not None}
    decisions = result.track_diagnostics.values()
    ambiguous = sum(1 for item in decisions if item.get("vote_decision") in {"ambiguous_vote", "no_majority"})
    cluster_sizes: dict[str, int] = {}
    for diagnostic in decisions:
        for label, count in diagnostic.get("cluster_votes", {}).items():
            cluster_sizes[label] = cluster_sizes.get(label, 0) + int(count)
    metrics = {
        "track_count": len(track_ids),
        "assigned_track_count": len(assigned),
        "coverage": len(assigned) / len(track_ids),
        "unknown_rate": (len(track_ids) - len(assigned)) / len(track_ids),
        "ambiguous_rate": ambiguous / len(track_ids),
        "cluster_sizes": cluster_sizes,
    }
    truth = {record.track_id: record.ground_truth for record in records if record.ground_truth is not None}
    if truth:
        comparable = [(track_id, assigned[track_id], truth[track_id]) for track_id in assigned if track_id in truth]
        predicted = sorted({label for _, label, _ in comparable})
        actual = sorted({label for _, _, label in comparable})
        best = 0
        for predicted_label in predicted:
            matches = [actual_label for _, label, actual_label in comparable if label == predicted_label]
            best += max((matches.count(label) for label in actual), default=0)
        metrics["ground_truth_track_count"] = len(truth)
        metrics["accuracy_on_assigned"] = best / len(comparable) if comparable else None
        metrics["correct_track_count"] = best
    return metrics


def evaluate_model(records: list[CropRecord], embedder, *, crops=None, team_count: int = 2, minimum_samples: int = 3, assignment_margin: float = 0.03) -> dict:
    """Run the unchanged global clustering and track voting path."""

    crops = _read_crops(records) if crops is None else crops
    embeddings = embedder.embed_crops(crops)
    observations = [
        TeamObservation(detection_id=index, track_id=record.track_id, embedding=embedding)
        for index, (record, embedding) in enumerate(zip(records, embeddings, strict=True), 1)
    ]
    result = identify_teams(
        observations,
        clusterer=KMeansTeamClusterer(),
        team_count=team_count,
        backend=embedder.name,
        minimum_samples=minimum_samples,
        assignment_margin=assignment_margin,
    )
    return {"result": result.to_dict(), "metrics": _metrics(records, result)}


def _labels(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("labels must be a JSON object mapping track IDs to team labels")
    return {str(key): str(value) for key, value in payload.items()}


def main(argv: list[str] | None = None, *, embedder_factory: Callable[[str], object] = build_team_embedder) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="pipeline output directory or team_debug.json")
    parser.add_argument("output", type=Path, help="JSON report path")
    parser.add_argument("--labels", type=Path, help="JSON object mapping track IDs to ground-truth team labels")
    parser.add_argument("--baseline-model-id", default=DEFAULT_SIGLIP_MODEL_ID)
    parser.add_argument("--candidate-model-id", default=DEFAULT_CANDIDATE_MODEL_ID)
    parser.add_argument("--team-count", type=int, default=2)
    parser.add_argument("--minimum-samples", type=int, default=3)
    parser.add_argument("--assignment-margin", type=float, default=0.03)
    parser.add_argument("--dry-run", action="store_true", help="validate metadata and crops without loading models")
    args = parser.parse_args(argv)

    records = load_crop_records(args.input, _labels(args.labels))
    report = {
        "input": str(args.input),
        "record_count": len(records),
        "records": [{"track_id": r.track_id, "frame_index": r.frame_index, "crop": str(r.crop_path)} for r in records],
        "models": {"baseline": args.baseline_model_id, "candidate": args.candidate_model_id},
    }
    if not args.dry_run:
        crops = _read_crops(records)
        report["evaluations"] = {
            name: evaluate_model(
                records,
                embedder_factory(model_id),
                crops=crops,
                team_count=args.team_count,
                minimum_samples=args.minimum_samples,
                assignment_margin=args.assignment_margin,
            )
            for name, model_id in (("baseline", args.baseline_model_id), ("candidate", args.candidate_model_id))
        }
        baseline = report["evaluations"]["baseline"]["result"]["assignments"]
        candidate = report["evaluations"]["candidate"]["result"]["assignments"]
        report["disagreements"] = sum(a != b for a, b in zip(baseline, candidate, strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validated {len(records)} selected crops; wrote {args.output}")
    if not args.dry_run:
        for name, evaluation in report["evaluations"].items():
            print(f"{name}: coverage={evaluation['metrics']['coverage']:.3f}, unknown={evaluation['metrics']['unknown_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
