"""Jersey-number evidence for the SAM3/V2 stats boundary.

This module deliberately has no tracking dependency.  V2 rows already contain
the authoritative identity; the video is only decoded at the configured sample
cadence to obtain predictor crops.
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.number_evidence import NumberEvidenceSample
from pipeline.number_predictor import build_number_predictor, normalize_number_prediction


def _sampled_rows(rows: list[dict], stride: int) -> list[dict]:
    return [row for row in rows if int(row["frame_index"]) % stride == 0 and row.get("stable_player_id")]


def _crop(frame, xyxy):
    height, width = frame.shape[:2]
    x1 = max(0, min(width, int(round(float(xyxy[0])))))
    y1 = max(0, min(height, int(round(float(xyxy[1])))))
    x2 = max(x1, min(width, int(round(float(xyxy[2])))))
    y2 = max(y1, min(height, int(round(float(xyxy[3])))))
    return frame[y1:y2, x1:x2].copy()


def _torso_roi(xyxy):
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    width, height = x2 - x1, y2 - y1
    return (x1 + width * 0.2, y1 + height * 0.2,
            x2 - width * 0.2, y1 + height * 0.65)


def _mask_for_row(row, masks_by_observation, clean_masks_by_observation):
    key = row.get("observation_key")
    if key is None:
        return None, "none"
    if clean_masks_by_observation and key in clean_masks_by_observation:
        return clean_masks_by_observation[key], "clean_mask"
    if masks_by_observation and key in masks_by_observation:
        return masks_by_observation[key], "raw_mask"
    return None, "none"


def _masked_torso_crop(frame, player_xyxy, mask, *, mask_kind):
    roi = _torso_roi(player_xyxy)
    crop = _crop(frame, roi)
    if mask is None:
        return crop, roi, {"association_method": "torso_roi_fallback", "mask_used": False}
    import numpy as np
    full_mask = np.asarray(mask)
    if full_mask.ndim != 2 or full_mask.shape[:2] != frame.shape[:2]:
        return crop, roi, {"association_method": "torso_roi_fallback", "mask_used": False,
                           "mask_status": "invalid"}
    x1, y1, x2, y2 = (int(round(value)) for value in roi)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi_mask = full_mask[y1:y2, x1:x2].astype(bool)
    if not roi_mask.any():
        return crop, roi, {"association_method": "torso_roi_fallback", "mask_used": False,
                           "mask_status": "empty"}
    crop = crop[:roi_mask.shape[0], :roi_mask.shape[1]].copy()
    crop[~roi_mask] = 0
    coverage = float(roi_mask.mean())
    return crop, roi, {"association_method": "masked_torso_roi", "mask_used": True,
                       "mask_kind": mask_kind, "mask_coverage": round(coverage, 4)}


_NUMBER_ARTIFACTS = ("numbers.json", "number_predictions.json", "number_prediction_cases.jsonl", "number_manifest.json")


def _remove_artifacts(output_dir: Path) -> None:
    for name in _NUMBER_ARTIFACTS:
        (output_dir / name).unlink(missing_ok=True)


def _aggregate(records: list[dict], minimum_support: int = 2) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["stable_player_id"]].append(record)
    output = []
    for stable_id in sorted(grouped):
        player_records = grouped[stable_id]
        accepted = [r for r in player_records if r["accepted"] and r["tentative_number"] != "unknown"]
        counts = Counter(r["tentative_number"] for r in accepted)
        winners = [number for number, count in counts.items() if count == max(counts.values())] if counts else []
        winner = winners[0] if len(winners) == 1 else "unknown"
        support_count = counts.get(winner, 0)
        if winner == "unknown":
            status = "ambiguous" if accepted else "unknown"
        elif support_count < minimum_support:
            status = "ambiguous"
        elif len(winners) != 1:
            status = "ambiguous"
        else:
            status = "confirmed"
        output.append({
            "stable_player_id": stable_id,
            "number": winner if status == "confirmed" else "unknown",
            "status": status,
            "support_count": support_count,
            "support": [r for r in accepted if r["tentative_number"] == winner],
            "conflicts": [r for r in accepted if r["tentative_number"] != winner] if winner != "unknown" else accepted,
            "ignored": [r for r in player_records if not (r["accepted"] and r["tentative_number"] != "unknown")],
        })
    return output


def run_v2_number_evidence(
    video_path: Path,
    rows: list[dict],
    config,
    output_dir: Path,
    *,
    predictor=None,
    masks_by_observation=None,
    clean_masks_by_observation=None,
) -> bool:
    """Predict sampled player crops and write V2-owned number artifacts."""
    sampled = _sampled_rows(rows, max(1, int(config.frame_stride)))
    if not sampled:
        _remove_artifacts(output_dir)
        return False

    _remove_artifacts(output_dir)
    predictor = build_number_predictor(config) if predictor is None else predictor
    import cv2

    by_frame = defaultdict(list)
    for row in sampled:
        by_frame[int(row["frame_index"])].append(row)
    capture = cv2.VideoCapture(str(video_path))
    records = []
    try:
        for frame_index in sorted(by_frame):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            for row in by_frame[frame_index]:
                player_xyxy = tuple(float(value) for value in row["xyxy"])
                mask, mask_kind = _mask_for_row(row, masks_by_observation, clean_masks_by_observation)
                crop, number_xyxy, evidence = _masked_torso_crop(
                    frame, player_xyxy, mask, mask_kind=mask_kind
                )
                evidence["player_xyxy"] = list(player_xyxy)
                evidence["number_xyxy"] = list(number_xyxy)
                sample = NumberEvidenceSample(
                    track_id=int(row["track_id"]), frame_index=frame_index,
                    track_state=str(row.get("track_state", "tracked")),
                    player_xyxy=player_xyxy, number_xyxy=tuple(number_xyxy),
                    player_confidence=row.get("confidence"), number_confidence=None,
                    overlap_ratio=float(evidence.get("mask_coverage", 0.0)),
                    player_overlap_ratio=float(evidence.get("mask_coverage", 0.0)),
                    contains_number_center=bool(evidence.get("mask_used", False)), crop_image=crop,
                )
                prediction = predictor.predict(sample)
                number, accepted = normalize_number_prediction(
                    prediction, threshold=config.number_predictor_score_threshold
                )
                records.append({
                    "frame_index": frame_index,
                    "stable_player_id": row["stable_player_id"],
                    "raw_track_id": int(row["track_id"]),
                    "tentative_number": number,
                    "raw_text": prediction.raw_text,
                    "accepted": accepted,
                    "score": prediction.score,
                    "raw_response": prediction.raw_response,
                    "evidence": {**evidence, "source": "sam31_v2_observations"},
                })
    except Exception:
        _remove_artifacts(output_dir)
        raise
    finally:
        capture.release()
    if not records:
        _remove_artifacts(output_dir)
        return False
    (output_dir / "number_predictions.json").write_text(json.dumps({"predictions": records}, indent=2), encoding="utf-8")
    (output_dir / "number_prediction_cases.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (output_dir / "numbers.json").write_text(json.dumps({
        "source": "sam31_v2_observations", "identity_authority": "stable_player_id",
        "sample_stride": int(config.frame_stride), "players": _aggregate(records),
    }, indent=2), encoding="utf-8")
    artifact_names = ["numbers.json", "number_predictions.json", "number_prediction_cases.jsonl"]
    artifacts = {}
    for name in artifact_names:
        content = (output_dir / name).read_bytes()
        artifacts[name] = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
    (output_dir / "number_manifest.json").write_text(json.dumps({
        "source": "sam31_v2_observations", "sampled_frames": sorted(by_frame),
        "prediction_count": len(records), "artifacts": artifacts,
    }, indent=2), encoding="utf-8")
    return bool(records)
