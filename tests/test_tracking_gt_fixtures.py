"""Validates the MOTA/IDF1 sample fixtures only; the metrics themselves are a separate follow-up."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tracking_gt"
VISIBILITY = {"visible", "partial_occlusion", "full_occlusion"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def expected():
    return json.loads((FIXTURES / "expected_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gt():
    records = load_jsonl(FIXTURES / "gt.jsonl")
    return records[0], [r for r in records if r["record_type"] == "box"], records


def test_gt_schema(gt, expected):
    header, boxes, records = gt
    assert records[0]["record_type"] == "header" and sum(r["record_type"] == "header" for r in records) == 1
    assert {"annotation_version", "clip", "fps", "width", "height", "start_frame", "num_frames"} <= set(header)
    assert header["num_frames"] == expected["num_frames"]
    seen = set()
    for record in records[1:]:
        assert record["record_type"] in {"box", "ignore"}
        assert 0 <= record["frame_index"] < header["num_frames"]
        x1, y1, x2, y2 = record["bbox"]
        assert 0 <= x1 < x2 <= header["width"] and 0 <= y1 < y2 <= header["height"]
        if record["record_type"] == "box":
            assert record["visibility"] in VISIBILITY and record["gt_id"]
            key = (record["frame_index"], record["gt_id"])
            assert key not in seen, "a gt_id appears once per frame"
            seen.add(key)
        else:
            assert record["reason"]
    assert len(boxes) == expected["num_gt_boxes"]
    assert sorted({b["gt_id"] for b in boxes}) == expected["gt_ids"]


def test_predictions_schema_and_counts(gt, expected):
    header = gt[0]
    for name, case in expected["cases"].items():
        rows = load_jsonl(FIXTURES / case["prediction"])
        assert len(rows) == case["num_pred_boxes"], name
        keys = set()
        for row in rows:
            assert set(row) == {"frame_index", "raw_track_id", "bbox_xyxy"}
            assert 0 <= row["frame_index"] < header["num_frames"]
            x1, y1, x2, y2 = row["bbox_xyxy"]
            assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1  # normalised, like tracks.jsonl
            assert (row["frame_index"], row["raw_track_id"]) not in keys
            keys.add((row["frame_index"], row["raw_track_id"]))


def exact_match_counts(gt_boxes, header, rows):
    """Fixtures reuse GT boxes verbatim, so exact box equality stands in for IoU matching here."""
    w, h = header["width"], header["height"]
    by_frame = {}
    for row in rows:
        by_frame.setdefault(row["frame_index"], []).append(row)
    tp = fp = idsw = 0
    last = {}
    matched_pred = set()
    for box in sorted(gt_boxes, key=lambda b: b["frame_index"]):
        x1, y1, x2, y2 = box["bbox"]
        target = [round(x1 / w, 6), round(y1 / h, 6), round(x2 / w, 6), round(y2 / h, 6)]
        hit = next((r for r in by_frame.get(box["frame_index"], []) if r["bbox_xyxy"] == target), None)
        if hit is None:
            continue
        tp += 1
        matched_pred.add((hit["frame_index"], hit["raw_track_id"]))
        if box["gt_id"] in last and last[box["gt_id"]] != hit["raw_track_id"]:
            idsw += 1
        last[box["gt_id"]] = hit["raw_track_id"]
    fp = len(rows) - len(matched_pred)
    return len(gt_boxes) - tp, fp, idsw


def test_expected_metrics_arithmetic(gt, expected):
    header, boxes, _ = gt
    for name, case in expected["cases"].items():
        rows = load_jsonl(FIXTURES / case["prediction"])
        fn, fp, idsw = exact_match_counts(boxes, header, rows)
        assert (fn, fp, idsw) == (case["fn"], case["fp"], case["idsw"]), name
        n_gt = expected["num_gt_boxes"]
        assert case["idfp"] == case["num_pred_boxes"] - case["idtp"], name
        assert case["idfn"] == n_gt - case["idtp"], name
        assert case["mota"] == pytest.approx(1 - (fn + fp + idsw) / n_gt, abs=1e-6), name
        idtp, idfp, idfn = case["idtp"], case["idfp"], case["idfn"]
        assert case["idf1"] == pytest.approx(2 * idtp / (2 * idtp + idfp + idfn), abs=1e-6), name
