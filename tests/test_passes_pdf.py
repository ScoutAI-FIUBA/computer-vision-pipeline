import json
from pathlib import Path

from scripts.ops import passes_pdf as pp
from scripts.ops import validation_report as vr


def _job(tmp_path: Path, passes: list[tuple[str, str, float]], jerseys: dict[str, str]) -> Path:
    job = tmp_path / "job1"
    job.mkdir()
    (job / "possession_events.jsonl").write_text("\n".join(json.dumps({"frame_index": i, "status": "lost"}) for i in range(10)))
    (job / "numbers.json").write_text(json.dumps({"players": [
        {"stable_player_id": pid, "number": n, "status": "confirmed"} for pid, n in jerseys.items()]}))
    (job / "results.json").write_text(json.dumps({"video": {"fps": 30.0, "duration_seconds": 1.0}}))
    events = [{"type": "pass", "timestamp_seconds": t, "from_stable_player_id": a, "to_stable_player_id": b} for a, b, t in passes]
    (tmp_path / "job1-events.json").write_text(json.dumps({"events": events}))
    return job


def test_a_pass_counts_only_when_both_jerseys_match(tmp_path):
    job = _job(tmp_path, [("p1", "p2", 1.0), ("p2", "p3", 2.0)], {"p1": "7", "p2": "30", "p3": "6"})
    truth = {"passes": [{"from": "7", "to": "30"}, {"from": "30", "to": "7"}]}
    clip = pp.load_clip("c", job, None, truth)
    assert clip["matched"] == 1
    assert clip["sequence_found"] == "7→30, 30→6"
    assert clip["sequence_real"] == "7→30, 30→7"


def test_passes_with_unread_jerseys_never_match(tmp_path):
    job = _job(tmp_path, [("p1", "p2", 1.0)], {})
    clip = pp.load_clip("c", job, None, {"passes": [{"from": "7", "to": "30"}]})
    assert clip["matched"] == 0 and clip["sequence_found"] == "?→?"


def test_a_truth_pass_with_nothing_to_match_on_is_not_counted(tmp_path):
    job = _job(tmp_path, [("p1", "p2", 1.0)], {"p1": "7", "p2": "30"})
    assert pp.load_clip("c", job, None, {"passes": [{}]})["matched"] == 0


def test_each_detected_pass_is_used_once(tmp_path):
    job = _job(tmp_path, [("p1", "p2", 1.0)], {"p1": "7", "p2": "30"})
    clip = pp.load_clip("c", job, None, {"passes": [{"from": "7", "to": "30"}, {"from": "7", "to": "30"}]})
    assert clip["matched"] == 1


def test_shots_without_times_are_matched_in_order(tmp_path):
    folder = tmp_path / "s"
    folder.mkdir()
    shots = [{"start_timestamp_seconds": 2.0, "outcome": "missed", "shooter_confirmed_number": "14"},
             {"start_timestamp_seconds": 9.0, "outcome": "missed", "shooter_confirmed_number": "7"}]
    (folder / "projected_shots.json").write_text(json.dumps({"shots": shots}))
    truth = {"shots": [{"outcome": "missed", "jersey_number": "14"}, {"outcome": "missed", "jersey_number": "7"}]}
    rows = vr.evaluate_clip("c", folder, truth)["rows"]
    assert [r["found"]["time_seconds"] for r in rows] == [2.0, 9.0]
    assert all(r["outcome_ok"] and r["number_ok"] for r in rows)
