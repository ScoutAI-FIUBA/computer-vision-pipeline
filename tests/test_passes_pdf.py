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


def test_only_human_verified_passes_count_and_jersey_pairs_alone_never_do(tmp_path):
    job = _job(tmp_path, [("p1", "p2", 1.0), ("p2", "p3", 2.0), ("p3", "p1", 3.0)], {"p1": "7", "p2": "30", "p3": "6"})
    truth = {"passes": [{"from": "7", "to": "30"}, {"from": "30", "to": "6"}, {"from": "6", "to": "7"}],
             "verdicts": [{"kind": "pass", "time_seconds": 1.0, "verdict": "real"},
                          {"kind": "pass", "time_seconds": 2.0, "verdict": "real_camiseta_mal"}]}
    clip = pp.load_clip("c", job, None, truth)
    assert clip["matched"] == 1
    assert len(clip["real_wrong_jersey"]) == 1
    assert [e["timestamp_seconds"] for e in clip["unverified"]] == [3.0]
    assert clip["sequence_found"] == "7→30, 30→6, 6→7"


def test_a_pair_that_matches_the_list_is_not_credited_without_a_verdict(tmp_path):
    job = _job(tmp_path, [("p1", "p2", 1.0)], {"p1": "7", "p2": "30"})
    clip = pp.load_clip("c", job, None, {"passes": [{"from": "7", "to": "30"}]})
    assert clip["matched"] == 0 and len(clip["unverified"]) == 1


def test_a_verdict_is_attached_by_time_and_type(tmp_path):
    job = _job(tmp_path, [("p1", "p2", 5.0)], {"p1": "7", "p2": "30"})
    truth = {"passes": [], "verdicts": [{"kind": "pass", "time_seconds": 5.2, "verdict": "falsa"},
                                         {"kind": "pass", "time_seconds": 9.0, "verdict": "real"}]}
    clip = pp.load_clip("c", job, None, truth)
    assert [e["verdict"] for e in clip["pass_events"]] == ["falsa"] and len(clip["false_passes"]) == 1


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


def test_shots_without_times_and_a_different_count_are_not_scored(tmp_path):
    folder = tmp_path / "s"
    folder.mkdir()
    shots = [{"start_timestamp_seconds": t, "outcome": "missed", "shooter_confirmed_number": "21"} for t in (10.4, 14.9, 19.2)]
    (folder / "projected_shots.json").write_text(json.dumps({"shots": shots}))
    result = vr.evaluate_clip("c", folder, {"shots": [{"outcome": "missed", "jersey_number": "21"}]})
    assert result["status"] == "evaluado_ambiguo"
    assert result["truth_shots"] == 1 and len(result["detections"]) == 3 and "rows" not in result


def test_a_one_digit_misread_is_reported_as_a_near_miss(tmp_path):
    job = _job(tmp_path, [("p1", "p2", 1.0)], {"p1": "48", "p2": "21"})
    clip = pp.load_clip("c", job, None, {"passes": [{"from": "49", "to": "21"}]})
    assert clip["matched"] == 0
    assert clip["near_misses"] == ["49→21 detectado como 48→21"]


def test_a_completely_different_pass_is_not_a_near_miss(tmp_path):
    job = _job(tmp_path, [("p1", "p2", 1.0)], {"p1": "7", "p2": "30"})
    assert pp.load_clip("c", job, None, {"passes": [{"from": "49", "to": "21"}]})["near_misses"] == []
