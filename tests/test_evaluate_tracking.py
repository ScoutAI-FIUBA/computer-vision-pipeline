from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tracking import evaluate_tracking as et

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "tracking" / "evaluate_tracking.py"
EXAMPLE = REPO / "future_tasks" / "tracking_review_events_example.jsonl"


def write_jsonl(path: Path, records) -> Path:
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records))
    return path


def ev(clip, event, frame=10, track=1, **extra):
    return {"clip": clip, "event": event, "frame_index": frame, "track_id": track, **extra}


def run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_default_cli_output_is_preserved(tmp_path):
    review = write_jsonl(tmp_path / "r.jsonl", [
        ev("c1", "id_switch"), ev("c1", "full_loss"), ev("c1", "successful_recovery"),
        ev("c1", "full_loss"),
    ])
    result = run_cli(review)
    assert result.returncode == 0
    assert result.stdout == (
        "Tracking baseline review\n"
        f"Review file: {review}\n"
        "Clip filter: all\n"
        "Review events: 4\n"
        "ID switches: 1\n"
        "Full track losses: 2\n"
        "Successful recoveries: 1\n"
        "Unrecovered losses: 1\n"
        "Occlusion recovery rate: 1/3 (33.3%)\n"
    )


def test_default_cli_no_outcomes_and_aliases_and_clip_filter(tmp_path):
    review = write_jsonl(tmp_path / "r.jsonl", [
        {"video": "a", "type": "switch"}, ev("b", "recovery"), "",
    ])
    out = run_cli(review, "--clip", "a").stdout
    assert "Clip filter: a" in out and "ID switches: 1" in out
    assert "Occlusion recovery rate: n/a (no loss/recovery events recorded)" in out


def test_track_context_preserved(tmp_path):
    review = write_jsonl(tmp_path / "r.jsonl", [ev("c", "id_switch")])
    tracks = write_jsonl(tmp_path / "t.jsonl", [
        {"frame_index": 0, "track_id": 1}, {"frame_index": 5, "track_id": 2, "track_state": "predicted"},
    ])
    out = run_cli(review, "--tracks-jsonl", tracks).stdout
    assert "Track context: sampled_frames=2 unique_track_ids=2 predicted_rows=1" in out
    assert "Track frame range: 0..5" in out


def test_per_clip_and_aggregate_counts(tmp_path):
    review = et.load_review(write_jsonl(tmp_path / "r.jsonl", [
        ev("a", "id_switch"), ev("a", "id_switch"), ev("b", "full_loss"), ev("b", "successful_recovery"),
    ]), None)
    per_clip = et.summarize_by_clip(review)
    assert per_clip["a"]["id_switch"] == 2 and per_clip["a"]["recovery_rate"] is None
    assert per_clip["b"]["recovery_rate"] == 0.5 and per_clip["b"]["unrecovered_losses"] == 0
    assert et.count_events(review.events)["id_switch"] == 2
    out = run_cli(tmp_path / "r.jsonl", "--per-clip").stdout
    assert "  a: events=2 id_switch=2" in out and "  b: events=2" in out


def test_json_output(tmp_path):
    review = write_jsonl(tmp_path / "r.jsonl", [ev("a", "id_switch"), ev("b", "full_loss")])
    data = json.loads(run_cli(review, "--json").stdout)
    assert data["aggregate"]["id_switch"] == 1 and data["aggregate"]["full_loss"] == 1
    assert set(data["clips"]) == {"a", "b"}
    assert data["track_context"] is None


def test_stable_player_id_field(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", [
        ev("a", "id_switch", track=4000005, stable_player_id="player-008"), ev("a", "full_loss"),
    ])
    events = et.load_review_events(path, None)
    assert events[0].stable_player_id == "player-008" and events[0].track_id == 4000005
    assert events[1].stable_player_id is None
    agg = et.count_events(events)
    assert agg["stable_player_ids"] == ["player-008"] and agg["events_with_stable_player_id"] == 1


def test_stable_player_id_must_be_string(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", [ev("a", "id_switch", stable_player_id=8)])
    with pytest.raises(ValueError, match="line 1.*stable_player_id"):
        et.load_review_events(path, None)


def test_clip_review_template_records_are_not_events(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", [
        {"record_type": "clip_review", "clip": "a", "id_switch": "pending", "full_loss": "pending"},
        ev("b", "id_switch"),
    ])
    review = et.load_review(path, None)
    assert len(review.events) == 1 and review.declared_clips == {"a", "b"}
    assert review.pending_clip_reviews == 1


def test_baseline_vs_candidate_delta(tmp_path):
    base = et.load_review(write_jsonl(tmp_path / "b.jsonl", [
        ev("a", "id_switch"), ev("a", "id_switch"), ev("a", "full_loss"),
        ev("b", "full_loss"), ev("only_base", "id_switch"),
    ]), None)
    cand = et.load_review(write_jsonl(tmp_path / "c.jsonl", [
        ev("a", "id_switch"), ev("a", "successful_recovery"),
        ev("b", "full_loss"), ev("only_cand", "full_loss"),
    ]), None)
    result = et.compare_reviews(base, cand)
    assert result["clips"]["a"]["delta"] == {"id_switch": -1, "full_loss": -1, "successful_recovery": 1}
    assert result["clips"]["b"]["delta"] == {"id_switch": 0, "full_loss": 0, "successful_recovery": 0}
    assert result["total_shared_clips"]["delta"] == {"id_switch": -1, "full_loss": -1, "successful_recovery": 1}
    assert result["one_sided_clips"] == ["only_base", "only_cand"]
    assert result["clips"]["only_base"]["only_in"] == "baseline"
    assert result["clips"]["only_cand"]["delta"] is None


def test_compare_cli_text_and_json(tmp_path):
    b = write_jsonl(tmp_path / "b.jsonl", [ev("a", "id_switch"), ev("x", "full_loss")])
    c = write_jsonl(tmp_path / "c.jsonl", [ev("a", "successful_recovery")])
    text = run_cli("--baseline", b, "--candidate", c).stdout
    assert "Clip a: id_switch=1->0 (-1)" in text and "successful_recovery=0->1 (+1)" in text
    assert "WARNING: clips present on one side only: x" in text
    data = json.loads(run_cli("--baseline", b, "--candidate", c, "--json").stdout)
    assert data["one_sided_clips"] == ["x"]


def test_compare_requires_both_sides(tmp_path):
    b = write_jsonl(tmp_path / "b.jsonl", [ev("a", "id_switch")])
    result = run_cli("--baseline", b)
    assert result.returncode == 1 and "together" in result.stderr


def test_bad_line_reports_line_number(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", [ev("a", "id_switch"), "", "{not json", ev("a", "teleport")])
    with pytest.raises(ValueError, match="at line 3"):
        et.load_review_events(path, None)
    result = run_cli(path)
    assert result.returncode == 1 and "line 3" in result.stderr


def test_validate_lists_all_errors(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", [
        ev("a", "id_switch"), "{not json", ev("a", "teleport"), "[1]",
        ev("a", "id_switch", frame=-3), ev("a", "full_loss", frame="abc"),
    ])
    errors = et.validate_review_file(path)
    assert [e.split(":")[0] for e in errors] == ["line 2", "line 3", "line 4", "line 5", "line 6"]
    result = run_cli(path, "--validate")
    assert result.returncode == 1 and result.stderr.count("INVALID") == 5
    good = write_jsonl(tmp_path / "g.jsonl", [ev("a", "id_switch")])
    assert run_cli(good, "--validate").returncode == 0


def test_example_file_is_valid_and_runs():
    assert et.validate_review_file(EXAMPLE) == []
    result = run_cli(EXAMPLE, "--per-clip")
    assert result.returncode == 0 and "crossing_clip_01" in result.stdout
