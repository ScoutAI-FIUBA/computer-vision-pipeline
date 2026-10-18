import json
from pathlib import Path

from scripts.ops import validation_report as vr


def _write(folder: Path, shots: list[dict], **extra) -> Path:
    folder.mkdir(parents=True)
    (folder / "projected_shots.json").write_text(json.dumps({"shots": shots}))
    for name, payload in extra.items():
        (folder / f"{name}.json").write_text(json.dumps(payload))
    return folder


def _shot(t, outcome="made", number=None, shooter="player-001", **more):
    return {"start_timestamp_seconds": t, "outcome": outcome, "shooter_confirmed_number": number,
            "shooter_stable_player_id": shooter, **more}


def test_correct_shot_is_matched_with_outcome_and_number(tmp_path):
    folder = _write(tmp_path / "a", [_shot(6.6, number="7")])
    result = vr.evaluate_clip("a", folder, {"shots": [{"outcome": "made", "jersey_number": "7", "approx_time_seconds": 6.5}]})
    row = result["rows"][0]
    assert result["matched"] == 1 and row["outcome_ok"] and row["number_ok"] and row["shooter_attributed"]


def test_wrong_outcome_and_wrong_number_are_reported(tmp_path):
    folder = _write(tmp_path / "a", [_shot(6.6, outcome="missed", number="9")])
    row = vr.evaluate_clip("a", folder, {"shots": [{"outcome": "made", "jersey_number": "7"}]})["rows"][0]
    assert row["outcome_ok"] is False and row["number_ok"] is False


def test_unread_number_is_not_counted_as_wrong(tmp_path):
    folder = _write(tmp_path / "a", [_shot(6.6, number=None)])
    row = vr.evaluate_clip("a", folder, {"shots": [{"outcome": "made", "jersey_number": "7"}]})["rows"][0]
    assert row["number_ok"] is None


def test_missed_and_spurious_shots(tmp_path):
    folder = _write(tmp_path / "a", [_shot(20.0)])
    result = vr.evaluate_clip("a", folder, {"tolerance_seconds": 2.0, "shots": [{"outcome": "made", "approx_time_seconds": 5.0}]})
    assert result["missed_by_system"] == 1 and len(result["spurious"]) == 1


def test_clip_with_no_shots_flags_any_detection(tmp_path):
    folder = _write(tmp_path / "a", [_shot(3.0)])
    assert len(vr.evaluate_clip("a", folder, {"shots": []})["spurious"]) == 1


def test_missing_results_and_missing_truth_are_not_invented(tmp_path):
    assert vr.evaluate_clip("a", tmp_path / "nope", {"shots": []})["status"] == "sin_resultados"
    folder = _write(tmp_path / "b", [_shot(1.0)])
    assert vr.evaluate_clip("b", folder, None)["status"] == "sin_verdad_terreno"


def test_cost_uses_reported_timings_and_rate_only():
    assert vr._cost({"timings": {"a": 1800, "b": 1800}, "duration_seconds": 60}, 2.5) == ("2.50 USD", "2.50 USD")
    assert vr._cost({"timings": {"a": 10}}, None) == ("n/d", "n/d")
    assert vr._cost(None, 2.5) == ("n/d", "n/d")


def test_report_is_rendered_in_spanish_with_readable_rows(tmp_path):
    folder = _write(tmp_path / "a", [_shot(6.6, number="7")])
    result = vr.evaluate_clip("brown", folder, {"description": "Un triple.", "shots": [{"outcome": "made", "jersey_number": "7"}]})
    text = vr.render([result], None)
    assert "convertido, camiseta 7" in text and "Límites" in text and "{'" not in text


def test_truth_without_an_outcome_does_not_judge_the_outcome(tmp_path):
    folder = _write(tmp_path / "a", [_shot(6.6, number="30")])
    result = vr.evaluate_clip("a", folder, {"shots": [{"jersey_number": "30"}]})
    assert result["rows"][0]["outcome_ok"] is None and result["rows"][0]["number_ok"] is True
    assert "n/d" in vr.render([result], None)


def test_review_note_is_shown(tmp_path):
    folder = _write(tmp_path / "a", [_shot(6.6, number="8")])
    result = vr.evaluate_clip("d", folder, {"review_note": "Tirador verificado a mano.", "shots": [{"outcome": "made", "jersey_number": "9"}]})
    assert result["rows"][0]["number_ok"] is False
    assert "Nota de revisión: Tirador verificado a mano." in vr.render([result], None)
