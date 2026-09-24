import json
import pytest
from pipeline.calibration import evaluate_clips, load_clips, write_report


def test_calibration_counts_unknown_and_holder_errors():
    clips = [{"clip_id": "fixture-1", "frames": [
        {"frame_index": 0, "predicted_holder": "p1", "ground_truth_holder": "p1"},
        {"frame_index": 1, "predicted_holder": "p1", "ground_truth_holder": "UNKNOWN"},
        {"frame_index": 2, "predicted_holder": "UNKNOWN", "ground_truth_holder": "p2"},
        {"frame_index": 3, "predicted_holder": "p2", "ground_truth_holder": "p2",
         "predicted_state": "GRACE"},
    ]}]
    metrics = evaluate_clips(clips)["metrics"]
    assert metrics["holder_accuracy"] == .5
    assert metrics["false_holder_rate"] == 1.0
    assert metrics["unknown_rate"] == .25
    assert metrics["grace_recovery_rate"] == 1.0


def test_recovery_thresholds_are_higher_is_better():
    clips = [{"clip_id": "recovery", "difficulty": "medium", "frames": [
        {"frame_index": 0, "predicted_holder": "UNKNOWN", "ground_truth_holder": "UNKNOWN"},
        {"frame_index": 1, "predicted_holder": "p1", "ground_truth_holder": "p1",
         "predicted_state": "GRACE"},
    ]}]
    report = evaluate_clips(clips, thresholds={"grace_recovery_rate": .8,
                                               "unknown_recovery_rate": .8})
    assert report["metrics"]["grace_recovery_rate"] == 1.0
    assert report["promotion_gate"]["checks"]["grace_recovery_rate"] is True
    assert report["metrics"]["unknown_recovery_rate"] == 1.0
    assert report["promotion_gate"]["checks"]["unknown_recovery_rate"] is True
def test_calibration_event_hooks_and_deterministic_report(tmp_path):
    clip = {"clip_id": "events", "frames": [],
            "primitive_events": [{"event_type": "release", "frame_index": 2}],
            "ground_truth_primitive_events": [{"event_type": "release", "frame_index": 2},
                                               {"event_type": "reception", "frame_index": 3}],
            "semantic_events": [{"event_type": "PASS", "frame_index": 2}],
            "ground_truth_semantic_events": [{"event_type": "PASS", "frame_index": 2}]}
    first = evaluate_clips([clip], profile="pilot")
    second = evaluate_clips([clip], profile="pilot")
    assert first == second
    assert first["metrics"]["primitive_precision"] == 1.0
    assert first["metrics"]["primitive_recall"] == .5
    assert first["metrics"]["semantic_f1"] == 1.0
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(clip) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"
    write_report(path, output, profile="pilot")
    written = json.loads(output.read_text())
    assert written["metrics"] == first["metrics"]
    assert written["profile"]["name"] == first["profile"]["name"]
    assert len(written["input_sha256"]) == 64
    assert load_clips(path) == [clip]
def test_event_metrics_do_not_match_events_across_clip_boundaries():
    clips = [
        {"clip_id": "predicted", "frames": [],
         "primitive_events": [{"event_type": "release", "frame_index": 2}]},
        {"clip_id": "truth", "frames": [],
         "ground_truth_primitive_events": [{"event_type": "release", "frame_index": 2}]},
    ]
    assert evaluate_clips(clips)["metrics"]["primitive_f1"] == 0.0


@pytest.mark.parametrize("events", [[{"frame_index": 1}], [{"event_type": "release", "frame_index": "1"}], [None]])
def test_malformed_event_evidence_cannot_score_perfect_f1(events):
    clip = {"clip_id": "bad-events", "frames": [],
            "primitive_events": events,
            "ground_truth_primitive_events": [{"event_type": "release", "frame_index": 1}]}
    report = evaluate_clips([clip])
    assert report["metrics"]["primitive_f1"] is None
    assert report["promotion_gate"]["checks"]["primitive_f1"] is False
def test_recovery_resets_at_clip_boundaries_and_report_has_rollout_fields():
    clips = [
        {"clip_id": "easy-1", "difficulty": "easy", "frames": [
            {"frame_index": 0, "predicted_holder": "UNKNOWN", "ground_truth_holder": "UNKNOWN"},
        ]},
        {"clip_id": "hard-1", "difficulty": "hard", "frames": [
            {"frame_index": 0, "predicted_holder": "p2", "ground_truth_holder": "p2"},
        ]},
    ]
    report = evaluate_clips(
        clips, thresholds={"holder_accuracy": .5}, feature_flag="v3",
        benchmark_reference="calibration-fixture-v1", configuration={"fps": 10},
        comparison={"legacy": {"holder_accuracy": .4}, "v2": {"holder_accuracy": .45}},
    )
    assert report["metrics"]["unknown_recovery_rate"] is None
    assert report["metrics"]["coverage"] == 1.0
    assert report["metrics"]["contested_rate"] == 0.0
    assert report["fixture_classification"] == {"easy": 1, "medium": 0, "hard": 1}
    assert report["event_counts"]["semantic"]["predicted"] == 0
    assert report["event_counts"]["primitive"]["ground_truth_by_type"] == {}
    assert report["legacy_v2_comparison"]["legacy"]["holder_accuracy"] == .4
    assert report["promotion_gate"]["decision"] == "REJECT"
def test_traceability_and_gate_can_promote_labeled_fixture(tmp_path):
    clip = {"clip_id": "medium-1", "difficulty": "medium", "frames": [
        {"frame_index": 0, "predicted_holder": "p1", "ground_truth_holder": "p1"},
    ]}
    source = tmp_path / "clips.jsonl"
    source.write_text(json.dumps(clip) + "\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    report = write_report(source, report_path, feature_flag="v3", configuration={"fps": 15})
    assert report["promotion_gate"]["decision"] == "REJECT"
    assert report["authoritative"] is False
    assert report["traceability"]["feature_flag"] == "v3"
    assert report["traceability"]["configuration"] == {"fps": 15}
    assert report["benchmark_input_sha256"] != report["input_sha256"]
def _complete_clips(count=10):
    clips = []
    for index in range(count):
        clips.append({"clip_id": f"clip-{index}", "difficulty": ("easy", "medium", "hard")[index % 3],
                      "frames": [{"frame_index": n, "predicted_holder": "p1", "ground_truth_holder": "p1"} for n in (0, 1)] +
                               [{"frame_index": 2, "predicted_holder": "UNKNOWN", "ground_truth_holder": "UNKNOWN"}],
                      "primitive_events": [{"event_type": "release", "frame_index": 0}], "ground_truth_primitive_events": [{"event_type": "release", "frame_index": 0}],
                      "semantic_events": [{"event_type": "PASS", "frame_index": 0}], "ground_truth_semantic_events": [{"event_type": "PASS", "frame_index": 0}]})
    return clips
def test_promotion_requires_complete_evidence_and_exact_decisions():
    comparison = {"legacy": {"holder_accuracy": .7}, "v2": {"holder_accuracy": .7}}
    complete = evaluate_clips(_complete_clips(), comparison=comparison, replay_evidence=True)
    assert complete["promotion_gate"]["decision"] == "PROMOTE"
    assert complete["authoritative"] is True
    assert evaluate_clips(_complete_clips(9), comparison=comparison, replay_evidence=True)["promotion_gate"]["decision"] == "REJECT"
    no_difficulty = _complete_clips()
    for clip in no_difficulty:
        clip["difficulty"] = "medium"
    assert evaluate_clips(no_difficulty, comparison=comparison, replay_evidence=True)["promotion_gate"]["decision"] == "REJECT"
    assert evaluate_clips(_complete_clips(), replay_evidence=True)["promotion_gate"]["decision"] == "REJECT"
    assert evaluate_clips(_complete_clips(), comparison={"legacy": {}, "v2": {}}, replay_evidence=True)["promotion_gate"]["decision"] == "REJECT"


@pytest.mark.parametrize("comparison", [
    {"legacy": {"holder_accuracy": "0.9"}, "v2": {"holder_accuracy": .9}},
    {"legacy": ["holder_accuracy", .9], "v2": {"holder_accuracy": .9}},
    {"legacy": {"holder_accuracy": float("nan")}, "v2": {"holder_accuracy": .9}},
])
def test_malformed_comparison_evidence_cannot_promote(comparison):
    report = evaluate_clips(_complete_clips(), comparison=comparison, replay_evidence=True)
    assert report["promotion_gate"]["checks"]["legacy_v2_comparison"] is False
    assert report["promotion_gate"]["decision"] == "REJECT"


def test_missing_or_failing_required_metrics_reject():
    comparison = {"legacy": {}, "v2": {}}
    missing_clips = _complete_clips()
    for clip in missing_clips:
        for frame in clip["frames"]:
            frame.pop("ground_truth_holder")
    missing = evaluate_clips(missing_clips, comparison=comparison, replay_evidence=True)
    assert missing["promotion_gate"]["decision"] == "REJECT"
    bad = _complete_clips()
    for clip in bad:
        for frame in clip["frames"]:
            frame["predicted_holder"] = "p2"
    assert evaluate_clips(bad, comparison=comparison, replay_evidence=True)["promotion_gate"]["decision"] == "REJECT"
def test_threshold_configuration_is_validated():
    with pytest.raises(ValueError, match="thresholds"):
        evaluate_clips(_complete_clips(), thresholds={"holder_accuracy": 2})


@pytest.mark.parametrize(("clips", "message"), [
    ([{"clip_id": "", "frames": []}], "each clip requires"),
    ([{"clip_id": "same", "frames": []}, {"clip_id": "same", "frames": []}], "clip_id values"),
])
def test_clip_identity_validation_is_exact(clips, message):
    with pytest.raises(ValueError, match=message):
        evaluate_clips(clips)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"profile": "missing"}, "unknown calibration profile"),
    ({"minimum_clips": 0}, "clip limits"),
    ({"minimum_clips": 11, "maximum_clips": 10}, "clip limits"),
    ({"calibration_mode": "legacy"}, "calibration_mode"),
    ({"calibration_mode": "benchmark", "minimum_clips": 2}, "benchmark minimum"),
    ({"thresholds": {"not_a_metric": .5}}, "thresholds"),
])
def test_calibration_configuration_validation_is_exact(kwargs, message):
    with pytest.raises(ValueError, match=message):
        evaluate_clips(_complete_clips(), **kwargs)
def test_clip_bounds_are_traceable_and_non_authoritative_mode_is_explicit(tmp_path):
    clips = _complete_clips()
    comparison = {"legacy": {"holder_accuracy": .7}, "v2": {"holder_accuracy": .7}}
    source = tmp_path / "clips.jsonl"
    source.write_text("\n".join(json.dumps(clip) for clip in clips) + "\n", encoding="utf-8")
    first = write_report(source, tmp_path / "first.json", comparison=comparison,
                         minimum_clips=10, maximum_clips=12)
    second = write_report(source, tmp_path / "second.json", comparison=comparison,
                          minimum_clips=10, maximum_clips=30)
    assert first["promotion_gate"]["decision"] == "PROMOTE"
    assert first["traceability"]["clip_bounds"] == {"minimum": 10, "maximum": 12}
    assert first["benchmark_input_sha256"] != second["benchmark_input_sha256"]
    exploratory = evaluate_clips(clips[:1], calibration_mode="non_authoritative",
                                  minimum_clips=1, maximum_clips=30)
    assert exploratory["promotion_gate"]["decision"] == "REJECT"
    assert exploratory["authoritative"] is False
