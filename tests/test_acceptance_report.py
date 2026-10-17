import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ops import acceptance_report as ar

TRACK_GEN = "t" * 64
STATS_GEN = "s" * 64

LOG = """\
\x1b[1mInitialized. View run at\x1b[0m
https://modal.com/apps/someworkspace/main/ap-SECRET123
 └── Created function sam31_dependency_smoke_test.
/usr/lib/sam3/model_builder.py:8: UserWarning: pkg_resources is deprecated
/usr/lib/sam3/model_builder.py:8: UserWarning: pkg_resources is deprecated
INFO 2026-10-12 03:03:52,694 2 sam3_base_predictor.py: 164: started new session abc
propagate_in_video:  10%|#  | 15/150 [00:01<00:08, 15.66it/s]\rpropagate_in_video: 100%
INFO 2026-10-12 03:04:12,862 2 sam3_base_predictor.py: 464: removed session abc
         CUDA:0 (NVIDIA A100-SXM4-80GB, 81153MiB)
Authorization: Bearer abc.def-123 user me@example.com at /home/fede/secret/dir
[v2-stats] stage generation validation/hash checks complete: elapsed=311.31s
[v2-stats] in-memory tracking payload: 12 rows
[v2-stats] court projection complete: 12 projected rows
[v2-stats] TrackNet progress: phase complete (100%; processed 6/6 frames)
[v2-stats] shot detection progress: phase complete (100%; processed 6/6 frames)
[v2-stats] stats complete
{
  "generation": "@STATS@",
  "status": "complete",
  "tracking_generation": "@TRACK@"
}
App completed. View run at
https://modal.com/apps/someworkspace/main/ap-SECRET123
""".replace("@STATS@", STATS_GEN).replace("@TRACK@", TRACK_GEN)


def _write_gen(job: Path, stage: str, gen: str, files: dict, extra=None):
    gen_dir = job / stage / "generations" / gen
    gen_dir.mkdir(parents=True)
    artifacts = {}
    for name, data in files.items():
        (gen_dir / name).write_bytes(data)
        artifacts[name] = {
            "key": name, "path": name,
            "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
        }
    manifest = {"artifacts": artifacts, "complete": True, "generation": gen,
                "frame_bounds": [0, 5], "stage": stage}
    manifest.update(extra or {})
    (gen_dir / "manifest.json").write_text(json.dumps(manifest))
    (job / stage / "current.json").write_text(json.dumps({"generation": gen, "stage": stage}))
    return gen_dir


@pytest.fixture
def job(tmp_path):
    root = tmp_path / "job-x"
    clip = b"fake-video-bytes"
    sha = hashlib.sha256(clip).hexdigest()
    (root / "inputs").mkdir(parents=True)
    (root / "inputs" / f"{sha}.mp4").write_bytes(clip)
    (root / "job.json").write_text(json.dumps({
        "job_id": "job-x", "state": "complete", "stage": "stats", "errors": [],
        "attempts": {"stats": 1, "tracking": 1}, "input_fingerprint": f"sha256:{sha}",
        "tracking_generation": TRACK_GEN, "stats_generation": STATS_GEN,
    }))
    tgen = _write_gen(root, "tracking", TRACK_GEN, {"tracks.jsonl": b"a\nb\n"})
    (tgen / "tracking-manifest.json").write_text(json.dumps({
        "observation_count": 12,
        "video": {"fps": 30.0, "frame_count": 6, "width": 64, "height": 48},
        "config": {"prompt": "player", "frame_start": 0, "frame_end": 5},
    }))
    ball = "\n".join(json.dumps({"frame_index": i, "track_state": "lost" if i else "tracked"}) for i in range(6))
    sgen = _write_gen(root, "stats", STATS_GEN, {"ball_tracks.jsonl": ball.encode(), "teams.json": b"{}"})
    (sgen / "results.json").write_text(json.dumps({
        "video": {"duration_seconds": 0.2}, "config": {"tracknet_enabled": True},
    }))
    return root


def test_parse_log_lines():
    log = ar.parse_log(LOG)
    assert log["stage_elapsed_seconds"] == {"stage generation validation/hash checks": 311.31}
    assert log["tracking_rows"] == 12 and log["projected_rows"] == 12
    assert log["stats_complete"] and log["app_completed"]
    assert log["phase_complete"]["TrackNet"]["processed_frames"] == 6
    assert log["final_json"]["status"] == "complete"
    assert log["final_json"]["generation"] == STATS_GEN
    assert log["sam3_sessions"] == 1 and log["sam3_sessions_removed"] == 1
    assert log["sam3_log_span_seconds"] == pytest.approx(20.168)
    assert log["created_functions"] == ["sam31_dependency_smoke_test"]


def test_warnings_deduplicated_and_no_progress_noise():
    warnings = ar.parse_log(LOG)["warnings"]
    assert sum("pkg_resources" in w for w in warnings) == 1
    assert not any("propagate_in_video" in w for w in warnings)


def test_gpu_extraction():
    gpus = ar.parse_log(LOG)["gpus"]
    assert gpus == [{"device": "CUDA:0", "name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81153}]
    assert ar.parse_log("nothing here\n")["gpus"] == []


def test_hash_verification_passes(job):
    v = ar.verify_generation(job / "stats" / "generations" / STATS_GEN)
    assert v["listed_count"] == 2 and v["verified_ok"] == 2
    assert v["mismatched"] == [] and v["missing"] == []


def test_tamper_detection(job):
    gen = job / "stats" / "generations" / STATS_GEN
    (gen / "teams.json").write_bytes(b"{tampered}")
    v = ar.verify_generation(gen)
    assert v["mismatched"] == ["teams.json"]
    assert v["size_mismatched"] == ["teams.json"]
    rep = ar.build_report(LOG, job, commit="abc123")
    assert rep["hash_verification_ok"] is False
    status = {c["item"]: c["status"] for c in rep["checklist"]}
    assert status["Artifact hashes match the manifests"] == ar.FAIL


def test_missing_file_detection(job):
    gen = job / "tracking" / "generations" / TRACK_GEN
    (gen / "tracks.jsonl").unlink()
    v = ar.verify_generation(gen)
    assert v["missing"] == ["tracks.jsonl"] and v["verified_ok"] == 0


def test_redaction_rules():
    counts = {}
    text = (
        "see https://modal.com/apps/ws/main/ap-1 token gho_abcdef123456 ghp_abcdef123456 "
        "key sk-abcdefgh1234 Authorization: Bearer abc.def me@example.com /home/fede/tpf/x"
    )
    out = ar.redact_text(text, counts)
    for leaked in ("modal.com", "gho_", "ghp_", "sk-abc", "abc.def", "me@example.com", "/home/fede"):
        assert leaked not in out
    assert "<modal-url>" in out and "<email>" in out and "<home>/tpf/x" in out
    assert sum(counts.values()) >= 7
    assert ar.redact_obj({"a": ["/home/bob/x"]}) == {"a": ["<home>/x"]}


def test_report_is_redacted_and_says_so(job):
    rep = ar.build_report(LOG, job, commit="abc123")
    blob = json.dumps(rep) + ar.render_markdown(rep)
    for leaked in ("modal.com", "ap-SECRET123", "me@example.com", "/home/fede", "abc.def-123"):
        assert leaked not in blob
    assert rep["redaction"]["applied"] is True and rep["redaction"]["replacements"] >= 4
    assert "Redaction: applied" in ar.render_markdown(rep)


def test_checklist_statuses(job):
    rep = ar.build_report(LOG, job, commit="abc123")
    status = {c["item"]: c["status"] for c in rep["checklist"]}
    by_prefix = lambda p: next(v for k, v in status.items() if k.startswith(p))
    assert by_prefix("SAM3.1 dependency smoke") == ar.NOT_RUN
    assert by_prefix("A short live V2 run") == ar.PASS
    assert by_prefix("The run publishes a non-empty") == ar.PASS
    assert by_prefix("TrackNet produces") == ar.PASS
    assert by_prefix("One-command staged") == ar.PASS
    assert by_prefix("Repeating the same stage") == ar.NOT_RUN
    assert by_prefix("Stats-only retry") == ar.NOT_RUN
    assert by_prefix("Downloaded artifacts render") == ar.NOT_RUN
    assert by_prefix("GPU time, timeout") == ar.NOT_MEASURED
    assert by_prefix("Configured external") == ar.NOT_MEASURED


def test_smoke_test_pass_when_log_shows_it(job):
    rep = ar.build_report(LOG + "sam31 dependency smoke test passed\n", job, commit="x")
    first = rep["checklist"][0]
    assert first["status"] == ar.PASS


def test_replay_output_status(job, tmp_path):
    out = tmp_path / "replay"
    out.mkdir()
    (out / "frame.png").write_bytes(b"png")
    rep = ar.build_report(LOG, job, commit="x", replay_output=str(out))
    assert next(c for c in rep["checklist"] if c["item"].startswith("Downloaded"))["status"] == ar.PASS
    rep = ar.build_report(LOG, job, commit="x", replay_output=str(tmp_path / "nope"))
    assert next(c for c in rep["checklist"] if c["item"].startswith("Downloaded"))["status"] == ar.FAIL


def test_failed_run_is_fail_not_pass(job):
    data = json.loads((job / "job.json").read_text())
    data["state"] = "failed"
    (job / "job.json").write_text(json.dumps(data))
    rep = ar.build_report(LOG, job, commit="x")
    assert rep["checklist"][1]["status"] == ar.FAIL


def test_json_keys_and_values(job):
    rep = ar.build_report(LOG, job, commit="abc123")
    for key in ("job_id", "commit", "input_clip", "gpu_and_configuration", "job", "durations",
                "artifact_summary", "hash_verification", "hash_verification_ok", "storage",
                "warnings", "not_measured", "checklist", "redaction"):
        assert key in rep
    json.dumps(rep)
    assert rep["job_id"] == "job-x" and rep["commit"]["sha"] == "abc123"
    assert rep["input_clip"]["sha256_matches_fingerprint"] is True
    assert rep["input_clip"]["frame_count"] == 6
    assert rep["job"]["consistency"] == {
        "log_stats_generation_matches_job": True, "log_tracking_generation_matches_job": True}
    assert rep["durations"]["total_job_seconds"] == ar.NOT_MEASURED
    assert rep["not_measured"]["cost"]
    assert rep["storage"]["stats"]["manifest_listed_bytes"] > 0
    assert rep["ball_tracks"] == {"rows": 6, "non_lost_rows": 1}


def test_markdown_minimal_job(tmp_path):
    job = tmp_path / "empty"
    job.mkdir()
    (job / "job.json").write_text(json.dumps({"job_id": "min", "state": "running"}))
    rep = ar.build_report("nothing useful\n", job, commit="c0ffee")
    md = ar.render_markdown(rep)
    assert md.startswith("# V2 acceptance report: min")
    assert "NOT_MEASURED" in md and "Acceptance checklist" in md
    assert rep["hash_verification_ok"] is False


def test_cli_json_and_markdown(job, tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text(LOG)
    assert ar.main(["--log", str(log), "--job", str(job), "--commit", "abc", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["job_id"] == "job-x"
    assert ar.main(["--log", str(log), "--job", str(job), "--commit", "abc"]) == 0
    assert "## Acceptance checklist" in capsys.readouterr().out
    (job / "stats" / "generations" / STATS_GEN / "teams.json").write_bytes(b"bad")
    assert ar.main(["--log", str(log), "--job", str(job), "--commit", "abc"]) == 1


# ------------------------------------------------------------ new evidence
def _status(rep, prefix):
    return next(c for c in rep["checklist"] if c["item"].startswith(prefix))


def _stats_dir(job):
    return job / "stats" / "generations" / STATS_GEN


def _track_rows(n_with, n_without):
    rows = [{"frame_index": i, "stable_player_id": f"player-{i}"} for i in range(n_with)]
    rows += [{"frame_index": i} for i in range(n_without)]
    return "\n".join(json.dumps(r) for r in rows).encode()


def _set_tracking(job, manifest_extra, tracks=None):
    gen = job / "tracking" / "generations" / TRACK_GEN
    path = gen / "tracking-manifest.json"
    data = json.loads(path.read_text())
    data.update(manifest_extra)
    path.write_text(json.dumps(data))
    if tracks is not None:
        (_stats_dir(job) / "tracks.jsonl").write_bytes(tracks)


def test_identity_rejected_without_stable_ids_fails(job):
    _set_tracking(job, {
        "acceptance": {"tracking_identity": "rejected"},
        "acceptance_reasons": ["identity_authority:missing_stage_receipt:reid"],
    }, tracks=_track_rows(0, 4))
    rep = ar.build_report(LOG, job, commit="x")
    ident = rep["tracking_identity"]
    assert ident["rows_total"] == 4 and ident["rows_with_stable_player_id"] == 0
    assert ident["rows_without_stable_player_id"] == 4
    row = _status(rep, "Stable player identity")
    assert row["status"] == ar.FAIL
    assert "0 of 4" in row["evidence"] and "missing_stage_receipt" in row["evidence"]
    assert "absent" in ar.render_markdown(rep)  # excluded-observation fields predate this run


def test_identity_accepted_with_stable_ids_passes(job):
    _set_tracking(job, {
        "acceptance": {"tracking_identity": "accepted"},
        "identity_excluded_observations": 3,
        "identity_excluded_observation_keys": ["k1", "k2", "k3"],
    }, tracks=_track_rows(5, 0))
    rep = ar.build_report(LOG, job, commit="x")
    assert _status(rep, "Stable player identity")["status"] == ar.PASS
    assert rep["tracking_identity"]["identity_excluded_observations"] == 3
    assert rep["tracking_identity"]["tracks_source"] == "stats/tracks.jsonl"


def test_identity_excluded_fields_absent_is_not_zero(job):
    rep = ar.build_report(LOG, job, commit="x")
    assert rep["tracking_identity"]["identity_excluded_observations"] is None
    assert "identity_excluded_observations: absent" in ar.render_markdown(rep)
    assert _status(rep, "Stable player identity")["status"] == ar.NOT_MEASURED  # fixture rows unparseable


def test_identity_falls_back_to_tracking_tracks(job):
    (job / "tracking" / "generations" / TRACK_GEN / "tracks.jsonl").write_bytes(_track_rows(2, 1))
    rep = ar.build_report(LOG, job, commit="x")
    assert rep["tracking_identity"]["tracks_source"] == "tracking/tracks.jsonl"
    assert rep["tracking_identity"]["rows_with_stable_player_id"] == 2


def _set_results(job, **extra):
    path = _stats_dir(job) / "results.json"
    data = json.loads(path.read_text())
    data.update(extra)
    path.write_text(json.dumps(data))
    return data


def test_numbers_missing_artifact_fails_with_explanation(job):
    _set_results(job, numbers_file=None, config={"number_predictor_backend": "roboflow-hosted"})
    rep = ar.build_report(LOG, job, commit="x")
    row = _status(rep, "Jersey number OCR")
    assert row["status"] == ar.FAIL
    assert ("configured (NUMBER_PREDICTOR_BACKEND=roboflow-hosted in the image) "
            "but no numbers artifact was produced") in row["evidence"]
    # the integrations row must not claim success and must point at the OCR row
    ext = _status(rep, "Configured external")
    assert ext["status"] == ar.NOT_MEASURED and "Jersey number OCR ran" in ext["evidence"]


def test_numbers_file_named_but_missing_on_disk_fails(job):
    _set_results(job, numbers_file="numbers.json")
    assert _status(ar.build_report(LOG, job, commit="x"), "Jersey number OCR")["status"] == ar.FAIL


def test_numbers_artifact_present_passes(job):
    (_stats_dir(job) / "numbers.json").write_text("{}")
    _set_results(job, numbers_file="numbers.json")
    row = _status(ar.build_report(LOG, job, commit="x"), "Jersey number OCR")
    assert row["status"] == ar.PASS and "not that its numbers are correct" in row["evidence"]


def test_numbers_not_measured_without_results(job):
    (_stats_dir(job) / "results.json").unlink()
    assert _status(ar.build_report(LOG, job, commit="x"), "Jersey number OCR")["status"] == ar.NOT_MEASURED


def _write_outcomes(job, events, shots):
    gen = _stats_dir(job)
    (gen / "shot_events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (gen / "projected_shots.json").write_text(json.dumps({"shots": shots}))


def test_outcome_summary_counts(job):
    _write_outcomes(
        job,
        [{"outcome": "made", "outcome_source": "ball_in_basket"},
         {"outcome": "made", "outcome_source": "rim_trajectory"},
         {"outcome": "missed", "outcome_source": "shot_detector"},
         {"outcome": "missed"}],
        [{"shooter_track_id": 1, "shooter_attribution_method": "action_box_vote"},
         {"shooter_track_id": None, "shooter_stable_player_id": None, "shooter_attribution_method": "none"},
         {"shooter_stable_player_id": "player-001", "shooter_attribution_method": "action_box_vote"}],
    )
    out = ar.build_report(LOG, job, commit="x")["outcome_summary"]
    assert out["events"] == 4 and out["projected_shots"] == 3
    assert out["by_outcome"] == {"made": 2, "missed": 2}
    assert out["by_outcome_source"] == {"absent": 1, "ball_in_basket": 1, "rim_trajectory": 1, "shot_detector": 1}
    assert out["by_shooter_attribution_method"] == {"action_box_vote": 2, "none": 1}
    assert out["shots_with_null_shooter"] == 1
    assert out["ball_tracks"] == {"rows": 6, "non_lost_rows": 1}
    assert "Outcome summary" in ar.render_markdown(ar.build_report(LOG, job, commit="x"))


def test_outcome_fields_absent_when_not_emitted(job):
    _write_outcomes(job, [{"outcome": "made"}], [{"shooter_track_id": 1}])
    out = ar.build_report(LOG, job, commit="x")["outcome_summary"]
    assert out["by_outcome_source"] is None and out["by_shooter_attribution_method"] is None
    assert out["shots_with_null_shooter"] == 0
    assert "absent (field not emitted)" in ar.render_markdown(ar.build_report(LOG, job, commit="x"))


def test_outcome_summary_without_artifacts(job):
    out = ar.build_report(LOG, job, commit="x")["outcome_summary"]
    assert out["events"] is None and out["shots_with_null_shooter"] is None


def _mp4(path, frames):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 10 % 255, dtype=np.uint8))
    writer.release()
    assert path.is_file()


def test_replay_mp4_frame_count_matches(job, tmp_path):
    video = tmp_path / "replay.mp4"
    _mp4(video, 6)  # frame_bounds [0, 5]
    row = _status(ar.build_report(LOG, job, commit="x", replay_output=str(video)), "Downloaded")
    assert row["status"] == ar.PASS and "6 frames" in row["evidence"]


def test_replay_mp4_frame_count_mismatch_fails(job, tmp_path):
    video = tmp_path / "replay.mp4"
    _mp4(video, 4)
    row = _status(ar.build_report(LOG, job, commit="x", replay_output=str(video)), "Downloaded")
    assert row["status"] == ar.FAIL and "4 frames" in row["evidence"] and "imply 6" in row["evidence"]


def test_replay_mp4_found_inside_directory(job, tmp_path):
    out = tmp_path / "render"
    out.mkdir()
    _mp4(out / "debug.mp4", 6)
    assert _status(ar.build_report(LOG, job, commit="x", replay_output=str(out)), "Downloaded")["status"] == ar.PASS


def test_replay_corrupt_mp4_fails(job, tmp_path):
    video = tmp_path / "replay.mp4"
    video.write_bytes(b"not a video")
    row = _status(ar.build_report(LOG, job, commit="x", replay_output=str(video)), "Downloaded")
    assert row["status"] == ar.FAIL and "could not be opened" in row["evidence"]


def test_replay_absent_stays_not_run(job):
    assert _status(ar.build_report(LOG, job, commit="x"), "Downloaded")["status"] == ar.NOT_RUN


def test_notes_embedded_and_redacted(job):
    notes = "## Findings\nOCR key leaked sk-abcdefgh1234 mailed to me@example.com from /home/fede/x\n"
    rep = ar.build_report(LOG, job, commit="x", notes=notes)
    md = ar.render_markdown(rep)
    assert "## Operational findings" in md and "## Findings" in md
    for leaked in ("sk-abcdefgh1234", "me@example.com", "/home/fede/x"):
        assert leaked not in md and leaked not in json.dumps(rep)
    assert "<token>" in md and "<email>" in md


def test_no_notes_invents_nothing(job):
    rep = ar.build_report(LOG, job, commit="x")
    assert rep["operational_notes"] is None
    assert "No operational notes were provided" in ar.render_markdown(rep)


def test_gpu_strings_per_log(job):
    other = LOG.replace("NVIDIA A100-SXM4-80GB, 81153MiB", "NVIDIA A100 80GB PCIe, 81153MiB")
    rep = ar.build_report(LOG, job, commit="x", also_logs={"second.log": other})
    cfg = rep["gpu_and_configuration"]
    assert cfg["gpu_strings_by_log"] == {
        "primary": ["NVIDIA A100-SXM4-80GB, 81153MiB"], "second.log": ["NVIDIA A100 80GB PCIe, 81153MiB"]}
    assert cfg["gpu_variants_differ_across_logs"] is True
    assert "different GPU variants" in ar.render_markdown(rep)
    same = ar.build_report(LOG, job, commit="x", also_logs={"same.log": LOG})
    assert same["gpu_and_configuration"]["gpu_variants_differ_across_logs"] is False


def test_diff_mode(job, tmp_path, capsys):
    a = ar.build_report(LOG, job, commit="x")
    _set_tracking(job, {"acceptance": {"tracking_identity": "accepted"}}, tracks=_track_rows(3, 0))
    (_stats_dir(job) / "numbers.json").write_text("{}")
    _set_results(job, numbers_file="numbers.json")
    b = ar.build_report(LOG, job, commit="x")
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    assert ar.main(["--diff", str(pa), str(pb)]) == 0
    out = capsys.readouterr().out
    assert "checklist: Jersey number OCR ran" in out and "FAIL" in out and "PASS" in out
    assert "| tracking_identity | absent | \"accepted\" |" in out
    assert "numbers_file" in out and "rows_with_stable_player_id" in out
    assert ar.main(["--diff", str(pa), str(pa)]) == 0
    assert "no differences" in capsys.readouterr().out


def test_diff_mode_unreadable_still_exits_zero(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    assert ar.main(["--diff", str(bad), str(tmp_path / "missing.json")]) == 0
    assert "diff skipped" in capsys.readouterr().out


def test_cli_requires_log_and_job_without_diff():
    with pytest.raises(SystemExit):
        ar.main([])


def test_cli_notes_and_also_log(job, tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text(LOG)
    notes = tmp_path / "notes.md"
    notes.write_text("manual finding sk-abcdefgh1234\n")
    assert ar.main(["--log", str(log), "--job", str(job), "--commit", "a", "--notes", str(notes),
                    "--also-log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "manual finding <token>" in out and "GPU string, run.log log" in out


# ------------------------------------------------- performance and cost
def _metrics(stage, elapsed, margin, timeout=3600, api=None, **extra):
    doc = {
        "schema": "v2-stage-metrics-1", "stage": stage,
        "started_at": "2026-10-12T03:00:00Z", "finished_at": "2026-10-12T03:05:00Z",
        "elapsed_seconds": elapsed, "phases": {"load_model_seconds": 10.5, "inference_seconds": elapsed - 10.5},
        "gpu": {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81153},
        "function_timeout_seconds": timeout, "timeout_margin_seconds": margin,
        "container": {"cpu": 4, "memory_mib": 32768},
        "frames": {"requested": 6, "processed": 6},
        "api_calls": api or {k: None for k in ar.API_KEYS}, "attempt": 1,
    }
    doc.update(extra)
    return doc


def _put_metrics(job, stage, gen, doc):
    (job / stage / "generations" / gen / "stage_metrics.json").write_text(json.dumps(doc))


TIMING = (
    "[v2-timing] stage=tracking started=2026-10-12T03:00:00Z finished=2026-10-12T03:02:00Z "
    "elapsed=120.0 timeout_margin=3480.0 gpu=NVIDIA A100-SXM4-80GB\n"
    "[v2-timing] stage=stats started=2026-10-12T03:02:10Z finished=2026-10-12T03:03:10Z "
    "elapsed=60.5 timeout_margin=3539.5 gpu=none\n"
    "[v2-timing] job id=job-x total_wall_seconds=250.0\n"
)


@pytest.fixture
def mjob(job):
    _put_metrics(job, "tracking", TRACK_GEN, _metrics("tracking", 120.0, 3480.0))
    _put_metrics(job, "stats", STATS_GEN, _metrics(
        "stats", 60.0, 540.0, timeout=600,
        api={"roboflow_shot_detection_frames": 6, "roboflow_court_detection_frames": None,
             "roboflow_number_detector_frames": 0, "roboflow_ocr_crops": None}))
    return job


def test_parse_timing_lines():
    log = ar.parse_log(LOG + TIMING)
    assert [t["stage"] for t in log["timing_stages"]] == ["tracking", "stats"]
    t = log["timing_stages"][0]
    assert t["elapsed_seconds"] == 120.0 and t["timeout_margin_seconds"] == 3480.0
    assert t["gpu"] == "NVIDIA A100-SXM4-80GB" and log["timing_stages"][1]["gpu"] is None
    assert log["timing_jobs"] == [{"job_id": "job-x", "total_wall_seconds": 250.0}]
    assert ar.parse_log("[v2-timing] job id=none total_wall_seconds=3.5\n")["timing_jobs"][0]["job_id"] is None
    assert ar.parse_log(LOG)["timing_stages"] == []


def test_stage_metrics_parsed_and_arithmetic(mjob):
    p = ar.build_report(LOG + TIMING, mjob, commit="x")["performance"]
    assert p["stage_metrics"]["status"] == "MEASURED"
    assert p["stages"]["tracking"]["source"] == "stage_metrics.json"
    assert p["stages"]["tracking"]["phases"]["load_model_seconds"] == 10.5
    assert p["elapsed_sum_seconds"] == 180.0 and p["elapsed_sum_complete"] is True
    assert p["total_wall_seconds"] == 250.0 and p["overhead_seconds"] == 70.0
    assert p["stages"]["stats"]["timeout_used_percent"] == 10.0
    t = p["timeout"]
    assert t["smallest_margin_seconds"] == 540.0 and t["smallest_margin_stage"] == "stats"
    assert t["percent_of_timeout_used_at_smallest_margin"] == 10.0
    assert p["gpu"]["names"] == ["NVIDIA A100-SXM4-80GB"] and p["gpu"]["memory_mib"] == [81153]
    assert p["stages"]["tracking"]["log_crosscheck"]["agrees"] is True
    assert p["stages"]["stats"]["log_crosscheck"] == {
        "log_elapsed_seconds": 60.5, "difference_seconds": 0.5, "agrees": True}  # within 1 s
    assert p["generations"]["total_count"] == 2 and p["generations"]["by_stage"]["tracking"]["count"] == 1


def test_null_api_calls_stay_unknown(mjob):
    p = ar.build_report(LOG, mjob, commit="x")["performance"]
    totals = p["api_calls"]["totals"]
    assert totals["roboflow_ocr_crops"]["total"] is None
    assert totals["roboflow_shot_detection_frames"] == {"total": 6, "unknown_in_stages": ["tracking"], "partial": True}
    assert totals["roboflow_number_detector_frames"]["total"] == 0
    md = ar.render_markdown(ar.build_report(LOG, mjob, commit="x"))
    assert "roboflow_ocr_crops: total unknown" in md


def test_current_generation_metrics_win(mjob):
    other = "o" * 64
    gdir = mjob / "stats" / "generations" / other
    gdir.mkdir()
    (gdir / "stage_metrics.json").write_text(json.dumps(_metrics("stats", 999.0, 1.0)))
    p = ar.build_report(LOG, mjob, commit="x")["performance"]
    assert p["stages"]["stats"]["elapsed_seconds"] == 60.0
    assert p["generations"]["by_stage"]["stats"]["count"] == 2


def test_old_run_is_not_measured_with_reason(job):
    rep = ar.build_report(LOG, job, commit="x")
    p = rep["performance"]
    assert p["stage_metrics"] == {
        "status": ar.NOT_MEASURED, "reason": "run predates stage metrics",
        "source": "stage_metrics.json per generation or [v2-timing] log lines"}
    assert p["elapsed_sum_seconds"] is None and p["timeout"] is None
    assert p["cost"]["status"] == ar.NOT_MEASURED
    assert _status(rep, "Operational metrics")["status"] == ar.NOT_MEASURED
    assert "run predates stage metrics" in _status(rep, "Operational metrics")["evidence"]
    md = ar.render_markdown(rep)
    assert "## Performance and cost" in md and "run predates stage metrics" in md
    assert "311.31 s" in md  # hash-check duration still shown


def test_log_lines_are_fallback(job):
    p = ar.build_report(LOG + TIMING, job, commit="x")["performance"]
    assert p["stages"]["tracking"]["source"] == "log [v2-timing]"
    assert p["elapsed_sum_seconds"] == 180.5 and p["overhead_seconds"] == pytest.approx(69.5)
    assert p["stages"]["stats"]["function_timeout_derived"] is True
    assert p["stages"]["stats"]["gpu"] is None


def test_cost_not_measured_without_rate(mjob):
    c = ar.build_report(LOG + TIMING, mjob, commit="x")["performance"]["cost"]
    assert c["status"] == ar.NOT_MEASURED and c["reason"] == "pass the rate from the Modal plan"
    assert c["per_video_minute"]["status"] == ar.NOT_MEASURED


def test_cost_arithmetic_with_rate_and_video_seconds(mjob):
    rep = ar.build_report(LOG + TIMING, mjob, commit="x", gpu_rate_usd_per_hour=3.6, video_seconds=120.0)
    c = rep["performance"]["cost"]
    assert c["estimated_usd_gpu_stage_time"] == pytest.approx(180 / 3600 * 3.6)  # 0.18
    assert c["estimated_usd_upper_bound_wall_time"] == pytest.approx(250 / 3600 * 3.6)  # 0.25
    pm = c["per_video_minute"]
    assert pm["usd_per_video_minute_gpu_stage_time"] == pytest.approx(0.09)
    assert pm["usd_per_video_minute_upper_bound"] == pytest.approx(0.125)
    assert c["video_seconds_source"] == "--video-seconds"
    md = ar.render_markdown(rep)
    assert "not the invoice" in md and "cold start" in md


def test_video_seconds_derived_from_manifest(mjob):
    c = ar.build_report(LOG, mjob, commit="x", gpu_rate_usd_per_hour=3.6)["performance"]["cost"]
    assert c["video_seconds"] == pytest.approx(6 / 30.0)  # frame_bounds [0, 5] at 30 fps
    assert "frame_bounds" in c["video_seconds_source"]
    assert c["estimated_usd_upper_bound_wall_time"] is None  # no wall line in this log


def test_video_seconds_not_measured_without_fps(mjob):
    tm_path = mjob / "tracking" / "generations" / TRACK_GEN / "tracking-manifest.json"
    tm_path.write_text(json.dumps({"observation_count": 12}))
    c = ar.build_report(LOG, mjob, commit="x", gpu_rate_usd_per_hour=3.6)["performance"]["cost"]
    assert c["video_seconds"] is None and c["per_video_minute"]["status"] == ar.NOT_MEASURED


def test_rate_without_timings_stays_not_measured(job):
    c = ar.build_report(LOG, job, commit="x", gpu_rate_usd_per_hour=2.5)["performance"]["cost"]
    assert c["status"] == ar.NOT_MEASURED and "run predates stage metrics" in c["reason"]
    assert "estimated_usd_gpu_stage_time" not in c


def test_volume_listing_parsing():
    assert ar.parse_volume_listing('{"a/x": 100, "a/y": 250}')["total_bytes"] == 350
    text = (
        "┃ Filename        ┃ Type ┃ Created/Modified ┃ Size     ┃\n"
        "│ job/tracking    │ dir  │ 2026-10-12       │          │\n"
        "│ job/job.json    │ file │ 2026-10-12       │ 1.5 KiB  │\n"
        "│ job/a.mp4       │ file │ 2026-10-12       │ 2 MiB    │\n"
    )
    r = ar.parse_volume_listing(text)
    assert r["total_bytes"] == 1536 + 2 * 1024**2 and r["approximate"] is True and r["files"] == 2
    plain = ar.parse_volume_listing("job/a 1000\njob/b 24\nnot a size here\n")
    assert plain["total_bytes"] == 1024 and plain["unparsed_lines"] == 1
    assert ar.parse_volume_listing("garbage\n")["total_bytes"] is None


def test_volume_growth_from_two_listings(mjob, tmp_path):
    before, after = tmp_path / "b.json", tmp_path / "a.txt"
    before.write_text(json.dumps({"job/inputs/x.mp4": 1000}))
    after.write_text("job/inputs/x.mp4 1000\njob/tracking/g/tracks.jsonl 5000\n")
    rep = ar.build_report(LOG, mjob, commit="x", volume_listing_before=str(before), volume_listing_after=str(after))
    g = rep["performance"]["volume_growth"]
    assert g["status"] == "MEASURED" and g["growth_bytes"] == 5000
    assert "Volume growth: 5000 bytes" in ar.render_markdown(rep)
    nog = ar.build_report(LOG, mjob, commit="x", volume_listing_before=str(tmp_path / "missing"),
                          volume_listing_after=str(after))["performance"]["volume_growth"]
    assert nog["status"] == ar.NOT_MEASURED and "published_bytes_as_downloaded" in nog


def test_volume_growth_not_measured_without_listings(mjob):
    g = ar.build_report(LOG, mjob, commit="x")["performance"]["volume_growth"]
    assert g["status"] == ar.NOT_MEASURED and g["published_bytes_as_downloaded"] > 0


def test_volume_storage_cost(mjob, tmp_path):
    p = ar.build_report(LOG, mjob, commit="x")["performance"]
    assert p["volume_cost"]["status"] == ar.NOT_MEASURED and "rate" in p["volume_cost"]["reason"]
    before, after = tmp_path / "b.json", tmp_path / "a.json"
    before.write_text("{}")
    after.write_text(json.dumps({"f": 2 * 1024**3}))
    kw = dict(volume_listing_before=str(before), volume_listing_after=str(after), volume_rate_usd_per_gib_month=0.5)
    vc = ar.build_report(LOG, mjob, commit="x", volume_included_gib=0.5, **kw)["performance"]["volume_cost"]
    assert vc["billable_gib"] == pytest.approx(1.5) and vc["usd_per_month"] == pytest.approx(0.75)
    vc = ar.build_report(LOG, mjob, commit="x", volume_included_gib=5, **kw)["performance"]["volume_cost"]
    assert vc["usd_per_month"] == 0


def test_retries_and_errors_reported(mjob):
    data = json.loads((mjob / "job.json").read_text())
    data["attempts"] = {"tracking": 2, "stats": 1}
    data["errors"] = ["tracking failed at /home/fede/x"]
    (mjob / "job.json").write_text(json.dumps(data))
    rep = ar.build_report(LOG, mjob, commit="x")
    r = rep["performance"]["retries"]
    assert r["retried_stages"] == ["tracking"] and r["recorded_error_count"] == 1
    assert "/home/fede" not in json.dumps(rep) and "<home>" in json.dumps(rep["performance"])


def test_reused_stage_without_metrics_is_listed(mjob):
    (mjob / "stats" / "generations" / STATS_GEN / "stage_metrics.json").unlink()
    rep = ar.build_report(LOG, mjob, commit="x")
    sm = rep["performance"]["stage_metrics"]
    assert sm["status"] == "MEASURED" and sm["stages_without_metrics"] == ["stats"]
    assert rep["performance"]["elapsed_sum_complete"] is False
    assert _status(rep, "Operational metrics")["status"] == ar.NOT_MEASURED
    assert "stats" in _status(rep, "Operational metrics")["evidence"]


def test_checklist_ops_row_statuses(mjob, tmp_path):
    row = lambda rep: _status(rep, "GPU time, timeout")
    rep = ar.build_report(LOG, mjob, commit="x")
    assert row(rep)["status"] == ar.NOT_MEASURED and "Volume growth" in row(rep)["evidence"]
    assert "GPU/stage time" not in row(rep)["evidence"] and "download size" not in row(rep)["evidence"].split("Present")[0]
    assert _status(rep, "Operational metrics")["status"] == ar.PASS
    before, after = tmp_path / "b.json", tmp_path / "a.json"
    before.write_text("{}")
    after.write_text('{"f": 10}')
    rep = ar.build_report(LOG, mjob, commit="x", volume_listing_before=str(before), volume_listing_after=str(after))
    assert row(rep)["status"] == ar.PASS


def test_ops_row_lists_all_missing_for_old_run(job):
    ev = _status(ar.build_report(LOG, job, commit="x"), "GPU time, timeout")
    assert ev["status"] == ar.NOT_MEASURED
    assert "run predates stage metrics" in ev["evidence"] and "timeout margin" in ev["evidence"]
    assert "Volume growth" in ev["evidence"]


def test_new_text_is_redacted(mjob, tmp_path):
    data = json.loads((mjob / "job.json").read_text())
    data["errors"] = ["boom me@example.com https://modal.com/apps/ws/main/ap-9 /home/fede/z"]
    (mjob / "job.json").write_text(json.dumps(data))
    log = tmp_path / "run.log"
    log.write_text(LOG + TIMING)
    out = tmp_path / "out.md"
    assert ar.main(["--log", str(log), "--job", str(mjob), "--commit", "x", "--gpu-rate-usd-per-hour", "2.5",
                    "--output", str(out)]) == 0
    text = out.read_text()
    for leaked in ("me@example.com", "modal.com", "/home/fede"):
        assert leaked not in text
    assert "<email>" in text and "Cost per video minute" in text


def test_diff_includes_performance_fields(mjob, job, tmp_path, capsys):
    a = ar.build_report(LOG + TIMING, mjob, commit="x", gpu_rate_usd_per_hour=3.6, video_seconds=120.0)
    _put_metrics(mjob, "stats", STATS_GEN, _metrics("stats", 90.0, 510.0, timeout=600))
    b = ar.build_report(LOG + TIMING, mjob, commit="x", gpu_rate_usd_per_hour=3.6, video_seconds=120.0)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    assert ar.main(["--diff", str(pa), str(pb)]) == 0
    out = capsys.readouterr().out
    assert "perf.elapsed_seconds.stats | 60.0 | 90.0" in out
    assert "perf.smallest_timeout_margin_seconds | 540.0 | 510.0" in out
    assert "perf.cost_usd_stage_time" in out and "perf.elapsed_sum_seconds" in out
    assert ar.main(["--diff", str(pa), str(pa)]) == 0
    assert "no differences" in capsys.readouterr().out
