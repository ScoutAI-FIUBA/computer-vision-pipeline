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
