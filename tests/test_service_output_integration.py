"""WP-A (service quality) fed with the run_info that WP-B (GPU app) really builds.

Each package is tested alone against hand-built dictionaries for the other side; this is the
only place where the real producer and the real consumer meet.
"""
import json
from pathlib import Path

import pytest

import modal_app_v2 as gpu
import modal_app_v2_service as service

JOBS = Path(__file__).parent / "fixtures" / "service_output" / "jobs"
RERUN = JOBS / "3931fd9f"  # clip_027 re-run: every file present, court projection worked
FAILED = JOBS / "c9fad24b"  # clip_027 first run: court projection silently produced nothing


def _inputs(folder: Path):
    results = json.loads((folder / "results.json").read_text())
    meta = json.loads((folder / "meta.json").read_text())
    return results, meta


def _run_info(*, court, overflow=None, seams=None, segments=9):
    return gpu._build_run_info(
        court_counters=court, seam_diagnostics=seams or [], overflow_diagnostics=overflow or [],
        segment_count=segments, segment_overlap=12, acceptance_reasons=["identity_ok"],
        input_sha256="ab" * 32,
    )


def _quality(folder: Path, run_info, *, uploaded=None):
    results, meta = _inputs(folder)
    return service.build_quality(
        results, uploaded if uploaded is not None else list(meta["files"]), [], run_info,
        acceptance=meta["acceptance"], team_assignments=meta["team_assignments"],
    )


def test_a_clean_court_run_reads_ok_for_the_court_stage():
    info = _run_info(court={"frames_total": 703, "frames_projected": 703, "frames_failed": 0, "errors_by_type": {}})
    quality = _quality(RERUN, info)
    court = quality["stages"]["court_projection"]
    assert (court["frames_total"], court["frames_projected"], court["frames_failed"]) == (703, 703, 0)
    assert quality["level"] != "critical"
    assert not any(r["code"].startswith("court") for r in quality["reasons"])


def test_a_few_failed_frames_stay_below_the_degraded_threshold_a_many_do_not():
    few = _run_info(court={"frames_total": 703, "frames_projected": 690, "frames_failed": 13, "errors_by_type": {"HTTPCallErrorError": 13}})
    many = _run_info(court={"frames_total": 703, "frames_projected": 600, "frames_failed": 103, "errors_by_type": {"HTTPCallErrorError": 103}})
    assert not any(r["code"].startswith("court") for r in _quality(RERUN, few)["reasons"])
    reasons = _quality(RERUN, many)["reasons"]
    assert any(r["code"].startswith("court") and r["severity"] == "degraded" for r in reasons)
    assert _quality(RERUN, many)["stages"]["court_projection"]["errors_by_type"] == {"HTTPCallErrorError": 103}


def test_the_silent_clip_027_failure_is_critical_with_or_without_run_info():
    nothing = _run_info(court={"frames_total": 703, "frames_projected": 0, "frames_failed": 703, "errors_by_type": {"HTTPCallErrorError": 1}})
    for info in (nothing, None):
        quality = _quality(FAILED, info)
        assert quality["level"] == "critical"
        assert any(r["code"] == "court_projection_empty" for r in quality["reasons"])


def test_tracking_counts_flow_through_and_overflow_degrades():
    overflow = [{"event": "segment_object_overflow", "dropped_raw_ids": [1, 2]}]
    seams = [{"event": "matched"}, {"event": "matched"}, {"event": "ambiguous"}, {"event": "skipped"}]
    info = _run_info(court={"frames_total": 1, "frames_projected": 1, "frames_failed": 0, "errors_by_type": {}},
                     overflow=overflow, seams=seams)
    tracking = _quality(RERUN, info)["stages"]["tracking"]
    assert tracking == {"segments": 9, "overflow_dropped_ids": 2, "seams": 3, "seams_linked": 2}
    assert any(r["code"].startswith("overflow") or "overflow" in r["code"] for r in _quality(RERUN, info)["reasons"])


def test_provenance_and_the_whole_summary_serialise():
    info = _run_info(court={"frames_total": 703, "frames_projected": 703, "frames_failed": 0, "errors_by_type": {}})
    results, meta = _inputs(RERUN)
    provenance = service.build_provenance(results, engine="v5", window=96, prompt="basketball player on the court", run_info=info, v3_generation="6b6b616c3c06ceb6")
    assert provenance["segment_overlap"] == 12 and provenance["input_sha256"] == "ab" * 32
    assert provenance["possession_v3_generation"] == "6b6b616c3c06ceb6"
    json.dumps({"quality": _quality(RERUN, info), "provenance": provenance, "contract_version": 2})


def test_absent_run_info_never_invents_values():
    quality = _quality(RERUN, None)
    assert quality["stages"]["court_projection"]["frames_failed"] is None
    assert quality["stages"]["tracking"]["segments"] is None
