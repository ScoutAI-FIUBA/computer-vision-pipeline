from __future__ import annotations

import json
import types

import modal_app_v2

RUN_INFO_KEYS = {
    "court", "tracking", "acceptance_reasons", "input_sha256", "segment_overlap",
    "pipeline_git_sha", "sam31_commit",
}


def _process_fn():
    cls = next(v for k, v in modal_app_v2.PipelineV2.__dict__.items() if k.startswith("_sync_original"))._user_cls
    return cls, cls.process.__dict__[next(k for k in cls.process.__dict__ if k.startswith("_sync_original"))].raw_f


def test_process_payload_carries_json_serialisable_run_info(monkeypatch):
    import numpy as np

    monkeypatch.setenv("PIPELINE_GIT_SHA", "abc1234")
    row = {
        "observation_key": "60:7", "frame_index": 60, "raw_track_id": 7, "stable_player_id": "player-007",
        "bbox_xyxy": [10, 10, 30, 50], "track_state": "tracked",
    }
    monkeypatch.setattr(modal_app_v2, "_track_video", lambda *_a: ([dict(row)], {"60:7": np.ones((2, 2), dtype=np.uint8)}))
    monkeypatch.setattr(modal_app_v2, "_load_video_frames", lambda *_a: {})
    monkeypatch.setattr(modal_app_v2, "_video_metadata", lambda _p: {})
    monkeypatch.setattr(
        modal_app_v2, "run_v2",
        lambda rows, **_k: types.SimpleNamespace(
            observations=rows, acceptance={}, acceptance_reasons=["identity_excluded_invalid_rows:2"],
        ),
    )

    def fake_stats(_video, output_dir, _observations, *, court_run_info, **_kwargs):
        court_run_info.update(frames_total=10, frames_projected=9, frames_failed=1, errors_by_type={"HTTPCallErrorError": 1})
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text("{}", encoding="utf-8")
        (output_dir / "tracks.jsonl").write_text("", encoding="utf-8")
        return types.SimpleNamespace(to_dict=lambda: {
            "tracks_file": "tracks.jsonl", "teams_file": None, "numbers_file": None, "shot_events_file": None,
        })

    monkeypatch.setattr(modal_app_v2, "_run_v2_stats", fake_stats)
    cls, process = _process_fn()
    monkeypatch.setattr(cls, "_ensure_predictor", lambda _self: object())

    payload = process(cls(), video_bytes=b"fake-video", frame_start=60, frame_end=60, return_artifacts=True)

    json.dumps({k: v for k, v in payload.items() if k != "artifacts"})
    run_info = payload["run_info"]
    assert set(run_info) == RUN_INFO_KEYS
    assert run_info["court"] == {
        "frames_total": 10, "frames_projected": 9, "frames_failed": 1, "errors_by_type": {"HTTPCallErrorError": 1},
    }
    assert run_info["tracking"] == {"segments": None, "overflow_dropped_ids": None, "seams": None, "seams_linked": None}
    assert run_info["acceptance_reasons"] == ["identity_excluded_invalid_rows:2"]
    import hashlib
    assert run_info["input_sha256"] == hashlib.sha256(b"fake-video").hexdigest()
    assert run_info["segment_overlap"] is None
    assert run_info["pipeline_git_sha"] == "abc1234"
    assert run_info["sam31_commit"] == modal_app_v2.SAM31_COMMIT
    assert {"acceptance", "timings", "gpu", "frames", "artifacts"} <= set(payload)


def test_build_run_info_counts_segmented_diagnostics_without_bodies():
    info = modal_app_v2._build_run_info(
        court_counters={},
        seam_diagnostics=[
            {"event": "matched"}, {"event": "matched"}, {"event": "ambiguous"}, {"event": "ownership"},
        ],
        overflow_diagnostics=[
            {"event": "segment_object_overflow", "dropped_raw_ids": [3, 4], "ranking": [{"big": 1}]},
            {"event": "segment_object_overflow", "dropped_raw_ids": [9]},
        ],
        segment_count=4, segment_overlap=20, acceptance_reasons=None, input_sha256="x",
    )
    assert info["tracking"] == {"segments": 4, "overflow_dropped_ids": 3, "seams": 3, "seams_linked": 2}
    assert info["court"] == {"frames_total": None, "frames_projected": None, "frames_failed": None, "errors_by_type": None}
    assert info["acceptance_reasons"] == [] and info["segment_overlap"] == 20
    json.dumps(info)


def test_pipeline_git_sha_falls_back_to_none_without_env_or_git(monkeypatch):
    monkeypatch.delenv("PIPELINE_GIT_SHA", raising=False)

    def no_git(*_a, **_k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(modal_app_v2.subprocess, "run", no_git)
    assert modal_app_v2._deploy_git_sha() is None
    assert modal_app_v2._build_run_info(
        court_counters={}, seam_diagnostics=None, overflow_diagnostics=None, segment_count=None,
        segment_overlap=None, acceptance_reasons=[], input_sha256=None,
    )["pipeline_git_sha"] is None


def test_deploy_git_sha_reads_git_when_env_unset(monkeypatch):
    monkeypatch.delenv("PIPELINE_GIT_SHA", raising=False)
    monkeypatch.setattr(
        modal_app_v2.subprocess, "run",
        lambda *_a, **_k: types.SimpleNamespace(returncode=0, stdout="deadbee\n"),
    )
    assert modal_app_v2._deploy_git_sha() == "deadbee"
