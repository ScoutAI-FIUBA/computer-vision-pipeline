from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tracking import prepare_eval_dataset as ped

FPS = 10.0


def make_video(path: Path, frames: int = 120, fps: float = FPS, size=(64, 48)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), (i * 2) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "src.mp4")


def run_prepare(video, out, windows=("a:10:70:crossing:hard",), **kwargs):
    defaults = dict(
        videos=[video], windows=[ped.parse_window(w) for w in windows], output_dir=out,
        target_fps=15, frame_stride=10, run_labels=("baseline", "candidate"), notes={},
        sheet_frames=6, repo_root=Path(__file__).resolve().parents[1],
    )
    defaults.update(kwargs)
    return ped.prepare_dataset(**defaults)


def test_probe_video(video):
    info = ped.probe_video(video)
    assert (info.width, info.height, info.frame_count) == (64, 48, 120)
    assert info.fps == pytest.approx(FPS)
    assert info.duration_seconds == pytest.approx(12.0)


def test_parse_window_ok_and_optional_source():
    w = ped.parse_window("c1:10:70:full_occlusion:easy:src")
    assert (w.clip, w.start_frame, w.end_frame, w.scenario, w.difficulty, w.source) == (
        "c1", 10, 70, "full_occlusion", "easy", "src")
    assert ped.parse_window("c1:10:70:crossing:easy").source is None


@pytest.mark.parametrize("spec", [
    "c1:10:70:crossing", "c1:x:70:crossing:easy", "c1:70:10:crossing:easy",
    "c1:10:70:dunk:easy", "c1:10:70:crossing:brutal", "bad/name:10:70:crossing:easy",
])
def test_parse_window_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        ped.parse_window(spec)


def test_window_shorter_than_five_seconds_rejected(video):
    info = ped.probe_video(video)
    with pytest.raises(ValueError, match="at least 5"):
        ped.validate_window(ped.parse_window("c:0:49:crossing:easy"), info)  # 4.9 s
    ped.validate_window(ped.parse_window("c:0:50:crossing:easy"), info)  # exactly 5 s


def test_window_beyond_video_rejected(video):
    with pytest.raises(ValueError, match="only 120 frames"):
        ped.validate_window(ped.parse_window("c:100:200:crossing:easy"), ped.probe_video(video))


def test_resolve_source_requires_disambiguation(tmp_path, video):
    other = make_video(tmp_path / "other.mp4")
    w = ped.parse_window("c:0:60:crossing:easy")
    with pytest.raises(ValueError, match="trailing :source"):
        ped.resolve_window_source(w, [video, other])
    w2 = ped.parse_window("c:0:60:crossing:easy:other")
    assert ped.resolve_window_source(w2, [video, other]) == other


def test_discover_videos_from_directory(tmp_path, video):
    (tmp_path / "notes.txt").write_text("x")
    assert ped.discover_videos([], tmp_path) == [video]


def test_prepare_writes_manifest_clip_sheet_and_dirs(tmp_path, video):
    out = tmp_path / "ds"
    manifest = run_prepare(video, out, notes={"a": "same player split near frame 40"})
    on_disk = json.loads((out / ped.MANIFEST_NAME).read_text())
    assert on_disk == manifest
    assert ped.validate_manifest(on_disk) == []
    entry = on_disk["clips"][0]
    for field in ped.REQUIRED_CLIP_FIELDS:
        assert field in entry
    assert entry["path"] == "clips/a.mp4"  # deterministic relative path
    assert (entry["start_frame"], entry["end_frame"]) == (10, 70)
    assert entry["duration_seconds"] == pytest.approx(6.0)
    assert (entry["width"], entry["height"]) == (64, 48)
    assert entry["configuration"] == {"target_fps": 15, "frame_stride": 10}
    assert entry["review_status"] == "pending"
    assert entry["notes"] == "same player split near frame 40"
    assert entry["encoding"]["reencoded"] is True
    assert len(entry["source_commit"]) in (40, 7) or entry["source_commit"] == "unknown"
    clip_info = ped.probe_video(out / entry["path"])
    assert clip_info.frame_count == 60
    assert (out / entry["contact_sheet"]).stat().st_size > 0
    assert entry["contact_sheet_frames"][0] == 10  # labelled with source frame numbers
    for label in ("baseline", "candidate"):
        assert (out / "runs" / label / "a" / "annotated").is_dir()


def test_manifest_scenario_is_only_what_human_gave(tmp_path, video):
    manifest = run_prepare(video, tmp_path / "ds", windows=("a:0:60:camera_motion:easy",))
    assert manifest["clips"][0]["scenario"] == "camera_motion"


def test_review_template_is_all_pending(tmp_path, video):
    out = tmp_path / "ds"
    run_prepare(video, out)
    lines = (out / ped.REVIEW_NAME).read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["record_type"] == "clip_review"
    assert record["clip"] == "a" and record["declared_scenario"] == "crossing"
    for field in ("scenario_present", "id_switch", "full_loss", "recovery", "stable_identity_involved"):
        assert record[field] == "pending"
    assert record["stable_player_id"] is None


def test_refuses_to_overwrite_dataset(tmp_path, video):
    out = tmp_path / "ds"
    run_prepare(video, out)
    with pytest.raises(FileExistsError):
        run_prepare(video, out)


def test_run_dirs_are_immutable(tmp_path):
    ped.create_run_dirs(tmp_path, "baseline", ["a"])
    marker = tmp_path / "runs" / "baseline" / "a" / "annotated" / "tracks.jsonl"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        ped.create_run_dirs(tmp_path, "baseline", ["a"])
    assert marker.read_text() == "keep"
    ped.create_run_dirs(tmp_path, "candidate", ["a"])  # separate label is fine


def test_nothing_written_when_a_window_is_too_short(tmp_path, video):
    out = tmp_path / "ds"
    with pytest.raises(ValueError, match="at least 5"):
        run_prepare(video, out, windows=("a:0:60:crossing:easy", "b:0:20:crossing:easy"))
    assert not out.exists()


def test_contact_sheet_from_synthetic_video(tmp_path, video):
    sheet = tmp_path / "sheet.jpg"
    indices = ped.make_contact_sheet(video, sheet, count=8, columns=4, thumb_width=32)
    assert len(indices) == 8 and indices[0] == 0 and indices[-1] == 119
    image = cv2.imread(str(sheet))
    assert image.shape[1] == 4 * 32 and image.shape[0] == 2 * 24


def test_suggested_run_command_is_staged_and_marked(tmp_path, video):
    manifest = run_prepare(video, tmp_path / "ds")
    entry = manifest["clips"][0]
    cmd = entry["suggested_run_command"]
    assert cmd == entry["suggested_run_commands"]["baseline"]
    assert cmd.startswith("modal run modal_app_v2.py --staged --job-id eval-a-baseline ")
    assert f"--video-path {(tmp_path / 'ds' / 'clips' / 'a.mp4').resolve()}" in cmd
    assert "--output-dir" not in cmd
    # 60-frame clip: manifest end is exclusive, Modal's --frame-end is inclusive.
    assert "--frame-start 0 --frame-end 59" in cmd
    assert entry["frame_count"] == 60 and entry["modal_frame_end_inclusive"] == 59
    assert "--segmented --segment-window 150 --segment-overlap 20 --possession-engine v5" in cmd
    assert entry["suggested_job_ids"] == {"baseline": "eval-a-baseline", "candidate": "eval-a-candidate"}
    assert "--job-id eval-a-candidate " in entry["suggested_run_commands"]["candidate"]
    assert entry["command_executed"] is False
    assert "NOT EXECUTED" in manifest["suggested_run_command_note"]


def test_validate_manifest_catches_problems(tmp_path, video):
    manifest = run_prepare(video, tmp_path / "ds")
    assert ped.validate_manifest({"clips": []})
    bad = json.loads(json.dumps(manifest))
    bad["clips"][0]["scenario"] = "dunk"
    bad["clips"][0]["duration_seconds"] = 2.0
    del bad["clips"][0]["source_commit"]
    assert any("missing fields source_commit" in e for e in ped.validate_manifest(bad))
    bad = json.loads(json.dumps(manifest))
    bad["clips"][0]["scenario"] = "dunk"
    bad["clips"][0]["duration_seconds"] = 2.0
    errors = ped.validate_manifest(bad)
    assert any("scenario" in e for e in errors) and any("duration_seconds" in e for e in errors)


def test_cli_validate_mode(tmp_path, video, capsys):
    out = tmp_path / "ds"
    run_prepare(video, out)
    assert ped.main(["--validate", str(out / ped.MANIFEST_NAME)]) == 0
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"clips": [{"clip": "x"}]}))
    assert ped.main(["--validate", str(broken)]) == 1


def test_cli_end_to_end_and_overwrite_refusal(tmp_path, video, capsys):
    out = tmp_path / "ds"
    argv = ["--output-dir", str(out), "--source", str(video), "--window", "a:0:60:crossing:easy"]
    assert ped.main(argv) == 0
    assert "NOT EXECUTED" in capsys.readouterr().out
    assert ped.main(argv) == 1
    assert ped.main(["--output-dir", str(tmp_path / "ds2"), "--source", str(video),
                     "--window", "a:0:40:crossing:easy"]) == 1
