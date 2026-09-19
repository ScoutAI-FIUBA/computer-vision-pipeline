from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_v2_image_isolated_and_uses_only_modal_model_storage():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")

    assert "modal_app.py" not in source
    assert "facebookresearch/sam3" in source
    assert "facebookresearch/sam2" not in source
    assert "R2" not in source
    assert "r2_key" not in source
    assert "output_prefix" not in source
    assert "R2Storage" not in source
    assert '"HF_HOME":' in source
    assert "HF_TOKEN" in source
    assert source.count("modal.Image") == 1


def test_entrypoint_accepts_nonempty_video_and_validates_public_values():
    from modal_app_v2 import validate_entrypoint_args

    validate_entrypoint_args(video_bytes=b"video", prompt="player", frame_start=0, frame_end=1)

    with pytest.raises(ValueError, match="non-empty"):
        validate_entrypoint_args(video_bytes=b"", prompt="player", frame_start=0, frame_end=1)
    with pytest.raises(ValueError, match="prompt"):
        validate_entrypoint_args(video_bytes=b"video", prompt=" ", frame_start=0, frame_end=1)
    with pytest.raises(ValueError, match="frame"):
        validate_entrypoint_args(video_bytes=b"video", prompt="player", frame_start=2, frame_end=1)


@pytest.mark.parametrize(
    ("video_bytes", "prompt", "frame_start", "frame_end"),
    [
        ("video", "player", 0, 1),
        (b"video", "player", -1, 1),
        (b"video", "player", True, 1),
        (b"video", "player", 0, 1.0),
    ],
)
def test_entrypoint_rejects_wrong_public_types(video_bytes, prompt, frame_start, frame_end):
    from modal_app_v2 import validate_entrypoint_args

    with pytest.raises(ValueError):
        validate_entrypoint_args(
            video_bytes=video_bytes,
            prompt=prompt,
            frame_start=frame_start,
            frame_end=frame_end,
        )


def test_direct_artifact_mapping_returns_relative_bytes(tmp_path: Path):
    from modal_app_v2 import collect_artifacts

    (tmp_path / "manifest.json").write_text('{"complete": false}', encoding="utf-8")
    (tmp_path / "masks").mkdir()
    (tmp_path / "masks" / "1.png").write_bytes(b"png")

    artifacts = collect_artifacts(tmp_path)

    assert artifacts == {
        "manifest.json": b'{"complete": false}',
        "masks/1.png": b"png",
    }
    assert all(not Path(name).is_absolute() for name in artifacts)


def test_direct_artifact_mapping_rejects_configured_size_limit(tmp_path: Path):
    from modal_app_v2 import collect_artifacts

    (tmp_path / "tracks.jsonl").write_bytes(b"1234")

    with pytest.raises(ValueError, match="exceeds configured artifact return limit"):
        collect_artifacts(tmp_path, max_bytes=3)


def test_direct_artifact_mapping_reads_size_limit_from_environment(tmp_path: Path, monkeypatch):
    from modal_app_v2 import collect_artifacts

    (tmp_path / "tracks.jsonl").write_bytes(b"1234")
    monkeypatch.setenv("SAM31_MAX_ARTIFACT_BYTES", "3")

    with pytest.raises(ValueError, match="exceeds configured artifact return limit"):
        collect_artifacts(tmp_path)


def test_direct_artifact_mapping_rejects_symlinked_file_outside_output_root(tmp_path: Path):
    from modal_app_v2 import collect_artifacts

    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    outside = tmp_path / "secret.bin"
    outside.write_bytes(b"secret")
    (output_dir / "tracks.jsonl").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink|outside"):
        collect_artifacts(output_dir)


def test_local_output_writer_recreates_nested_artifact_paths(tmp_path: Path):
    from modal_app_v2 import write_artifacts

    write_artifacts(
        {"manifest.json": b"{}", "masks/1.png": b"png"},
        tmp_path / "output",
    )

    assert (tmp_path / "output" / "manifest.json").read_bytes() == b"{}"
    assert (tmp_path / "output" / "masks" / "1.png").read_bytes() == b"png"


@pytest.mark.parametrize("relative_path", ["", ".", "/tmp/out", "../escape", "masks/../../escape", "bad\\path"])
def test_local_output_writer_rejects_unsafe_artifact_paths(tmp_path: Path, relative_path: str):
    from modal_app_v2 import write_artifacts

    with pytest.raises(ValueError, match="artifact path"):
        write_artifacts({relative_path: b"nope"}, tmp_path / "output")

    assert not (tmp_path / "escape").exists()


def test_local_entrypoint_writes_remote_artifact_bytes(tmp_path: Path, monkeypatch, capsys):
    import modal_app_v2

    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")

    class FakeProcess:
        def remote(self, **kwargs):
            assert kwargs["video_bytes"] == b"video"
            assert kwargs["return_artifacts"] is True
            return {"acceptance": {"tracking_identity": "unvalidated"}, "artifacts": {"manifest.json": b"{}"}}

    class FakePipeline:
        process = FakeProcess()

    monkeypatch.setattr(modal_app_v2, "PipelineV2", FakePipeline)
    modal_app_v2.main(video_path=str(video), output_dir=str(tmp_path / "output"))

    assert (tmp_path / "output" / "manifest.json").read_bytes() == b"{}"
    assert "tracking_identity" in capsys.readouterr().out


def test_v2_process_returns_artifacts_and_never_remote_debug_video():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")

    assert 'payload["artifacts"]' in source
    assert "return_artifacts" in source
    assert "debug.mp4" not in source
    assert "upload" not in source.lower()


def test_failed_or_incomplete_local_manifest_is_returned_without_claiming_complete(tmp_path: Path):
    from modal_app_v2 import collect_artifacts

    manifest = {"complete": False, "truncated": True, "transfer_errors": ["artifact too large"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    returned = json.loads(collect_artifacts(tmp_path)["manifest.json"])

    assert returned == manifest


def test_legacy_modal_file_is_not_imported_by_v2():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    assert "import modal_app" not in source
