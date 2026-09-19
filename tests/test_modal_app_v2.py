from __future__ import annotations

from pathlib import Path

import pytest


def test_v2_image_isolated_and_uses_sam3_without_legacy_image_dependencies():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")

    assert "modal_app.py" not in source
    assert "facebookresearch/sam3" in source
    assert "facebookresearch/sam2" not in source
    assert '"HF_HOME":' in source
    assert "HF_TOKEN" in source
    assert source.count("modal.Image") == 1


def test_entrypoint_requires_one_video_source_and_an_artifact_prefix():
    from modal_app_v2 import validate_entrypoint_args

    validate_entrypoint_args(video_bytes=b"video", r2_key=None, output_prefix="runs/job")
    validate_entrypoint_args(video_bytes=None, r2_key="input.mp4", output_prefix="runs/job")

    with pytest.raises(ValueError, match="exactly one"):
        validate_entrypoint_args(video_bytes=b"video", r2_key="input.mp4", output_prefix="runs/job")
    with pytest.raises(ValueError, match="exactly one"):
        validate_entrypoint_args(video_bytes=None, r2_key=None, output_prefix="runs/job")
    with pytest.raises(ValueError, match="output_prefix"):
        validate_entrypoint_args(video_bytes=b"video", r2_key=None, output_prefix="")


def test_artifact_response_returns_retrieval_references_not_file_bytes():
    from modal_app_v2 import artifact_retrieval_response

    response = artifact_retrieval_response(
        {"complete": True, "artifacts": {"manifest": {"key": "runs/job/manifest.json"}}},
        ["runs/job/manifest.json", "runs/job/tracks.jsonl"],
    )

    assert response == {
        "manifest": {"complete": True, "key": "runs/job/manifest.json"},
        "artifact_keys": ["runs/job/manifest.json", "runs/job/tracks.jsonl"],
    }
    assert "artifacts" not in response


def test_v2_upload_reports_transfer_errors_and_never_creates_debug_video(tmp_path: Path, monkeypatch):
    from modal_app_v2 import upload_artifacts

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    class BrokenStorage:
        def upload_directory(self, _directory, _prefix):
            raise OSError("R2 unavailable")

    monkeypatch.setattr("modal_app_v2.R2Storage", BrokenStorage)
    keys, errors = upload_artifacts(tmp_path, "runs/job")
    assert keys == []
    assert errors == ["OSError: R2 unavailable"]
    assert not (tmp_path / "debug.mp4").exists()

    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    assert "debug.mp4" not in source


def test_legacy_modal_file_is_not_imported_by_v2():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    assert "import modal_app" not in source
