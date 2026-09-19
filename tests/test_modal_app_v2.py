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


@pytest.mark.parametrize(
    ("video_bytes", "prompt", "frame_start", "frame_end"),
    [
        (b"", "player", 0, 1),
        (b"video", " ", 0, 1),
        (b"video", "player", -1, 1),
        (b"video", "player", 2, 1),
        (b"video", "player", True, 1),
        (b"video", "player", 0, 1.0),
    ],
)
def test_entrypoint_rejects_invalid_public_values(video_bytes, prompt, frame_start, frame_end):
    from modal_app_v2 import validate_entrypoint_args

    with pytest.raises(ValueError):
        validate_entrypoint_args(
            video_bytes=video_bytes,
            r2_key=None,
            output_prefix="runs/job",
            prompt=prompt,
            frame_start=frame_start,
            frame_end=frame_end,
        )


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


def test_artifact_response_cannot_advertise_complete_with_transfer_errors():
    from modal_app_v2 import artifact_retrieval_response

    response = artifact_retrieval_response(
        {
            "complete": True,
            "truncated": False,
            "transfer_errors": ["upload failed"],
            "artifacts": {"manifest": {"key": "runs/job/manifest.json"}},
        },
        ["runs/job/manifest.json"],
    )

    assert response["manifest"]["complete"] is False


def test_v2_upload_reports_transfer_errors_and_never_creates_debug_video(tmp_path: Path, monkeypatch):
    from modal_app_v2 import upload_artifacts

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    class BrokenStorage:
        def upload(self, _source, _key, _content_type=None):
            raise OSError("R2 unavailable")

    monkeypatch.setattr("modal_app_v2.R2Storage", BrokenStorage)
    keys, errors = upload_artifacts(tmp_path, "runs/job")
    assert keys == []
    assert errors == ["OSError: R2 unavailable"]
    assert not (tmp_path / "debug.mp4").exists()

    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    assert "debug.mp4" not in source


def test_failed_upload_overwrites_manifest_with_incomplete_state(tmp_path: Path, monkeypatch):
    from modal_app_v2 import upload_artifacts

    (tmp_path / "manifest.json").write_text(
        '{"complete": true, "truncated": false, "transfer_errors": [], '
        '"artifacts": {"manifest": {"key": "runs/job/manifest.json"}}}',
        encoding="utf-8",
    )
    (tmp_path / "tracks.jsonl").write_text("{}\n", encoding="utf-8")
    uploaded_manifests = []

    class BrokenStorage:
        def upload(self, source, key, content_type=None):
            if key.endswith("manifest.json"):
                uploaded_manifests.append(source.read_text(encoding="utf-8"))
            if key.endswith("tracks.jsonl"):
                raise OSError("R2 unavailable")
            return key

    monkeypatch.setattr("modal_app_v2.R2Storage", BrokenStorage)
    keys, errors = upload_artifacts(tmp_path, "runs/job")

    assert keys == ["runs/job/manifest.json"]
    assert errors == ["OSError: R2 unavailable"]
    assert uploaded_manifests
    failed_manifest = __import__("json").loads(uploaded_manifests[-1])
    assert failed_manifest["complete"] is False
    assert failed_manifest["truncated"] is True
    assert failed_manifest["transfer_errors"] == ["OSError: R2 unavailable"]


def test_successful_upload_republishes_final_complete_manifest(tmp_path: Path, monkeypatch):
    from modal_app_v2 import upload_artifacts

    (tmp_path / "manifest.json").write_text(
        '{"complete": true, "truncated": false, "transfer_errors": [], '
        '"artifacts": {"manifest": {"key": "runs/job/manifest.json"}}}',
        encoding="utf-8",
    )
    (tmp_path / "tracks.jsonl").write_text("{}\n", encoding="utf-8")
    remote_manifests = []

    class RecordingStorage:
        def upload(self, source, key, content_type=None):
            if key.endswith("manifest.json"):
                remote_manifests.append(__import__("json").loads(source.read_text(encoding="utf-8")))
            return key

    monkeypatch.setattr("modal_app_v2.R2Storage", RecordingStorage)
    keys, errors = upload_artifacts(tmp_path, "runs/job")

    assert errors == []
    assert keys == ["runs/job/manifest.json", "runs/job/tracks.jsonl"]
    assert [manifest["complete"] for manifest in remote_manifests] == [False, True]
    assert remote_manifests[-1]["truncated"] is False
    assert remote_manifests[-1]["transfer_errors"] == []


def test_session_output_rejects_cardinality_mismatches():
    from modal_app_v2 import _session_objects

    with pytest.raises(RuntimeError, match="boxes"):
        _session_objects({"out_obj_ids": [1], "out_boxes_xywh": []})
    with pytest.raises(RuntimeError, match="probabilities"):
        _session_objects({"out_obj_ids": [1], "out_boxes_xywh": [[0, 0, 1, 1]], "out_probs": []})
    with pytest.raises(RuntimeError, match="masks"):
        _session_objects({"out_obj_ids": [1], "out_boxes_xywh": [[0, 0, 1, 1]], "out_binary_masks": []})


def test_session_rows_include_explicit_mask_references_and_metadata_helpers():
    from modal_app_v2 import _add_mask_references, _effective_run_metadata

    rows = [{"observation_key": "60:3"}]
    _add_mask_references(rows, {"60:3": object()}, "runs/job")
    assert rows[0]["mask_ref"] == "runs/job/masks/" + __import__("hashlib").sha256(b"60:3").hexdigest() + ".png"
    assert _effective_run_metadata("player", 60, 130) == {
        "prompt": "player",
        "frame_start": 60,
        "frame_end": 130,
    }


def test_legacy_modal_file_is_not_imported_by_v2():
    source = (Path(__file__).parents[1] / "modal_app_v2.py").read_text(encoding="utf-8")
    assert "import modal_app" not in source
