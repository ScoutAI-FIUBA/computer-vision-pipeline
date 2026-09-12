from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ball"))

from publish_tracknet_processed import (
    _build_upload_batches,
    _manifest_put_command,
    _remote_path_exists,
    _volume_put_command,
    publish_processed_dataset,
    validate_dataset_id,
)


def _runner_that_materializes(output_root: Path, commands: list[list[str]]):
    uploaded: set[str] = set()

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "preprocess.py" in command[1]:
            (output_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (output_root / "manifest.json").write_text(
                '{"fingerprint": "immutable-fingerprint", "generated_splits": ["train", "val", "test"]}',
                encoding="utf-8",
            )
            for split in ("train", "val", "test"):
                match_root = output_root / split / "match01"
                (match_root / "video").mkdir(parents=True)
                (match_root / "csv").mkdir()
                (match_root / "frame" / "clip01").mkdir(parents=True)
                (match_root / "video" / "clip01.mp4").touch()
                (match_root / "csv" / "clip01_ball.csv").touch()
                (match_root / "frame" / "clip01" / "0.png").touch()
            return subprocess.CompletedProcess(command, 0)
        if command[:4] == ["modal", "volume", "put", "tracknet-data"]:
            source = Path(command[-2])
            destination = command[-1].rstrip("/")
            uploaded.add(f"{destination}/{source.name}" if source.is_dir() else destination)
            return subprocess.CompletedProcess(command, 0)
        if command[:4] == ["modal", "volume", "get", "tracknet-data"]:
            Path(command[-1]).write_text((output_root / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)
        remote_path = command[-1].rstrip("/")
        exists = any(
            remote_path == uploaded_path or remote_path.startswith(f"{uploaded_path}/") or uploaded_path.startswith(f"{remote_path}/")
            for uploaded_path in uploaded
        )
        return subprocess.CompletedProcess(command, 0, stdout='["exists"]' if exists else "[]")

    return run


def _materialize_processed_root(output_root: Path) -> None:
    (output_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        '{"fingerprint": "immutable-fingerprint", "generated_splits": ["train", "val", "test"]}',
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        match_root = output_root / split / "match01"
        (match_root / "video").mkdir(parents=True)
        (match_root / "csv").mkdir()
        (match_root / "frame" / "clip01").mkdir(parents=True)
        (match_root / "video" / "clip01.mp4").touch()
        (match_root / "csv" / "clip01_ball.csv").touch()
        (match_root / "frame" / "clip01" / "0.png").touch()


def test_build_upload_batches_is_deterministic_and_manifest_is_last(tmp_path: Path) -> None:
    output_root = tmp_path / "processed" / "v1"
    for split, games in (("test", ("game_02",)), ("train", ("game_02", "game_01")), ("val", ("game_03",))):
        for game in games:
            (output_root / split / game).mkdir(parents=True)

    batches = _build_upload_batches(output_root, "v1")

    assert batches == [
        (output_root / "train" / "game_01", "/processed/v1/train/"),
        (output_root / "train" / "game_02", "/processed/v1/train/"),
        (output_root / "val" / "game_03", "/processed/v1/val/"),
        (output_root / "test" / "game_02", "/processed/v1/test/"),
        (output_root / "manifest.json", "/processed/v1/manifest.json"),
    ]
    assert all(not destination.rstrip("/").endswith(source.name) for source, destination in batches[:-1])


def test_upload_command_construction_keeps_manifest_after_data_batches(tmp_path: Path) -> None:
    source = tmp_path / "train" / "game_01"
    assert _volume_put_command(source, "/processed/v1/train/") == [
        "modal", "volume", "put", "tracknet-data", str(source), "/processed/v1/train/"
    ]
    assert _manifest_put_command(tmp_path / "manifest.json", "/processed/v1/manifest.json")[-1] == "/processed/v1/manifest.json"


@pytest.mark.parametrize("dataset_id", ["", "../escape", "with space", "name/child"])
def test_validate_dataset_id_rejects_unsafe_path_components(dataset_id: str) -> None:
    with pytest.raises(ValueError, match="dataset ID"):
        validate_dataset_id(dataset_id)


def test_remote_preflight_treats_missing_path_cli_error_as_absent() -> None:
    command: list[str] = []

    def runner(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        command.extend(args)
        return subprocess.CompletedProcess(args, 1, stderr="No such file or directory")

    assert not _remote_path_exists("/processed/v1", runner)
    assert command == ["modal", "volume", "ls", "--json", "tracknet-data", "/processed/v1"]


def test_remote_preflight_treats_modal_missing_path_stdout_as_absent() -> None:
    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            1,
            command,
            output="╭ Error\n╰─ No such file or directory",
            stderr="modal volume ls failed",
        )

    assert not _remote_path_exists("/processed/v1", runner)


def test_remote_preflight_propagates_real_cli_exception() -> None:
    error = subprocess.CalledProcessError(1, ["modal", "volume", "ls"], stderr="Authentication failed")

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise error

    with pytest.raises(subprocess.CalledProcessError) as raised:
        _remote_path_exists("/processed/v1", runner)

    assert raised.value is error


def test_remote_preflight_detects_existing_path() -> None:
    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout='["v1"]')

    assert _remote_path_exists("/processed/v1", runner)


def test_remote_preflight_raises_clear_error_for_cli_failure() -> None:
    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stderr="Authentication failed")

    with pytest.raises(ValueError, match="modal volume ls failed with exit code 1: Authentication failed"):
        _remote_path_exists("/processed/v1", runner)


def test_publish_preprocesses_validates_and_uploads_new_id_without_force(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    output_root = tmp_path / "processed" / "v1"

    manifest = publish_processed_dataset(
        raw_dataset_root=tmp_path / "raw",
        dataset_id="v1",
        processed_base=tmp_path / "processed",
        runner=_runner_that_materializes(output_root, commands),
    )

    assert manifest["fingerprint"] == "immutable-fingerprint"
    assert commands[0][1:] == ["TrackNetV3/preprocess.py", str(tmp_path / "raw"), "--output-root", str(output_root)]
    assert commands[1] == ["modal", "volume", "ls", "--json", "tracknet-data", "/processed/v1"]
    assert [command for command in commands if command[:4] == ["modal", "volume", "put", "tracknet-data"]] == [
        ["modal", "volume", "put", "tracknet-data", str(output_root / "train" / "match01"), "/processed/v1/train/"],
        ["modal", "volume", "put", "tracknet-data", str(output_root / "val" / "match01"), "/processed/v1/val/"],
        ["modal", "volume", "put", "tracknet-data", str(output_root / "test" / "match01"), "/processed/v1/test/"],
        ["modal", "volume", "put", "tracknet-data", str(output_root / "manifest.json"), "/processed/v1/manifest.json"],
    ]
    assert commands[-4:] == [
        ["modal", "volume", "ls", "--json", "tracknet-data", "/processed/v1/manifest.json"],
        ["modal", "volume", "ls", "--json", "tracknet-data", "/processed/v1/train/match01"],
        ["modal", "volume", "ls", "--json", "tracknet-data", "/processed/v1/val/match01"],
        ["modal", "volume", "ls", "--json", "tracknet-data", "/processed/v1/test/match01"],
    ]


def test_publish_reuses_valid_existing_local_output(tmp_path: Path) -> None:
    staging_root = tmp_path / "processed" / "v1"
    _materialize_processed_root(staging_root)
    commands: list[list[str]] = []

    manifest = publish_processed_dataset(
        tmp_path / "raw", "v1", processed_base=tmp_path / "processed",
        runner=_runner_that_materializes(staging_root, commands),
    )

    assert manifest["fingerprint"] == "immutable-fingerprint"
    assert all("preprocess.py" not in command for command in commands)
    assert commands[0] == ["modal", "volume", "ls", "--json", "tracknet-data", "/processed/v1"]


def test_publish_rejects_invalid_existing_local_output(tmp_path: Path) -> None:
    staging_root = tmp_path / "processed" / "v1"
    staging_root.mkdir(parents=True)
    commands: list[list[str]] = []

    with pytest.raises(ValueError, match="existing local processed dataset.*invalid"):
        publish_processed_dataset(
            tmp_path / "raw", "v1", processed_base=tmp_path / "processed",
            runner=lambda command, **_: commands.append(command),
        )

    assert commands == []


def test_publish_reuse_preserves_remote_collision_and_upload_sequence(tmp_path: Path) -> None:
    output_root = tmp_path / "processed" / "v1"
    _materialize_processed_root(output_root)
    commands: list[list[str]] = []

    publish_processed_dataset(
        tmp_path / "raw", "v1", processed_base=tmp_path / "processed",
        runner=_runner_that_materializes(output_root, commands),
    )

    assert [command for command in commands if command[:4] == ["modal", "volume", "put", "tracknet-data"]][-1][-1] == "/processed/v1/manifest.json"


def test_publish_rejects_a_dataset_id_claimed_by_another_local_publisher(tmp_path: Path) -> None:
    processed_base = tmp_path / "processed"
    processed_base.mkdir()
    (processed_base / ".v1.publish.lock").write_text("publishing", encoding="utf-8")

    with pytest.raises(ValueError, match="already being published"):
        publish_processed_dataset(tmp_path / "raw", "v1", processed_base=processed_base)


def test_publish_releases_its_claim_when_preprocessing_fails(tmp_path: Path) -> None:
    processed_base = tmp_path / "processed"

    def failing_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        publish_processed_dataset(tmp_path / "raw", "v1", processed_base=processed_base, runner=failing_runner)

    assert not (processed_base / ".v1.publish.lock").exists()


def test_publish_dry_run_does_not_upload(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    output_root = tmp_path / "processed" / "v1"

    publish_processed_dataset(
        raw_dataset_root=tmp_path / "raw",
        dataset_id="v1",
        processed_base=tmp_path / "processed",
        dry_run=True,
        runner=_runner_that_materializes(output_root, commands),
    )

    assert all(command[:3] != ["modal", "volume", "put"] for command in commands)


def test_publish_rejects_existing_remote_id_before_upload(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    output_root = tmp_path / "processed" / "v1"

    def remote_collision_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "preprocess.py" in command[1]:
            return _runner_that_materializes(output_root, [])(command)
        return subprocess.CompletedProcess(command, 0, stdout='["v1"]')

    with pytest.raises(ValueError, match="could not read remote manifest"):
        publish_processed_dataset(tmp_path / "raw", "v1", processed_base=tmp_path / "processed", runner=remote_collision_runner)

    assert len([command for command in commands if "preprocess.py" in command[1]]) == 1
    assert all(command[:3] != ["modal", "volume", "put"] for command in commands)


def test_publish_rejects_partial_publication_with_different_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "processed" / "v1"
    _materialize_processed_root(output_root)
    commands: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "preprocess.py" in command[1]:
            return subprocess.CompletedProcess(command, 0)
        if command[:4] == ["modal", "volume", "get", "tracknet-data"]:
            Path(command[-1]).write_text('{"fingerprint": "different"}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 0, stdout='["exists"]')

    with pytest.raises(ValueError, match="differs from the local publication"):
        publish_processed_dataset(tmp_path / "raw", "v1", processed_base=tmp_path / "processed", runner=runner)

    assert not any(command[:3] == ["modal", "volume", "put"] for command in commands)


def test_publish_rejects_remote_split_content_without_final_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "processed" / "v1"
    _materialize_processed_root(output_root)
    commands: list[list[str]] = []

    def partial_remote_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "preprocess.py" in command[1]:
            return subprocess.CompletedProcess(command, 0)
        if command[-1] == "/processed/v1/manifest.json":
            return subprocess.CompletedProcess(command, 1, stderr="No such file or directory")
        if command[-1] == "/processed/v1":
            return subprocess.CompletedProcess(command, 0, stdout='["train"]')
        return subprocess.CompletedProcess(command, 0, stdout='["match01"]')

    with pytest.raises(ValueError, match="lacks its final manifest.*new immutable dataset ID"):
        publish_processed_dataset(tmp_path / "raw", "v1", processed_base=tmp_path / "processed", runner=partial_remote_runner)

    assert not any(command[:3] == ["modal", "volume", "put"] for command in commands)


def test_publish_validation_failure_prevents_remote_commands(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="[]")

    with pytest.raises(ValueError, match="manifest"):
        publish_processed_dataset(tmp_path / "raw", "v1", processed_base=tmp_path / "processed", runner=runner)

    assert len(commands) == 1
    assert "preprocess.py" in commands[0][1]


def test_publish_fails_when_remote_verification_is_incomplete(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    output_root = tmp_path / "processed" / "v1"

    def incomplete_remote_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "preprocess.py" in command[1]:
            return _runner_that_materializes(output_root, [])(command)
        if command[:4] == ["modal", "volume", "get", "tracknet-data"]:
            Path(command[-1]).write_text((output_root / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)
        if command[-1].startswith("/processed/v1/val"):
            return subprocess.CompletedProcess(command, 0, stdout="[]")
        return subprocess.CompletedProcess(command, 0, stdout="[]" if command[-1] == "/processed/v1" else '["exists"]')

    with pytest.raises(ValueError, match="/processed/v1/val"):
        publish_processed_dataset(tmp_path / "raw", "v1", processed_base=tmp_path / "processed", runner=incomplete_remote_runner)

    assert any(
        command[:4] == ["modal", "volume", "put", "tracknet-data"]
        and command[-2].endswith("/val/match01")
        for command in commands
    )
