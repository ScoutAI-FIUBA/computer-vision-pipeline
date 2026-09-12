from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "TrackNetV3"))

from basketball_data import validate_processed_root


VOLUME_NAME = "tracknet-data"
SAFE_DATASET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
MISSING_REMOTE_PATH_MARKER = "no such file or directory"
Runner = Callable[..., subprocess.CompletedProcess[str]]


def validate_dataset_id(dataset_id: str) -> str:
    if not SAFE_DATASET_ID.fullmatch(dataset_id):
        raise ValueError(f"unsafe dataset ID {dataset_id!r}; choose a new dataset ID using letters, digits, dots, hyphens, or underscores")
    return dataset_id


def _volume_ls_command(remote_path: str) -> list[str]:
    return ["modal", "volume", "ls", "--json", VOLUME_NAME, remote_path]


def _volume_put_command(source: Path, remote_path: str) -> list[str]:
    return ["modal", "volume", "put", VOLUME_NAME, str(source), remote_path]


def _manifest_put_command(source: Path, remote_path: str) -> list[str]:
    return _volume_put_command(source, remote_path)


def _build_upload_batches(output_root: Path, dataset_id: str) -> list[tuple[Path, str]]:
    batches = [
        (game, f"/processed/{dataset_id}/{split}/")
        for split in ("train", "val", "test")
        for game in sorted((path for path in (output_root / split).iterdir() if path.is_dir()), key=lambda path: path.name)
    ]
    batches.append((output_root / "manifest.json", f"/processed/{dataset_id}/manifest.json"))
    return batches


def _remote_manifest_matches(remote_path: str, manifest: dict, runner: Runner) -> bool | None:
    if not _remote_path_exists(remote_path, runner):
        return None
    with TemporaryDirectory() as temporary_directory:
        local_manifest = Path(temporary_directory) / "manifest.json"
        runner(["modal", "volume", "get", VOLUME_NAME, remote_path, str(local_manifest)], check=True)
        try:
            remote_manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read remote manifest {remote_path}") from exc
    if remote_manifest != manifest:
        raise ValueError(f"remote manifest at {remote_path} differs from the local publication")
    return True


def _remote_game_path(source: Path, remote_path: str) -> str:
    return f"{remote_path.rstrip('/')}/{source.name}"


def _verify_remote_dataset_complete(
    remote_root: str,
    manifest_path: str,
    manifest: dict,
    batches: list[tuple[Path, str]],
    runner: Runner,
) -> None:
    if not _remote_manifest_matches(manifest_path, manifest, runner):
        raise ValueError(
            f"remote dataset {remote_root} has remote content but lacks its final manifest; "
            "clean it up or publish under a new immutable dataset ID"
        )
    missing = [
        _remote_game_path(source, remote_path)
        for source, remote_path in batches[:-1]
        if not _remote_path_exists(_remote_game_path(source, remote_path), runner)
    ]
    if missing:
        raise ValueError(
            f"remote dataset {remote_root} has a manifest but is missing expected game content "
            f"({missing[0]}); clean it up or publish under a new immutable dataset ID"
        )


def _contains_missing_path_marker(*outputs: object) -> bool:
    return MISSING_REMOTE_PATH_MARKER in "\n".join(str(output) for output in outputs if output).casefold()


def _remote_path_exists(remote_path: str, runner: Runner) -> bool:
    command = _volume_ls_command(remote_path)
    try:
        result = runner(command, check=False, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        if _contains_missing_path_marker(exc.stdout, exc.stderr, exc):
            return False
        raise
    if result.returncode:
        if _contains_missing_path_marker(result.stdout, result.stderr):
            return False
        error = (result.stderr or "").strip()
        detail = f": {error}" if error else ""
        raise ValueError(f"could not verify remote path {remote_path}; modal volume ls failed with exit code {result.returncode}{detail}")
    try:
        return bool(json.loads(result.stdout or "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not verify remote path {remote_path}") from exc


def _claim_local_publish_lock(processed_base: Path, dataset_id: str) -> Path:
    processed_base.mkdir(parents=True, exist_ok=True)
    lock_path = processed_base / f".{dataset_id}.publish.lock"
    try:
        with lock_path.open("x", encoding="utf-8"):
            pass
    except FileExistsError as exc:
        raise ValueError(f"processed dataset {dataset_id!r} is already being published locally") from exc
    return lock_path


def publish_processed_dataset(
    raw_dataset_root: Path | str,
    dataset_id: str,
    *,
    processed_base: Path | str = Path("processed"),
    dry_run: bool = False,
    runner: Runner = subprocess.run,
) -> dict:
    dataset_id = validate_dataset_id(dataset_id)
    processed_base = Path(processed_base)
    output_root = processed_base / dataset_id
    lock_path = _claim_local_publish_lock(processed_base, dataset_id)
    try:
        if output_root.exists():
            try:
                manifest = validate_processed_root(output_root)
            except ValueError as exc:
                raise ValueError(f"existing local processed dataset at {output_root} is invalid: {exc}") from exc
        else:
            runner(
                [sys.executable, "TrackNetV3/preprocess.py", str(raw_dataset_root), "--output-root", str(output_root)],
                check=True,
            )
            manifest = validate_processed_root(output_root)
        remote_root = f"/processed/{dataset_id}"
        batches = _build_upload_batches(output_root, dataset_id)
        if _remote_path_exists(remote_root, runner):
            _verify_remote_dataset_complete(
                remote_root, f"{remote_root}/manifest.json", manifest, batches, runner
            )
            return manifest
        if dry_run:
            return manifest

        for source, remote_path in batches[:-1]:
            batch_path = _remote_game_path(source, remote_path)
            print(f"uploading {batch_path}")
            if not _remote_path_exists(batch_path, runner):
                runner(_volume_put_command(source, remote_path), check=True)
            else:
                print(f"skipping existing {batch_path}")
        manifest_source, manifest_path = batches[-1]
        print(f"uploading {manifest_path}")
        if not _remote_manifest_matches(manifest_path, manifest, runner):
            runner(_manifest_put_command(manifest_source, manifest_path), check=True)
        for path in (f"{remote_root}/manifest.json", *(_remote_game_path(source, remote_path) for source, remote_path in batches[:-1])):
            if not _remote_path_exists(path, runner):
                raise ValueError(f"remote upload verification failed: {path} is missing")
        return manifest
    finally:
        lock_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess and publish an immutable TrackNet dataset to the Modal Volume.")
    parser.add_argument("--raw-dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--processed-base", type=Path, default=Path("processed"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        publish_processed_dataset(
            args.raw_dataset_root,
            args.dataset_id,
            processed_base=args.processed_base,
            dry_run=args.dry_run,
        )
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"published processed dataset {args.dataset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
