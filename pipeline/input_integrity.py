"""Input integrity checks for staged pipeline jobs."""

from __future__ import annotations

import hashlib
from pathlib import Path


def verify_sha256_file(path: Path, fingerprint: str) -> Path:
    """Fail closed when a SHA-256 fingerprint does not match a file."""
    digest = fingerprint.removeprefix("sha256:")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"invalid input fingerprint: {fingerprint}")
    if not path.is_file():
        raise ValueError(f"immutable job input is missing for fingerprint: {fingerprint}")

    actual = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            actual.update(chunk)
    if actual.hexdigest() != digest:
        raise ValueError(f"immutable job input fingerprint mismatch: {fingerprint}")
    return path
