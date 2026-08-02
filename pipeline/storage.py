"""Cloudflare R2 object storage.

Plain boto3, no Modal. Mirrors the conventions the Render backend already uses
in `back/src/infrastructure/storage.py`: R2 is S3-compatible, region is "auto",
and signature version is s3v4.

Kept out of `runner.py` on purpose — the pipeline itself deals in local paths, so
it stays runnable with no cloud credentials at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class R2Settings:
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    endpoint_url: str

    @classmethod
    def from_env(cls) -> "R2Settings":
        missing = [
            name
            for name in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME", "R2_ENDPOINT_URL")
            if not (os.getenv(name) or "").strip()
        ]
        if missing:
            raise RuntimeError(f"Missing R2 environment variables: {', '.join(missing)}")
        return cls(
            access_key_id=os.environ["R2_ACCESS_KEY_ID"].strip(),
            secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"].strip(),
            bucket_name=os.environ["R2_BUCKET_NAME"].strip(),
            endpoint_url=os.environ["R2_ENDPOINT_URL"].strip(),
        )


class R2Storage:
    """Download inputs and upload outputs for one job."""

    def __init__(self, settings: R2Settings | None = None) -> None:
        import boto3
        from botocore.client import Config

        self._settings = settings or R2Settings.from_env()
        self._client = boto3.client(
            "s3",
            endpoint_url=self._settings.endpoint_url,
            aws_access_key_id=self._settings.access_key_id,
            aws_secret_access_key=self._settings.secret_access_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    def download(self, key: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self._settings.bucket_name, key, str(destination))
        return destination

    def upload(self, source: Path, key: str, content_type: str | None = None) -> str:
        extra = {"ContentType": content_type} if content_type else {}
        self._client.upload_file(str(source), self._settings.bucket_name, key, ExtraArgs=extra)
        return key

    def upload_directory(self, directory: Path, key_prefix: str) -> list[str]:
        """Upload every file in `directory` under `key_prefix/`. Returns the keys."""

        content_types = {".json": "application/json", ".jsonl": "application/x-ndjson", ".mp4": "video/mp4"}
        keys = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(directory).as_posix()
            key = f"{key_prefix.rstrip('/')}/{relative}"
            keys.append(self.upload(path, key, content_types.get(path.suffix)))
        return keys
