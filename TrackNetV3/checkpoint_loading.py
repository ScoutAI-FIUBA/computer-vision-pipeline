from __future__ import annotations

import inspect
from pathlib import Path


class UnsafeCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be loaded without pickle execution."""


def _is_within_root(path: Path, root: Path | str | None) -> bool:
    if root is None:
        return False
    try:
        path.resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def load_checkpoint(path: Path | str, *, map_location: str = "cpu", trusted_root: Path | str | None = None):
    """Load a checkpoint safely, allowing pickle only for an explicit trusted root."""
    import torch

    checkpoint_path = Path(path)
    try:
        supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        supports_weights_only = False

    if supports_weights_only:
        try:
            return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        except Exception as exc:
            raise UnsafeCheckpointError(
                f"checkpoint must load safely with torch.load(..., weights_only=True): {checkpoint_path}"
            ) from exc

    if _is_within_root(checkpoint_path, trusted_root):
        return torch.load(checkpoint_path, map_location=map_location)
    raise UnsafeCheckpointError(
        f"checkpoint requires torch.load(..., weights_only=True) or an explicit trusted root: {checkpoint_path}"
    )
