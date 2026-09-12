from __future__ import annotations

import inspect
import json
from pathlib import Path
from statistics import mean, median


class TrainingStateError(RuntimeError):
    pass


def save_training_config(output_path: Path | str, config: dict) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_finetune_state(
    *,
    model_name: str,
    seq_len: int,
    bg_mode: str,
    dataset_root: str,
    save_dir: str,
    adapted_data_manifest: str | None,
    pretrained_tracknet: Path | str | None,
    resume_checkpoint: Path | str | None,
    device: str = "cpu",
) -> dict:
    if pretrained_tracknet and resume_checkpoint:
        raise TrainingStateError("incompatible checkpoint mode: pretrained_tracknet and resume_checkpoint are mutually exclusive")

    if pretrained_tracknet:
        payload = _load_checkpoint_payload(pretrained_tracknet, device=device, trust_mode="pretrained")
        _assert_checkpoint_matches(payload, model_name=model_name, seq_len=seq_len, bg_mode=bg_mode)
        return {
            "mode": "pretrained",
            "start_epoch": 0,
            "best_metric": None,
            "model_state_dict": payload["model"],
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "history": [],
        }

    if resume_checkpoint:
        payload = _load_checkpoint_payload(resume_checkpoint, device=device, trust_mode="resume", trusted_root=save_dir)
        _assert_checkpoint_matches(
            payload,
            model_name=model_name,
            seq_len=seq_len,
            bg_mode=bg_mode,
            dataset_root=dataset_root,
            save_dir=save_dir,
            adapted_data_manifest=adapted_data_manifest,
        )
        return {
            "mode": "resume",
            "start_epoch": int(payload.get("epoch", -1)) + 1,
            "best_metric": payload.get("best_metric"),
            "model_state_dict": payload["model"],
            "optimizer_state_dict": payload.get("optimizer"),
            "scheduler_state_dict": payload.get("scheduler"),
            "history": payload.get("history", []),
        }

    return {
        "mode": "fresh",
        "start_epoch": 0,
        "best_metric": None,
        "model_state_dict": None,
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
        "history": [],
    }


def compute_validation_metrics(records: list[dict] | None = None, summary: dict | None = None) -> dict:
    if records is None and summary is None:
        raise ValueError("validation metrics require records or summary")

    if records is not None:
        visible = [record for record in records if int(record["visibility"]) == 1]
        visible_count = len(visible)
        detected = [record for record in visible if int(record["predicted_visibility"]) == 1]
        distances = [float(record["distance"]) for record in detected if record.get("distance") is not None]
        return _build_metric_payload(visible_count=visible_count, detected_count=len(detected), distances=distances)

    tp = float(summary.get("TP", 0.0))
    fp1 = float(summary.get("FP1", 0.0))
    fn = float(summary.get("FN", 0.0))
    visible_count = tp + fp1 + fn
    detected_count = tp + fp1
    p_at_20 = tp / visible_count if visible_count else 0.0
    detection_rate = detected_count / visible_count if visible_count else 0.0
    return {
        "visible_frames": visible_count,
        "detected_visible_frames": detected_count,
        "p_at_10": summary.get("p_at_10"),
        "p_at_20": p_at_20,
        "p_at_40": summary.get("p_at_40"),
        "detection_rate": detection_rate,
        "mean_error": summary.get("mean_error"),
        "median_error": summary.get("median_error", float("inf")),
    }


def select_best_checkpoint(baseline_metrics: dict, candidates: list[dict], detection_rate_guard: float = 0.02) -> dict:
    minimum_detection_rate = float(baseline_metrics["detection_rate"]) - detection_rate_guard
    allowed = [candidate for candidate in candidates if float(candidate["detection_rate"]) >= minimum_detection_rate]
    if not allowed:
        raise TrainingStateError("guard rejected every checkpoint candidate")
    return sorted(
        allowed,
        key=lambda candidate: (
            -float(candidate["p_at_20"]),
            -float(candidate["detection_rate"]),
            float(candidate.get("median_error", float("inf"))),
        ),
    )[0]


def append_history(history: list[dict], *, epoch: int, train_loss: float, val_loss: float, metrics: dict, metadata: dict | None = None) -> list[dict]:
    updated = list(history)
    updated.append(
        {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "metrics": metrics,
            **({"metadata": dict(metadata)} if metadata else {}),
        }
    )
    return updated


def _build_metric_payload(*, visible_count: int, detected_count: int, distances: list[float]) -> dict:
    p_at_10 = sum(distance <= 10.0 for distance in distances) / visible_count if visible_count else 0.0
    p_at_20 = sum(distance <= 20.0 for distance in distances) / visible_count if visible_count else 0.0
    p_at_40 = sum(distance <= 40.0 for distance in distances) / visible_count if visible_count else 0.0
    detection_rate = detected_count / visible_count if visible_count else 0.0
    return {
        "visible_frames": visible_count,
        "detected_visible_frames": detected_count,
        "p_at_10": p_at_10,
        "p_at_20": p_at_20,
        "p_at_40": p_at_40,
        "detection_rate": detection_rate,
        "mean_error": mean(distances) if distances else None,
        "median_error": median(distances) if distances else float("inf"),
    }


def _assert_checkpoint_matches(payload: dict, **expected: object) -> None:
    param_dict = payload.get("param_dict", {})
    source = payload | param_dict
    mismatches = []
    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        if source.get(key) != expected_value:
            mismatches.append(f"{key}={source.get(key)!r} != {expected_value!r}")
    if mismatches:
        raise TrainingStateError(f"incompatible checkpoint: {', '.join(mismatches)}")


def _load_checkpoint_payload(path: Path | str, *, device: str, trust_mode: str, trusted_root: Path | str | None = None) -> dict:
    checkpoint_path = Path(path)
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        import torch

        if _supports_weights_only(torch.load):
            try:
                return torch.load(checkpoint_path, map_location=device, weights_only=True)
            except Exception as exc:
                raise TrainingStateError(
                    f"{trust_mode} checkpoint must load safely with torch.load(..., weights_only=True): {checkpoint_path}"
                ) from exc

        if trust_mode == "resume" and _is_within_root(checkpoint_path, trusted_root):
            return torch.load(checkpoint_path, map_location=device)

        if trust_mode == "pretrained":
            raise TrainingStateError(
                f"pretrained checkpoint requires torch.load(..., weights_only=True); upgrade torch or provide a safer checkpoint: {checkpoint_path}"
            )

        raise TrainingStateError(
            f"resume checkpoint fallback is trusted-input only and requires a checkpoint under save_dir: {checkpoint_path}"
        )


def _supports_weights_only(loader: object) -> bool:
    try:
        return "weights_only" in inspect.signature(loader).parameters
    except (TypeError, ValueError):
        return False


def _is_within_root(path: Path | str, root: Path | str | None) -> bool:
    if root is None:
        return False
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return False
    return True
