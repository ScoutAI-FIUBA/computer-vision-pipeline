from pathlib import Path


DOC_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ball" / "FINETUNING_TRACKNETV3_REPO.md"


def test_documented_publish_and_processed_modal_dry_run_commands() -> None:
    document = DOC_PATH.read_text(encoding="utf-8")

    assert "python scripts/ball/publish_tracknet_processed.py" in document
    assert "--dataset-id basketball-v1" in document
    assert "processed/basketball-v1" in document
    assert "modal run modal_tracknet_finetune.py::run_dry" in document
    assert "--processed-root /mnt/tracknet/processed/basketball-v1" in document


def test_documented_immutable_id_and_shared_root_safety_rules() -> None:
    document = DOC_PATH.read_text(encoding="utf-8")

    assert "/processed/<dataset-id>/<split>/" in document
    assert "unused dataset ID" in document
    assert "new dataset ID" in document
    assert "No overwrite path" in document
    assert "serializes" in document
    assert "runs/<run-id>/" in document


def test_documented_single_publisher_contract() -> None:
    document = DOC_PATH.read_text(encoding="utf-8")

    assert "Supported publication contract" in document
    assert "same-machine concurrent calls" in document
    assert "cross-machine publishers must serialize manually" in document
