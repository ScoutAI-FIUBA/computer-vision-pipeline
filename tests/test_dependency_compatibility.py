"""Static checks for the production Modal dependency set."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "ftfy": "6.3.1",
    "open_clip_torch": "2.32.0",
    "transformers": "5.0.0",
    "huggingface_hub": "1.7.1",
}
BOXMOT = "22.0.0"
TORCH_PINS = {"torch": "2.5.1", "torchvision": "0.20.1"}


def _pin(text: str, package: str) -> str:
    match = re.search(rf"(?:^|\s|\")({re.escape(package)})==([^\s\"',]+)", text)
    assert match, f"missing pinned {package} dependency"
    return match.group(2)


def test_default_production_dependencies_are_pinned_in_requirements_and_modal():
    sources = [(ROOT / name).read_text() for name in ("requirements.txt", "requirements-team-ab.txt", "modal_app.py")]

    for package, version in EXPECTED.items():
        for source in sources:
            assert _pin(source, package) == version


def test_fashionsiglip_uses_the_official_open_clip_distribution_name():
    for name in ("requirements.txt", "requirements-team-ab.txt", "modal_app.py"):
        source = (ROOT / name).read_text()
        assert "open_clip==" not in source
        assert _pin(source, "open_clip_torch") == EXPECTED["open_clip_torch"]


def test_open_clip_runtime_import_is_available_without_loading_a_model():
    try:
        open_clip = importlib.import_module("open_clip")
    except ModuleNotFoundError as exc:
        pytest.skip(
            f"optional runtime dependency is absent ({exc.name}); install "
            "open_clip_torch==2.32.0 to run this import check"
        )

    assert open_clip.__name__ == "open_clip"
    assert hasattr(open_clip, "create_model")


def test_boxmot_and_transformers_huggingface_pins_are_coherent():
    hub = Version(EXPECTED["huggingface_hub"])
    requirements = (ROOT / "requirements.txt").read_text()
    modal = (ROOT / "modal_app.py").read_text()
    assert _pin(requirements, "boxmot") == BOXMOT
    assert _pin(modal, "boxmot") == BOXMOT
    assert SpecifierSet(">=1.7.1").contains(hub)  # boxmot==22.0.0
    assert SpecifierSet(">=1.3.0,<2.0").contains(hub)  # transformers==5.0.0


def test_torch_pins_match_production_team_ab_modal_and_sam2():
    sources = [
        (ROOT / "requirements.txt").read_text(),
        (ROOT / "requirements-team-ab.txt").read_text(),
        (ROOT / "requirements-sam2.txt").read_text(),
        (ROOT / "modal_app.py").read_text(),
    ]
    for package, version in TORCH_PINS.items():
        for source in sources:
            assert _pin(source, package) == version
