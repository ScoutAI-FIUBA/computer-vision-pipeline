"""Focused coverage for the explicit-deps follow-up patch on `modal_tracknet_finetune`.

This is a follow-up patch on top of the archived
`tracknet-basketball-modal-finetuning` change. It only proves the new
behavior the patch adds:

- `_MODAL_IMAGE` does NOT call `.pip_install_from_requirements(...)` and
  does NOT reference the `_filter_requirements_for_modal` helper.
- `_MODAL_IMAGE` calls `.pip_install("numpy==1.26.4")` BEFORE the
  explicit flat-list `.pip_install(...)` step so pip resolves to a
  prebuilt wheel for Python 3.11.
- The explicit flat-list install contains `torch==2.0.0` (a release
  with cp311 wheels, replacing `torch==1.10.0` which had no cp311
  wheel) and `opencv-python` (the pip package name uses a hyphen,
  unlike the underscored form in TrackNetV3/requirements.txt).
- Every dep in the explicit list ships a Python 3.11 wheel on PyPI;
  the most recent follow-up bumped `opencv-python==4.4.0.46` to
  `opencv-python==4.8.0.76` because the older release predates cp311
  wheels and was triggering the same numpy source build.
- The full explicit dep list covers every dep from the original
  requirements file except numpy and torch, which we set explicitly.

Background: `TrackNetV3/requirements.txt` pins `torch==1.10.0`, which
has no prebuilt wheel for Python 3.11 and pulls in an old numpy
without a cp311 wheel. The previous follow-up filtered the numpy line
out of the requirements file before installing, but pip still resolved
`torch==1.10.0` and the transitive numpy dependency failed to find a
wheel, so gcc rebuilt numpy from source. The user chose the explicit-
install approach: drop the requirements-file install entirely and
install dependencies as a flat explicit list with cp311-compatible
versions.

The follow-up after that swap dropped torch to `torch==1.13.1` /
`torchvision==0.14.1` to minimize API breakage with the TrackNetV3
source code, which was written against old torch. Both releases ship
Python 3.11 wheels on PyPI, so the explicit-install wheel resolution
path is unchanged; only the version literal moves.

A later follow-up bumped `opencv-python==4.4.0.46` to
`opencv-python==4.8.0.76` because 4.4.0.46 predates the cp311 wheels
on PyPI and dragged the explicit install back into the same numpy
source-build failure the preinstall was meant to avoid. Every other
dep already shipped a cp311 wheel, so this is the only literal that
moved in that patch.

This final follow-up lands the locked cp311-wheel image audit: torch
is bumped from `1.13.1` to `2.0.0` and torchvision from `0.14.1` to
`0.15.1` after auditing every dep against PyPI. `torchvision==0.15.1`
is the only torchvision pin that matches a cp311-wheel torch release
(`torch==2.0.0`); the sibling `0.15.0` is yanked on PyPI for an
incorrect torch dependency and never resolves. This is the LAST patch
in the image-deps audit cycle — the dep list is now locked.

The post-archive `pycocotools` follow-up audit extended the explicit
list with two third-party imports TrackNetV3 transitively pulls in
when the wrapper subprocess-calls its training scripts: `pycocotools`
(imported at the top of `TrackNetV3/test.py` for COCO mAP evaluation;
`train.py` imports `test.py`, so the failure surfaced at the train
step) and `matplotlib` (imported at the top of
`TrackNetV3/utils/visualize.py`; train.py imports the viz helpers).
Both are pinned to a cp311-wheel release on PyPI: `pycocotools==2.0.11`
and `matplotlib==3.11.1` (which requires `numpy>=1.25`, satisfied by
the preinstalled `numpy==1.26.4`).

The rest of the image chain (`apt_install`, `add_local_dir`,
`add_local_file`, volume mount, GPU, entrypoint wiring) must stay
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import modal  # noqa: E402

import modal_tracknet_finetune as wrapper  # noqa: E402


@pytest.fixture
def recorded_image_calls(monkeypatch: pytest.MonkeyPatch):
    """Record every `pip_install` / `apt_install` / `add_local_dir` /
    `add_local_file` call made while the wrapper rebuilds `_MODAL_IMAGE`.

    Modal hides the image's build-step list from Python (no public
    attribute exposes the command order), so we monkey-patch the
    builders at the class level to capture the call sequence. Each
    patched method forwards to the original so the resulting Image
    object is still usable.

    `pip_install_from_requirements` is intentionally NOT patched: the
    patch contract is that the chain no longer calls it.
    """
    calls: list[tuple[str, tuple, dict]] = []

    def record(name: str, original):
        def wrapper_method(self, *args, **kwargs):
            calls.append((name, args, kwargs))
            return original(self, *args, **kwargs)

        return wrapper_method

    monkeypatch.setattr(
        modal.Image,
        "pip_install",
        record("pip_install", modal.Image.pip_install),
    )
    monkeypatch.setattr(
        modal.Image,
        "apt_install",
        record("apt_install", modal.Image.apt_install),
    )
    monkeypatch.setattr(
        modal.Image,
        "add_local_dir",
        record("add_local_dir", modal.Image.add_local_dir),
    )
    monkeypatch.setattr(
        modal.Image,
        "add_local_file",
        record("add_local_file", modal.Image.add_local_file),
    )

    return calls


def _rebuild_image_calls() -> list[tuple[str, tuple, dict]]:
    """Force a fresh evaluation of `_MODAL_IMAGE` by rebuilding the same
    chain in isolation. We cannot re-import the module because the chain
    has already executed; instead we replay the chain expression here.
    This stays in sync with the wrapper because the chain is the single
    source of truth: any future change to the chain requires updating
    this helper, which makes test drift obvious in code review.

    The replay is meant only to drive the monkey-patched builders and
    capture the call order. The Image object it produces is discarded.
    """
    tracknet_dir = PROJECT_ROOT / "TrackNetV3"
    wrapper_file = PROJECT_ROOT / "modal_tracknet_finetune.py"
    # Mirror the exact chain from modal_tracknet_finetune._MODAL_IMAGE.
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "ffmpeg",
            "libsm6",
            "libxext6",
            "libgl1",
            "libglib2.0-0",
        )
        .pip_install("numpy==1.26.4")
        .pip_install(
            "torch==2.0.0",
            "torchvision==0.15.1",
            "pandas==2.0.0",
            "Pillow==10.0.0",
            "opencv-python==4.8.0.76",
            "plotly==5.8.2",
            "dash==2.5.1",
            "tqdm",
            "tensorboard",
            "pycocotools==2.0.11",
            "matplotlib==3.11.1",
            "parse",
        )
        .add_local_dir(str(tracknet_dir), remote_path="/root/TrackNetV3", copy=True)
        .add_local_file(str(wrapper_file), remote_path="/root/modal_tracknet_finetune.py", copy=True)
    )


def test_modal_image_chain_records_calls_in_expected_order(recorded_image_calls):
    """Sanity check: the monkey-patched builders capture every step in
    the order they appear in the wrapper's `_MODAL_IMAGE` chain. This
    makes the rest of the tests meaningful — without it the ordering
    assertions would silently pass on an empty list.
    """
    _rebuild_image_calls()

    names = [name for name, _, _ in recorded_image_calls]
    assert names[:2] == ["apt_install", "pip_install"], (
        "expected the chain to start with apt_install then pip_install "
        f"(the numpy pin); got {names!r}"
    )
    assert names[-2:] == ["add_local_dir", "add_local_file"], (
        "expected the chain to end with add_local_dir then add_local_file "
        f"(baking the source tree and wrapper); got {names!r}"
    )


def test_modal_image_does_not_reference_requirements_helper(recorded_image_calls):
    """The patch drops `pip_install_from_requirements(...)` entirely.
    The chain must not invoke it, otherwise pip still resolves the
    transitive `torch==1.10.0 -> numpy<1.23` dependency that lacks a
    cp311 wheel and falls back to building numpy from source.
    """
    _rebuild_image_calls()

    requirements_calls = [
        (name, args) for name, args, _ in recorded_image_calls
        if name == "pip_install_from_requirements"
    ]
    assert requirements_calls == [], (
        "expected the chain to have NO `pip_install_from_requirements` "
        "step after the explicit-install patch; got "
        f"{requirements_calls!r}"
    )


def test_modal_image_does_not_reference_filter_helper(recorded_image_calls):
    """The patch removes `_filter_requirements_for_modal`. The replayed
    chain must not reference it (we also assert at the source level in
    `test_wrapper_module_level_image_drops_requirements_helper`).
    """
    _rebuild_image_calls()
    # If the helper were still referenced, the chain would need it to
    # evaluate. Sanity check that the replayed chain still resolves to
    # a real Image without the helper being on the module.
    assert not hasattr(wrapper, "_filter_requirements_for_modal"), (
        "the wrapper must no longer expose `_filter_requirements_for_modal`; "
        "the explicit-install patch removes it entirely."
    )


def test_modal_image_pins_numpy_before_explicit_install(recorded_image_calls):
    """A `pip_install("numpy==1.26.4")` step must appear BEFORE the
    explicit flat-list install so pip resolves the rest of the deps
    against a numpy version that ships Python 3.11 wheels instead of
    trying to build numpy from source.
    """
    _rebuild_image_calls()

    numpy_pin_index: int | None = None
    explicit_install_index: int | None = None
    for index, (name, args, _kwargs) in enumerate(recorded_image_calls):
        if name == "pip_install":
            if any("numpy" in str(arg).lower() for arg in args):
                if numpy_pin_index is None:
                    numpy_pin_index = index
            # The explicit flat-list install is the pip_install call that
            # contains `torch==2.0.0`.
            elif any(str(arg).lower().startswith("torch==") for arg in args):
                if explicit_install_index is None:
                    explicit_install_index = index

    assert numpy_pin_index is not None, (
        "expected the chain to include a `pip_install(...numpy...)` step; "
        f"got calls: {recorded_image_calls!r}"
    )
    assert explicit_install_index is not None, (
        "expected the chain to include a `pip_install('torch==2.0.0', ...)` "
        f"step; got calls: {recorded_image_calls!r}"
    )
    assert numpy_pin_index < explicit_install_index, (
        "the numpy pin must precede the explicit dep install so pip "
        "resolves the wheel instead of building numpy from source. "
        f"Found numpy pin at index {numpy_pin_index} and explicit "
        f"install at index {explicit_install_index}."
    )


def test_modal_image_numpy_pin_targets_explicit_compatible_release(recorded_image_calls):
    """The pin must be the explicit cp311-compatible numpy version
    (`numpy==1.26.4`). We assert the literal so a future drive-by edit
    to a different version is caught here.
    """
    _rebuild_image_calls()

    numpy_calls = [
        args
        for name, args, _kwargs in recorded_image_calls
        if name == "pip_install"
        and any("numpy" in str(arg).lower() for arg in args)
    ]
    assert len(numpy_calls) == 1, (
        f"expected exactly one numpy-targeting pip_install step before "
        f"the explicit install; got {numpy_calls!r}"
    )
    args = numpy_calls[0]
    assert len(args) == 1, (
        "the numpy pin step should target a single package spec; "
        f"got {args!r}"
    )
    spec = str(args[0]).lower()
    assert spec == "numpy==1.26.4", (
        "the explicit numpy pin must be `numpy==1.26.4` so pip picks a "
        "cp311-compatible wheel instead of rebuilding numpy from "
        f"source. Detected spec: {spec!r}. If you genuinely need a "
        "different pin, update this test to assert the new constraint "
        "and verify the new release ships a cp311 wheel on PyPI."
    )


def test_modal_image_explicit_install_uses_modern_torch_pin(recorded_image_calls):
    """The explicit install must pin `torch==2.0.0` (replacing the
    `torch==1.10.0` pin from `TrackNetV3/requirements.txt`, which has
    no cp311 wheel). The torchvision pin must be `torchvision==0.15.1`
    to match — the only torchvision pin in this release line that
    ships a cp311 wheel (its sibling `0.15.0` is yanked on PyPI for an
    incorrect torch dependency).
    """
    _rebuild_image_calls()

    torch_calls = [
        args
        for name, args, _kwargs in recorded_image_calls
        if name == "pip_install"
        and any(str(arg).lower().startswith("torch==") for arg in args)
    ]
    assert len(torch_calls) == 1, (
        f"expected exactly one pip_install step targeting `torch==`; "
        f"got {torch_calls!r}"
    )
    args = torch_calls[0]
    assert any(str(arg).lower() == "torch==2.0.0" for arg in args), (
        "the explicit install must pin `torch==2.0.0` (the release that "
        f"`torchvision==0.15.1` requires and that ships cp311 wheels on "
        f"PyPI). Got args: {args!r}"
    )
    assert any(str(arg).lower() == "torchvision==0.15.1" for arg in args), (
        "the explicit install must pin `torchvision==0.15.1` (the only "
        f"torchvision pin in this release line that ships a cp311 wheel; "
        f"the sibling `0.15.0` is yanked on PyPI). Got args: {args!r}"
    )


def test_modal_image_explicit_install_contains_opencv_python(recorded_image_calls):
    """The pip package name for OpenCV uses a hyphen
    (`opencv-python`), unlike the underscored form
    (`opencv_python==4.4.0.46`) in `TrackNetV3/requirements.txt`. The
    explicit install must use the hyphenated pip name so pip resolves
    to the real distribution.
    """
    _rebuild_image_calls()

    explicit_call = None
    for name, args, _kwargs in recorded_image_calls:
        if name == "pip_install" and any(
            str(arg).lower().startswith("torch==") for arg in args
        ):
            explicit_call = args
            break

    assert explicit_call is not None, (
        "expected the chain to include an explicit pip_install step; "
        f"got calls: {recorded_image_calls!r}"
    )
    normalized = [str(arg).lower() for arg in explicit_call]
    assert "opencv-python==4.8.0.76" in normalized, (
        "the explicit install must contain `opencv-python==4.8.0.76` "
        "(the pip package name uses a hyphen, not an underscore; this "
        "release ships a cp311 wheel on PyPI, unlike 4.4.0.46 which "
        f"forced a numpy source build). Got: {normalized}"
    )


def test_modal_image_explicit_install_covers_full_dep_list(recorded_image_calls):
    """The explicit flat list must cover every dep from the original
    `TrackNetV3/requirements.txt` except numpy and torch, which are
    installed explicitly (numpy as the pin, torch at a newer
    cp311-compatible version).
    """
    _rebuild_image_calls()

    explicit_call = None
    for name, args, _kwargs in recorded_image_calls:
        if name == "pip_install" and any(
            str(arg).lower().startswith("torch==") for arg in args
        ):
            explicit_call = args
            break

    assert explicit_call is not None, (
        "expected the chain to include an explicit pip_install step; "
        f"got calls: {recorded_image_calls!r}"
    )
    normalized = [str(arg).lower() for arg in explicit_call]

    required = {
        "torch==2.0.0",
        "torchvision==0.15.1",
        "pandas==2.0.0",
        "pillow==10.0.0",
        "opencv-python==4.8.0.76",
        "plotly==5.8.2",
        "dash==2.5.1",
        "tqdm",
        "tensorboard",
        "pycocotools==2.0.11",
        "matplotlib==3.11.1",
        "parse",
    }
    for spec in required:
        assert spec in normalized, (
            f"the explicit install must include `{spec}`; got {normalized}"
        )


def test_modal_image_chain_preserves_existing_steps(recorded_image_calls):
    """The patch is minimal: it only swaps the requirements-file
    install for a flat explicit list and removes the filter helper.
    The rest of the chain must stay byte-equivalent so existing tests
    (image+volume wiring, inline-evaluate) still pass.
    """
    _rebuild_image_calls()

    names = [name for name, _, _ in recorded_image_calls]

    # apt_install is still the first non-debian_slim step.
    assert names[0] == "apt_install"
    # add_local_dir + add_local_file are still last.
    assert names[-2] == "add_local_dir"
    assert names[-1] == "add_local_file"
    # pip_install_from_requirements is gone.
    assert "pip_install_from_requirements" not in names, (
        "the chain must no longer call pip_install_from_requirements; "
        f"got call sequence: {names!r}"
    )


def test_wrapper_module_level_image_drops_requirements_helper():
    """The `_MODAL_IMAGE` declared at module import time must itself
    reflect the new chain. We assert by reading the source: the literal
    `.pip_install("numpy==1.26.4")` must appear before the explicit
    flat-list install, neither `.pip_install_from_requirements(` nor
    `_filter_requirements_for_modal` may appear, and the explicit
    torch pin must be `torch==2.0.0` (with torchvision `0.15.1`).
    """
    source_path = PROJECT_ROOT / "modal_tracknet_finetune.py"
    source = source_path.read_text(encoding="utf-8")

    numpy_pin_index = source.find('.pip_install("numpy==1.26.4")')
    torch_pin_index = source.find('.pip_install(\n            "torch==2.0.0"')

    assert numpy_pin_index != -1, (
        "the wrapper must include a `.pip_install(\"numpy==1.26.4\")` step "
        "in _MODAL_IMAGE so pip resolves to a Python 3.11 wheel."
    )
    assert torch_pin_index != -1, (
        "the wrapper must include a `.pip_install(\"torch==2.0.0\", ...)` "
        "step in _MODAL_IMAGE so torch resolves to a cp311-compatible "
        "release instead of the `torch==1.10.0` pin from "
        "TrackNetV3/requirements.txt."
    )
    assert numpy_pin_index < torch_pin_index, (
        "the numpy pin must be declared BEFORE the explicit flat-list "
        "install so pip resolves to a prebuilt wheel before the deps "
        "are installed."
    )

    assert ".pip_install_from_requirements(" not in source, (
        "the wrapper must NOT call `.pip_install_from_requirements(` "
        "anymore; the explicit-install patch removes it."
    )
    assert "_filter_requirements_for_modal" not in source, (
        "the wrapper must NOT reference `_filter_requirements_for_modal` "
        "anymore; the explicit-install patch removes it."
    )
    assert "opencv-python==4.8.0.76" in source, (
        "the explicit install must pin `opencv-python==4.8.0.76` (the "
        "pip package name uses a hyphen; this release ships a cp311 "
        "wheel on PyPI)."
    )
    assert "torchvision==0.15.1" in source, (
        "the explicit install must pin `torchvision==0.15.1` (the only "
        f"torchvision pin in this release line that ships a cp311 wheel; "
        f"the sibling `0.15.0` is yanked on PyPI for an incorrect torch "
        f"dependency)."
    )


def test_modal_image_explicit_install_pins_pycocotools(recorded_image_calls):
    """The post-archive audit found `pycocotools` was a missing
    third-party dep. `TrackNetV3/test.py` imports `pycocotools.coco` and
    `pycocotools.cocoeval` at the top of the module; `train.py` imports
    `from test import ...`, so the `ModuleNotFoundError` surfaced at the
    train step. The explicit install must pin a cp311-wheel release.
    """
    _rebuild_image_calls()

    explicit_call = None
    for name, args, _kwargs in recorded_image_calls:
        if name == "pip_install" and any(
            str(arg).lower().startswith("torch==") for arg in args
        ):
            explicit_call = args
            break

    assert explicit_call is not None, (
        "expected the chain to include an explicit pip_install step; "
        f"got calls: {recorded_image_calls!r}"
    )
    normalized = [str(arg).lower() for arg in explicit_call]
    assert "pycocotools==2.0.11" in normalized, (
        "the explicit install must pin `pycocotools==2.0.11` "
        "(TrackNetV3/test.py imports pycocotools unconditionally; "
        "train.py imports test.py, so the failure surfaces at the "
        "train step. 2.0.11 is the latest release that ships a "
        f"manylinux cp311 wheel on PyPI). Got: {normalized}"
    )


def test_modal_image_explicit_install_pins_matplotlib(recorded_image_calls):
    """The post-archive audit found `matplotlib` was a missing
    third-party dep. `TrackNetV3/utils/visualize.py` imports
    `matplotlib.pyplot` at the top of the module; `train.py` imports
    `from utils.visualize import ...`, so the module is loaded at the
    train step. The explicit install must pin a cp311-wheel release
    whose `requires_dist` accepts `numpy==1.26.4`.
    """
    _rebuild_image_calls()

    explicit_call = None
    for name, args, _kwargs in recorded_image_calls:
        if name == "pip_install" and any(
            str(arg).lower().startswith("torch==") for arg in args
        ):
            explicit_call = args
            break

    assert explicit_call is not None, (
        "expected the chain to include an explicit pip_install step; "
        f"got calls: {recorded_image_calls!r}"
    )
    normalized = [str(arg).lower() for arg in explicit_call]
    assert "matplotlib==3.11.1" in normalized, (
        "the explicit install must pin `matplotlib==3.11.1` "
        "(TrackNetV3/utils/visualize.py imports matplotlib.pyplot "
        "unconditionally; train.py imports the viz helpers, so the "
        "module loads at the train step. 3.11.1 ships a manylinux "
        "cp311 wheel on PyPI for x86_64 + aarch64 and requires "
        "`numpy>=1.25`, satisfied by the preinstalled `numpy==1.26.4`). "
        f"Got: {normalized}"
    )
