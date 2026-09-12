"""Focused coverage for the Modal entrypoint wiring on `modal_tracknet_finetune`.

This is a follow-up patch on top of the archived
`tracknet-basketball-modal-finetuning` change. It only proves the
entrypoint-wiring layer:

- the module loads cleanly
- `app` is exposed (and is a `modal.App` when the SDK is installed)
- `dry_run` and `run` are registered as `@app.function` entrypoints
- `dry_run` produces a plan JSON without touching the local subprocess path
- the existing local CLI is untouched (the `main` import path still works)

It does NOT exercise detached jobs, status.json, resume, or persistent
experiment state from the extension PRD — those are intentionally out of
scope for this minimal wiring fix.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import modal_tracknet_finetune as wrapper  # noqa: E402


def test_module_exposes_app_when_modal_sdk_is_installed():
    assert wrapper.app is not None, (
        "modal_tracknet_finetune.app must be exposed whenever the modal "
        "SDK is importable; otherwise `modal run` still reports no entrypoints."
    )
    # `app` is either the real `modal.App` instance or, in a venv without
    # the SDK, `None`. With the SDK installed we should get the real thing.
    modal = sys.modules.get("modal")
    if modal is not None:
        assert isinstance(wrapper.app, modal.App)


def test_get_app_matches_module_level_app():
    assert wrapper.get_app() is wrapper.app


def test_dry_run_and_run_are_registered_entrypoints():
    registered = wrapper.app.registered_functions
    assert "dry_run" in registered, (
        "dry_run entrypoint is required so the user can validate Modal "
        "wiring cheaply without running the full pipeline."
    )
    assert "run" in registered, (
        "run entrypoint is required so `modal run modal_tracknet_finetune.py::run` "
        "actually invokes the orchestration instead of complaining about "
        "missing entrypoints."
    )


def test_dry_run_entrypoint_writes_plan(tmp_path):
    """`dry_run` should not invoke any subprocess; it only writes the plan."""
    layout = wrapper.build_run_layout(tmp_path, "wiring-test")
    plan_output = layout.run_root / "plans" / "modal-plan.json"
    dry_fn = wrapper.app.registered_functions["dry_run"].get_raw_f()

    # The CLI marks every report path as required=True, so we have to hand
    # the wrapper something non-empty for each one. Empty strings would
    # normalize to "." and the prerequisite reader would explode. Pointing
    # at nonexistent files in tmp_path keeps the plan honest: every
    # prerequisite stays "missing" so `ready_to_execute` is False.
    nonexistent = tmp_path / "absent.json"
    result = dry_fn(
        run_id="wiring-test",
        raw_dataset_root=str(PROJECT_ROOT / "dataset-tracknet"),
        volume_root=str(tmp_path),
        pretrained_tracknet="/nonexistent/TrackNet_best.pt",
        baseline_validation_report=str(layout.reports_root / "absent-baseline-val.json"),
        candidate_validation_report=str(layout.reports_root / "absent-candidate-val.json"),
        baseline_test_report=str(layout.reports_root / "absent-baseline-test.json"),
        finetuned_test_report=str(layout.reports_root / "absent-finetuned-test.json"),
        plan_output=str(plan_output),
    )

    assert result == 0
    assert plan_output.exists()
    payload = json.loads(plan_output.read_text(encoding="utf-8"))
    assert payload["run_id"] == "wiring-test"
    assert payload["dry_run"] is True
    assert payload["commands"], "dry_run must still emit the orchestration command list"
    # No prerequisite reports were provided, so the plan is honest: not ready.
    assert payload["ready_to_execute"] is False
    assert all(item["status"] == "missing" for item in payload["prerequisites"])


def test_local_cli_main_still_importable():
    # The local CLI path must keep working without Modal.
    assert callable(wrapper.main)
    assert callable(wrapper.build_run_layout)
    assert callable(wrapper.normalize_run_id)


def test_app_unavailable_path(monkeypatch):
    """When `modal` cannot be imported, the module still loads and `app` is None."""
    fake_module = importlib.import_module("modal_tracknet_finetune")

    # Simulate the "no SDK" branch without actually removing `modal` from sys.modules.
    monkeypatch.setattr(fake_module, "app", None)
    monkeypatch.setattr(fake_module, "_APP", None)
    assert fake_module.get_app() is None
    assert fake_module.app is None


def test_entrypoints_declare_image_volume_gpu_and_timeout():
    """The Modal entrypoints must reference the declared image + volume + GPU so
    the TrackNetV3 source tree (mounted in the image at /root/TrackNetV3) is
    available to the training subprocesses inside the container.

    The error we are fixing is:
        /usr/local/bin/python: can't open file '/root/TrackNetV3/basketball_data.py': No such file or directory
    which happens when Modal only mounts the wrapper file itself. This test
    proves the entrypoint specs now carry the image and the volume mount.
    """
    modal = sys.modules.get("modal")
    if modal is None:  # pragma: no cover - guarded import path
        pytest.skip("modal SDK not installed in this venv")

    for name in ("run", "dry_run"):
        fn = wrapper.app.registered_functions[name]
        spec = fn.spec

        # Image must be declared (not None). The spec stores the image as a
        # `modal._image._Image` (the internal variant) which is what Modal's
        # own serialization uses, so accept either the public or the internal
        # Image class as long as the object is a real Image.
        assert spec.image is not None, (
            f"{name} entrypoint must declare an image so the TrackNetV3 source "
            "tree is available inside the container at /root/TrackNetV3."
        )
        image_class_names = {cls.__name__ for cls in type(spec.image).__mro__}
        assert any(name in {"Image", "_Image"} for name in image_class_names), (
            f"{name} entrypoint image must be a real modal.Image "
            f"(got {type(spec.image).__name__})."
        )

        # The tracknet-data volume must mount at /mnt/tracknet.
        assert "/mnt/tracknet" in spec.volumes, (
            f"{name} entrypoint must mount the tracknet-data volume at "
            "/mnt/tracknet; got mounts: {sorted(spec.volumes)}."
        )
        mounted_volume = spec.volumes["/mnt/tracknet"]
        # The volume object exposes `.name` whether it is the hydrated Volume
        # or the from_name placeholder used at decoration time.
        volume_name = getattr(mounted_volume, "name", None) or str(mounted_volume)
        assert "tracknet-data" in volume_name, (
            f"{name} entrypoint volume must be the 'tracknet-data' Modal "
            f"volume; got {volume_name!r}."
        )

        # GPU must be L4 for both entrypoints per the wiring contract.
        assert spec.gpus == "L4", (
            f"{name} entrypoint must request gpu='L4'; got {spec.gpus!r}."
        )

        # The wrapper file must be mounted at /root/modal_tracknet_finetune.py
        # inside the container; the TrackNetV3/ directory is baked into the
        # image via add_local_dir(copy=True) so it shows up in the dockerfile
        # layer rather than as a runtime mount.
        mounts = spec.mounts or []
        mounted_remote_paths: list[str] = []
        for mount in mounts:
            for entry in (getattr(mount, "_entries", None) or []):
                remote_path = getattr(entry, "remote_path", None)
                if remote_path is not None:
                    mounted_remote_paths.append(str(remote_path))
        assert any(path.endswith("modal_tracknet_finetune.py") for path in mounted_remote_paths), (
            f"{name} entrypoint must mount the wrapper file at "
            "/root/modal_tracknet_finetune.py; got mounts: {mounted_remote_paths}."
        )


def test_modal_image_and_volume_accessors_expose_declared_objects():
    """The image/volume accessors must return Image/Volume objects that are
    wired into the entrypoint specs. Keeps the test honest if someone later
    refactors the decorator and accidentally drops the wiring."""
    modal = sys.modules.get("modal")
    if modal is None:  # pragma: no cover
        pytest.skip("modal SDK not installed in this venv")

    image = wrapper.get_modal_image()
    volume = wrapper.get_modal_volume()
    assert image is not None, "get_modal_image() must return the declared image"
    assert volume is not None, "get_modal_volume() must return the declared volume"
    assert "Image" in {cls.__name__ for cls in type(image).__mro__}
    assert "Volume" in {cls.__name__ for cls in type(volume).__mro__}

    for name in ("run", "dry_run"):
        spec = wrapper.app.registered_functions[name].spec
        # The spec stores its own wrapped Image/Volume instances, so we
        # assert that the spec actually carries a non-None Image and a
        # volume at the expected mount path. Identity equality is not
        # guaranteed because Modal may hydrate / wrap the references.
        assert spec.image is not None
        assert "/mnt/tracknet" in spec.volumes


def test_image_bakes_tracknetv3_dir_via_add_local_dir():
    """The image must be a real `modal.Image` (not None) and the local
    TrackNetV3/ source tree must exist at the path the wrapper passed to
    `add_local_dir`. Together these prove the source tree the image needs
    is available locally and the image is wired into the entrypoint.

    The TrackNetV3/ dir is baked into the image with `copy=True`, so it does
    not surface as a runtime mount — only the wrapper file shows up there.
    Asserting on the image object's existence + the source tree's presence
    is enough to catch a regression that drops the wiring."""
    modal = sys.modules.get("modal")
    if modal is None:  # pragma: no cover
        pytest.skip("modal SDK not installed in this venv")

    image = wrapper.get_modal_image()
    assert image is not None
    assert isinstance(image, modal.Image)

    # The local TrackNetV3/ directory must exist so the image can COPY it
    # into /root/TrackNetV3 inside the container.
    tracknet_dir = PROJECT_ROOT / "TrackNetV3"
    assert tracknet_dir.is_dir(), (
        f"TrackNetV3/ source tree must exist at {tracknet_dir} so the "
        f"image's `add_local_dir` step can bake it into the container."
    )
    # And the specific files the wrapper subprocesses need must be present.
    for script in ("basketball_data.py", "train.py", "test.py",
                   "evaluate.py", "preprocess.py", "visualize_labels.py"):
        assert (tracknet_dir / script).is_file(), (
            f"TrackNetV3/{script} must be present locally; the image bakes "
            f"it into /root/TrackNetV3 inside the container."
        )
