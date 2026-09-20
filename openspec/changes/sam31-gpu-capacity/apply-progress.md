# Apply Progress: SAM 3.1 GPU Capacity

## Work Unit

- Boundary: Change only the Modal V2 class GPU capacity and its focused regression test.
- Delivery: single conventional commit; legacy `modal_app.py` remains unchanged.
- Rollback: revert the single GPU-capacity commit.
- Scope guard: model logic, allocator settings, R2, renderer, and game validation were not changed.

## Completed Tasks

- [x] Changed `modal_app_v2.py` V2 class from `gpu="L4"` to `gpu="A100-40GB"`.
- [x] Added a regression test asserting V2 uses A100-40GB and legacy remains on L4.
- [x] Strengthened the GPU regression with AST inspection of the `PipelineV2` and legacy `Pipeline` decorators.
- [x] Required the AST regression to match the specific `app.cls` decorator receiver before binding GPU values to each class.
- [x] Ran focused and full pytest suites.

## TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|------|-----------|-------|------------|-----|-------|-------------|----------|
| GPU capacity boundary | `tests/test_modal_app_v2.py` | Unit/static boundary | ✅ 45/45 | ✅ New assertion failed before implementation | ✅ Focused test and V2 suite passed | ✅ V2 A100-40GB and legacy L4 cases | ➖ None needed |
| GPU test structural strengthening | `tests/test_modal_app_v2.py` | Unit/static boundary | ✅ 46/46 | ✅ Test-first replacement of string matching | ✅ Focused, V2, and full suites passed | ✅ Class-specific V2 and legacy decorator extraction | ➖ None needed |
| GPU decorator receiver strengthening | `tests/test_modal_app_v2.py` | Unit/static boundary | ✅ 46/46 | ✅ Test-first receiver constraint | ✅ Focused, V2, and full suites passed | ✅ Rejects non-`app.cls` decorators; binds both class values | ➖ None needed |

## Verification

- Focused regression: `.venv/bin/python -m pytest tests/test_modal_app_v2.py::test_v2_and_legacy_gpu_capacity_are_explicitly_separate -q` — `1 passed`.
- V2 suite: `.venv/bin/python -m pytest tests/test_modal_app_v2.py -q` — `46 passed`.
- Full suite: `.venv/bin/python -m pytest -q` — `359 passed, 1 skipped`.
- Structural regression: `.venv/bin/python -m pytest tests/test_modal_app_v2.py::test_v2_and_legacy_gpu_capacity_are_explicitly_separate -q` — `1 passed`.
- V2 suite after structural strengthening: `.venv/bin/python -m pytest tests/test_modal_app_v2.py -q` — `46 passed`.
- Full suite after structural strengthening: `.venv/bin/python -m pytest -q` — `359 passed, 1 skipped`.
- Receiver regression: `.venv/bin/python -m pytest tests/test_modal_app_v2.py::test_v2_and_legacy_gpu_capacity_are_explicitly_separate -q` — `1 passed`.
- V2 suite after receiver strengthening: `.venv/bin/python -m pytest tests/test_modal_app_v2.py -q` — `46 passed`.
- Full suite after receiver strengthening: `.venv/bin/python -m pytest -q` — `359 passed, 1 skipped`.

## Remaining Tasks

- [x] Run the full pytest suite and record the result.
- [x] Create the single conventional commit: `fix(modal): assign A100 capacity to V2`.
- [x] Create the new conventional test commit: `test(modal): strengthen GPU capacity regression`.
- [x] Create the final conventional test commit: `test(modal): require app GPU decorator binding`.
