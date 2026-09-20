# Apply Progress: SAM 3.1 GPU Capacity

## Work Unit

- Boundary: Change only the Modal V2 class GPU capacity and its focused regression test.
- Delivery: cumulative history across `fccf054`, `16be854`, and `f25272d`; legacy `modal_app.py` remains unchanged.
- Rollback: revert the three GPU-capacity commits in reverse order if the complete unit must be removed.
- Scope guard: model logic, allocator settings, R2, renderer, and game validation were not changed.

## Completed Tasks

- [x] `fccf054` changed `modal_app_v2.py` V2 class from `gpu="L4"` to `gpu="A100-40GB"` and added the basic V2/legacy regression test in the same commit.
- [x] `16be854` strengthened the regression with AST inspection of the `PipelineV2` and legacy `Pipeline` decorators.
- [x] `f25272d` required the AST regression to match the specific `app.cls` decorator receiver before binding GPU values to each class.
- [x] Ran focused and full pytest suites.
- [x] Corrected this progress record to distinguish the historical implementation and test-strengthening commits; no unsupported historical TDD ordering is claimed.

## TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|------|-----------|-------|------------|-----|-------|-------------|----------|
| GPU capacity boundary (`fccf054`) | `tests/test_modal_app_v2.py` | Unit/static boundary | Not recorded | Not proven — production pin and basic test landed together | ✅ Post-commit focused, V2, and full suites passed | ✅ Basic V2 A100-40GB and legacy L4 assertions | ➖ None needed |
| GPU test structural strengthening (`16be854`) | `tests/test_modal_app_v2.py` | Unit/static boundary | ✅ 46/46 | Not applicable — test-only strengthening of existing behavior | ✅ Focused, V2, and full suites passed | ✅ Class-specific V2 and legacy decorator extraction | ➖ None needed |
| GPU decorator receiver strengthening (`f25272d`) | `tests/test_modal_app_v2.py` | Unit/static boundary | ✅ 46/46 | Not applicable — test-only strengthening of existing behavior | ✅ Focused, V2, and full suites passed | ✅ Requires `app.cls` and binds both class values | ➖ None needed |

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
- [x] Create the documentation-only history correction commit: `docs(sdd): correct GPU capacity history`.
