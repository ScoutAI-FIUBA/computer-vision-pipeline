# Apply Progress: SAM 3.1 Player Tracking V2

## Work Unit

- Boundary: Phase 1 foundation/contracts and adapter boundary, tasks 1.1–1.3; Phase 2 artifact packaging, tasks 2.1–2.2.
- Delivery: feature-branch-chain strategy implemented as ordered local atomic commits; no PR.
- Rollback: revert the single work-unit commit.
- Scope guard: `modal_app.py` untouched; runner, Modal wiring, remote rendering, and local renderer deferred.

## Completed Tasks

- [x] 1.1 Added strict-TDD contract tests for normalization, identity separation, deterministic keys, and bounded retention.
- [x] 1.2 Added the dependency-free `Sam31MultiplexTracker` adapter with injected session lifecycle, prompt boundary, normalized rows, bounded coasting, and termination.
- [x] 1.3 Added V2-only SAM 3.1 configuration settings and environment parsing while preserving legacy defaults.
- [x] 2.1 Added strict-TDD artifact-contract tests for manifest, JSONL, lossless masks, metadata, retrieval references, and transfer failures.
- [x] 2.2 Added deterministic artifact packaging in `pipeline/results.py`, optional manifest references on `PipelineResults`, and legacy serialization preservation.

## TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|------|-----------|-------|------------|-----|-------|-------------|----------|
| 1.1 | `tests/test_sam31_tracking.py` | Unit | N/A (new) | ✅ Missing module failure | ✅ 3 focused tests passed | ✅ Multiple objects, xywh, null identity | ✅ Contract kept as pure normalization helpers |
| 1.2 | `tests/test_sam31_tracking.py` | Unit | N/A (new) | ✅ Missing module failure | ✅ 3 focused tests passed | ✅ Prompt/lifecycle, bounded loss, pre-start guard | ✅ Session dependency injected; no model import at module load |
| 1.3 | `tests/test_config.py` | Unit | ✅ 23/23 | ✅ Missing V2 fields/config behavior | ✅ 24 config tests passed, then 26 after triangulation | ✅ Environment parsing and invalid range/retention cases | ✅ Reused existing env parsing conventions |
| 2.1 | `tests/test_results_artifacts.py` | Unit | ✅ 30 prior focused tests | ✅ Import failure for missing `package_artifacts` | ✅ 4 artifact tests passed | ✅ Complete package, missing mask/transfer failure, duplicate keys, legacy payload | ✅ Kept contract assertions focused on persisted/retrievable values |
| 2.2 | `tests/test_results_artifacts.py` | Unit | ✅ 4/4 artifact tests | ✅ Same missing production API | ✅ 4 artifact tests passed; 74 legacy/results tests passed | ✅ PNG round-trip and incomplete transfer paths | ✅ Small pure reference helper; no bulk inline return |

## Verification

- Focused command: `.venv/bin/python -m pytest tests/test_results_artifacts.py tests/test_ball_tracking.py tests/test_shot_events.py tests/test_number_evidence.py`
- Result: `74 passed`.
- Full suite: deferred to verification phase.

## Remaining Tasks

- [x] 2.1–2.2 Artifact packaging contract and `pipeline/results.py` behavior.
- [ ] 2.3–2.5 Runner integration and Modal V2 wiring.
- [ ] 3.1–3.3 Local renderer, validation, and final rollback/full-suite verification.
