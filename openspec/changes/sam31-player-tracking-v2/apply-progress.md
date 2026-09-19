# Apply Progress: SAM 3.1 Player Tracking V2

## Work Unit

- Boundary: Phase 1 foundation/contracts and adapter boundary, tasks 1.1–1.3.
- Delivery: feature-branch-chain strategy implemented as one local atomic commit; no PR.
- Rollback: revert the single work-unit commit.
- Scope guard: `modal_app.py` untouched; Modal wiring, full runner integration, remote artifact packaging, and local renderer deferred.

## Completed Tasks

- [x] 1.1 Added strict-TDD contract tests for normalization, identity separation, deterministic keys, and bounded retention.
- [x] 1.2 Added the dependency-free `Sam31MultiplexTracker` adapter with injected session lifecycle, prompt boundary, normalized rows, bounded coasting, and termination.
- [x] 1.3 Added V2-only SAM 3.1 configuration settings and environment parsing while preserving legacy defaults.

## TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|------|-----------|-------|------------|-----|-------|-------------|----------|
| 1.1 | `tests/test_sam31_tracking.py` | Unit | N/A (new) | ✅ Missing module failure | ✅ 6 focused tests passed | ✅ Multiple objects, xywh, null identity, lifecycle/loss cases | ✅ Contract kept as pure normalization helpers |
| 1.2 | `tests/test_sam31_tracking.py` | Unit | N/A (new) | ✅ Missing module failure | ✅ 6 focused tests passed | ✅ Prompt/lifecycle, bounded loss, pre-start guard | ✅ Session dependency injected; no model import at module load |
| 1.3 | `tests/test_config.py` | Unit | ✅ 23/23 | ✅ Missing V2 fields/config behavior | ✅ 24 config tests passed, then 26 after triangulation | ✅ Environment parsing and invalid range/retention cases | ✅ Reused existing env parsing conventions |

## Verification

- Focused command: `.venv/bin/python -m pytest tests/test_sam31_tracking.py tests/test_config.py`
- Result: `30 passed`.
- Full suite: deferred to verification phase.

## Remaining Tasks

- [ ] 2.1–2.5 Artifact packaging, runner integration, and Modal V2 wiring.
- [ ] 3.1–3.3 Local renderer, validation, and final rollback/full-suite verification.
