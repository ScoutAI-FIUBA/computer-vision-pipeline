# Apply Progress: SAM 3.1 Player Tracking V2

## Work Unit

- Boundary: Phase 1 foundation/contracts and adapter boundary, tasks 1.1–1.3; Phase 2 artifact packaging, tasks 2.1–2.2; runner wiring task 2.3.
- Delivery: feature-branch-chain strategy implemented as ordered local atomic commits; no PR.
- Rollback: revert the artifact work-unit commits `0242d74` and the follow-up integrity fix commit.
- Scope guard: `modal_app.py` untouched; Modal wiring, remote rendering, and local renderer deferred.

## Completed Tasks

- [x] 1.1 Added strict-TDD contract tests for normalization, identity separation, deterministic keys, and bounded retention.
- [x] 1.2 Added the dependency-free `Sam31MultiplexTracker` adapter with injected session lifecycle, prompt boundary, normalized rows, bounded coasting, and termination.
- [x] 1.3 Added V2-only SAM 3.1 configuration settings and environment parsing while preserving legacy defaults.
- [x] 2.1 Added strict-TDD artifact-contract tests for manifest, JSONL, lossless masks, metadata, retrieval references, and transfer failures.
- [x] 2.2 Added deterministic artifact packaging in `pipeline/results.py`, optional manifest references on `PipelineResults`, and legacy serialization preservation.
- [x] 2.3 Added a truthful `RunnerV2`/`run_v2` adapter: identity acceptance is derived internally, repository ReID/team contracts are invoked when frame data is available, and unsupported video-stream stages remain deferred.

## TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|------|-----------|-------|------------|-----|-------|-------------|----------|
| 1.1 | `tests/test_sam31_tracking.py` | Unit | N/A (new) | ✅ Missing module failure | ✅ 3 focused tests passed | ✅ Multiple objects, xywh, null identity | ✅ Contract kept as pure normalization helpers |
| 1.2 | `tests/test_sam31_tracking.py` | Unit | N/A (new) | ✅ Missing module failure | ✅ 3 focused tests passed | ✅ Prompt/lifecycle, bounded loss, pre-start guard | ✅ Session dependency injected; no model import at module load |
| 1.3 | `tests/test_config.py` | Unit | ✅ 23/23 | ✅ Missing V2 fields/config behavior | ✅ 24 config tests passed, then 26 after triangulation | ✅ Environment parsing and invalid range/retention cases | ✅ Reused existing env parsing conventions |
| 2.1 | `tests/test_results_artifacts.py` | Unit | N/A (new test file) | ✅ Import failure for missing `package_artifacts` | ✅ 4 artifact tests passed | ✅ Complete package, missing mask/transfer failure, duplicate keys, legacy payload | ✅ Kept contract assertions focused on persisted/retrievable values |
| 2.2 | `tests/test_results_artifacts.py` | Unit | N/A (initial baseline not captured before first implementation) | ✅ Missing production API | ✅ 4 artifact tests passed; 70 legacy result tests passed afterward | ✅ PNG round-trip and incomplete transfer paths | ✅ Small pure reference helper; no bulk inline return |
| 2.3 | `tests/test_runner_v2.py` | Unit | ✅ 4/4 baseline | ✅ New contract tests failed against caller-controlled gate/callback API | ✅ 6 focused tests passed | ✅ Empty input, malformed raw ID, missing mapping, conflicting IDs, ReID, team contracts | ✅ Narrow adapter; removed unsupported arbitrary callbacks |

### Review Fix Evidence

| Fix | Test File | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|---|---|---|---|---|---|---|
| Manifest integrity, collision-free masks, incomplete packages | `tests/test_results_artifacts.py` | ✅ 74 focused/result tests passed before edits | ✅ 3 authoritative review tests failed | ✅ 6 artifact tests passed | ✅ Prefix/no-prefix manifests, adversarial keys, missing masks and transfer errors | ✅ Canonical self-reference-excluding hash helper |
| Runner identity/seam audit findings | `tests/test_runner_v2.py` | ✅ 4/4 prior runner tests passed | ✅ 6 replacement contract tests failed against old API | ✅ 6 runner tests passed | ✅ Production ReID/team contracts and rejection cases | ✅ Removed caller-controlled acceptance and arbitrary stage callbacks |

## Verification

- Focused command: `.venv/bin/python -m pytest tests/test_runner_v2.py`
- Result: `6 passed`.
- Relevant existing tests: `.venv/bin/python -m pytest tests/test_runner_v2.py tests/test_sam31_tracking.py tests/test_results_artifacts.py tests/test_reid_backends.py tests/test_team_id.py tests/test_player_positions.py` — `61 passed`.
- CI configuration: no repository CI workflow/configuration was found; none was added.
- Full suite: deferred to verification phase.

## Remaining Tasks

- [x] 2.1–2.3 Artifact packaging and truthful runner adapter boundary.
- [ ] 2.4–2.5 Modal V2 wiring and broader runner/Modal tests.
- [ ] 3.1–3.3 Local renderer, validation, and final rollback/full-suite verification.
