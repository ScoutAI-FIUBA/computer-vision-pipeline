# Apply Progress: SAM 3.1 Player Tracking V2

## Work Unit

- Boundary: Phase 1 foundation/contracts and adapter boundary, tasks 1.1–1.3; Phase 2 artifact packaging, tasks 2.1–2.2; runner wiring task 2.3; Modal V2 boundary task 2.4.
- Delivery: feature-branch-chain strategy implemented as ordered local atomic commits; no PR.
- Rollback: revert this Modal V2 work-unit commit plus the prior artifact/runner commits; `modal_app.py` remains the legacy fallback.
- Scope guard: `modal_app.py` untouched; remote rendering and local renderer deferred.

## Completed Tasks

- [x] 1.1 Added strict-TDD contract tests for normalization, identity separation, deterministic keys, and bounded retention.
- [x] 1.2 Added the dependency-free `Sam31MultiplexTracker` adapter with injected session lifecycle, prompt boundary, normalized rows, bounded coasting, and termination.
- [x] 1.3 Added V2-only SAM 3.1 configuration settings and environment parsing while preserving legacy defaults.
- [x] 2.1 Added strict-TDD artifact-contract tests for manifest, JSONL, lossless masks, metadata, retrieval references, and transfer failures.
- [x] 2.2 Added deterministic artifact packaging in `pipeline/results.py`, optional manifest references on `PipelineResults`, and legacy serialization preservation.
- [x] 2.3 Added a truthful validation-only `RunnerV2`/`run_v2` adapter: identity evidence is structurally validated, repository ReID/team contracts are invoked when frame data is available, and acceptance plus unsupported video-stream stages remain deferred.
- [x] 2.4 Added isolated `modal_app_v2.py` with the official SAM 3.1 multiplex predictor boundary, HF cache/auth, R2 input/output transfer, raw artifact packaging, retrieval-key response, and no remote video rendering. V2 remains validation-only because `run_v2` is called without an identity authority.
- [x] 2.4 boundary remediation: validated public inputs before lazy predictor/GPU initialization, rejected SAM cardinality mismatches, added provenance/video/config/benchmark metadata and mask references, and made R2 failure manifests explicitly incomplete before and after transfer attempts.

## TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|------|-----------|-------|------------|-----|-------|-------------|----------|
| 1.1 | `tests/test_sam31_tracking.py` | Unit | N/A (new) | ✅ Missing module failure | ✅ 3 focused tests passed | ✅ Multiple objects, xywh, null identity | ✅ Contract kept as pure normalization helpers |
| 1.2 | `tests/test_sam31_tracking.py` | Unit | N/A (new) | ✅ Missing module failure | ✅ 3 focused tests passed | ✅ Prompt/lifecycle, bounded loss, pre-start guard | ✅ Session dependency injected; no model import at module load |
| 1.3 | `tests/test_config.py` | Unit | ✅ 23/23 | ✅ Missing V2 fields/config behavior | ✅ 24 config tests passed, then 26 after triangulation | ✅ Environment parsing and invalid range/retention cases | ✅ Reused existing env parsing conventions |
| 2.1 | `tests/test_results_artifacts.py` | Unit | N/A (new test file) | ✅ Import failure for missing `package_artifacts` | ✅ 4 artifact tests passed | ✅ Complete package, missing mask/transfer failure, duplicate keys, legacy payload | ✅ Kept contract assertions focused on persisted/retrievable values |
| 2.2 | `tests/test_results_artifacts.py` | Unit | N/A (initial baseline not captured before first implementation) | ✅ Missing production API | ✅ 4 artifact tests passed; 70 legacy result tests passed afterward | ✅ PNG round-trip and incomplete transfer paths | ✅ Small pure reference helper; no bulk inline return |
| 2.3 | `tests/test_runner_v2.py` | Unit | ✅ 14/14 baseline | ✅ Strengthened rejected-observation assertion | ✅ 14 focused tests passed | ✅ Each rejected observation excludes stable identity, ReID, and team-derived fields | ✅ Centralized derived-field scrubbing |
| 2.4 | `tests/test_modal_app_v2.py` | Unit/static boundary | N/A (new files) | ✅ Missing module and helpers failed | ✅ 5 focused tests passed | ✅ Bytes/R2 argument paths, transfer failure, retrieval-only response, isolation | ✅ Thin helpers; Modal/model calls remain lazy at runtime |

### Review Fix Evidence

| Fix | Test File | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|---|---|---|---|---|---|---|
| Manifest integrity, collision-free masks, incomplete packages | `tests/test_results_artifacts.py` | ✅ 74 focused/result tests passed before edits | ✅ 3 authoritative review tests failed | ✅ 6 artifact tests passed | ✅ Prefix/no-prefix manifests, adversarial keys, missing masks and transfer errors | ✅ Canonical self-reference-excluding hash helper |
| Runner identity/seam audit findings | `tests/test_runner_v2.py` | ✅ 4/4 prior runner tests passed | ✅ 6 replacement contract tests failed against old API | ✅ 11 runner tests passed | ✅ Production ReID/team contracts and malformed/conflicting identity cases | ✅ Removed caller-controlled acceptance and arbitrary stage callbacks |
| Team assignment collision audit | `tests/test_runner_v2.py` | ✅ 11/11 prior runner tests passed | ✅ Conflicting same-stable team test failed by silent overwrite | ✅ 13 runner tests passed | ✅ Conflicting assignments rejected; consistent assignments retained | ✅ Deterministic conflict reasons and no partial team output |
| Atomic rejected-run audit | `tests/test_runner_v2.py` | ✅ 13/13 prior runner tests passed | ✅ ReID output leaked beside conflicting team validation | ✅ 14 runner tests passed | ✅ ReID + conflicting team assignment yields empty downstream output | ✅ Rejected result is not consumable as partial derived data |
| Rejected observation leakage audit | `tests/test_runner_v2.py` | ✅ 14/14 prior runner tests passed | ✅ `reid_embedding` remained in rejected observations | ✅ 14 runner tests passed | ✅ Derived observation fields scrubbed while raw evidence remains | ✅ Centralized scrub set prevents consumer confusion |
| Rejected derived-field assertion | `tests/test_runner_v2.py` | ✅ 14/14 prior runner tests passed | ✅ Assertion strengthened from ReID-only to all derived fields | ✅ 14 runner tests passed | ✅ `stable_player_id`, `reid_embedding`, and `team_id` all absent | ✅ Test now checks the complete rejection contract |
| Populated team-field rejection coverage | `tests/test_runner_v2.py` | ✅ 14/14 prior runner tests passed | ✅ Regression setup strengthened with pre-existing `team_id` | ✅ 14 runner tests passed | ✅ Populated team field is scrubbed alongside identity/ReID fields | ✅ No production change required |
| Modal V2 critical boundary findings | `tests/test_modal_app_v2.py` | ✅ 5/5 prior Modal tests passed | ✅ 9 regression tests failed against the permissive boundary | ✅ 15 Modal tests passed | ✅ Invalid inputs, upload failure, metadata/mask references, and all cardinality paths | ✅ Lazy predictor initialization; manifest state is written before transfer |

## Verification

- Focused command: `.venv/bin/python -m pytest tests/test_runner_v2.py`
- Result: `14 passed`.
- Modal V2 focused command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py`
- Modal V2 result: `5 passed`.
- Modal V2 remediation result: `15 passed`.
- Relevant existing tests: `.venv/bin/python -m pytest tests/test_runner_v2.py tests/test_sam31_tracking.py tests/test_results_artifacts.py tests/test_reid_backends.py tests/test_team_id.py tests/test_player_positions.py` — `69 passed`.
- Remediation regression command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py tests/test_runner_v2.py tests/test_results_artifacts.py tests/test_sam31_tracking.py` — `41 passed`.
- CI configuration: no repository CI workflow/configuration was found; none was added.
- Full suite: deferred to verification phase.

## Remaining Tasks

- [x] 2.1–2.3 Artifact packaging and truthful runner adapter boundary.
- [x] 2.4 Modal V2 wiring and raw artifact retrieval boundary.
- [ ] 2.5 Broader runner/Modal tests.
- [ ] 3.1–3.3 Local renderer, validation, and final rollback/full-suite verification.
