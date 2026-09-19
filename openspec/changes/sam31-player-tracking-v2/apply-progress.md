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
- [x] 2.4 Added isolated `modal_app_v2.py` with the official SAM 3.1 multiplex predictor boundary, HF cache/auth, ephemeral processing, direct byte-artifact return, and no remote video rendering. V2 remains validation-only because `run_v2` is called without an identity authority.
- [x] 2.4 boundary remediation: validated public inputs before lazy predictor/GPU initialization, rejected SAM cardinality mismatches, and added provenance/video/config/benchmark metadata and mask references.
- [x] 2.4 direct-return correction: removed the unintended external-storage boundary; local `--video-path` bytes are sent to Modal, returned artifacts are `{relative_path: bytes}`, and local `--output-dir` writing mirrors V1.
- [x] 2.4 setuptools compatibility remediation: kept the V2 image pinned to `setuptools<82` and added a static dependency-contract regression test for the SAM 3.1 `pkg_resources` requirement.
- [x] `sam31-runtime-deps`: tightened the V2 image to `setuptools<81`, added `einops`, and disabled optional FlashAttention 3 for the baseline predictor with `use_fa3=False`.

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
| Modal V2 finalized manifest publication | `tests/test_modal_app_v2.py` | ✅ 15/15 prior Modal tests passed | ✅ Success-path regression observed only the incomplete marker remotely | ✅ 16 Modal tests passed | ✅ Initial incomplete marker plus finalized complete manifest; failed uploads remain incomplete | ✅ Final manifest upload is inside the transfer transaction |
| Modal V2 direct artifact return | `tests/test_modal_app_v2.py` | ✅ 16/16 prior Modal tests passed | ✅ R2-oriented boundary tests failed against the direct-return contract | ✅ 12 direct-return tests passed | ✅ Relative byte mapping, nested local writing, incomplete local manifest, and input bytes | ✅ Removed external-storage helpers and retained ephemeral packaging |
| Modal V2 artifact size/path boundary | `tests/test_modal_app_v2.py` | ✅ 12/12 direct-return tests passed | ✅ Size-limit and unsafe-path regressions failed before implementation | ✅ 20 Modal tests passed | ✅ Explicit/env size limits, absolute/traversal/invalid paths, and nested masks | ✅ Preflight total size before reading bytes; validate all paths before writing |
| Modal V2 symlink collection boundary | `tests/test_modal_app_v2.py` | ✅ 20/20 prior Modal tests passed | ✅ Outside symlink regression read bytes before implementation | ✅ 21 Modal tests passed | ✅ Symlinked outside file rejected while regular nested files remain valid | ✅ Reject symlinks and resolved escapes before stat/read |
| Modal V2 symlink-directory boundary | `tests/test_modal_app_v2.py` | ✅ 21/21 prior Modal tests passed | ✅ Symlinked directory was silently skipped before implementation | ✅ 22 Modal tests passed | ✅ Symlinked directories rejected while regular nested directories remain valid | ✅ Check symlinks before `is_file()` traversal filtering |
| Modal V2 setuptools compatibility | `tests/test_modal_app_v2.py` | ✅ 27/27 prior Modal tests passed | ✅ Regression failed when the pin was temporarily removed | ✅ 28 Modal tests passed | ➖ Single structural dependency contract; no branching behavior | ✅ Parsed `pip_install` calls without importing Modal |
| SAM31 runtime dependency baseline | `tests/test_modal_app_v2.py` | ✅ 28/28 prior Modal tests passed | ✅ Setuptools/einops/FlashAttention contracts failed before implementation | ✅ 29 Modal tests passed | ➖ Structural image/predictor contracts; no branching behavior | ✅ Minimal AST-based static checks |

## Verification

- Focused command: `.venv/bin/python -m pytest tests/test_runner_v2.py`
- Result: `14 passed`.
- Modal V2 focused command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py`
- Modal V2 result: `5 passed`.
- Modal V2 remediation result: `15 passed`.
- Modal V2 final manifest result: `16 passed`.
- Modal V2 direct-return result: `12 passed`.
- Modal V2 size/path result: `20 passed`.
- Modal V2 symlink result: `21 passed`.
- Modal V2 symlink-directory result: `22 passed`.
- Modal V2 setuptools compatibility result: `.venv/bin/python -m pytest tests/test_modal_app_v2.py` — `28 passed`.
- Relevant existing tests: `.venv/bin/python -m pytest tests/test_runner_v2.py tests/test_sam31_tracking.py tests/test_results_artifacts.py tests/test_reid_backends.py tests/test_team_id.py tests/test_player_positions.py` — `69 passed`.
- Remediation regression command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py tests/test_runner_v2.py tests/test_results_artifacts.py tests/test_sam31_tracking.py` — `48 passed`.
- Setuptools compatibility regression command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py tests/test_runner_v2.py tests/test_results_artifacts.py tests/test_sam31_tracking.py` — `54 passed`.
- `sam31-runtime-deps` focused command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py` — `29 passed`.
- `sam31-runtime-deps` relevant command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py tests/test_runner_v2.py tests/test_results_artifacts.py tests/test_sam31_tracking.py` — `55 passed`.
- CI configuration: no repository CI workflow/configuration was found; none was added.
- Full suite: deferred to verification phase.

## Remaining Tasks

- [x] 2.1–2.3 Artifact packaging and truthful runner adapter boundary.
- [x] 2.4 Modal V2 wiring and raw artifact retrieval boundary.
- [x] 2.5 Added controlled-fake process coverage for complete direct artifacts, lossless masks, manifest config/benchmark metadata, explicit size-limit failure, SAM cardinality errors, and existing local nested-path, acceptance, no-R2, no-debug-video, and legacy-isolation contracts.
- [ ] 3.1–3.3 Local renderer, validation, and final rollback/full-suite verification.

## Task 2.5 TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|------|-----------|-------|------------|-----|-------|-------------|----------|
| 2.5 | `tests/test_modal_app_v2.py`, `tests/test_runner_v2.py` | Unit with controlled process seam | ✅ 40/40 | ✅ Process/cardinality tests failed before seam correction | ✅ 27 Modal tests passed; 14 runner tests passed | ✅ Complete accepted fields/mask bytes, rejected process gating, size failure, cardinality mismatch, rejection/deferred paths | ✅ Reused existing artifact writer/runner contracts; no production change |
| 2.4 compatibility remediation | `tests/test_modal_app_v2.py` | Unit/static boundary | ✅ 27/27 | ✅ Pin assertion failed with unbounded `setuptools` | ✅ 28 Modal tests passed | ➖ Structural contract has one valid output | ✅ Minimal AST-based static check |

## Verification

- Task 2.5 baseline command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py tests/test_runner_v2.py` — `40 passed` (26 Modal, 14 runner).
- Task 2.5 final command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py tests/test_runner_v2.py` — `41 passed` (27 Modal, 14 runner).
- No real Modal deployment, GPU inference, `game_03` validation, renderer, Phase 3 work, or `modal_app.py` modification was performed.
