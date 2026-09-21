# Apply Progress: SAM 3.1 Player Tracking V2

## Work Unit

- Boundary: Phase 1 foundation/contracts and adapter boundary, tasks 1.1–1.3; Phase 2 artifact packaging, tasks 2.1–2.2; runner wiring task 2.3; Modal V2 boundary task 2.4.
- Delivery: feature-branch-chain strategy implemented as ordered local atomic commits; no PR.
- Rollback: revert this Modal V2 work-unit commit plus the prior artifact/runner commits; `modal_app.py` remains the legacy fallback.
- Scope guard: `modal_app.py` untouched; remote rendering remains deferred.

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
- [x] `sam31-runtime-deps` runtime coverage: added controlled fake `sam3.model_builder` modules and executed `PipelineV2._ensure_predictor()` to verify the builder receives `use_fa3=False` and is cached.
- [x] `scipy-numpy-compat`: kept the V2 image on `scipy==1.14.1` and added a focused AST regression test for the NumPy 1.26-compatible pin.
- [x] `sam31-import-smoke`: added the reviewed SAM3 commit pin, official import dependencies, and a GPU/model-free Modal dependency smoke test with focused structural coverage.
- [x] `sam31-import-smoke` hardening: extracted injectable `check_sam31_imports`, preserved independent attempts/failures, and made the Modal wrapper raise a concise failure summary without model construction.
- [x] `sam31-import-smoke` review fixes: extracted injectable `run_sam31_dependency_smoke` wrapper logic with runtime failure coverage and documented the exact-image import command in `SETUP.md`.
- [x] `sam31-session-compat`: added `_start_session` in `modal_app_v2.py`, which inspects the predictor model's actual `init_state` signature and removes only unsupported optional session kwargs before normal `handle_request` registration; `_track_video` uses the helper and retains add/propagate/close lifecycle.
- [x] `sam31-session-compat` test strengthening: added focused `_track_video` lifecycle/session-ID assertions and unchanged genuine start-session exception propagation coverage; production code unchanged.
- [x] `sam31-session-compat-fix`: temporarily wraps model `init_state` methods whenever SAM3 `handle_request` rebuilds any unsupported keyword, filters every rebuilt keyword against the original bound signature, restores the bound original method on success/failure, and preserves the original resource path/session ID.
- [x] `sam31-output-envelope`: `_track_video` now passes `output.get("outputs", output)` to `_session_objects`, preserving the outer `frame_index`; empty output mappings normalize to zero objects while non-empty cardinality errors remain explicit; direct output compatibility is covered.
- [x] 3.1 Added the local-only `scripts/tracking/debug_v2.py` renderer and synthetic artifact/video tests. It reads raw `tracks.jsonl` observations, resolves lossless mask references, renders deterministic raw-track colors, `RAW: <id>` labels, boxes, and masks, preserves empty frames, handles missing masks, and supports inclusive source frame bounds. No Modal, remote execution, ReID, Team ID, or statistics code was added.
- [x] 3.1 review fixes: the renderer now reads `manifest.json` when present and rejects incomplete/truncated packages, transfer errors, or manifest-listed missing artifacts by default; `--allow-incomplete` is an explicit unsafe override. Synthetic tests use distinct source-frame content and behavioral pixel assertions for masks, labels, and inclusive frame selection.
- [x] 3.1 final boundary fixes: `manifest.json` is now required by default, with `--allow-incomplete` as the only bypass; missing tracks, masks directories, malformed manifests, and valid empty packages have controlled validation coverage. Tasks 3.2 and 3.3 remain pending.
- [x] 3.1 final review fixes: manifest `complete`/`truncated` fields now require real booleans; nested artifact keys, transfer errors, and missing lists require controlled types; CLI coverage proves `--allow-incomplete` reaches the render flow.
- [x] `renderer-normalized-bboxes`: normalized `bbox_xyxy` values whose four coordinates are all in `[0, 1]` are scaled by the current frame width/height before drawing; any coordinate outside that range retains pixel-coordinate behavior. Mask overlays and raw labels remain rendered.

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
| SAM31 runtime predictor coverage | `tests/test_modal_app_v2.py` | ✅ 29/29 prior Modal tests passed | ✅ Runtime test exercises the pre-existing path and would fail without the keyword | ✅ 30 Modal tests passed | ✅ Builder invocation plus cached second call | ✅ Controlled module seam; no production change |
| scipy-numpy-compat | `tests/test_modal_app_v2.py` | ✅ 37/37 prior Modal tests passed | ✅ New exact SciPy pin assertion written before verification | ✅ 38 Modal tests passed | ➖ Single structural dependency contract | ✅ Minimal AST-based assertion; no production change |

## `sam31-import-smoke` TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|------|-----------|-------|------------|-----|-------|-------------|----------|
| `sam31-import-smoke` | `tests/test_modal_app_v2.py` | Unit/static boundary | ✅ 33/33 | ✅ Dependency/pin/smoke assertions failed before implementation | ✅ 33 Modal tests passed | ✅ Seven module names, callable builder, and no model construction | ✅ One image and one independent import loop |
| `sam31-import-smoke` hardening | `tests/test_modal_app_v2.py` | Unit | ✅ 33/33 | ✅ Injectable helper test failed before extraction | ✅ 35 Modal tests passed | ✅ Selected import failures plus non-callable builder while all modules were attempted | ✅ Small pure helper; Modal wrapper only formats/raises |
| `sam31-import-smoke` review fixes | `tests/test_modal_app_v2.py`, `SETUP.md` | Unit/docs | ✅ 35/35 | ✅ Runtime wrapper test failed before injectable wrapper logic | ✅ 37 Modal tests passed | ✅ Failure and success wrapper paths; command purpose and GPU distinction documented | ✅ Modal function delegates to the same minimal wrapper |
| `sam31-session-compat` | `tests/test_modal_app_v2.py` | Unit | ✅ 38/38 | ✅ Import failed before `_start_session` existed | ✅ 40 Modal tests passed | ✅ `init_state` accepting and rejecting `offload_state_to_cpu`; session id and registration kwargs verified | ✅ Small signature filter; official builder and predictor lifecycle unchanged |
| `sam31-session-compat` test strengthening | `tests/test_modal_app_v2.py` | Unit | ✅ 40/40 | ✅ New lifecycle/error assertions added against the active implementation | ✅ 42 Modal tests passed | ✅ Session ID propagation, ordered cleanup, and unchanged start exception | ✅ Tests-only; no production change required |
| `sam31-session-compat-fix` | `tests/test_modal_app_v2.py` | Unit | ✅ 44/44 | ✅ Intermediate signature passed unsupported kwarg before generalized wrapper | ✅ 45 Modal tests passed | ✅ Supported/unsupported/intermediate signatures plus success/failure restoration and session ID | ✅ Every request kwarg is checked; bound original captured once; wrapper removed in `finally`; exceptions propagate |
| `sam31-output-envelope` | `tests/test_modal_app_v2.py` | Unit | ✅ 47/47 existing tests | ✅ Nested envelope, empty output, and malformed cardinality cases failed before implementation | ✅ 50 Modal tests passed | ✅ Official envelope, empty frame, malformed non-empty response, and direct-output compatibility | ✅ Minimal fallback plus `_sequence_is_empty`; upstream and legacy boundaries untouched |

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
- `sam31-runtime-deps` runtime-test focused command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py` — `30 passed`.
- `sam31-runtime-deps` runtime-test relevant command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py tests/test_runner_v2.py tests/test_results_artifacts.py tests/test_sam31_tracking.py` — `56 passed`.
- `scipy-numpy-compat` focused command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py` — `38 passed`.
- `scipy-numpy-compat` relevant command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py tests/test_dependency_compatibility.py` — `42 passed, 1 skipped`.
- `sam31-import-smoke` focused command: `PYTHONPATH=. uv run --with pytest==8.3.5 pytest tests/test_modal_app_v2.py -q` — `33 passed`.
- `sam31-import-smoke` relevant command: `PYTHONPATH=. uv run --with pytest==8.3.5 pytest tests/test_modal_app_v2.py tests/test_dependency_compatibility.py -q` — `37 passed, 1 skipped`.
- Full suite: `PYTHONPATH=. uv run --with pytest==8.3.5 pytest -q` — `346 passed, 1 skipped`.
- `sam31-import-smoke` hardening focused command: `PYTHONPATH=. uv run --with pytest==8.3.5 pytest tests/test_modal_app_v2.py -q` — `35 passed`.
- `sam31-import-smoke` hardening relevant command: `PYTHONPATH=. uv run --with pytest==8.3.5 pytest tests/test_modal_app_v2.py tests/test_dependency_compatibility.py -q` — `39 passed, 1 skipped`.
- Full suite after hardening: `PYTHONPATH=. uv run --with pytest==8.3.5 pytest -q` — `348 passed, 1 skipped`.
- `sam31-import-smoke` review-fix focused command: `PYTHONPATH=. uv run --with pytest==8.3.5 pytest tests/test_modal_app_v2.py -q` — `37 passed`.
- `sam31-import-smoke` review-fix relevant command: `PYTHONPATH=. uv run --with pytest==8.3.5 pytest tests/test_modal_app_v2.py tests/test_dependency_compatibility.py -q` — `41 passed, 1 skipped`.
- Full suite after review fixes: `PYTHONPATH=. uv run --with pytest==8.3.5 pytest -q` — `350 passed, 1 skipped`.
- `sam31-output-envelope` focused command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py -q` — `50 passed`.
- `sam31-output-envelope` full suite: `.venv/bin/python -m pytest -q` — `363 passed, 1 skipped`.
- CI configuration: no repository CI workflow/configuration was found; none was added.
- Full suite was executed for `sam31-import-smoke`; final renderer/rollback verification remains deferred.

## Remaining Tasks

- [x] 2.1–2.3 Artifact packaging and truthful runner adapter boundary.
- [x] 2.4 Modal V2 wiring and raw artifact retrieval boundary.
- [x] 2.5 Added controlled-fake process coverage for complete direct artifacts, lossless masks, manifest config/benchmark metadata, explicit size-limit failure, SAM cardinality errors, and existing local nested-path, acceptance, no-R2, no-debug-video, and legacy-isolation contracts.
- [x] `sam31-import-smoke` added explicit `pycocotools`, `Pillow`, and `psutil` dependencies; pinned SAM3 to `2345a4a`; and added `sam31_dependency_smoke_test` without model construction or GPU requirements.
- [x] `sam31-import-smoke` hardening added controlled import-failure and non-callable-builder coverage while preserving all dependency, pin, isolation, and no-download contracts.
- [x] `sam31-import-smoke` review fixes added runtime wrapper failure/success coverage and documented `modal run modal_app_v2.py::sam31_dependency_smoke_test` in `SETUP.md`.
- [x] `sam31-session-compat` added signature-compatible session registration coverage for both official-model signature variants.
- [x] `sam31-session-compat-fix` corrected compatibility for SAM3 base `handle_request` request rebuilding, generalized per-keyword filtering, and added intermediate-signature coverage.
- [x] `sam31-output-envelope` corrected official nested propagation output handling and empty-frame normalization without changing SAM3.1 upstream code.
- [x] 3.1 Local renderer and focused synthetic artifact/video coverage.
- [ ] 3.2 Validate `game_03`, approved prompts, and frames 60–130 with complete returned artifacts and identity-first acceptance.
- [ ] 3.3 Record ordered conventional commits, verify unchanged `modal_app.py` rollback, and run `.venv/bin/python -m pytest`; create no PRs.

## Phase 3 Task 3.1 TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|---|---|---|---|---|---|---|---|
| 3.1 | `tests/test_debug_v2.py` | Unit with synthetic video/artifacts | ✅ 10/10 baseline | ✅ Boolean/nested-type and CLI regressions | ✅ 20 focused tests passed | ✅ Real booleans, malformed nested fields, CLI override forwarding, plus prior rendering/empty/error paths | ✅ Centralized strict validation; no pipeline changes |

## `renderer-normalized-bboxes` TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|---|---|---|---|---|---|---|---|
| `renderer-normalized-bboxes` | `tests/test_debug_v2.py` | Unit with synthetic frame/artifacts | ✅ 21/21 | ✅ Full-label/corner and non-finite regression tests written; inf cases failed before guard | ✅ Focused renderer suite passed | ✅ All four corners, full `RAW: 7` label including digit, mask overlay, pixel boxes, inf, and NaN | ✅ Finite-value guard reuses existing skip-observation behavior |

## Task 2.5 TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|------|-----------|-------|------------|-----|-------|-------------|----------|
| 2.5 | `tests/test_modal_app_v2.py`, `tests/test_runner_v2.py` | Unit with controlled process seam | ✅ 40/40 | ✅ Process/cardinality tests failed before seam correction | ✅ 27 Modal tests passed; 14 runner tests passed | ✅ Complete accepted fields/mask bytes, rejected process gating, size failure, cardinality mismatch, rejection/deferred paths | ✅ Reused existing artifact writer/runner contracts; no production change |
| 2.4 compatibility remediation | `tests/test_modal_app_v2.py` | Unit/static boundary | ✅ 27/27 | ✅ Pin assertion failed with unbounded `setuptools` | ✅ 28 Modal tests passed | ➖ Structural contract has one valid output | ✅ Minimal AST-based static check |

## Verification

- Task 2.5 baseline command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py tests/test_runner_v2.py` — `40 passed` (26 Modal, 14 runner).
- Task 2.5 final command: `.venv/bin/python -m pytest tests/test_modal_app_v2.py tests/test_runner_v2.py` — `41 passed` (27 Modal, 14 runner).
- No real Modal deployment, GPU inference, or `game_03` validation was performed; `modal_app.py` remains unchanged.
- Task 3.1 focused command: `.venv/bin/python -m pytest tests/test_debug_v2.py -q` — `3 passed`.
- Task 3.1 relevant command: `.venv/bin/python -m pytest tests/test_debug_v2.py tests/test_results_artifacts.py -q` — `9 passed`.
- Repository suite after task 3.1: `.venv/bin/python -m pytest -q` — `367 passed, 1 skipped`.
- Task 3.1 review-fix focused command: `.venv/bin/python -m pytest tests/test_debug_v2.py -q` — `6 passed`.
- Task 3.1 final-boundary focused command: `.venv/bin/python -m pytest tests/test_debug_v2.py -q` — `10 passed`.
- Task 3.1 strict-type/CLI focused command: `.venv/bin/python -m pytest tests/test_debug_v2.py -q` — `20 passed`.
- `renderer-normalized-bboxes` focused command: `.venv/bin/python -m pytest tests/test_debug_v2.py -q` — `21 passed`.
- `renderer-normalized-bboxes` review-fix focused command: `.venv/bin/python -m pytest tests/test_debug_v2.py -q` — `26 passed`.
- `renderer-normalized-bboxes` label assertion: deterministic OpenCV-rendered ROI comparison now covers the full `RAW: 7` label including its digit/ID, alongside all four normalized bbox corners.
- Task 3.1 remains local-only; no Modal deployment, GPU inference, `game_03` validation, ReID/Team ID/statistics, or `modal_app.py` modification was performed.
- `sam31-import-smoke` also made no real Modal deployment or model download; the smoke function only imports dependencies and validates builder callability.
- `sam31-session-compat` made no upstream or vendored package changes; only `modal_app_v2.py` and its focused fake-predictor tests changed.
- `sam31-session-compat` test strengthening changed tests and cumulative progress only; the production helper required no modification.
- `sam31-session-compat-fix` changed only `modal_app_v2.py`, its focused fake-predictor tests, and this progress record; no upstream/legacy/R2/renderer/game files were touched.
- `sam31-output-envelope` changed only `modal_app_v2.py`, `tests/test_modal_app_v2.py`, this progress record, and the task checkbox; no upstream/legacy/R2/renderer/game files were touched.

## `sam31-memory-mitigation` Work Unit

- Boundary: Configure the official SAM 3.1 multiplex predictor for ten objects/streams and request CPU video offload during session initialization; preserve signature filtering for older `init_state` variants.
- Delivery: One local conventional commit on the active native branch; no PR and no branch reset/acquisition.
- Scope guard: A100-40GB remains unchanged; no model, legacy `modal_app.py`, R2, renderer, or game changes.

### Completed

- [x] `_start_session` includes `offload_video_to_cpu=True` and filters it when the bound `init_state` does not support it.
- [x] `_ensure_predictor` calls `build_sam3_multiplex_video_predictor` with `max_num_objects=10`, `multiplex_count=10`, and `use_fa3=False`.
- [x] Added focused propagation/filtering and builder-configuration regression coverage.

### TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|---|---|---|---|---|---|---|---|
| `sam31-memory-mitigation` | `tests/test_modal_app_v2.py` | Unit | ✅ 50/50 | ✅ 4 failures before implementation | ✅ 51 passed | ✅ Supported and unsupported `offload_video_to_cpu`; cached builder call | ✅ Minimal request/builder changes |

### Verification

- Focused: `.venv/bin/python -m pytest tests/test_modal_app_v2.py -q` — `51 passed`.
- Full: `.venv/bin/python -m pytest -q` — `364 passed, 1 skipped`.

## `sam31-gpu80` Work Unit

- Boundary: Move only the V2 Modal `PipelineV2` class from `A100-40GB` to `A100-80GB`; retain the legacy `modal_app.py` `L4` binding and all model, checkpoint, offload, renderer, R2, and game behavior.
- Delivery: One local conventional commit on the active native branch; no branch acquisition or reset.
- Scope guard: Only `modal_app_v2.py`, `tests/test_modal_app_v2.py`, and this progress record changed.

### Completed

- [x] Updated the exact AST GPU regression test to require V2 `A100-80GB` while requiring legacy `L4`.
- [x] Changed only the V2 Modal class GPU binding to `A100-80GB`; legacy `modal_app.py` remains unchanged.

### TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|---|---|---|---|---|---|---|---|
| `sam31-gpu80` | `tests/test_modal_app_v2.py` | Unit/static boundary | ✅ 51/51 | ✅ Updated AST expectation failed (`1 failed`) | ✅ 51 focused tests passed | ➖ Skipped: purely structural GPU configuration with one expected output and legacy-L4 assertion | ➖ None needed |

### Verification

- Focused red: `.venv/bin/python -m pytest tests/test_modal_app_v2.py::test_v2_and_legacy_gpu_capacity_are_explicitly_separate -q` — expected `1 failed` before the production change.
- Focused green: `.venv/bin/python -m pytest tests/test_modal_app_v2.py -q` — `51 passed`.

## `sam31-multiplex-restore` Work Unit

- Boundary: Restore checkpoint-compatible SAM 3.1 multiplex capacity while retaining FA3 disablement, CPU video offload, and A100-40GB execution.
- Delivery: One local conventional commit on the active native branch; no branch reset/acquisition and no PR.
- Scope guard: Only `modal_app_v2.py`, its focused configuration test, and this progress record changed; legacy, R2, renderer, game validation, and model checkpoint remain untouched.

### Completed

- [x] Restored `_ensure_predictor` to `max_num_objects=16` and `multiplex_count=16`.
- [x] Retained `use_fa3=False`, `offload_video_to_cpu=True`, and `gpu="A100-40GB"`.
- [x] Updated focused builder-configuration regression coverage.

### TDD Cycle Evidence

| Task | Test File | Layer | Safety Net | RED | GREEN | TRIANGULATE | REFACTOR |
|---|---|---|---|---|---|---|---|
| `sam31-multiplex-restore` | `tests/test_modal_app_v2.py` | Unit/static boundary | ✅ 51/51 | ✅ Updated expected capacity failed against 10/10 | ✅ 51 passed | ➖ Structural configuration contract; cached second-call assertion retained | ✅ Minimal two-value configuration change |

### Verification

- Focused: `.venv/bin/python -m pytest tests/test_modal_app_v2.py -q` — `51 passed`.
- Full: `.venv/bin/python -m pytest -q` — `364 passed, 1 skipped`.
