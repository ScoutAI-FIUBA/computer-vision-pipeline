# Tasks: SAM 3.1 Player Tracking V2

## Review Workload Forecast

Estimated changed lines: 650–900
400-line budget risk: High
Suggested split: adapter/artifacts → runner/Modal → local validation
Delivery strategy: ask-on-risk; local ordered commits, no PRs

Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: feature-branch-chain
400-line budget risk: High

### Suggested Work Units

| Unit | Deliverable | Boundary |
|---|---|---|
| 1 | Mask/ReID/identity/team adapters plus tests | Local commit 1 |
| 2 | Runner/Modal wiring and artifacts | Local commit 2; `modal_app.py` untouched |
| 3 | Renderer and `game_03` validation | Local commit 3; no PR |

## Phase 1: Foundation / Contracts

- [x] 1.1 **RED:** Test SAM3.1 normalization, raw/stable identity separation, deterministic keys, and bounded retention in `tests/test_sam31_tracking.py`.
- [x] 1.2 **GREEN:** Create `pipeline/sam31_tracking.py` with multiplex lifecycle, prompts, normalized observations, bounded termination, and safe fragmentation.
- [x] 1.3 Add V2 settings to `pipeline/config.py`; preserve legacy defaults and test parsing.

## Phase 2: Artifact Packaging and Core Pipeline

- [x] 2.1 **RED:** Test manifest, `tracks.jsonl`, lossless masks, metadata, relative paths, and incomplete-transfer errors.
- [x] 2.2 **GREEN:** Extend `pipeline/results.py`; preserve legacy serialization.
- [x] 2.3 Create validation-only `pipeline/runner_v2.py`; defer statistics.
- [x] 2.4 Implement `modal_app_v2.py` ephemeral output and direct artifacts; never return `debug.mp4`.
- [x] 2.5 Test return completeness, local writes, size limits, boundaries, acceptance, and no R2 dependency.
- [x] 2.6 Add `sam31-session-compat`; cover both `init_state` signatures.
- [x] `sam31-output-envelope`: preserve nested outputs, outer `frame_index`, empty frames, and direct-output compatibility.

## Phase 3: Mask-Authoritative Identity and Team Wiring

- [x] 3.1 **RED/GREEN:** Add `MaskAppearanceAdapter` in `pipeline/runner_v2.py`: require full-frame non-empty matching masks, use bbox only as ROI, zero outside-mask pixels, and never fall back.
- [x] 3.2 Reuse `pipeline/reid.py::Embedder.embed(frame, boxes)` with masked frames/ROI boxes; V2 passes tracked frames/masks at runtime, and invalid masks yield no ReID or team evidence.
- [x] 3.3 Add conservative `RunnerV2Seams.identity`: merge only at/above `similarity_threshold=0.90` and within `max_frame_gap=5`; preserve `raw_track_id`, reject duplicate observation keys, and fragment ambiguity when authority is absent or evidence is weak.
- [x] 3.4 Run stable-first through `pipeline/team_embedder.py`/`pipeline/team_id.py`, pass `player_mask`, aggregate by stable ID, and atomically reject conflicts.
- [x] 3.4 follow-up: accept stable-player Team labels only at a dominant vote share of at least 0.85; reject lower shares and ties.

## Phase 4: V2 Runtime and Debug Integration

- [x] 4.1 Extend `modal_app_v2.py`/`RunnerV2` using real `modal_app.py` patterns: class lifecycle/model reuse, local video bytes, `TemporaryDirectory`, direct `artifacts: {relative_path: bytes}`, local writes, progress, and error handling.
- [x] 4.2 Keep legacy isolation: `modal_app.py` MUST NOT change; V2 dependencies and paths stay in V2 files.
- [x] 4.3 Preserve fields in `pipeline/results.py`; render raw/stable/team labels, masks, and rejection state in `scripts/tracking/debug_v2.py`.

## Phase 5: Local Debug and Validation

- [x] 5.1 Implement local retrieval/writing in `scripts/tracking/debug_v2.py` for tracks, masks, manifest, config, and benchmark metadata.
- [x] 5.2 Add focused pytest coverage in `tests/test_runner_v2.py`, `tests/test_modal_app_v2.py`, `tests/test_results_artifacts.py`, and `tests/test_debug_v2.py`.
- [x] 5.3 Validate `game_03`, approved prompts, frames 60–130, complete artifacts, deterministic masks, identity-first acceptance, and local labels.
- [x] 5.4 Record ordered conventional commits, verify unchanged `modal_app.py`, run `.venv/bin/python -m pytest`, and create no PRs.
- [x] 5.5 **RED/GREEN:** Add the explicit post-resolution identity-authority stage, wire it into V2 Modal, and cover accepted, rejected/scrubbed, and no-authority compatibility paths in runner and Modal boundary tests. Preserve the 0.85 Team policy.
- [x] 5.6 **RED/GREEN:** Harden authority acceptance with runner-produced stable/ReID/Team stage receipts; reject forged prepopulated fields and diagnose missing receipts without changing no-authority behavior or atomic scrubbing.
- [x] 5.7 **RED/GREEN:** Enforce opaque provenance for `RuntimeStageReceipts`; add multi-row acceptance and partial-stage-failure regressions proving all rows are checked and rejected results are scrubbed atomically.

## Status

Phase 1 and Phase 2 through 2.6 plus `sam31-output-envelope` are complete. Tasks 3.1–3.4, including V2 frame/mask runtime wiring, conservative stable identity, and stable-first Team ID, plus tasks 4.1–4.3, 5.2, 5.3, and 5.5–5.7 focused coverage, are complete. Completed commits and no-PR policy are preserved.

Review follow-up: `PipelineV2.load_models()` now has controlled fake-backend cold-container coverage; no production or legacy behavior changed. The focused V2 baseline is 157 tests. The final fresh native `game_03` run supplied runtime identity-authority evidence, passed the 0.85 Team-majority policy, and was accepted; see `verify-report.md` for the complete evidence.

## Post-completion Follow-up — SAM3 Propagation Quality Gating

- [ ] **Follow-up optimization — SAM3 propagation quality gating (not a renderer workaround):** Add propagation-time quality gates that reject observations with low mask fill inside the tracked bbox or sudden bbox-growth drift; when a gate trips, mark the propagated track invalid and reinitialize or refresh it from a trusted detector seed/session state. Add focused regression coverage for low-fill rejection, abrupt-growth rejection, refresh/reinitialization, and normal propagation. Keep the fix in the SAM3 tracking/runner quality boundary; do not change `scripts/tracking/debug_v2.py` or any renderer behavior to hide drift.

**Dependency:** completed SAM3.1 V2 propagation/artifact contract and representative `game_03` validation artifacts. **Acceptance:** drift is rejected before identity/ReID/Team evidence is accepted, the track is refreshed or reinitialized deterministically, and renderer output remains unchanged.

## Recommended Next Pipeline Task

- [ ] **Task 8.1 — Define the versioned statistics contract and validate provenance:** specify event fields/status vocabulary and validate player-track, team, ball, and court-coordinate provenance, including unknown/lost states, before implementing additional shot or possession logic.

**Rationale:** tracking and team validation are now accepted, while statistics/shots are the roadmap's explicit next phase and remain deferred. Starting with the contract prevents shot, possession, and pass logic from creating incompatible artifacts or claiming attribution when evidence is missing.

**Dependencies:** accepted SAM3.1 tracking/team artifacts; existing ball-signal output; court homography/projected coordinates where available. Shot make/miss implementation follows this contract; possession and passes remain downstream of validated evidence.
