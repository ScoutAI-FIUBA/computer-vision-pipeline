# Tasks: SAM 3.1 Player Tracking V2

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 650–900 |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | Adapter/artifacts → runner/Modal direct return → local validation |
| Delivery strategy | ask-on-risk |
| Chain strategy | feature-branch-chain (local ordered commits; no PRs) |

Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: feature-branch-chain
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | SAM3.1 adapter and raw artifact contract | Local commit 1 | Base feature branch; tests included; rollback independently |
| 2 | Runner and Modal direct artifact return | Local commit 2 | Base commit 1; `modal_app.py` untouched |
| 3 | Local rendering and benchmark acceptance | Local commit 3 | Base commit 2; no PR creation |

## Phase 1: Foundation / Contracts

- [x] 1.1 **RED:** Test SAM3.1 normalization, raw/stable identity separation, deterministic keys, and bounded retention in `tests/test_sam31_tracking.py`.
- [x] 1.2 **GREEN:** Create `pipeline/sam31_tracking.py` with multiplex lifecycle, prompts, normalized observations, bounded termination, and safe fragmentation.
- [x] 1.3 Add V2 settings to `pipeline/config.py`; preserve legacy defaults and test configuration parsing.

## Phase 2: Artifact Packaging and Core Pipeline

- [x] 2.1 **RED:** Test manifest, `tracks.jsonl`, lossless masks, config/benchmark metadata, relative artifact paths, and explicit incomplete-transfer errors.
- [x] 2.2 **GREEN:** Extend `pipeline/results.py` with those artifact references and packaging behaviors; preserve legacy serialization.
- [x] 2.3 Create validation-only `pipeline/runner_v2.py` wiring identity evidence and supported ReID/team seams; defer statistics stages.
- [x] 2.4 Implement `modal_app_v2.py` using ephemeral container output and `artifacts: {relative_path: bytes}` for `modal run --video-path --output-dir`; use Modal Volumes only for model, checkpoint, and cache data, and never return `debug.mp4`.
- [x] 2.5 Test direct-return completeness, local output writing, short-clip/size-limit failure without silent truncation, image boundaries, acceptance gating, and no R2 dependency in `tests/test_runner_v2.py` and `tests/test_modal_app_v2.py`.
- [x] 2.6 Add the active `sam31-session-compat` helper to filter unsupported SAM 3.1 session-init kwargs without modifying the official builder or vendored package; cover both `init_state` signatures.
- [x] `sam31-output-envelope` pass official nested SAM3.1 outputs to `_session_objects`, preserve outer `frame_index`, normalize empty frames, and retain direct-output compatibility with strict-TDD coverage.

## Phase 3: Local Debug and Validation

- [x] 3.1 Implement local retrieval/writing in `scripts/tracking/debug_v2.py`; read returned raw tracks, lossless masks, manifest, config, and benchmark metadata beside the original video to render debug output locally.
- [ ] 3.2 Validate `game_03`, approved prompts, and frames 60–130 with complete returned artifacts and identity-first acceptance.
- [ ] 3.3 Record ordered conventional commits, verify unchanged `modal_app.py` rollback, and run `.venv/bin/python -m pytest`; create no PRs.

## Status

Phase 1 and Phase 2 work through task 2.6 plus `sam31-output-envelope` are complete; Phase 3 remains pending.
