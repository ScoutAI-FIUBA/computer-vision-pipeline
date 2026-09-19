# Tasks: SAM 3.1 Player Tracking V2

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 650–900 |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | Adapter/artifacts → full runner/wiring → local renderer/validation |
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
| 2 | Full V2 pipeline and Modal wiring | Local commit 2 | Base commit 1; `modal_app.py` untouched |
| 3 | Local rendering and benchmark acceptance | Local commit 3 | Base commit 2; no PR creation for now |

## Phase 1: Foundation / Contracts

- [x] 1.1 **RED:** Add tests for SAM3.1 output normalization, `raw_track_id`/`stable_player_id` separation, deterministic observation keys, and bounded retention in `tests/test_sam31_tracking.py`.
- [x] 1.2 **GREEN:** Create `pipeline/sam31_tracking.py` with multiplex lifecycle, prompt handling, normalized observations, bounded loss termination, and fragmentation-over-contamination behavior.
- [x] 1.3 Add V2-only prompt, frame-range, retention, artifact, and acceptance settings to `pipeline/config.py`; preserve legacy defaults and add config tests.

## Phase 2: Artifact Packaging and Core Pipeline

- [x] 2.1 **RED:** Add artifact-contract tests for manifest, tracks JSONL, lossless masks, config/benchmark metadata, retrieval references, and explicit transfer errors.
- [x] 2.2 **GREEN:** Extend `pipeline/results.py` with those manifest/artifact references and packaging behaviors; keep legacy serialization intact.
- [x] 2.3 Create a truthful validation-only `pipeline/runner_v2.py` adapter that validates identity evidence, invokes the supported ReID/team contracts, and defers acceptance plus video-stream stages (TrackNet, court, shot, analytics, statistics) to later V2 orchestration.
- [x] 2.4 Create `modal_app_v2.py` with the isolated SAM3.1 image, runner entrypoint, and Modal/R2 upload-retrieval response; never render or return remote `debug.mp4`. Leave `modal_app.py` byte-for-byte untouched.
- [ ] 2.5 Add `tests/test_runner_v2.py` and `tests/test_modal_app_v2.py` for full wiring, artifact completeness/retrieval, no silent truncation, image boundary, and acceptance gating.

## Phase 3: Local Debug and Validation

- [ ] 3.1 Create `scripts/tracking/debug_v2.py` to download manifest/artifacts and render from the original video plus masks/JSONL; test reconstruction without remote video output.
- [ ] 3.2 Add `game_03` frames 60–130 validation with approved prompts and benchmark evidence; verify identity-first acceptance, zero cross-team contamination, and bounded-loss/fragmentation checks.
- [ ] 3.3 Record each work unit as an atomic conventional commit, verify rollback to unchanged `modal_app.py`, and run `.venv/bin/python -m pytest`; create no PRs.

## Status

Phase 1 foundation/contracts and Phase 2 artifact contract/packaging complete; ready for the next work unit.
