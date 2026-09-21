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
- [ ] 3.4 Run stable-first through `pipeline/team_embedder.py`/`pipeline/team_id.py`, pass `player_mask`, aggregate by stable ID, and atomically reject conflicts.

## Phase 4: V2 Runtime and Debug Integration

- [ ] 4.1 Extend `modal_app_v2.py`/`RunnerV2` using real `modal_app.py` patterns: class lifecycle/model reuse, local video bytes, `TemporaryDirectory`, direct `artifacts: {relative_path: bytes}`, local writes, progress, and error handling.
- [ ] 4.2 Keep legacy isolation: `modal_app.py` MUST NOT change; V2 dependencies and paths stay in V2 files.
- [ ] 4.3 Preserve fields in `pipeline/results.py`; render raw/stable/team labels, masks, and rejection state in `scripts/tracking/debug_v2.py`.

## Phase 5: Local Debug and Validation

- [x] 5.1 Implement local retrieval/writing in `scripts/tracking/debug_v2.py` for tracks, masks, manifest, config, and benchmark metadata.
- [ ] 5.2 Add focused pytest coverage in `tests/test_runner_v2.py`, `tests/test_modal_app_v2.py`, `tests/test_results_artifacts.py`, and `tests/test_debug_v2.py`.
- [ ] 5.3 Validate `game_03`, approved prompts, frames 60–130, complete artifacts, deterministic masks, identity-first acceptance, and local labels.
- [x] 5.4 Record ordered conventional commits, verify unchanged `modal_app.py`, run `.venv/bin/python -m pytest`, and create no PRs.

## Status

Phase 1 and Phase 2 through 2.6 plus `sam31-output-envelope` are complete. Tasks 3.1–3.3, including V2 frame/mask runtime wiring and conservative stable identity, are complete. Tasks 3.4, 4.1–4.3, 5.2, and 5.3 (focused validation and `game_03`) remain pending; completed commits and no-PR policy are preserved.
