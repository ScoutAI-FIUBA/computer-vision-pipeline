# Future Tasks: Basketball Video Analysis Pipeline

This backlog documents the remaining implementation work needed to evolve the
player/court pipeline into a fuller basketball video-analysis pipeline.

Ball-model training, TrackNet experiments, and ball-only evaluation remain a
separate experimental workstream under `scripts/ball/`. The production
TrackNet+SAM2 integration is complete for this phase and is documented in the
main pipeline/setup docs; keep only the model experimentation there instead of
mixing it into player tracking, team ID, jersey OCR, or statistics planning.

## Workstream Boundaries

| Workstream | Owns | Does not own |
|---|---|---|
| Player tracking | player detection, tracking identity, track exports | ball model training or ball-only inference experiments |
| Court geometry | court keypoints, homography, projected player positions | shot resolution logic by itself |
| Team ID / jersey OCR | per-track team and jersey evidence | ball detection |
| Ball model experiments | ball-only scripts, checkpoints, fine-tuning docs in `scripts/ball/` | production player/team/statistics integration |
| Shot / make-miss events | event windows, player/team association, shot charts | ball model training details |

## Current State

| Area | Status |
|---|---|
| Image detection | Implemented with Roboflow Inference and annotated image output. |
| Video detection | Implemented with frame sampling, dry-run cost estimation, sampled-only output, and annotated MP4 output. |
| Cost control | Basic controls exist: `--dry-run`, `--frame-stride`, `--max-frames`, `--sampled-output-only`. |
| Supervision integration | Implemented for image/video annotation with shared detection/keypoint annotation helpers. |
| Tracking | Implemented locally with official SAM2 entrypoint `scripts/tracking/track_players_sam2.py`. The older `scripts/tracking/video_pipeline.py` / `scripts/tracking/track_players_video.py` box-tracker path still exists for comparison and support tooling. |
| Ball detection | Completed for this phase: TrackNet is the production primary, with dense source-frame output and bounded ball-only SAM2 fallback. |
| Team identification | Completed for this phase in `pipeline/` and Modal. FashionSigLIP is the default backbone; end-of-run assignment writes `teams.json` and rewrites `tracks.jsonl`. |
| Jersey OCR | Implemented as a conservative detector-frame OCR-and-aggregation path. |
| Court keypoints / homography | Implemented locally for single-frame, sampled-video, homography debug, and projected-track export. Homography now rejects obviously bad point layouts before fitting. |
| SAM 2 tracking | Implemented and now the tracker path to use going forward. |

## Implementation Order

| Phase | Goal | Why now |
|---|---|---|
| 1 | Stabilize detection outputs | Everything depends on trustworthy detections. |
| 2 | Convert to Supervision | Avoid custom drawing/tracking glue. |
| 3 | Keep the existing box-tracker path available for comparison | Useful baseline and support path beside the official SAM2 tracker. |
| 4 | Add court keypoints and homography | Needed for positions, shot charts, and spatial stats. |
| 5 | Improve team identification quality and validation | The baseline is already wired; the remaining work is evidence quality, validation, and output confidence. |
| 6 | Add jersey OCR | Useful but noisy; depends on good crops/tracks. |
| 7 | Improve SAM2 robustness and operating cost | SAM2 is now the chosen tracker path; optimize it instead of deferring it. |
| 8 | Keep ball-model experimentation separate | Production TrackNet+SAM2 integration is complete; model training and ball-only evaluation remain under `scripts/ball/`. |
| 9 | Add make/miss and event logic | Depends on stable tracking, court geometry, and ball/rim signals. |
| 10 | Add evaluation and regression set | Needed before production. |
| 11 | Decide deployment path | Local/serverless/dedicated depends on measured cost and latency. |

## Dedicated Ball Detection Track

Use this backlog only to note integration boundaries.

- Experimental ball work lives under `scripts/ball/`.
- Keep ball-model details out of the player-tracking phases unless they change the integration contract.
- Prefer linking to the ball docs instead of duplicating training or dataset notes here.

References:

- `scripts/ball/FINETUNING.md`
- `scripts/ball/FINETUNING_TRACKNETV3_REPO.md`

## Phase 1 — Detection Foundation

### Task 1.1: Centralize model configuration

- [x] Move model IDs into a small config module or `.env` variables.
- [x] Support at least:
  - `ROBOFLOW_API_KEY`
  - `PLAYER_MODEL_ID`
  - `COURT_MODEL_ID`
  - `JERSEY_OCR_MODEL_ID`
- [x] Keep defaults aligned with the currently working player model.

Acceptance criteria:

- Running image/video detection still works without code edits.
- Changing model version does not require editing script internals.

### Task 1.2: Save raw inference outputs

- [x] Add optional `--save-json` to image and video scripts.
- [x] For video, save one JSON record per inferred frame.
- [x] Include frame index, timestamp, model ID, and predictions.

Acceptance criteria:

- Detection outputs can be inspected without re-calling Roboflow.
- Failed/interesting frames can be replayed offline.

Notes:

- Video raw inference output is now emitted as JSONL from `scripts/tracking/video_pipeline.py`
  and `scripts/tracking/track_players_video.py`.
- Current per-inference record includes `frame_index`, `timestamp_seconds`,
  `model_id`, and `predictions`.

### Task 1.3: Add frame extraction utility

- [ ] Add a script to extract sampled frames from a video.
- [ ] Support `--frame-stride`, `--max-frames`, and `--output-dir`.
- [ ] Do not call Roboflow from this script.

Acceptance criteria:

- A short target clip can be converted into reviewable frame images.
- The user can manually inspect whether the sample is representative.

## Phase 2 — Supervision Integration

### Task 2.1: Convert Roboflow outputs to `sv.Detections`

- [x] Add `supervision` to pinned dependencies.
- [x] Convert `InferenceHTTPClient` results with `sv.Detections.from_inference(result)`.
- [x] Replace manual detection filtering with Supervision filtering.

Acceptance criteria:

- Image script can annotate using `sv.BoxAnnotator` and `sv.LabelAnnotator`.
- Video script no longer needs custom box drawing logic.

### Task 2.2: Create annotation helpers

- [x] Create reusable helpers for labels, colors, confidence thresholds, and class filtering.
- [x] Keep helpers model-agnostic.

Acceptance criteria:

- Image and video scripts use the same annotation code.
- Adding court/player/jersey annotations does not duplicate drawing logic.

Notes:

- `scripts/shared/annotation_helpers.py` now owns shared detection filtering, label formatting,
  color lookup, and point/detection annotation helpers.
- `scripts/tracking/track_players_video.py`, `scripts/tracking/detect_players.py`,
  `scripts/tracking/detect_players_video.py`, `scripts/court/detect_court_keypoints.py`, and
  `scripts/court/detect_court_keypoints_video.py` now reuse the shared helpers instead
  of maintaining separate label or OpenCV text/color code paths.

## Phase 3 — Cheap Tracking First

### Task 3.1: Add ByteTrack tracking

- [x] Use `sv.ByteTrack` or the current recommended Supervision tracker API.
- [x] Track only detections from frames actually inferred.
- [x] Output track IDs in labels.

Acceptance criteria:

- [x] Sampled video output can show track IDs on simple clips.
- [x] Track IDs are saved in JSONL for later analysis.

Notes:

- Implemented locally in `scripts/tracking/track_players_video.py` and exposed through the
  unified entrypoint `scripts/tracking/video_pipeline.py`; not API-integrated yet.
- Default output writes only inferred frames to avoid implying continuous tracking across skipped frames.
- Tracking quality is improved but not solved: IDs still switch in crossings/occlusions, some tracks still die entirely, and `--frame-stride 1` is too slow/costly to treat as the default fix.
- The current tracker construction is still `sv.ByteTrack()`, and the deprecation warning means the tracker API choice should be revisited before treating this path as stable.

## Completed phases and historical notes

The earlier tracking, court, ball, and team sections below preserve historical
implementation notes. Ball detection and team identification are closed for
this phase; do not reopen those sections as the next implementation target
unless new evidence breaks their current output contracts.

## Statistics progress

The current shot slice now preserves legacy `shot_events.jsonl` and adds
conservative player/team attribution to `shot_evidence.jsonl`. Attribution
requires observed ball evidence plus exactly one consistent player/stable
identity whose existing track box contains the shot anchor; team IDs are copied
only when present and non-conflicting. Proximity and ball-only evidence are not
enough.

## Next immediate task: statistics detection

Implement the following in order during the next session:

1. Define a versioned statistics contract/schema and event status vocabulary.
2. Validate player-track, team-ID, ball-track, and court-coordinate provenance,
   including unknown, lost, and unassigned states.
3. Implement shot/event detection with explicit make/miss outcomes and
   reviewable unknown events.
4. Add possession only after player/team/ball evidence passes validation; the
   narrow `possession_events.jsonl` slice is now implemented. Keep passes deferred.
5. Add passes only after possession and transfer evidence support them; do not
   infer passes from proximity alone.
6. Select a small evaluation clip set and define practical metrics for event
   precision/recall, outcome accuracy, and provenance coverage.
7. Define and write the output artifacts, with references from `results.json`.
8. Add acceptance tests for schema validation, missing evidence, make/miss,
   possession, and rejected/unknown events.

The current `shot_events.jsonl` is a minimal existing artifact, not proof that
the statistics phase is implemented.

Historical next-task note (superseded):

**Previously recommended:** Task 3.3 — evaluate tracker robustness on
crossings/occlusions, then decide whether to replace or wrap the current tracker API.

Why this was previously next:

- The repo already has config centralization, raw inference JSONL output, and a
  unified video pipeline entrypoint for player detection + tracking.
- The main blocker is no longer missing wiring. It is identity robustness under
  real basketball crossings and occlusions.
- The repo now has enough debug output (`tracks.jsonl`, `court_keypoints.jsonl`,
  `projected_tracks.jsonl`, top-down visualization) to evaluate failures
  directly instead of guessing.
- `Task 1.3` (frame extraction utility) is still useful, but it is now a
  support tool rather than the main pipeline bottleneck.

### Task 3.2: Add track-level smoothing

- [x] Store detections per track.
- [x] Smooth boxes/labels over short windows.
- [x] Avoid drawing stale boxes by default unless explicitly requested.

Acceptance criteria:

- Boxes no longer appear visually misleading between sampled frames.
- Debug mode can show only sampled frames; tracking mode can show full video.

Notes:

- Implemented through `TrackSmoother` plus sampled-output defaults in
  `scripts/tracking/track_players_video.py` and `scripts/tracking/video_pipeline.py`.
- This improved visual stability, but it did not solve the harder identity
  problem during player crossings and occlusions.

### Task 3.3: Compare the legacy box-tracker path against SAM2

- [ ] Create a small evaluation clip set with occlusions and camera motion.
- [ ] Count ID switches, full track losses, and reassignment recoveries manually.
- [ ] Compare current `sv.ByteTrack()` behavior against the current recommended Supervision tracker API.
- [ ] Decide whether any legacy box-tracker path is still worth keeping once SAM2 baselines are stable.

Acceptance criteria:

- SAM2 remains the official path unless target-footage evaluation shows a clear reason to fall back.

Notes:

- `--frame-stride 1` is not an acceptable default answer by itself: it raises
  Roboflow cost/runtime and still does not guarantee identity consistency.
- The current torso-color appearance matching is only a cheap baseline. Treat it
  as a recovery heuristic, not a production identity solution.

## Phase 4 — Court Keypoints and Homography

### Task 4.1: Integrate court keypoint model

- [x] Add configurable model ID for `basketball-court-detection-2` in the local image/frame script.
- [x] Run court inference on a single frame.
- [x] Save keypoints and confidence values.
- [x] Run court inference on sampled video frames.

Acceptance criteria:

- Court keypoints are visible on annotated debug frames.
- The script reports when not enough keypoints are detected.

Notes:

- Implemented locally in `scripts/court/detect_court_keypoints.py` (single image) and `scripts/court/detect_court_keypoints_video.py` (sampled video). Not API-integrated yet.
- Default model is `COURT_MODEL_ID` or `basketball-court-detection-2/22`.
- Video script defaults: `--confidence 0.7` (model returns ~33 raw keypoints but most are noise <0.05; only ~10-12 are reliable >0.97), `--frame-stride 30`, `--sampled-output-only true`. Uses the same `parse_keypoints` helper as the image script.
- Image visualization now uses Supervision `sv.KeyPoints` + `sv.VertexAnnotator` + `sv.EdgeAnnotator` (see `to_keypoints` and `draw_with_annotators` in `scripts/court/detect_court_keypoints.py`). The edges list is a placeholder that connects consecutive parsed keypoints — replace with the canonical court skeleton once labels are mapped.

### Task 4.2: Compute homography

- [x] Map detected court keypoints to canonical court coordinates.
- [x] Use OpenCV `findHomography` with RANSAC.
- [x] Per-sampled-frame homography in a debug video (original | FIBA 2D court with projected keypoints).
- [ ] Project player foot points onto court coordinates.

Notes:

- Implemented locally in `scripts/court/build_court_homography.py` (RANSAC, 8 px
  threshold by default; `--self-test` for a synthetic round-trip check; no
  Roboflow call) and `scripts/court/build_court_homography_video.py` (sampled
  frames, side-by-side debug MP4 with frame index, status, inliers count and
  mean reprojection error in meters).
- Homography now performs geometric pre-validation on the matched image points
  and rejects degenerate layouts before calling `cv2.findHomography`.
- Projected player foot points are now exported from the tracking pipeline as
  JSONL when `--court-keypoints` is enabled.
- The court mapping still depends on the current template assumptions and needs
  continued validation before using projected coordinates for real stats.

Acceptance criteria:

- Player positions can be exported as court coordinates.
- Bad homography frames are rejected instead of silently producing wrong stats.

### Task 4.3: Generate shot/location debug view

- [x] Create a top-down court visualization.
- [x] Plot tracked player positions.
- [x] Save a PNG or simple MP4 overlay.

Acceptance criteria:

- A reviewer can visually confirm that court projection is plausible.

Notes:

- Implemented locally in `scripts/court/visualize_projected_tracks.py`.
- Reuses `draw_fiba_court` from `scripts/court/build_court_homography.py` and consumes the existing projected-track JSONL emitted by `scripts/tracking/track_players_video.py` / `scripts/tracking/video_pipeline.py`.
- Produces an offline top-down PNG summary by default and can optionally render a simple court-only MP4 for per-frame review, with no additional Roboflow calls.

## Phase 5 — Team identification (completed for this phase; historical detail)

### Task 5.1: Keep the current config and docs aligned with the deployed path

- [x] Document `PipelineConfig` as the source of truth.
- [x] Document the current team-ID environment variables and override surfaces.
- [x] Remove outdated notes that described team ID as not implemented or as a different algorithm.

Acceptance criteria:

- README, setup notes, and task docs all describe the same current team-ID flow.
- The documented knobs match `pipeline/config.py`, `pipeline/cli.py`, and `modal_app.py`.

### Task 5.2: Validate team-ID quality across multiple clips

- [ ] Create a small review set with at least two team-color combinations.
- [ ] Measure wrong-team assignments, unassigned tracks, and obviously contaminated evidence.
- [ ] Save example `team_debug/` artifacts for each recurring failure mode.

Acceptance criteria:

- Threshold changes are judged on more than one clip.
- Reviewers can point to concrete debug artifacts instead of tuning blindly.

### Task 5.3: Improve evidence quality before adding more classifier stages

- [ ] Review `team_debug.json` and `team_debug/` to find the most common contamination pattern.
- [ ] Tighten crop, mask, or sample-acceptance logic only where the debug artifacts justify it.
- [ ] Re-run the same multi-clip review set before changing defaults.

Acceptance criteria:

- Any new threshold or crop change is backed by before/after evidence.
- The pipeline keeps the same output contract: `results.json`, `tracks.jsonl`, `teams.json`, optional `annotated.mp4`.

## Open Tracking Issues

- Player IDs still switch during crossings and occlusions even after the recent
  persistence/reassignment work.
- Tracking can still be lost entirely in some crossings/occlusions.
- `sv.ByteTrack()` currently emits a deprecation warning, so the tracker API
  path itself should be treated as under review.
- The current appearance re-ID baseline is intentionally cheap and should not be
  mistaken for solved identity tracking.

See `TRACKING_ISSUES.md` for the focused issue breakdown and recommended next actions.
See `future_tasks/robust-player-tracking-plan.md` for the trackable rework plan,
phase order, and success criteria that should guide Task 3.3 and the follow-on
tracking changes.

### Task 3.4: Use the tracking rework plan as the working reference

- [ ] Use `future_tasks/robust-player-tracking-plan.md` as the planning reference for the tracking rework.
- [ ] Finish the evaluation-first slice before changing the tracking architecture more aggressively.

Acceptance criteria:

- Tracking changes are evaluated against the same clip set before and after tracker/re-ID changes.
- The backlog has one explicit reference doc for the tracking rework instead of scattered notes only.

## Phase 5 — Team Identification (completed for this phase; historical detail)

### Task 5.1: Start with color clustering baseline

- [ ] Crop tracked players.
- [ ] Focus on torso area when possible.
- [ ] Cluster dominant jersey colors in HSV/Lab.

Acceptance criteria:

- Tracks receive team labels on simple clips without SigLIP.
- Team labels are stable per track, not recalculated blindly per frame.

### Task 5.2: Add SigLIP embeddings if color fails

- [ ] Add optional SigLIP dependency set.
- [ ] Generate embeddings for player crops.
- [ ] Reduce with UMAP and cluster with K-Means.

Acceptance criteria:

- FashionSigLIP is the production default; explicit legacy timm override remains supported.
- Team assignment is measured against a small manually labeled clip set.

### Task 5.3: Track-level team voting

- [ ] Accumulate team predictions per track.
- [ ] Assign final team by majority/centroid over time.

Acceptance criteria:

- Team labels do not flicker frame to frame.

## Phase 6 — Jersey Number OCR

### Task 6.1: Integrate Roboflow jersey OCR model

- [ ] Add `JERSEY_OCR_MODEL_ID` config.
- [ ] Crop tracked player torso/back regions.
- [ ] Send only selected sharp crops to OCR to control cost.

Acceptance criteria:

- OCR calls are counted and printed like detection calls.
- OCR output is stored per track with frame index.

### Task 6.2: Add temporal voting for jersey numbers

- [ ] Accumulate OCR predictions per track.
- [ ] Filter low-confidence predictions.
- [ ] Select final jersey number by majority or weighted vote.

Acceptance criteria:

- One noisy crop does not overwrite a stable jersey number.

### Task 6.3: Evaluate SmolVLM2 vs lightweight classifier later

- [ ] Keep VLM path optional.
- [ ] Test a cheaper classifier only after collecting enough crops.
- [ ] Compare exact match and per-digit accuracy.

Acceptance criteria:

- Heavy OCR/VLM is justified by measured improvement, not assumption.

## Phase 7 — SAM 2 tracking (completed)

SAM2 is implemented as the official player tracker and as the bounded,
ball-only fallback for TrackNet misses. The production integration is complete
for this phase; this section is not an undecided implementation task.

### Completed integration

- [x] Initialize player masks from detector boxes and propagate them through the
  production player stream.
- [x] Use a separate bounded ball-only SAM2 pass for TrackNet misses.
- [x] Keep the legacy box-tracker path only as comparison/support tooling.

### Follow-up / deferred SAM2 optimization

These are optimization and evaluation tasks, not prerequisites for the next
phase:

- [ ] Measure SAM2 robustness, GPU memory, latency, and cost on representative
  clips.
- [ ] Tune player cadence, checkpoint choice, and fallback budgets only when
  measured evidence justifies a change.
- [ ] Revisit the legacy comparison path only if target-footage evaluation shows
  a clear production benefit.

## Phase 8 — Statistics detection (next session)

Ball detection is complete for this phase. Statistics work starts from the
usable ball signal/output contract provided by `scripts/ball/` and the
production pipeline.

### Task 8.1: Define the statistics contract and validate provenance

- [ ] Define versioned event/schema fields and status vocabulary.
- [ ] Validate player, team, ball, and court provenance, including unknown/lost states.
- [ ] Reference the current ball-signal source without duplicating TrackNet training notes here.

Acceptance criteria:

- The project has a clear local design before copying notebook code.

### Task 8.2: Implement shot event tracker with make/miss

- [ ] Track candidate jump-shot events.
- [ ] Associate ball/rim/player signals over time.
- [ ] Output event records with frame range, player track, team, and explicit make/miss/unknown status.

Acceptance criteria:

- Events are exported as JSON/CSV.
- False positives can be reviewed frame-by-frame.

### Task 8.3: Add possession and evidence-backed passes

- [ ] Add possession only after ball/player evidence is validated.
- [ ] Add passes only after transfer evidence is available; do not use proximity alone.

### Task 8.4: Evaluate and publish statistics artifacts

- [ ] Select small, representative evaluation clips and define practical metrics.
- [ ] Write stable statistics artifacts and reference them from `results.json`.
- [ ] Add acceptance tests for schema, provenance, make/miss, possession, and unknown/rejected events.

Acceptance criteria:

- Each detected shot has an approximate court coordinate.

## Phase 9 — Evaluation and Regression

### Task 9.1: Build a tiny regression clip set

- [ ] Save 3-5 short clips that represent target footage.
- [ ] Include easy, medium, and hard clips.
- [ ] Keep them small enough to run cheaply.

Acceptance criteria:

- Every pipeline change can be checked on the same clips.

### Task 9.2: Add metrics exports

- [ ] For detection: precision/recall/mAP where labels exist.
- [ ] For tracking: ID switches manually first, later HOTA/IDF1/MOTA.
- [ ] For OCR: exact match and per-digit accuracy.
- [ ] For team ID: track-level accuracy.

Acceptance criteria:

- The project can compare model changes without relying only on visual inspection.

### Task 9.3: Add cost and latency logging

- [ ] Count API calls by model.
- [ ] Log runtime per step.
- [ ] Estimate cost per minute of video.

Acceptance criteria:

- Before running a long video, the user can estimate spend and runtime.

## Phase 10 — Deployment Decision

### Task 10.1: Decide hosted vs local inference

- [ ] Compare Roboflow Serverless, local Inference server, and Dedicated GPU.
- [ ] Document cost, latency, and model support.

Acceptance criteria:

- Production deployment path is selected based on measured usage, not guesswork.

### Task 10.2: Split optional heavy dependencies

- [ ] Keep base requirements small.
- [ ] Add optional files:
  - `requirements-tracking.txt`
  - `requirements-sam2.txt`
  - `requirements-vlm.txt`
  - `requirements-dev.txt`

Acceptance criteria:

- The MVP can install without SAM 2 or VLM dependencies.

### Task 10.3: Add reproducible commands

- [ ] Document commands for image, video, dry-run, sampled output, tracking, OCR, and evaluation.
- [ ] Keep command examples in README or docs.

Acceptance criteria:

- A new developer can run the MVP without reading source code.

## Deferred Until Proven Necessary

- [ ] Further SAM2 robustness and cost optimization after statistics needs are clearer.
- [ ] SmolVLM2 in the hot path.
- [ ] Custom RF-DETR fine-tuning.
- [ ] ResNet-32 jersey classifier.
- [ ] Browser/Node.js inference.
- [ ] ONNX/TFLite export.

These are valid directions, but they should wait until the simpler pipeline fails on target footage.

## Next Immediate Task

Statistics detection, beginning with the contract/schema and provenance checks
listed in **Phase 8** above. Court projection is an existing dependency to
validate, not the next standalone implementation task.
