# Future Tasks: Basketball Video Analysis Pipeline

This backlog documents the implementation work needed to evolve the current Roboflow player-detection prototype into a full basketball video-analysis pipeline. The shortest useful path is: stabilize video detection, normalize model outputs with Supervision, use the current SAM2 tracking path, then add court geometry, team identification, jersey OCR, and later heavier event logic or VLM components.

## Current State

| Area | Status |
|---|---|
| Image detection | Implemented with Roboflow Inference and annotated image output. |
| Video detection | Implemented with frame sampling, dry-run cost estimation, sampled-only output, and annotated MP4 output. |
| Cost control | Basic controls exist: `--dry-run`, `--frame-stride`, `--max-frames`, `--sampled-output-only`. |
| Supervision integration | Implemented for image/video annotation with shared detection/keypoint annotation helpers. |
| Tracking | Implemented locally with official SAM2 entrypoint `scripts/tracking/track_players_sam2.py`. The older `scripts/tracking/video_pipeline.py` / `scripts/tracking/track_players_video.py` box-tracker path still exists for comparison and support tooling. |
| Team identification | Implemented experimentally in `pipeline/` and Modal. End-of-run assignment writes `teams.json` and rewrites `tracks.jsonl`. It is still not considered production-stable. |
| Jersey OCR | Not implemented yet. |
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
| 8 | Add make/miss and event logic | Depends on stable tracking, court geometry, and ball/rim signals. |
| 9 | Add evaluation and regression set | Needed before production. |
| 10 | Decide deployment path | Local/serverless/dedicated depends on measured cost and latency. |

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

## Recommended next task

**Recommended next implementation task: Task 3.3 — Evaluate tracker robustness on crossings/occlusions, then decide whether to replace or wrap the current tracker API.**

Why this is next now:

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

## Phase 5 — Team identification

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

## Phase 5 — Team Identification

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

- SigLIP path is optional and disabled by default.
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

## Phase 7 — SAM 2 Tracking Optional Layer

### Task 7.1: Create SAM 2 experiment branch/script

- [ ] Install SAM 2 in a separate optional requirements file.
- [ ] Initialize masks from player detector boxes.
- [ ] Propagate masks through a short clip.

Acceptance criteria:

- SAM 2 runs on a short local clip with documented GPU requirements.
- ByteTrack vs SAM 2 result is visually comparable.

### Task 7.2: Decide SAM 2 deployment path

- [ ] Measure GPU memory, latency, and setup complexity.
- [ ] Decide local GPU, Roboflow Dedicated, or skip.

Acceptance criteria:

- SAM 2 is either accepted with measurable benefit or explicitly deferred.

## Phase 8 — Make/Miss and Event Logic

### Task 8.1: Review Roboflow make/miss notebook behavior

- [ ] Document required model classes and signals.
- [ ] Identify how `BALL_IN_BASKET_CLASS_ID` is used.
- [ ] Identify dependencies on court homography and shot event tracking.

Acceptance criteria:

- The project has a clear local design before copying notebook code.

### Task 8.2: Implement shot event tracker

- [ ] Track candidate jump-shot events.
- [ ] Associate ball/rim/player signals over time.
- [ ] Output event records: frame range, player track, team, make/miss if known.

Acceptance criteria:

- Events are exported as JSON/CSV.
- False positives can be reviewed frame-by-frame.

### Task 8.3: Add shot chart export

- [ ] Use homography to locate shot origin.
- [ ] Render shot chart on top-down court.

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

- [ ] Full SAM 2 tracking in production.
- [ ] SmolVLM2 in the hot path.
- [ ] Custom RF-DETR fine-tuning.
- [ ] ResNet-32 jersey classifier.
- [ ] Browser/Node.js inference.
- [ ] ONNX/TFLite export.

These are valid directions, but they should wait until the simpler pipeline fails on target footage.

## Next Immediate Task

Wire the homography into the tracking pipeline so player positions can be
projected to FIBA court coordinates.

```text
court keypoints -> COURT_POINT_MAP -> cv2.findHomography
tracked player foot points -> cv2.perspectiveTransform -> court coordinates
```

Why this next: the local homography script is in place
(`scripts/court/build_court_homography.py`); the missing piece is using it on every
tracked frame so downstream code can reason in meters instead of pixels. The
mapping is still a best-guess, so any real measurement on the target court
should land in `COURT_POINT_MAP` before relying on shot charts or spatial
stats.
