# Future Tasks: Basketball Video Analysis Pipeline

This backlog documents the implementation work needed to evolve the current Roboflow player-detection prototype into a full basketball video-analysis pipeline. The shortest useful path is: stabilize video detection, normalize model outputs with Supervision, add tracking, then add court geometry, team identification, jersey OCR, and only later SAM 2 / VLM-heavy components.

## Current State

| Area | Status |
|---|---|
| Image detection | Implemented with Roboflow Inference and annotated image output. |
| Video detection | Implemented with frame sampling, dry-run cost estimation, sampled-only output, and annotated MP4 output. |
| Cost control | Basic controls exist: `--dry-run`, `--frame-stride`, `--max-frames`, `--sampled-output-only`. |
| Supervision integration | Not implemented yet. Current scripts draw manually. |
| Tracking | Not implemented yet. |
| Team identification | Not implemented yet. |
| Jersey OCR | Not implemented yet. |
| Court keypoints / homography | Not implemented yet. |
| SAM 2 tracking | Not implemented yet. Optional, not MVP-first. |

## Implementation Order

| Phase | Goal | Why now |
|---|---|---|
| 1 | Stabilize detection outputs | Everything depends on trustworthy detections. |
| 2 | Convert to Supervision | Avoid custom drawing/tracking glue. |
| 3 | Add ByteTrack | Cheap tracking before expensive SAM 2. |
| 4 | Add court keypoints and homography | Needed for positions, shot charts, and spatial stats. |
| 5 | Add team identification | Needed to separate possessions and stats. |
| 6 | Add jersey OCR | Useful but noisy; depends on good crops/tracks. |
| 7 | Add SAM 2 only if needed | Expensive; use when ByteTrack fails. |
| 8 | Add make/miss and event logic | Depends on stable tracking, court geometry, and ball/rim signals. |
| 9 | Add evaluation and regression set | Needed before production. |
| 10 | Decide deployment path | Local/serverless/dedicated depends on measured cost and latency. |

## Phase 1 — Detection Foundation

### Task 1.1: Centralize model configuration

- [ ] Move model IDs into a small config module or `.env` variables.
- [ ] Support at least:
  - `ROBOFLOW_API_KEY`
  - `PLAYER_MODEL_ID`
  - `COURT_MODEL_ID`
  - `JERSEY_OCR_MODEL_ID`
- [ ] Keep defaults aligned with the currently working player model.

Acceptance criteria:

- Running image/video detection still works without code edits.
- Changing model version does not require editing script internals.

### Task 1.2: Save raw inference outputs

- [ ] Add optional `--save-json` to image and video scripts.
- [ ] For video, save one JSON record per inferred frame.
- [ ] Include frame index, timestamp, model ID, and predictions.

Acceptance criteria:

- Detection outputs can be inspected without re-calling Roboflow.
- Failed/interesting frames can be replayed offline.

### Task 1.3: Add frame extraction utility

- [ ] Add a script to extract sampled frames from a video.
- [ ] Support `--frame-stride`, `--max-frames`, and `--output-dir`.
- [ ] Do not call Roboflow from this script.

Acceptance criteria:

- A short target clip can be converted into reviewable frame images.
- The user can manually inspect whether the sample is representative.

## Phase 2 — Supervision Integration

### Task 2.1: Convert Roboflow outputs to `sv.Detections`

- [ ] Add `supervision` to pinned dependencies.
- [ ] Convert `InferenceHTTPClient` results with `sv.Detections.from_inference(result)`.
- [ ] Replace manual detection filtering with Supervision filtering.

Acceptance criteria:

- Image script can annotate using `sv.BoxAnnotator` and `sv.LabelAnnotator`.
- Video script no longer needs custom box drawing logic.

### Task 2.2: Create annotation helpers

- [ ] Create reusable helpers for labels, colors, confidence thresholds, and class filtering.
- [ ] Keep helpers model-agnostic.

Acceptance criteria:

- Image and video scripts use the same annotation code.
- Adding court/player/jersey annotations does not duplicate drawing logic.

## Phase 3 — Cheap Tracking First

### Task 3.1: Add ByteTrack tracking

- [ ] Use `sv.ByteTrack` or the current recommended Supervision tracker API.
- [ ] Track only detections from frames actually inferred.
- [ ] Output track IDs in labels.

Acceptance criteria:

- Sampled video output shows stable track IDs on simple clips.
- Track IDs are saved in JSON/CSV for later analysis.

### Task 3.2: Add track-level smoothing

- [ ] Store detections per track.
- [ ] Smooth boxes/labels over short windows.
- [ ] Avoid drawing stale boxes by default unless explicitly requested.

Acceptance criteria:

- Boxes no longer appear visually misleading between sampled frames.
- Debug mode can show only sampled frames; tracking mode can show full video.

### Task 3.3: Compare ByteTrack vs SAM 2 need

- [ ] Create a small evaluation clip set with occlusions and camera motion.
- [ ] Count obvious ID switches manually.
- [ ] Decide whether SAM 2 is justified.

Acceptance criteria:

- SAM 2 is not added unless ByteTrack failure is proven on target footage.

## Phase 4 — Court Keypoints and Homography

### Task 4.1: Integrate court keypoint model

- [ ] Add model ID for `basketball-court-detection-2`.
- [ ] Run court inference on sampled frames.
- [ ] Save keypoints and confidence values.

Acceptance criteria:

- Court keypoints are visible on annotated debug frames.
- The script reports when not enough keypoints are detected.

### Task 4.2: Compute homography

- [ ] Map detected court keypoints to canonical court coordinates.
- [ ] Use OpenCV `findHomography` with RANSAC.
- [ ] Project player foot points onto court coordinates.

Acceptance criteria:

- Player positions can be exported as court coordinates.
- Bad homography frames are rejected instead of silently producing wrong stats.

### Task 4.3: Generate shot/location debug view

- [ ] Create a top-down court visualization.
- [ ] Plot tracked player positions.
- [ ] Save a PNG or simple MP4 overlay.

Acceptance criteria:

- A reviewer can visually confirm that court projection is plausible.

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

Implement Supervision conversion and annotation:

```text
Roboflow result -> sv.Detections -> BoxAnnotator/LabelAnnotator -> video output
```

Why this next: it deletes custom annotation code and prepares the project for ByteTrack, filtering, metrics, and dataset tooling.
