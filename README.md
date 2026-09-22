# computer-vision-pipeline

GPU-first basketball pipeline for Roboflow player detection, SAM2 tracking, end-of-run team identification, and conservative jersey-number confirmation.

## Current pipeline

The current production path is `pipeline/` plus `modal_app.py`.

High-level flow:

1. Detect players on the first frame of each segment.
2. Match detections back to active tracks and propagate them with SAM2.
3. Collect team evidence from the detector frame and from selected propagated frames.
4. Cluster accepted team samples at the end of the run and vote one `team_id` per track.
5. Collect detector-frame jersey-number evidence from the `number` class, run hosted OCR on selected number crops, aggregate final per-track numbers, then rewrite `tracks.jsonl`, write `teams.json`, `numbers.json`, and optionally render `annotated.mp4`.

`pipeline/` contains no Modal imports, so the same code runs from local CLI, tests, notebooks, or Modal.

## SAM3.1 V2 boundary

`modal_app_v2.py` keeps its SAM3.1 observation artifact at the root of the returned bundle (`tracks.jsonl` and `manifest.json`). It then delegates the same input video to `pipeline.runner.run_pipeline` in `stats/`, so the returned statistics artifacts are produced by the existing orchestration rather than translated from V2 rows.

`stats/results.json`, `stats/tracks.jsonl`, `stats/teams.json`, `stats/numbers.json`, `stats/shot_events.jsonl`, and `stats/possession_events.jsonl` therefore use the legacy contract and are referenced in the V2 manifest with relative paths, sizes, and hashes. This performs a second, independent stats run; it does not claim that SAM3.1 rows are already legacy stats rows. The stats pass uses its own `PipelineConfig`/environment scope rather than V2's `frame_start`/`frame_end`, so configure its frame limits explicitly when cost matters. The V2 image consequently also requires the legacy SAM2/Roboflow runtime and `ROBOFLOW_API_KEY`.

Render possession from a saved V2 bundle with the root SAM3.1 video overlay and the legacy stats events/tracks:

```bash
./.venv/bin/python scripts/tracking/debug_v2.py \
  --video-path path/to/input.mp4 \
  --artifact-dir outputs/v2-bundle \
  --stats-dir outputs/v2-bundle/stats \
  --output outputs/v2-bundle/possession-debug.mp4
```

The renderer matches possession events by exact `frame_index`. It displays only the event's authoritative `stable_player_id`, `team_id`, or explicit `UNKNOWN`/`LOST`/`UNASSIGNED` status; it never infers a holder.

## Phase status

- **Completed for this phase:** production ball detection and production team identification.
- **Current slice:** ball-track validation, shot-to-ball evidence, and conservative possession events are available offline. Passes remain deferred; missing evidence stays explicit.

## Dedicated player-position pipeline

There is now a separate homography pipeline for movement data: `python -m pipeline.player_positions` locally, or `modal run modal_app.py --pipeline player-positions ...` on Modal.

Quick path:

```bash
python -m pipeline.player_positions path/to/clip.mp4 \
  --output-dir outputs/clip-player-positions
```

Modal path with the existing entrypoint:

```bash
modal run modal_app.py \
  --pipeline player-positions \
  --video-path path/to/clip.mp4 \
  --output-dir outputs/clip-player-positions
```

What it does:

1. Runs a player-only tracking pass separate from the stats/shot pipeline.
2. Samples the video at 5 FPS and keeps stable `player_id` values across emitted frames.
3. Uses a stricter default player detector confidence (`0.8`) than the stats pipeline, while still allowing `--confidence` overrides from the dedicated CLI and Modal entrypoint.
4. Writes a machine-first JSONL plus, when rows are emitted, a top-down debug video for movement review.

Artifacts written to `--output-dir`:

| File | Meaning |
| --- | --- |
| `player_positions.jsonl` | One row per player per emitted frame for downstream replay or visualization tooling. |
| `player_positions_debug.mp4` | Optional top-down court video with per-player movement trails for quick visual debugging; omitted when no player-position rows are emitted. |

`player_positions.jsonl` rows include at least:

- `frame_index`
- `timestamp_seconds`
- `player_id`
- `team_id` when the end-of-run team pass can resolve it
- `bbox_xyxy`
- `image_anchor_xy`
- `court_xy_m`
- `confidence`
- homography debug fields such as `homography_frame_status`, `homography_inliers_count`, and `homography_reprojection_mean_m`

Important scope notes:

- This path is players only for now.
- It intentionally does not run shot-event logic.
- Team IDs remain best-effort because they still depend on the end-of-run clustering pass.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Recommended run path

The current production-style path is Modal through `modal_app.py`.

Estimate cost before spending anything:

```bash
python -m pipeline.cli path/to/clip.mp4 --dry-run
```

Run the current pipeline on Modal and save artifacts locally:

```bash
modal run modal_app.py \
  --video-path path/to/clip.mp4 \
  --output-dir outputs/check \
  --annotated
```

For local CLI runs, TrackNet is disabled by default (`TRACKNET_ENABLED=0` in
`PipelineConfig`); enable it explicitly with the checkpoint environment
variables. Modal production runs enable TrackNet by default. To use the hosted
ball-detection compatibility path without loading the TrackNet checkpoint,
pass `--no-tracknet-enabled` to `modal run`, or send
`"tracknet_enabled": false` to the HTTP submit endpoint.

Run the dedicated player-position pipeline on Modal and save artifacts locally:

```bash
modal run modal_app.py \
  --pipeline player-positions \
  --video-path path/to/clip.mp4 \
  --output-dir outputs/clip-player-positions
```

Enable hosted jersey OCR on the same run:

```bash
modal run modal_app.py \
  --video-path media/game.mp4 \
  --output-dir outputs/ocr-check \
  --annotated \
  --number-predictor-backend roboflow-hosted
```

Reproduce an older short comparison run with bounded cost:

```bash
modal run modal_app.py \
  --video-path media/game.mp4 \
  --output-dir outputs/check-repro \
  --annotated \
  --max-frames 100 \
  --frame-stride 10 \
  --min-team-samples-per-track 2
```

Expected local artifacts in `--output-dir`:

- `results.json`
- `tracks.jsonl`
- `teams.json`
- `numbers.json`
- `shot_events.jsonl`
- `shot_evidence.jsonl`
- `ball_tracks.jsonl` when TrackNet or the ball fallback emits ball rows
- `possession_events.jsonl` with one shared player-frame record per evaluated frame
- `annotated.mp4` when `--annotated` is set

### Possession contract

`possession_events.jsonl` is schema version 1. A `confirmed` record requires a
valid ball center inside or within 8 pixels of exactly one authoritative player
box for 0.5 seconds, with a valid player track identity. `team_id` is copied
only when row and final team evidence agree; otherwise it is `null`.

Statuses are `confirmed`, `unknown`, `lost`, and `unassigned`:

- `confirmed`: the temporal gate passed.
- `unknown`: ball geometry/state or player matching is ambiguous, invalid, or still pending confirmation.
- `lost`: the exact shared frame has missing or authoritative lost ball evidence.
- `unassigned`: one player matched, but its identity is missing or unresolved.

Gaps reset confirmation. The stage never interpolates missing ball rows, infers
possession from proximity when candidates are tied, or emits passes yet.

Expected local artifacts for `--pipeline player-positions`:

- `player_positions.jsonl`
- `player_positions_debug.mp4` when player-position rows were emitted

See `SETUP.md` for secrets, deployment, and the configuration surface.

## Configuration

`PipelineConfig` in `pipeline/config.py` is the single source of truth for the deployed pipeline.

Configuration sources, from broadest to narrowest:

1. Environment variables via `PipelineConfig.from_env()`.
2. Explicit local CLI overrides in `pipeline.cli`.
3. Explicit Modal/local-entrypoint overrides in `modal_app.py`.

Current environment-backed knobs:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROBOFLOW_API_KEY` | required | Roboflow private API key. |
| `ROBOFLOW_API_URL` | `https://serverless.roboflow.com` | Hosted inference endpoint. |
| `PLAYER_MODEL_ID` | `basketball-player-detection-3-ycjdo/16` | Player detector model. |
| `SHOT_TARGET_FPS` | unset | Optional shot-pass FPS target. When unset, the shot pass reuses tracking downsampling. |
| `SAM2_MODEL_ID` | `facebook/sam2-hiera-small` | SAM2 checkpoint. |
| `TEAM_EMBEDDER_BACKEND` | `siglip` | FashionSigLIP by default; `colour` or `none` remain supported overrides. |
| `TEAM_EMBEDDER_MODEL_ID` | `Marqo/marqo-fashionSigLIP` | FashionSigLIP model id when the backend is `siglip`. |
| `TEAM_COUNT` | `2` | Cluster count for team assignment. |
| `MIN_TEAM_SAMPLES_PER_TRACK` | `3` | Minimum selected samples before a track can be assigned. |
| `TEAM_SAMPLE_STRIDE` | `5` | Propagated-frame sampling interval for team evidence. Detector frames are always sampled. |
| `MIN_TEAM_SAMPLE_QUALITY` | `0.0` | Minimum quality score for a sample to count. |
| `MAX_TEAM_SAMPLES_PER_TRACK` | `12` | Maximum selected samples kept per track. |
| `TEAM_ASSIGNMENT_MARGIN` | `0.03` | Minimum winning vote margin before a track is assigned. |
| `TEAM_DEBUG_ENABLED` | `true` | Write team-debug crops, masks, and per-sample metadata. |
| `NUMBER_PREDICTOR_BACKEND` | `dummy` | `dummy` or `roboflow-hosted` jersey-number predictor. |
| `NUMBER_PREDICTOR_ROBOFLOW_MODEL_ID` | `basketball-jersey-numbers-ocr/6` | Hosted Roboflow OCR model id for jersey numbers. |
| `NUMBER_PREDICTOR_SCORE_THRESHOLD` | `0.7` | Minimum hosted OCR score before a read can count. |

The local CLI exposes the common run knobs:

- `--frame-stride`
- `--max-frames`
- `--confidence`
- `--target-fps`
- `--sam2-model-id`
- `--team-embedder-backend`
- `--team-embedder-model-id`
- `--team-count`
- `--min-team-samples-per-track`
- `--team-assignment-margin`
- `--annotated-video`

The Modal entrypoint exposes the full current override surface, including:

- `--pipeline` (`stats` or `player-positions`)
- `--output-dir`
- `--annotated`
- `--frame-stride`
- `--max-frames`
- `--target-fps`
- `--team-sample-stride`
- `--min-team-samples-per-track`
- `--min-team-sample-quality`
- `--max-team-samples-per-track`
- `--team-debug-enabled`
- `--number-predictor-backend`

## Team identification

Current behavior is simpler than the earlier design notes in `future_tasks/`.

What the code does today:

1. Extract a jersey-focused crop per accepted sample.
2. Build either a colour histogram, a FashionSigLIP embedding, or both depending on backend.
3. Reject unusable samples, keep the best `max_team_samples_per_track` samples per track, then cluster all selected samples globally.
4. Vote one label per track and leave the track unassigned when it is under-sampled, tied, or too close to the runner-up cluster.

FashionSigLIP is the production default backbone. The downstream global clustering, per-track voting, under-sampled/ambiguous handling, and unknown logic are unchanged. An explicit legacy timm model ID remains supported as an override; model IDs containing `/` use the OpenCLIP/Hugging Face FashionSigLIP path, while legacy timm IDs use the timm path.

There is no separate "colour-first then SigLIP refinement" stage in the current code. With the default backend, clustering uses the FashionSigLIP embedding when present and falls back to the colour signature only when an embedding is missing.

Current outputs from one pipeline run:

| File | Meaning |
| --- | --- |
| `results.json` | Run summary plus the resolved config used for the run, including output file pointers such as `teams_file`, `numbers_file`, and `shot_events_file`. |
| `tracks.jsonl` | One row per track per emitted frame, rewritten at the end with final `team_id`s. |
| `teams.json` | Team assignments, skipped/unassigned tracks, selected-sample counts, and per-track diagnostics. |
| `numbers.json` | Final per-track jersey-number confirmations and ambiguity details. |
| `shot_events.jsonl` | Detector-frame make/miss event records from the active pipeline path. Empty means no shot events were resolved for that clip. |
| `ball_tracks.jsonl` | One ball record per decoded source frame in the dense TrackNet stream, in source-frame order. Rows include `timestamp_seconds`; TrackNet detections are preferred, bounded SAM2 fallback rows use `track_state: "fallback"`, and an exhausted miss emits `track_state: "lost"` with null geometry. |
| `results.json.ball_provenance` | Backward-compatible validation metadata: backend, state counts, configured/missing coverage, lost coverage, and track-ID diagnostics. |
| `team_debug.json` | Per-track sample summaries when team debug is enabled. |
| `team_debug/` | Crops and masks for manual team-ID inspection when team debug is enabled. |

| `annotated.mp4` | Optional final render after team assignment/number aggregation, with confirmed jersey numbers when available. |

`shot_events.jsonl` keeps its legacy schema. The additive `shot_evidence.jsonl`
joins each event to ball evidence for its frame window and includes a
`player_identity` object. A player is assigned only when validated ball evidence
is observed and the shot anchor lies inside exactly one consistent player/stable
identity's track geometry in the event window. Team IDs are copied only from a
non-conflicting existing assignment; proximity and ball-only evidence never
assign a player or team.

### Offline team-backbone A/B evaluation

The production run's `team_debug.json` and `team_debug/` directory are a
read-only evaluation input. They contain the already-selected torso crops, so
the offline pass does not call Roboflow, SAM2, or Modal and does not change
production configuration.

Validate an artifact without loading models:

```bash
python scripts/team/evaluate_backbones.py outputs/check outputs/team-ab.json --dry-run
```

Install the normal production requirements and the optional A/B baseline
dependency before a non-dry-run evaluation:

```bash
python -m pip install -r requirements.txt -r requirements-team-ab.txt
```

Dry-run validation does not require `timm`, `transformers`, or model
downloads. The normal requirements provide the production FashionSigLIP
runtime; `requirements-team-ab.txt` adds the legacy timm baseline and keeps
the same pinned torch/torchvision, Transformers, and Hugging Face versions for
a coherent evaluation environment. This is a heavy CUDA-capable environment.

Run the current SigLIP baseline against FashionSigLIP (model downloads and
inference happen only when this command is explicitly run):

```bash
python scripts/team/evaluate_backbones.py outputs/check outputs/team-ab.json
```

`labels.json` is optional. Add `--labels labels.json` only when a JSON object
such as `{"12": "home", "19": "away"}` is available. Override either model explicitly with `--baseline-model-id` or
`--candidate-model-id`; the default candidate is `Marqo/marqo-fashionSigLIP`.
The report contains both assignments, coverage/unknown/ambiguous rates,
cluster sizes, disagreements, and label metrics when labels are supplied.

## Next session: statistics detection

Implement this work in dependency order, without adding a broad analytics framework:

1. Define the statistics contract and versioned event/schema fields.
2. Validate provenance for player tracks, team IDs, ball rows, and court coordinates, including unknown/lost states.
3. Implement shot/event detection with explicit `made` and `missed` outcomes and reviewable unknowns.
4. Add possession only after ball/player evidence is reliable enough to support it.
5. Add passes only after possession and transfer evidence are validated; do not infer passes from proximity alone.
6. Build a small evaluation clip set and metrics for detection quality, event outcomes, and provenance coverage.
7. Write stable output artifacts and document their ownership in `results.json`.
8. Add focused acceptance tests for schema validity, missing evidence, make/miss, possession, and rejected/unknown events.

## Shot events

The active `pipeline/` path now writes a minimal make/miss artifact.

What it does today:

1. Read detector-frame action/state classes from the player model. Optionally downsample the shot-only pass independently with `SHOT_TARGET_FPS`.
2. Start a candidate event when `jump_shot` or `layup_dunk` appears.
3. Mark the event as `made` when `ball_in_basket` appears during that window.
4. Mark the event as `missed` when the shot signal times out.
5. Mark the event as `unknown` at end-of-clip if the run ends before resolution.
6. Write conservative player/team attribution in the additive `shot_evidence.jsonl`
   artifact without changing `shot_events.jsonl`.

Current limits:

- `shot_events.jsonl` now keeps the best available shot-anchor point (`start_image_xy` / `start_shot_xyxy`) from the detector frame when the model returned a shot box.
- When court homography is available during the main pipeline run, `results.json` also exposes `projected_tracks_file`, `projected_shots_file`, and `projected_shots_image_file`.
- If no frame yields a usable homography, those projected-artifact keys are omitted and no projected-shot artifacts are written.
- The output can be empty for clips where the detector never emits the required action classes.
- Attribution remains unknown when ball evidence is missing/lost, multiple player
  boxes contain the anchor, stable identity evidence conflicts, or team evidence
  is absent/conflicting. Possession and pass attribution are intentionally still
  deferred.

## Jersey numbers

The pipeline now has a working jersey-number path.

What it does today:

1. On each detector frame, keep `player` detections and collect overlapping `number` detections as number evidence.
2. Crop the matched `number` box and send it to the hosted Roboflow OCR model when `NUMBER_PREDICTOR_BACKEND=roboflow-hosted`.
3. Store every OCR attempt in `number_predictions.json` and `number_prediction_cases.jsonl`.
4. Aggregate accepted reads per track at the end of the run and write confirmed/ambiguous results to `numbers.json`.
5. When an annotated video is requested, show confirmed jersey numbers in labels. If no number is confirmed, the overlay stays generic instead of falling back to internal track IDs.

Confirmation is intentionally conservative:

- A track is `confirmed` only when one accepted number wins cleanly.
- Conflicting accepted reads keep the track `ambiguous` instead of guessing.
- One accepted valid read can still confirm a number when there are no competing accepted reads for that track.
- Non-digit, low-score, or unsupported-length OCR reads are ignored.

Current number artifacts from one pipeline run:

| File | Meaning |
| --- | --- |
| `number_predictions.json` | All per-sample OCR outputs in one JSON payload. |
| `number_prediction_cases.jsonl` | Same OCR cases as JSONL for grep/review workflows. |
| `numbers.json` | Final per-track number status with `confirmed`, `ambiguous`, or `unknown`, plus support/conflict details. |

`results.json` now includes `numbers_file` and `shot_events_file`, and it is written only at the true end of the run after track, team, number, shot-event, and optional video outputs finish.

Current limits:

- OCR is detector-frame only; propagated frames are not sampled for number evidence.
- Partial digits and short noisy reads can still slip through as `unknown` or remain unresolved.
- There is no stronger temporal validation yet beyond per-track acceptance/conflict aggregation.
- The pipeline prefers leaving a track unresolved over forcing a wrong jersey number.

Tests that verify the config and team-ID plumbing without a GPU:

```bash
python -m pytest tests/test_config.py tests/test_team_id.py tests/test_number_evidence.py tests/test_shot_events.py
```

Read `future_tasks/TRACKING_ISSUES.md` before tuning tracking. Read `future_tasks/team-identification-architecture.md` before changing the team-ID logic.

## Local Scripts

These scripts are local pipeline steps first. The local CLI path is ahead of the API path.

Implementations now live under:

- `scripts/tracking/` for player detection, tracking, frame extraction, and tracking evaluation.
- `scripts/court/` for court-keypoint, homography, and projected-track tooling.
- `scripts/ball/` for ball-model experiments, training history, and ball-specific docs; the production TrackNet+SAM2 integration is documented above.
- `scripts/shared/` for shared config and annotation helpers.

Keep ball work separate from the player/team/OCR planning docs unless you are documenting an integration boundary.

## Script Roles

This repo currently has one official local tracking entrypoint plus a set of
focused support tools.

### Main entrypoints

Use these first.

| Script | Role | When to use it |
| --- | --- | --- |
| `scripts/tracking/track_players_sam2.py` | Official local SAM2 tracker | Run the detector-seeded SAM2 tracking pipeline with annotated video, track JSONL, raw inference JSONL, and the end-of-run team/number artifacts. |
| `scripts/tracking/video_pipeline.py` | ByteTrack/BoxMOT video pipeline | Run the older box-tracker pipeline when you need the existing ByteTrack, Deep OC-SORT, or BoT-SORT path. |
| `scripts/tracking/detect_players.py` | Main single-image detector | Run player detection on one local image/frame. |

### Local support tools

These are useful, but they are support/debug tools rather than the primary
workflow.

| Script | Role | When to use it |
| --- | --- | --- |
| `scripts/tracking/extract_video_frames.py` | Offline frame extraction | Sample a local video into JPG frames without calling Roboflow. |
| `scripts/tracking/detect_players_video.py` | Detection-only video helper | Inspect or run sampled video detection without tracking; also provides reusable video helpers. |
| `scripts/tracking/track_players_video.py` | Tracking core | Tracker adapter point used by `video_pipeline.py`; keeps ByteTrack and Deep OC-SORT backend logic in one place. |
| `scripts/tracking/track_players_sam2_experiment.py` | SAM2 implementation module | Keeps the current SAM2 CLI implementation that the stable wrapper calls. |
| `scripts/tracking/evaluate_tracking.py` | Baseline tracking review | Turn manual review events into baseline counts for ID switches, full losses, and successful recoveries. |
| `scripts/tracking/summarize_tracking_run.py` | Automatic track-file summary | Print technical summary metrics that are directly derivable from one `tracks.jsonl` file. |
| `scripts/court/detect_court_keypoints.py` | Court keypoints on one image | Test court landmarks on a single frame. |
| `scripts/court/detect_court_keypoints_video.py` | Court keypoints on sampled video | Evaluate court keypoint behavior across a clip. |
| `scripts/court/build_court_homography.py` | Homography from saved keypoints | Turn detected court keypoints into a FIBA homography and debug reprojection quality. |
| `scripts/court/build_court_homography_video.py` | Video homography debug | Build a side-by-side debug video for court projection quality checks. |
| `scripts/court/visualize_projected_tracks.py` | Top-down projected-track debug | Render projected player positions from JSONL onto a FIBA court as a PNG or simple MP4. |

### Shared infrastructure

| Script | Role |
| --- | --- |
| `scripts/shared/env_config.py` | Shared `.env`-backed configuration for model IDs, API key, and Roboflow endpoint. |

### Tracking evaluation baseline

Use `scripts/tracking/evaluate_tracking.py` to summarize manual review events against the current `tracks.jsonl` workflow.

Manual review input is one JSON object per line. Keep it small and explicit:

```json
{"clip":"crossing_clip_01","frame_index":120,"event":"id_switch","track_id":7,"notes":"blue guard swaps with wing after overlap"}
{"clip":"crossing_clip_01","frame_index":188,"event":"full_loss","track_id":12,"notes":"player disappears behind screener and returns with new ID"}
{"clip":"crossing_clip_01","frame_index":214,"event":"successful_recovery","track_id":5,"notes":"same player returns with original ID"}
```

Run it like this:

```bash
python3 scripts/tracking/evaluate_tracking.py review/tracking_events.jsonl --tracks-jsonl outputs/game_tracks.jsonl
```

The report prints the three baseline counts plus a simple occlusion recovery rate.

For an automatic summary from the track file alone:

```bash
python3 scripts/tracking/summarize_tracking_run.py outputs/game_tracks.jsonl
```

### Current direction

- Treat `scripts/tracking/track_players_sam2.py` as the canonical local tracking path.
- Keep court/homography tools integrated only where they already help the local pipeline (`--court-keypoints` plus projected-track JSONL), and keep the heavier debug/evaluation scripts separate.
- Add new stages to the pipeline by reusing these support tools first, then
  folding only the stable pieces into the main entrypoint.

## Current Status

Implemented now:

- Shared annotation helpers now cover player detection, tracking, and court keypoint visualization.
- `scripts/tracking/track_players_sam2.py` is now the official local tracker path for player tracking with annotated video, track JSONL, raw inference JSONL, and end-of-run team/number outputs.
- `scripts/tracking/video_pipeline.py` still exists for the older ByteTrack/BoxMOT path plus optional court keypoints and projected-track JSONL export.
- Homography fitting now includes geometric pre-validation before `cv2.findHomography`, so obviously degenerate keypoint layouts are rejected early.
- Projected-track debugging is available through `scripts/court/visualize_projected_tracks.py`.
- Tracking keeps short-lived predicted tracks for brief missed sampled detections.
- A lightweight torso-color appearance re-ID baseline can reassign some recently lost tracks.
- The main tracking pipeline can temporarily lower frame stride around close-player or overlapping-player interactions, with tuning now stored in `config/adaptive_frame_stride.toml`.

Known limitations:

- Player IDs still switch during crossings and occlusions.
- Some crossing/occlusion cases still lose tracking entirely.
- `--frame-stride 1` is slower and more expensive, and it still does not solve identity consistency by itself.
- Adaptive stride only reacts on sampled frames, so it can still miss risk that starts and ends entirely between two sampled frames.
- Current appearance re-ID is a cheap baseline, not a robust identity solution.
- The current `sv.ByteTrack()` usage should be revisited because the tracker API now emits a deprecation warning.
- Deep OC-SORT and BoT-SORT now route through BoxMOT and add a ReID-model dependency plus runtime cost compared with ByteTrack.

See `TRACKING_ISSUES.md` for the short list of unresolved tracking failures and recommended next actions.

## Integration roadmap

This is the intended path so future work stays understandable.

### 1. Keep as the main pipeline

These belong in `scripts/tracking/video_pipeline.py` because they are part of the core
"process one local video and inspect the result" workflow.

| Stage | Status | Notes |
| --- | --- | --- |
| Player detection | Integrated | Current base stage. |
| Tracker backend (`--tracker-backend`) | Legacy integrated path | `scripts/tracking/video_pipeline.py` still routes tracking through `scripts/tracking/track_players_video.py` for ByteTrack, Deep OC-SORT, and BoT-SORT, but SAM2 is now the official tracker path. |
| Raw inference JSONL | Integrated | One record per inferred frame. |
| Court keypoints | Partially integrated | Available through `--court-keypoints` in the local video pipeline, with separate debug scripts still retained. |
| Court homography / player projection | Partially integrated | Projected-track JSONL export exists in the local pipeline; broader downstream use still depends on trustworthy keypoints and mapping. |
| Team identification | Integrated | End-of-run global clustering plus per-track voting is active in the current pipeline. |
| Jersey OCR | Integrated | Detector-frame number evidence, hosted OCR prediction, and end-of-run per-track aggregation are active in the current pipeline. |
| Event logic / make-miss | Future | Depends on stable tracking + court geometry. |

### 2. Keep as external tools for now

These remain outside the main pipeline until their outputs are trustworthy and
their role is clear.

| Script | Why it stays external for now |
| --- | --- |
| `scripts/tracking/extract_video_frames.py` | Operator tooling for offline inspection, not part of inference/render. |
| `scripts/tracking/detect_players_video.py` | Detection-only helper and source of reusable video utilities. |
| `scripts/tracking/track_players_video.py` | Processing core used by the main pipeline, but still useful as a focused debug tool. |
| `scripts/court/detect_court_keypoints.py` | Single-frame evaluation tool for court landmarks. |
| `scripts/court/detect_court_keypoints_video.py` | Sampled-video evaluation tool for court landmarks. |
| `scripts/court/build_court_homography.py` | Local geometry/debug tool for checking court mapping quality. |
| `scripts/court/build_court_homography_video.py` | Side-by-side projection debug tool before main-pipeline integration. |

### 3. Candidates to merge or retire later

- `scripts/tracking/detect_players_video.py`: keep while it provides reusable helpers;
  retire it as a user-facing entrypoint once `scripts/tracking/video_pipeline.py` fully replaces
  it for normal video work.
- `scripts/tracking/track_players_video.py`: keep as the implementation core unless it
  becomes thin enough to fold directly into `scripts/tracking/video_pipeline.py` without making
  that file ugly.
- `scripts/tracking/track_players_sam2_experiment.py`: keep as the implementation module behind `scripts/tracking/track_players_sam2.py` unless it gets folded into the stable wrapper later.
- Court and homography scripts: keep separate until the integration path is
  proven; do not merge them early just to reduce file count.

### Rule of thumb

- If a script is used to process the normal local tracking workflow, it belongs in
  or behind `scripts/tracking/track_players_sam2.py`.
- If a script exists to inspect, validate, calibrate, or debug one stage, keep
  it as a support tool until the stage is stable.

Run the official SAM2 tracker, no Roboflow call:

```bash
python3 scripts/tracking/track_players_sam2.py media/game.mp4 --dry-run --frame-stride 15 --max-frames 300
```

Run the official SAM2 tracker:

```bash
python3 scripts/tracking/track_players_sam2.py media/game.mp4 --frame-stride 15 --max-frames 300
```

Install the full SAM2 tracker stack first:

```bash
python3 -m pip install -r requirements.txt
# install torch/torchvision from pytorch.org for your machine first
python3 -m pip install -r requirements-sam2.txt
```

Use the older box-tracker pipeline when you specifically want ByteTrack, Deep OC-SORT, or BoT-SORT:

```bash
python3 scripts/tracking/video_pipeline.py media/game.mp4 --frame-stride 30 --max-frames 300
```

Run the same pipeline with Deep OC-SORT through BoxMOT:

```bash
python3 scripts/tracking/video_pipeline.py media/game.mp4 \
  --tracker-backend deepocsort \
  --frame-stride 30 \
  --max-frames 300
```

Run the same pipeline with BoT-SORT through BoxMOT:

```bash
python3 scripts/tracking/video_pipeline.py media/game.mp4 \
  --tracker-backend botsort \
  --frame-stride 30 \
  --max-frames 300
```

Run the pipeline after editing the adaptive frame-stride file:

```bash
python3 scripts/tracking/video_pipeline.py media/game.mp4 \
  --frame-stride 30
```

Adaptive frame-stride workflow:

1. Edit `config/adaptive_frame_stride.toml`.
2. Set `enabled = true` when you want the adaptive mode on.
3. Re-run the same pipeline command.
4. Use `--adaptive-*` flags only when you want a one-run override without editing the file.

```bash
python3 scripts/tracking/video_pipeline.py media/game.mp4 \
  --frame-stride 30 \
  --max-frames 300
```

Adaptive stride quick path:

| Knob | What it does |
| --- | --- |
| `enabled` | Turns the feature on. `false` keeps the old fixed-stride behavior. |
| `risk_frame_stride` | Lower stride used after a risky sampled frame. |
| `risk_overlap_fraction` | Triggers risk when two player boxes overlap enough relative to the smaller box. |
| `risk_proximity_heights` | Triggers risk when two player centers are very close relative to player size. |
| `risk_hold_frames` | Keeps the lower stride on briefly so the pipeline does not flap every sampled frame. |

The default dry run still reports the base stride estimate. When adaptive mode is enabled in the file or via a one-run override, it also prints an adaptive call ceiling using the lower risk stride.

One-run override example:

```bash
python3 scripts/tracking/video_pipeline.py media/game.mp4 \
  --frame-stride 30 \
  --adaptive-frame-stride \
  --adaptive-risk-frame-stride 5
```

Deep OC-SORT notes:

- Install `boxmot==22.0.0` from `requirements.txt` before using `--tracker-backend deepocsort` or `--tracker-backend botsort`.
- Imports stay lazy, so ByteTrack users do not pay the BoxMOT import/runtime cost unless they select a BoxMOT backend.
- Edit `config/deepocsort.toml` first. That file is now the main place to tune Deep OC-SORT behavior.
- Edit `config/botsort.toml` first when you want to tune BoT-SORT behavior.
- The repo now defaults Deep OC-SORT and BoT-SORT to `osnet_x1_0_msmt17.pt`, matching the shipped BoxMOT config files.
- BoT-SORT currently reuses the same BoxMOT runtime override flags (`--deepocsort-reid-weights`, `--deepocsort-device`, `--deepocsort-half`) to avoid expanding the CLI surface.
- Edit `config/reid.toml` when you want to tune the lightweight torso-crop appearance path used for recent-track reassignment.
- `--tracker-config` defaults to `config/deepocsort.toml` for `deepocsort` and `config/botsort.toml` for `botsort`.
- `--reid-config` defaults to `config/reid.toml`.
- `--deepocsort-reid-weights`, `--deepocsort-device`, and `--deepocsort-half` still exist, but now they only override the chosen BoxMOT config file for one run.
- On first use, BoxMOT may need to download that checkpoint into its weights directory. Keep `device = "cpu"` unless your local torch + BoxMOT stack is already proven on CUDA.

Minimal BoT-SORT workflow:

1. Open `config/botsort.toml` and adjust the comments-backed values you care about.
2. Open `config/reid.toml` if you want to tune torso crop bounds or minimum crop/box sizes for ReID.
3. Run the pipeline with `--tracker-backend botsort`.
4. Only use the `--deepocsort-*` flags when you want a one-off BoxMOT runtime override without editing the file.

Minimal Deep OC-SORT workflow:

1. Open `config/deepocsort.toml` and adjust the comments-backed values you care about.
2. Open `config/reid.toml` if you want to tune torso crop bounds or minimum crop/box sizes for ReID.
3. Run the pipeline with `--tracker-backend deepocsort`.
4. Only use the `--deepocsort-*` flags when you want a one-off override without editing the file.

SAM2 tracking notes:

- SAM2 is now the official local tracker path. It still depends on the same Roboflow player detections for seeding; it is not a detector replacement.
- Install `requirements.txt` first because the SAM2 script also imports the main repo stack, then install PyTorch for your machine, then `requirements-sam2.txt`.
- This path is heavier than ByteTrack, Deep OC-SORT, or BoT-SORT, and it is most realistic on a short local clip with a GPU.
- The current implementation re-seeds SAM2 from detector boxes every `--frame-stride` frames and propagates masks between those refreshes.

Install the SAM2 dependencies:

```bash
python3 -m pip install -r requirements.txt
# install torch/torchvision from pytorch.org for your machine first
python3 -m pip install -r requirements-sam2.txt
```

Run the official detector-seeded SAM2 tracker on a short clip:

```bash
python3 scripts/tracking/track_players_sam2.py media/game.mp4 \
  --frame-stride 15 \
  --max-frames 300
```

Run the same clip summary on the output:

```bash
python3 scripts/tracking/summarize_tracking_run.py outputs/game_sam2_tracks.jsonl
```

ReID crop tuning quick path:

1. Edit `config/reid.toml`.
2. Start with `[reid.crop]` if you want to try a tighter or lower torso band.
3. Use `[reid.overlap]` to reject heavily occluded torso crops so local reassignment falls back to motion instead of contaminated color.
4. Use `[reid.quality]` to reject tiny boxes/crops that produce noisy appearance features.
5. Run the same command again; the pipeline loads the file on each run.

Run the main local video pipeline with court keypoints and projected-track export:

```bash
python3 scripts/tracking/video_pipeline.py media/game.mp4 \
  --court-keypoints \
  --frame-stride 30 \
  --max-frames 300
```

Main pipeline defaults:

- Output video: `outputs/<video-stem>_pipeline.mp4`.
- Track records: `outputs/<video-stem>_tracks.jsonl`.
- Raw inference JSONL: `outputs/<video-stem>_inference.jsonl`.
- With `--court-keypoints`, also writes `outputs/<video-stem>_court_keypoints.jsonl`.
- If at least one frame yields a usable homography, also writes `outputs/<video-stem>_projected_tracks.jsonl`, `outputs/<video-stem>_projected_shots.json`, and `outputs/<video-stem>_projected_shots.png`.
- `--save-json` defaults to `true`.
- `--sampled-output-only` defaults to `true`.
- `--persist-tracks` defaults to `true` with `--max-missing-sampled-frames 1`.
- `--smoothing-window` defaults to `1`.
- `--court-confidence` defaults to `0.7` when court keypoints are enabled.

Projected-track visualization from pipeline output:

```bash
python3 scripts/court/visualize_projected_tracks.py outputs/game_projected_tracks.jsonl --output-video outputs/game_projected_tracks_court.mp4
```

Visualizer defaults:

- Summary PNG: `outputs/<projected-jsonl-stem>_court_debug.png`.
- Optional `--output-video` renders a simple top-down MP4 without new Roboflow calls.
- `--pixels-per-meter` default `40`, `--point-radius` default `6`, `--output-fps` default `5.0`.

Detect court keypoints on one frame:

```bash
python3 scripts/court/detect_court_keypoints.py media/frame.jpg --confidence 0.0
```

Defaults:

- Output image: `outputs/<image-stem>_court_keypoints.jpg`.
- Keypoint JSON: `outputs/<image-stem>_court_keypoints.json` with `model_id`, `raw_result`, normalized `keypoints`, and a `keypoints_sv` summary (`xy`, `labels`, `class_ids`, `keypoint_confidence`).
- Court model: `COURT_MODEL_ID` or `basketball-court-detection-2/22`.
- Visualization uses Supervision `sv.KeyPoints` with `sv.VertexAnnotator` (yellow circles + label text) and `sv.EdgeAnnotator` driven by the shared semantic court template.
- Homography uses the shared authoritative label -> court-coordinate map in `scripts/court/court_template.py`.

Dry-run court keypoints on a video, no Roboflow call:

```bash
python3 scripts/court/detect_court_keypoints_video.py media/game.mp4 --dry-run --frame-stride 30 --max-frames 300
```

Run sampled-frame court keypoint detection on a video:

```bash
python3 scripts/court/detect_court_keypoints_video.py media/game.mp4 --frame-stride 30 --max-frames 300
```

Defaults:

- Output video: `outputs/<video-stem>_court_keypoints.mp4`.
- `--confidence` default `0.7` to filter noisy keypoints the model returns (use `0.0` to keep all).
- `--sampled-output-only` default `true`, so only Roboflow-inferred frames are written.
- `--frame-stride` default `30`; Roboflow calls are counted and printed at the end.
- Drawing style mirrors `detect_court_keypoints.py`: yellow circle + label, with `no reliable keypoints` overlay when the filter empties a frame.

Build a FIBA court homography from a keypoints JSON. This local-only step maps
detected court labels to FIBA (28m x 15m) coordinates. Output is a homography
matrix plus reprojection stats. The scripts share one explicit numbered court
template and use the authoritative label -> coordinate map. Labels missing
from that map stay semantic-only and are skipped. A geometric pre-validation
step now rejects obviously degenerate point layouts before homography fitting.
No Roboflow call is made.

```bash
python3 scripts/court/build_court_homography.py outputs/frame_court_keypoints.json \
  --court-image
```

Defaults:

- Output JSON: `outputs/<keypoints-stem>_homography.json`.
- With `--court-image`, also writes `outputs/<keypoints-stem>_court_2d.png` (a
  2D top-down FIBA court PNG with the projected keypoints, green = inlier,
  red = outlier).
- `--min-confidence` default `0.7` (filters model noise; raise to drop more).
- `--min-points` default `4` (minimum matched points to attempt a homography).
- `--reproj-threshold` default `8.0` (RANSAC threshold in pixels).
- Degenerate keypoint layouts are rejected before `cv2.findHomography` based on spread and convex-hull area checks.
- `--self-test` runs a synthetic 4-point round-trip check and exits, useful
  for CI / wiring checks without needing a real keypoints JSON.

Run sampled-frame court keypoint detection + FIBA homography on a video, no API call:

```bash
python3 scripts/court/build_court_homography_video.py media/game.mp4 --dry-run --frame-stride 30 --max-frames 300
```

Build a side-by-side debug video (original frame | FIBA 2D court with projected keypoints) per sampled frame. Reuses the same shared court template and RANSAC math as `scripts/court/build_court_homography.py`.

```bash
python3 scripts/court/build_court_homography_video.py media/game.mp4 --frame-stride 30 --max-frames 300
```

Defaults:

- Output video: `outputs/<video-stem>_homography.mp4`.
- Output is a side-by-side composite (original frame | FIBA 2D court), so the output frame width is `2*width + 4`.
- `--confidence` default `0.7`, `--frame-stride` default `30`, `--sampled-output-only` default `true`.
- `--min-points` default `4` (minimum matched points to fit a homography).
- `--reproj-threshold` default `8.0` (RANSAC threshold in pixels).
- Summary at the end: frames read, Roboflow calls, frames with homography, mean inliers per homography frame, mean reprojection error in meters.

Dry-run video tracking, no Roboflow call:

```bash
python3 scripts/tracking/track_players_video.py media/game.mp4 --dry-run --frame-stride 30 --max-frames 300
```

Run sampled-frame ByteTrack tracking:

```bash
python3 scripts/tracking/track_players_video.py media/game.mp4 --frame-stride 30 --max-frames 300
```

Defaults:

- Output video: `outputs/<video-stem>_tracked.mp4`.
- Track records: `outputs/<video-stem>_tracks.jsonl` with `frame_index`, `track_id`, `class_name`, `confidence`, and `xyxy`.
- With `--court-keypoints`, also writes per-frame court keypoints JSONL and projected-track JSONL.
- `--sampled-output-only` defaults to `true`, so only Roboflow-inferred frames are written and tracked.
- `--no-sampled-output-only` writes skipped frames unannotated; it does not pretend tracking is continuous between sampled detections.
- `--persist-tracks` defaults to `true` and keeps recent track IDs alive for brief missed sampled detections using predicted boxes.
- The current appearance reassignment is a lightweight torso-color baseline, not a full re-ID system.
- Tracking requires `ROBOFLOW_API_KEY` for non-dry runs.

## API

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

Returns:

```json
{"status":"ok"}
```

### `POST /detect/image`

Calls the Roboflow model and returns predictions plus a base64 JPEG annotation.

```bash
curl -X POST http://127.0.0.1:8000/detect/image \
  -F "file=@media/img.png"
```

Requires `ROBOFLOW_API_KEY`.

### `POST /detect/video`

Accepts a multipart video upload. Defaults are cost-aware: `dry_run=true`, `frame_stride=30`, and `sampled_output_only=true`.

Dry run, no Roboflow call:

```bash
curl -X POST http://127.0.0.1:8000/detect/video \
  -F "file=@media/game.mp4" \
  -F "dry_run=true" \
  -F "frame_stride=30" \
  -F "max_frames=300"
```

Process video and return a base64 MP4 annotation:

```bash
curl -X POST http://127.0.0.1:8000/detect/video \
  -F "file=@media/game.mp4" \
  -F "dry_run=false" \
  -F "frame_stride=30" \
  -F "max_frames=300" \
  -F "sampled_output_only=true" \
  -F "reuse_last_detections=false" \
  -F "confidence=0.0"
```

Video form fields:

- `dry_run`: default `true`; returns video info and estimated Roboflow calls only.
- `frame_stride`: default `30`; infer every N frames, must be at least 1.
- `max_frames`: optional; maximum frames to read, must be at least 1.
- `sampled_output_only`: default `true`; output only inferred frames.
- `reuse_last_detections`: default `false`; draw last detections on skipped frames when writing all frames.
- `confidence`: default `0.0`; minimum confidence to draw.
- `output_fps`: optional; must be greater than 0.

Processing requires `ROBOFLOW_API_KEY` and returns `model_id`, `frames_read`, `roboflow_calls`, `output_fps`, `content_type`, `filename`, and `annotated_video_base64`.
### Ball cadence

The player/SAM2 pipeline defaults to `15` FPS. In Modal production, TrackNet is
primary for the ball
and defaults to `BALL_ALL_FRAMES=1`, so every decoded source frame is processed
for ball windows while player cost stays at 15 FPS. This is independent of the
declared video FPS. On TrackNet misses, a bounded ball-only SAM2 pass bridges
dense frames; this adds fallback cost only during misses and does not alter the
player SAM2 stream. For legacy FPS-based ball sampling, disable the frame mode
and set its compatibility cadence. This mode uses strided source-frame
windows, not consecutive decoded frames:

```bash
modal run modal_app.py --video-path clips/match03.mp4 --no-ball-all-frames --ball-target-fps 15
```
