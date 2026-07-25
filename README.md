# computer-vision-pipeline

Minimal Roboflow/Supervision basketball pipeline for sampled video detection, tracking, and court-projection debugging.

## Setup

Install dependencies, set your Roboflow key, and run the API:

```bash
python3 -m pip install -r requirements.txt
printf 'ROBOFLOW_API_KEY=your_key_here\n' > .env
uvicorn api.main:app --reload
```

Video processing can make many Roboflow calls. Run `dry_run=true` first to estimate cost before processing.

## Local Scripts

These scripts are local pipeline steps first. The local CLI path is ahead of the API path.

Implementations now live under:

- `scripts/tracking/` for player detection, tracking, frame extraction, and tracking evaluation.
- `scripts/court/` for court-keypoint, homography, and projected-track tooling.
- `scripts/shared/` for shared config and annotation helpers.

## Script Roles

This repo currently has one official local tracking entrypoint plus a set of
focused support tools.

### Main entrypoints

Use these first.

| Script | Role | When to use it |
| --- | --- | --- |
| `scripts/tracking/track_players_sam2.py` | Official local SAM2 tracker | Run the detector-seeded SAM2 tracking pipeline with annotated video, track JSONL, and raw inference JSONL. |
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
- `scripts/tracking/track_players_sam2.py` is now the official local tracker path for player tracking with annotated video, track JSONL, and raw inference JSONL.
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
| Team identification | Future | Add after tracking and court projection are stable. |
| Jersey OCR | Future | Add after track crops and team separation are reliable. |
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
- With `--court-keypoints`, also writes `outputs/<video-stem>_court_keypoints.jsonl` and `outputs/<video-stem>_projected_tracks.jsonl`.
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
