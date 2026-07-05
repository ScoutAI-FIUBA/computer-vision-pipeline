# computer-vision-pipeline

Minimal Roboflow/Supervision prototype for basketball player detection.

## Setup

Install dependencies, set your Roboflow key, and run the API:

```bash
python3 -m pip install -r requirements.txt
printf 'ROBOFLOW_API_KEY=your_key_here\n' > .env
uvicorn api.main:app --reload
```

Video processing can make many Roboflow calls. Run `dry_run=true` first to estimate cost before processing.

## Local Scripts

These scripts are local pipeline steps only. The tracking and court keypoint scripts are not integrated into the API yet.

## Script Roles

This repo currently has one main local video entrypoint plus a set of focused
tools. Keep that distinction clear so the pipeline can evolve without turning
every experiment into the "official" path.

### Main entrypoints

Use these first.

| Script | Role | When to use it |
| --- | --- | --- |
| `scripts/video_pipeline.py` | Main local video pipeline | Run the current player-detection + tracking pipeline with annotated video, track JSONL, and raw inference JSONL. |
| `scripts/detect_players.py` | Main single-image detector | Run player detection on one local image/frame. |

### Local support tools

These are useful, but they are support/debug tools rather than the primary
workflow.

| Script | Role | When to use it |
| --- | --- | --- |
| `scripts/extract_video_frames.py` | Offline frame extraction | Sample a local video into JPG frames without calling Roboflow. |
| `scripts/detect_players_video.py` | Detection-only video helper | Inspect or run sampled video detection without tracking; also provides reusable video helpers. |
| `scripts/track_players_video.py` | Tracking core | Core ByteTrack processing used by `video_pipeline.py`; still useful directly for focused tracking/debug work. |
| `scripts/detect_court_keypoints.py` | Court keypoints on one image | Test court landmarks on a single frame. |
| `scripts/detect_court_keypoints_video.py` | Court keypoints on sampled video | Evaluate court keypoint behavior across a clip. |
| `scripts/build_court_homography.py` | Homography from saved keypoints | Turn detected court keypoints into a FIBA homography and debug reprojection quality. |
| `scripts/build_court_homography_video.py` | Video homography debug | Build a side-by-side debug video for court projection quality checks. |

### Shared infrastructure

| Script | Role |
| --- | --- |
| `scripts/env_config.py` | Shared `.env`-backed configuration for model IDs, API key, and Roboflow endpoint. |

### Current direction

- Treat `scripts/video_pipeline.py` as the canonical local video path.
- Treat court/homography scripts as separate debug/evaluation tools until they
  are intentionally integrated into the main pipeline.
- Add new stages to the pipeline by reusing these support tools first, then
  folding only the stable pieces into the main entrypoint.

## Integration roadmap

This is the intended path so future work stays understandable.

### 1. Keep as the main pipeline

These belong in `scripts/video_pipeline.py` because they are part of the core
"process one local video and inspect the result" workflow.

| Stage | Status | Notes |
| --- | --- | --- |
| Player detection | Integrated | Current base stage. |
| ByteTrack tracking | Integrated | Includes conservative smoothing and optional stale-track reuse. |
| Raw inference JSONL | Integrated | One record per inferred frame. |
| Court keypoints | Next integration candidate | Integrate only after the current debug flow is stable enough. |
| Court homography / player projection | Later integration candidate | Depends on trustworthy keypoint labels and court mapping. |
| Team identification | Future | Add after tracking and court projection are stable. |
| Jersey OCR | Future | Add after track crops and team separation are reliable. |
| Event logic / make-miss | Future | Depends on stable tracking + court geometry. |

### 2. Keep as external tools for now

These remain outside the main pipeline until their outputs are trustworthy and
their role is clear.

| Script | Why it stays external for now |
| --- | --- |
| `scripts/extract_video_frames.py` | Operator tooling for offline inspection, not part of inference/render. |
| `scripts/detect_players_video.py` | Detection-only helper and source of reusable video utilities. |
| `scripts/track_players_video.py` | Processing core used by the main pipeline, but still useful as a focused debug tool. |
| `scripts/detect_court_keypoints.py` | Single-frame evaluation tool for court landmarks. |
| `scripts/detect_court_keypoints_video.py` | Sampled-video evaluation tool for court landmarks. |
| `scripts/build_court_homography.py` | Local geometry/debug tool for checking court mapping quality. |
| `scripts/build_court_homography_video.py` | Side-by-side projection debug tool before main-pipeline integration. |

### 3. Candidates to merge or retire later

- `scripts/detect_players_video.py`: keep while it provides reusable helpers;
  retire it as a user-facing entrypoint once `video_pipeline.py` fully replaces
  it for normal video work.
- `scripts/track_players_video.py`: keep as the implementation core unless it
  becomes thin enough to fold directly into `video_pipeline.py` without making
  that file ugly.
- Court and homography scripts: keep separate until the integration path is
  proven; do not merge them early just to reduce file count.

### Rule of thumb

- If a script is used to process the normal local video workflow, it belongs in
  or behind `scripts/video_pipeline.py`.
- If a script exists to inspect, validate, calibrate, or debug one stage, keep
  it as a support tool until the stage is stable.

Detect court keypoints on one frame:

```bash
python3 scripts/detect_court_keypoints.py media/frame.jpg --confidence 0.0
```

Defaults:

- Output image: `outputs/<image-stem>_court_keypoints.jpg`.
- Keypoint JSON: `outputs/<image-stem>_court_keypoints.json` with `model_id`, `raw_result`, normalized `keypoints`, and a `keypoints_sv` summary (`xy`, `labels`, `class_ids`, `keypoint_confidence`).
- Court model: `COURT_MODEL_ID` or `basketball-court-detection-2/22`.
- Visualization uses Supervision `sv.KeyPoints` with `sv.VertexAnnotator` (yellow circles + label text) and `sv.EdgeAnnotator` driven by the shared semantic court template.
- Homography uses the shared authoritative label -> court-coordinate map in `scripts/court_template.py`.

Dry-run court keypoints on a video, no Roboflow call:

```bash
python3 scripts/detect_court_keypoints_video.py media/game.mp4 --dry-run --frame-stride 30 --max-frames 300
```

Run sampled-frame court keypoint detection on a video:

```bash
python3 scripts/detect_court_keypoints_video.py media/game.mp4 --frame-stride 30 --max-frames 300
```

Defaults:

- Output video: `outputs/<video-stem>_court_keypoints.mp4`.
- `--confidence` default `0.7` to filter noisy keypoints the model returns (use `0.0` to keep all).
- `--sampled-output-only` default `true`, so only Roboflow-inferred frames are written.
- `--frame-stride` default `30`; Roboflow calls are counted and printed at the end.
- Drawing style mirrors `detect_court_keypoints.py`: yellow circle + label, with `no reliable keypoints` overlay when the filter empties a frame.

Build a FIBA court homography from a keypoints JSON. This is the local-only
step that maps detected keypoint labels to FIBA (28m x 15m) court
coordinates. Output is a homography matrix plus reprojection stats. The
scripts share one explicit numbered court template and use the provided
authoritative label -> coordinate map. Labels still missing from that map stay
semantic-only and are skipped. No Roboflow call is made.

```bash
python3 scripts/build_court_homography.py outputs/frame_court_keypoints.json \
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
- `--self-test` runs a synthetic 4-point round-trip check and exits, useful
  for CI / wiring checks without needing a real keypoints JSON.

Run sampled-frame court keypoint detection + FIBA homography on a video, no API call:

```bash
python3 scripts/build_court_homography_video.py media/game.mp4 --dry-run --frame-stride 30 --max-frames 300
```

Build a side-by-side debug video (original frame | FIBA 2D court with projected keypoints) per sampled frame. Reuses the same shared court template and RANSAC math as `build_court_homography.py`.

```bash
python3 scripts/build_court_homography_video.py media/game.mp4 --frame-stride 30 --max-frames 300
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
python3 scripts/track_players_video.py media/game.mp4 --dry-run --frame-stride 30 --max-frames 300
```

Run sampled-frame ByteTrack tracking:

```bash
python3 scripts/track_players_video.py media/game.mp4 --frame-stride 30 --max-frames 300
```

Defaults:

- Output video: `outputs/<video-stem>_tracked.mp4`.
- Track records: `outputs/<video-stem>_tracks.jsonl` with `frame_index`, `track_id`, `class_name`, `confidence`, and `xyxy`.
- `--sampled-output-only` defaults to `true`, so only Roboflow-inferred frames are written and tracked.
- `--no-sampled-output-only` writes skipped frames unannotated; it does not pretend tracking is continuous between sampled detections.
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
