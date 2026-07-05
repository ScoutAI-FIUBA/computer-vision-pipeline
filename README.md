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

Detect court keypoints on one frame:

```bash
python3 scripts/detect_court_keypoints.py media/frame.jpg --confidence 0.0
```

Defaults:

- Output image: `outputs/<image-stem>_court_keypoints.jpg`.
- Keypoint JSON: `outputs/<image-stem>_court_keypoints.json` with `model_id`, `raw_result`, normalized `keypoints`, and a `keypoints_sv` summary (`xy`, `labels`, `class_ids`, `keypoint_confidence`).
- Court model: `COURT_MODEL_ID` or `basketball-court-detection-2/22`.
- Visualization uses Supervision `sv.KeyPoints` with `sv.VertexAnnotator` (yellow circles + label text) and `sv.EdgeAnnotator` (placeholder consecutive edges until labels are mapped to canonical court positions).
- Homography is pending until detected keypoint names are mapped to canonical court coordinates.

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
internal `COURT_POINT_MAP` is a **best-guess approximation** inferred from a
single broadcast-angle frame; replace it with a real measurement before
trusting downstream stats. No Roboflow call is made.

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

Build a side-by-side debug video (original frame | FIBA 2D court with projected keypoints) per sampled frame. Reuses the same `COURT_POINT_MAP` and RANSAC math as `build_court_homography.py`; the map is still a best-guess approximation, see that script's section.

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
