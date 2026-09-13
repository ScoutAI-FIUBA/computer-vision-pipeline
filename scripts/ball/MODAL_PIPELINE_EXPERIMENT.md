# TrackNet visual experiment

This is read-only and does not enter `pipeline/`. It uses the local fine-tuned
checkpoint, the official TrackNet shape recorded in that checkpoint (`seq_len=8`,
`bg_mode=concat`, input `512x288`), and `Frame,Visibility,X,Y` labels.

The causal window is `target-7..target`, with edge repetition at the beginning.
Every frame is emitted; no initial or final frame is silently dropped. Hosted
detector calls are bounded to 20 deterministic frames per clip by default and
use the same labeled frames for comparison. Missing credentials, dependencies, or
network access fail the comparison with a nonzero exit. Use `--tracknet-only` for
an explicit visualization-only run.

## Three known clips

```bash
./.venv/bin/python scripts/ball/modal_pipeline_experiment.py \
  processed/basketball-v1/test/match20 \
  processed/basketball-v1/train/match01 \
  processed/basketball-v1/train/match03 \
  --output-dir outputs/ball/modal-experiment
```

Plan without torch, credentials, or networking:

```bash
./.venv/bin/python scripts/ball/modal_pipeline_experiment.py \
  --processed-root processed/basketball-v1 \
  --clip-name test/match20 --clip-name train/match01 --clip-name train/match03 \
  --output-dir outputs/ball/modal-experiment --dry-run
```

Inspect the generated `*_tracknet_vs_detector.mp4` files with any video player.
The heatmap is blended over the source frame, yellow circles are TrackNet
candidates, green boxes/centers are the current detector, and the top-left text
contains the frame index and sequence. Red crosses are visible-ball labels.
JSON files beside the videos record the checkpoint, exact comparison frame
indices/paths, and output validation.

## Modal production smoke test

The production wrapper mounts the read-only `tracknet-data` Volume at
`/tracknet-data` and reads:
`/tracknet-data/tracknet-finetune/runs/basketball-v1-20ep-r3/checkpoints/TrackNet_best.pt`.
The exact bounded local invocation is:

```bash
modal run --env main modal_app.py --video-path clips/short.mp4 --max-frames 90 --annotated --output-dir outputs/modal-smoke
```

This downloads `tracks.jsonl`, `ball_tracks.jsonl`, `results.json`, and
`annotated.mp4` to `outputs/modal-smoke`. No checkpoint or dataset is baked
into the image.
