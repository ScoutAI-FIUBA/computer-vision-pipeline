# Modal setup

## TrackNet ball-only inspection

This standalone script uses only the local TrackNet checkpoint. It does not
load the player detector, SAM2, Roboflow, possession, pass, or Modal code.

```bash
uv run python scripts/ball/tracknet_ball_only.py clip_020.mp4 \
  --model /absolute/path/to/TrackNet_best.pt \
  --output-dir outputs/clip_020_ball
```

It writes `ball_tracks.jsonl` and `ball_only.mp4`. Use `--max-frames`,
`--frame-stride` or `--target-fps`, `--threshold`, `--min-distance`, and
`--max-candidates` to keep an inspection run small.

## Local TrackNet ball detection

The local pipeline defaults TrackNet to disabled. Enable it explicitly with
`TRACKNET_ENABLED=1` and point it at a local checkpoint (the file is not stored
in this repository):

```bash
export TRACKNET_ENABLED=1
export TRACKNET_CHECKPOINT=/absolute/path/to/TrackNet_best.pt
export TRACKNET_FALLBACK_BUDGET=12
python -m pipeline.cli clips/short.mp4 --max-frames 90 --output-dir outputs/tracknet
```

TrackNet uses 8-frame windows at 512x288 with `bg_mode=concat`. Modal production
enables TrackNet by default: all decoded source frames feed the dense TrackNet
stream, independent of the player output cadence. When TrackNet misses, ball-only SAM2 may carry
the last valid ball for at most `TRACKNET_FALLBACK_BUDGET` **emitted ball
frames** (default 12). The counter resets only on a real TrackNet detection,
not on fallback rows. After the hard cap, `ball_tracks.jsonl` emits one
`track_state: "lost"` row with stable public `track_id`, `xyxy: null`, and
`center_xy: null`; no box is sent to SAM2 until TrackNet seeds the ball again.
Every row has `timestamp_seconds` and is written in source-frame order.
`TRACKNET_SEED_RADIUS` and `TRACKNET_HEATMAP_THRESHOLD` tune seed size and
acceptance.

The dense ball pass keeps only its 8-frame TrackNet window and previous fallback
frame in memory. Dense misses use a separate two-frame, ball-only SAM2
fallback; the player SAM2/output stream remains at its separate 15 FPS default.

### V2 Modal stats

V2 also provides restartable tracking/stats primitives backed by the named `v2-job-data`
Volume. A job directory contains `job.json`, `input.mp4`, and immutable generation
directories under `tracking/` and `stats/`; manifests include hashes and fingerprints.
Retries reuse matching complete generations, while stale leases can be reclaimed. The
active lease rejects duplicate workers. The native `modal run` entrypoint exposes restartable
routing. Use `--staged` with `--job-id` for one command that runs tracking then stats, or use
`--stats-only --job-id` to retry only stats from an existing job. Remote logs and errors are
streamed by the normal Modal runner:

```bash
# Command 1: persist tracking artifacts/bundles
modal run modal_app_v2.py --tracking-only --job-id game-03 \
  --video-path clips/game_03.mp4

# Command 2: run stats from the persisted tracking bundle
modal run modal_app_v2.py --stats-only --job-id game-03
```

`--tracking-only`, `--staged`, and `--stats-only` are mutually exclusive; each requires
`--job-id`, and `--tracking-only` also requires `--video-path`. The existing monolithic
`process` method remains the default and rollback path when neither flag is supplied.

Replay a selected generation locally (read-only apart from the requested MP4 output):

```bash
python scripts/tracking/debug_v2.py --job-id /path/to/job \
  --generation GENERATION --output replay.mp4
```

The retention window for the job Volume is intentionally still an operational decision.
Measure timeout and GPU cost before enabling staged routing in production.

`modal_app_v2.py` uses the existing `tracknet-data` Modal Volume, mounted at
`/tracknet-data`, and enables TrackNet with this checkpoint path:

```text
/tracknet-data/tracknet-finetune/runs/basketball-v1-20ep-r3/checkpoints/TrackNet_best.pt
```

The V2 stats path does not load SAM2 or rerun player tracking. The Volume must
already contain that checkpoint; a missing file fails the job clearly.
`BALL_ALL_FRAMES=1` keeps the dense TrackNet pass independent of the V2 player
observation cadence. After a short run, inspect the returned
`stats/results.json` and `stats/ball_tracks.jsonl`:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/v2-bundle/stats")
results = json.loads((root / "results.json").read_text())
rows = [json.loads(line) for line in (root / "ball_tracks.jsonl").read_text().splitlines() if line.strip()]
print("tracknet_enabled:", results["config"]["tracknet_enabled"])
print("non_lost_ball_rows:", sum(row["track_state"] != "lost" for row in rows))
PY
```

The expected values are `tracknet_enabled: True` and at least one non-lost
ball row for a clip where TrackNet detects the ball.

If `COURT_MODEL_ID` is configured, V2 samples the authoritative observation
frames with that detector and writes `projected_tracks.jsonl`,
`projected_shots.json`, and `projected_shots.png` only when a valid homography
is available. These are optional: missing/invalid court evidence leaves the
three pointers out of `stats/results.json` and the V2 manifest. Emitted
manifest entries use relative paths, byte sizes, and SHA-256 hashes; projected
rows retain `track_id`, `stable_player_id`, `team_id`, and pixel coordinates.

V2 jersey numbers use the existing predictor settings but do not invoke the
legacy pipeline. On every `FRAME_STRIDE` sample, V2 calls the configured
basketball detector, associates its `number` box to the matching SAM3 player
row, and sends that number crop to OCR. Votes are grouped by `stable_player_id`;
two consistent accepted reads are required for confirmation. Local/offline
callers use `NUMBER_PREDICTOR_BACKEND=dummy` by default. The V2 Modal image uses
the configured `PLAYER_MODEL_ID` detector, hosted OCR, and the
`scoutai-roboflow` Modal secret. Inspect `stats/number_manifest.json`,
`stats/number_predictions.json`, and `stats/numbers.json`; increasing
`FRAME_STRIDE` reduces detector and OCR inference cost. If the detector returns
no `number` boxes, OCR is not invoked and no number artifacts are fabricated.

The V2 image opts its existing stats adapter into Possession V3 without changing
the legacy Modal image:

```bash
modal run modal_app_v2.py --video-path clips/game_03.mp4 \
  --frame-start 60 --frame-end 130 --output-dir outputs/v2-game
```

The returned `stats/` directory should contain `possession_v3_manifest.json` and
generation-scoped files under `possession_v3/` for possession rows, transitions,
and events. `modal_app.py` remains on the legacy possession engine, and local
or other callers still default to legacy unless they explicitly set
`POSSESSION_ENGINE=v3`.

### V4 and rollback commands

Run segmented V4 explicitly; the selection is sent to the Modal container and
recorded in the root manifest plus stats metadata:

```bash
SAM31_V2_SEGMENTED=1 modal run modal_app_v2.py \
  --video-path clips/game_03.mp4 --frame-start 60 --frame-end 130 \
  --segment-window 32 --segment-overlap 1 \
  --possession-engine v4 --output-dir outputs/v2-segmented-v4
```

V4 is exclusive: it does not run V3 in parallel. Roll back V2 with
`--possession-engine v3`; use `legacy` only with the general local
pipeline/config path, for example
`POSSESSION_ENGINE=legacy python -m pipeline.cli clips/game_03.mp4`.
Calibration is offline and is not run automatically.

For a V3-only saved-bundle debug render, pass both exact inputs:

```bash
./.venv/bin/python scripts/tracking/debug_v2.py \
  --video-path clips/game_03.mp4 \
  --artifact-dir outputs/v2-segmented-v4 \
  --v3-dir outputs/v2-segmented-v4/stats/possession_v3/generations/<generation-id> \
  --numbers-json outputs/v2-segmented-v4/stats/numbers.json \
  --output outputs/v2-segmented-v4/v3-number-debug.mp4
```

For bounded SAM3.1 V2 runs, seam stitching uses mask IoU first and bbox
IoU/center fallback to make one-to-one local-ID matches. Set
`SAM31_V2_SEAM_MIN_SCORE` and `SAM31_V2_SEAM_AMBIGUITY_MARGIN` to tune the
physical-evidence gate. Ambiguous, crossing, or missing overlap is kept as
separate qualified fragments; IDs are never joined from local-ID equality alone.
Segmented rows use `raw_track_id` for the namespaced segment-local SAM ID and
preserve the original model ID/key in `local_raw_track_id` and
`raw_observation_key`. The separate `canonical_raw_track_id` is used only for
seam grouping and identity resolution.
When the later overlap owner lacks a mask, the mask is omitted and diagnosed;
the earlier segment's mask is never reassigned to the later observation.

## Verification

Focused checks are useful while editing, but do not count as full verification.
The canonical full-suite command is the repository-wide guard from the project root;
run it and report its actual totals:

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

The configured `testpaths = tests` makes this command collect the complete test
suite; a focused path such as `tests/test_sam31_segments.py` is intentionally not
a substitute.

To review V3 state together with recognized jersey numbers without loading the
legacy possession overlays, render the saved bundle with explicit inputs:

```bash
./.venv/bin/python scripts/tracking/debug_v2.py \
  --video-path path/to/input.mp4 \
  --artifact-dir outputs/v2-game \
  --v3-dir outputs/v2-game/stats/possession_v3/generations/<generation-id> \
  --numbers-json outputs/v2-game/stats/numbers.json \
  --output outputs/v2-game/v3-number-debug.mp4
```

When using `--stats-dir`, the renderer automatically uses an existing
`numbers.json` from that directory and keeps the legacy possession overlays.
Omit `--numbers-json` when number overlays are not needed.

How to get the pipeline running on Modal GPUs. Follow it in order — each step
gets cheaper to debug if the previous one worked.

> **Budget context.** We are on $30/month of credits that has to last ~8 months
> plus a demo. Every step below is written to spend as little as possible while
> still proving the thing works. Do not skip the `--max-frames` on the first run.

---

## 1. One-time account setup

Use a virtualenv. Debian, Ubuntu and WSL refuse system-wide `pip install` with
`error: externally-managed-environment` (PEP 668), and you do not want these
packages system-wide anyway.

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements-dev.txt
```

Everything below assumes the venv is active — your prompt should start with
`(.venv)`. If you would rather not activate it, prefix commands with
`.venv/bin/` instead: `.venv/bin/modal run ...`.

> `requirements-dev.txt` is local tooling: Modal, pytest, OpenCV, python-dotenv,
> umap-learn, and inference-sdk. It does **not** install torch, SAM2, or the
> production FashionSigLIP/legacy BoxMOT stack. Install the heavier optional
> sets separately when needed: `requirements.txt` for the production team and
> legacy BoxMOT stack, and `requirements-sam2.txt` for SAM2.

```bash
modal setup
```

That opens a browser and writes a token to `~/.modal.toml`. Each teammate does
this once with their own account; the workspace is shared.

---

## 2. Create the Secrets

Secrets live in Modal, never in the repo. Run each of these once. Replace the
placeholder values — do **not** paste real keys into a file you might commit.

**Roboflow** — use your **private API key**, not the publishable one.

Roboflow issues two kinds of key. The publishable key is for `inference.js`
running in a browser against published models, and is safe to ship to clients.
The private key is the one the Python `inference_sdk` needs to reach our private
model, and it also grants full account access — including deleting trained
models. It stays server-side, always.

```bash
modal secret create scoutai-roboflow ROBOFLOW_API_KEY=your_private_key_here
```

> Never put the private key in the frontend repo. Vite inlines every `VITE_*`
> variable into the client bundle at build time, so it would be publicly
> readable. Note also that the free plan meters calls per account: the whole
> team shares one quota, so use `--dry-run` and `--max-frames` before big runs.

**Cloudflare R2** — same bucket and credentials the Render backend uses:

```bash
modal secret create scoutai-r2 R2_ACCESS_KEY_ID=your_key_id R2_SECRET_ACCESS_KEY=your_secret R2_BUCKET_NAME=your_bucket R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com
```

**Endpoint auth** — the shared bearer token the web endpoint checks. Generate a
random one, don't invent it by hand:

```bash
openssl rand -hex 32
```

```bash
modal secret create scoutai-auth MODAL_WEBHOOK_TOKEN=paste_the_generated_value_here
```

**Job status callback** — where to report progress back to the backend. Only
needed once the backend is deployed; `modal run` works without it, and jobs
simply log their status instead of reporting it.

The token must match `PROCESSING_CALLBACK_TOKEN` in the backend's environment.

```bash
modal secret create scoutai-callback PROCESSING_CALLBACK_URL=https://your-backend.onrender.com/internal/jobs/status PROCESSING_CALLBACK_TOKEN=paste_the_backend_token_here
```

Confirm they exist:

```bash
modal secret list
```

---

## 3. Validate the SAM3.1 import image without GPU inference

Run the dependency smoke test in the exact image defined by
`modal_app_v2.py`:

```bash
modal run modal_app_v2.py::sam31_dependency_smoke_test
```

This checks the official SAM3.1 import path (`pycocotools.mask`, `PIL`,
`psutil`, `einops`, `scipy`, OpenCV, and `sam3.model_builder`) and verifies
that the predictor builder is callable. It does **not** construct the model,
download model weights, or require a GPU.

This is an import/environment check only. It does not replace the GPU
predictor smoke or inference path: `PipelineV2.process` initializes the
predictor lazily when the normal V2 video command is run.

For a V2 review run, add `--return-diagnostics` to the local entrypoint. The
JSON response then includes opt-in `debug_diagnostics` with Team vote counts;
normal responses and returned `tracks.jsonl` artifacts do not include it:

```bash
modal run modal_app_v2.py --video-path clips/game_03.mp4 \
  --frame-start 60 --frame-end 130 --return-diagnostics
```

To opt into bounded SAM3.1 sessions while retaining the 16-object model limit:

```bash
SAM31_V2_SEGMENTED=1 modal run modal_app_v2.py \
  --video-path clips/game_03.mp4 --frame-start 60 --frame-end 130 \
  --segment-window 32 --segment-overlap 1 --output-dir outputs/v2-segmented
```

The later segment owns shared frames. Temporary clips preserve source dimensions
and FPS, and their local SAM IDs are namespaced before the global deduplication,
ReID, and Team ID stages. To roll back, omit `SAM31_V2_SEGMENTED`; the original
full-video session is the default.

---

## 4. First run — smallest possible

This builds the image (slow the first time, ~5-10 min, cached afterwards),
loads SAM2, and processes **90 frames only**.

```bash
modal run modal_app.py --video-path clips/short.mp4 --max-frames 90
```

Modal production enables TrackNet by default. To skip loading its checkpoint
and retain the hosted ball-detection compatibility path, use the explicit
override:

```bash
modal run modal_app.py --video-path clips/short.mp4 --no-tracknet-enabled
```

`--video-path` uploads a local file straight into the container, so this works
before any R2 wiring exists. Keep the test clip small — it travels as a function
argument.

What you should see: the image build, `SAM2 loaded: facebook/sam2-hiera-small on
cuda`, progress logs, then a JSON summary with `frames_processed: 90` and a
`tracks_preview` array.

FashionSigLIP is the default team-ID backbone. Its Modal image and production
requirements include `ftfy==6.3.1` and `open_clip_torch==2.32.0` in addition to
the aligned Transformers/Hugging Face pins. `open_clip_torch` provides the
`open_clip` import used by Marqo's dynamic modeling code. The Modal image
targets Python 3.12; a local Python 3.13 environment is not proof that the
Modal runtime will resolve the same stack.

If it fails, the likely causes in order:

| Symptom | Cause |
|---|---|
| Image build or FashionSigLIP import fails | Keep `transformers==5.0.0`, `huggingface_hub==1.7.1`, `ftfy==6.3.1`, and `open_clip_torch==2.32.0` aligned in the production/team requirements and `modal_app.py`; verify the runtime is Python 3.12. |
| `Secret not found` | Name typo in step 2. Names must match `scoutai-roboflow`, `scoutai-r2`, `scoutai-auth` exactly. |
| `ModuleNotFoundError: pipeline` | `.add_local_python_source("pipeline")` missing from the image definition. |
| Roboflow 401/403 | Bad key, or the model id in `.env.example` doesn't match your workspace. |

---

## 5. Check what it cost

```bash
modal app list
```

Then open the run in the Modal dashboard and look at GPU seconds. Do this after
the first real run and write the number down — it is how you predict whether a
full game is affordable before you try one.

Rough expectation on L4 with `hiera-small`: a 10-minute clip lands somewhere
around 20-30 minutes of GPU time. Measure rather than trust that.

---

## 6. Scale up gradually

Once 90 frames works, increase deliberately. Do not jump straight to a full game.

```bash
modal run modal_app.py --video-path clips/short.mp4 --max-frames 900
```

Reading from R2 instead of uploading, and writing results back:

```bash
modal run modal_app.py --r2-key videos/TEAM_ID/VIDEO_ID --output-prefix results/VIDEO_ID
```

---

## 7. Deploy the endpoint

Only needed when something else has to call the pipeline over HTTP. `modal run`
does not require this.

```bash
modal deploy modal_app.py
```

That prints a URL for `submit`. Test that auth works — the first call must be
rejected, the second accepted:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://YOUR-ENDPOINT.modal.run -H 'Content-Type: application/json' -d '{"r2_key":"test"}'
```

Expect `401`. Now with the token:

```bash
curl -X POST https://YOUR-ENDPOINT.modal.run -H "Authorization: Bearer YOUR_TOKEN" -H 'Content-Type: application/json' -d '{"r2_key":"videos/TEAM_ID/VIDEO_ID","output_prefix":"results/VIDEO_ID"}'
```

The submit payload accepts `tracknet_enabled` and defaults it to `true` for
backward compatibility. Set it to `false` to use hosted ball detection without
loading the TrackNet checkpoint.

Expect `{"call_id": "...", "status": "queued"}` immediately — it does not wait
for the GPU.

---

## 8. Configuration quick path

`pipeline/config.py` is the source of truth for the deployed pipeline.

How configuration is resolved:

1. `PipelineConfig.from_env()` loads environment variables and optional `.env`.
2. `pipeline.cli` overrides a smaller local subset for ad hoc runs.
3. `modal_app.py` exposes the full current remote override surface.

Core environment variables:

| Variable | Default | Notes |
|---|---|---|
| `ROBOFLOW_API_KEY` | required | Private Roboflow key. |
| `ROBOFLOW_API_URL` | `https://serverless.roboflow.com` | Hosted inference endpoint. |
| `PLAYER_MODEL_ID` | `basketball-player-detection-3-ycjdo/16` | Player detector model id. |
| `SAM2_MODEL_ID` | `facebook/sam2-hiera-small` | Default SAM2 checkpoint. |
| `TEAM_EMBEDDER_BACKEND` | `siglip` | FashionSigLIP by default; `colour` or `none` remain supported overrides. |
| `TEAM_EMBEDDER_MODEL_ID` | `Marqo/marqo-fashionSigLIP` | FashionSigLIP model id for `siglip`. |
| `TEAM_COUNT` | `2` | Current clustering target. |
| `MIN_TEAM_SAMPLES_PER_TRACK` | `3` | Minimum selected samples before assigning a track. |
| `TEAM_SAMPLE_STRIDE` | `5` | Propagated-frame team-evidence sampling rate. |
| `MIN_TEAM_SAMPLE_QUALITY` | `0.0` | Minimum quality score for an accepted sample. |
| `MAX_TEAM_SAMPLES_PER_TRACK` | `12` | Per-track cap before global clustering. |
| `TEAM_ASSIGNMENT_MARGIN` | `0.03` | Required vote margin for a final assignment. |
| `TEAM_DEBUG_ENABLED` | `true` | Writes `team_debug.json` plus crop/mask artifacts. |
| `NUMBER_PREDICTOR_BACKEND` | `dummy` | `dummy` or `roboflow-hosted` jersey-number predictor. |
| `NUMBER_PREDICTOR_ROBOFLOW_MODEL_ID` | `basketball-jersey-numbers-ocr/6` | Hosted Roboflow OCR model id for jersey numbers. |
| `NUMBER_PREDICTOR_SCORE_THRESHOLD` | `0.7` | Minimum OCR score before a read can count. |

Local CLI overrides available today:

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

Modal overrides additionally expose:

- `team_sample_stride`
- `min_team_sample_quality`
- `max_team_samples_per_track`
- `team_debug_enabled`
- `number_predictor_backend`
- `max_missed_detections`
- `appearance_weight`

Current team-and-number flow, as implemented:

1. Collect team evidence on each detector frame.
2. Collect extra evidence on propagated frames every `TEAM_SAMPLE_STRIDE` offsets.
3. Keep only accepted samples above `MIN_TEAM_SAMPLE_QUALITY`, then select the best `MAX_TEAM_SAMPLES_PER_TRACK` samples per track.
4. Cluster all selected samples globally and vote one label per track.
5. Leave tracks unassigned when they are under-sampled or ambiguous.
6. On detector frames, collect overlapping `number` detections, run hosted OCR when enabled, and save every OCR case to `number_predictions.json` plus `number_prediction_cases.jsonl`.
7. Aggregate accepted OCR reads per track into `numbers.json`, confirming only clean winners and leaving conflicts ambiguous.

Output contract to expect from a successful run:

- `results.json` now includes `numbers_file` and is written only after all final artifacts are done.
- `numbers.json` contains the final per-track jersey-number decisions.
- `annotated.mp4`, when enabled, can show confirmed jersey numbers in labels and does not fall back to internal track IDs when no number is confirmed.

FashionSigLIP is the default `siglip` backbone. The downstream clustering,
voting, and unknown/unassigned logic is unchanged; an explicit legacy timm
model ID remains available as an override. Several older notes in
`future_tasks/` describe a more elaborate planned design. The code today is one
global clustering pass plus per-track voting for teams, plus a detector-frame
OCR-and-aggregation path for jersey numbers — not a larger temporal validation
system yet.

## Phase status and next session

- **Completed for this phase:** production ball detection and team identification.
- **Next session:** implement statistics detection. Start with the schema and
  provenance checks, then shot make/miss, possession, evidence-backed passes,
  evaluation clips/metrics, output artifacts, and acceptance tests. Statistics
  are not implemented yet.

---

## 9. The cost knobs, and what they cost

All of these live in `modal_app.py` and `pipeline/config.py`, each with a `COST:`
comment next to it. Summary:

| Knob | Default | Why |
|---|---|---|
| `gpu` | `L4` | ~1/4 the price of an A100. Peak VRAM is a few hundred MB because SAM2 only ever sees one ~16-frame segment, so a bigger GPU buys nothing. |
| `max_containers` | `2` | Hard ceiling on parallel GPUs. Stops a retry loop from spinning up ten of them. |
| `scaledown_window` | `60`s | Idle containers are billed. Long enough to reuse a warm container between back-to-back jobs, short enough that forgetting about one costs cents. |
| `timeout` | `1800`s | A hung job cannot bill forever. Raise when you start on full games. |
| `sam2_model_id` | `hiera-small` | The image encoder runs on **every** frame, so this is the biggest single driver of GPU time. Switch to `hiera-large` for final demo runs only. |
| `target_fps` | `15` | **Biggest lever.** SAM2's encoder runs on every processed frame, so cost tracks frames processed, not video length. 60fps source at 15fps is a 4x saving. |
| `frame_stride` | `15` | Detector calls are metered by Roboflow. Note that **lowering this makes tracking worse**, not better — see below. |
| `max_missed_detections` | `2` | Detector calls a track may miss before it is retired. Without it, one missed detection permanently destroys a track. |
| `write_annotated_video` | `False` | Encoding an MP4 of every frame is pure cost when the real output is `tracks.jsonl`. Turn on to debug, off otherwise. |

### Measured configurations

All on the same 9.7s test clip (583 frames, 1280x720 @ 60fps, ~10 players).
Defaults are the "cheap" row.

| Config | Wall | Distinct IDs | Extra IDs per detector call | 10-min clip | Full game |
|---|---|---|---|---|---|
| Colab notebook config (60fps, hiera-large, no fixes) | 164 s | 30 | 0.51 | ~$2.25 | ~$11 |
| 60fps + hiera-large + fixes | 178 s | 16 | 0.15 | ~$2.45 | ~$12 |
| **15fps + hiera-small + fixes (default)** | **38 s** | **11** | **0.10** | **~$0.52** | **~$2.50** |

Raw ID counts are not comparable across frame rates — 60fps makes 39 detector
calls against 15fps's 10, and each call can mint an ID. Use the per-call column.

To reproduce the expensive end:

```bash
modal run modal_app.py --video-path clip.mp4 --native-fps --sam2-model-id facebook/sam2-hiera-large --annotated --output-dir outputs/best
```

Two results that contradict the obvious guess, both measured:

- **`hiera-large` costs only ~7% more than `hiera-small`.** Checkpoint choice is
  nearly free; sampling rate is the only lever that moves cost much (~4x).
- **Full quality does not clearly beat the cheap default on identity.** Reviewed
  visually and judged not better. Prefer the cheap default until there is
  evidence otherwise.

### `frame_stride` is U-shaped — 10 is the minimum

Neither "more detections" nor "fewer detections" is better. Measured on the test
clip at the default 15fps sampling, with carry-forward and re-ID on, counting
distinct track IDs for 10 players on court:

| stride | Detector calls | 2.5s | 5.0s | 6.7s | 9.7s |
|---|---|---|---|---|---|
| 4 | 37 | 10 | 12 | 15 | 18 |
| **10 (default)** | **15** | **10** | **10** | **10** | **10** |
| 15 | 10 | 10 | 10 | 11 | 11 |

Stride 10 is the only setting that stays correct for the whole clip. Both
directions away from it get worse, for different reasons:

- **Too low**: SAM2 is re-seeded from scratch at every detector call, so a player
  the detector misses for one frame is erased and comes back with a new ID. More
  calls means more chances to drop a track. (`max_missed_detections` reduces this
  but does not eliminate it.)
- **Too high**: SAM2 propagates unaided for longer and identities drift.

Two traps when comparing against the Colab notebook:

- **Always read stride together with `target_fps`.** Stride 10 at 15fps sampling
  is a 0.67s detector interval; stride 10 at `--native-fps` on 60fps footage is
  0.17s. Same number, 4x apart — and at native fps stride 10 measured **20** IDs,
  not 10.
- **Short clips hide everything.** At 2.5s every configuration above scores 10.
  Differences only appear past ~5s, so never judge tracking on a short clip.

Two safety habits worth keeping:

- Set a spend limit in the Modal dashboard. It is the only backstop that works
  when someone leaves a loop running overnight.
- Always `--max-frames` on a new clip before running it whole.

---

## 10. Running without Modal

The pipeline is plain Python and imports nothing from Modal. On a machine with a
GPU:

```bash
python -m pipeline.cli clips/short.mp4 --max-frames 90
```

Estimate cost without running or spending anything (works with no GPU, no SAM2,
no API key):

```bash
python -m pipeline.cli clips/short.mp4 --dry-run
```

Tests for the segment windowing, which also need no GPU:

```bash
python -m pytest tests/test_video_segments.py
```
Modal production ball detection uses `BALL_ALL_FRAMES=1`: TrackNet consumes
every decoded source frame independently of the player `target_fps=15`. TrackNet
uses true 8-frame windows on that dense stream; SAM2 only fills bounded TrackNet
misses. Local `PipelineConfig` runs keep `tracknet_enabled=False` unless
`TRACKNET_ENABLED=1` is set explicitly.
For compatibility with the old FPS-derived mode, disable all-frame mode and
set the ball cadence. This compatibility mode uses strided source-frame
windows, not consecutive decoded frames:

```bash
modal run modal_app.py --video-path clips/match03.mp4 --no-ball-all-frames --ball-target-fps 15
```

Dense mode costs in proportion to the source frame count, while leaving
player/SAM2 and court/team work at 15 FPS. Modal carries the mode through
`BALL_ALL_FRAMES`; local SAM2 is not required for the Modal command.
