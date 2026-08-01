# Modal setup

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

> `requirements-dev.txt` is intentionally small: Modal, pytest, and OpenCV. It
> does **not** install torch or SAM2, because those run in the container. If you
> want the legacy BoxMOT scripts too, `pip install -r requirements.txt`
> separately — that one does pull torch and the CUDA toolkit, several GB.

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

Confirm all three exist:

```bash
modal secret list
```

---

## 3. First run — smallest possible

This builds the image (slow the first time, ~5-10 min, cached afterwards),
loads SAM2, and processes **90 frames only**.

```bash
modal run modal_app.py --video-path clips/short.mp4 --max-frames 90
```

`--video-path` uploads a local file straight into the container, so this works
before any R2 wiring exists. Keep the test clip small — it travels as a function
argument.

What you should see: the image build, `SAM2 loaded: facebook/sam2-hiera-small on
cuda`, progress logs, then a JSON summary with `frames_processed: 90` and a
`tracks_preview` array.

If it fails, the likely causes in order:

| Symptom | Cause |
|---|---|
| Image build fails on `sam2` install | Version conflict between `torch`, `huggingface_hub`, and SAM2's own pins. Adjust in `modal_app.py` and `requirements-sam2.txt` **together**. |
| `Secret not found` | Name typo in step 2. Names must match `scoutai-roboflow`, `scoutai-r2`, `scoutai-auth` exactly. |
| `ModuleNotFoundError: pipeline` | `.add_local_python_source("pipeline")` missing from the image definition. |
| Roboflow 401/403 | Bad key, or the model id in `.env.example` doesn't match your workspace. |

---

## 4. Check what it cost

```bash
modal app list
```

Then open the run in the Modal dashboard and look at GPU seconds. Do this after
the first real run and write the number down — it is how you predict whether a
full game is affordable before you try one.

Rough expectation on L4 with `hiera-small`: a 10-minute clip lands somewhere
around 20-30 minutes of GPU time. Measure rather than trust that.

---

## 5. Scale up gradually

Once 90 frames works, increase deliberately. Do not jump straight to a full game.

```bash
modal run modal_app.py --video-path clips/short.mp4 --max-frames 900
```

Reading from R2 instead of uploading, and writing results back:

```bash
modal run modal_app.py --r2-key videos/TEAM_ID/VIDEO_ID --output-prefix results/VIDEO_ID
```

---

## 6. Deploy the endpoint

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

Expect `{"call_id": "...", "status": "queued"}` immediately — it does not wait
for the GPU.

---

## 7. Configuration quick path

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
| `TEAM_EMBEDDER_BACKEND` | `siglip` | `siglip`, `colour`, or `none`. |
| `TEAM_EMBEDDER_MODEL_ID` | `vit_base_patch16_siglip_224` | Used only for `siglip`. |
| `TEAM_COUNT` | `2` | Current clustering target. |
| `MIN_TEAM_SAMPLES_PER_TRACK` | `3` | Minimum selected samples before assigning a track. |
| `TEAM_SAMPLE_STRIDE` | `5` | Propagated-frame team-evidence sampling rate. |
| `MIN_TEAM_SAMPLE_QUALITY` | `0.0` | Minimum quality score for an accepted sample. |
| `MAX_TEAM_SAMPLES_PER_TRACK` | `12` | Per-track cap before global clustering. |
| `TEAM_ASSIGNMENT_MARGIN` | `0.03` | Required vote margin for a final assignment. |
| `TEAM_DEBUG_ENABLED` | `true` | Writes `team_debug.json` plus crop/mask artifacts. |

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
- `max_missed_detections`
- `appearance_weight`

Current team-ID flow, as implemented:

1. Collect team evidence on each detector frame.
2. Collect extra evidence on propagated frames every `TEAM_SAMPLE_STRIDE` offsets.
3. Keep only accepted samples above `MIN_TEAM_SAMPLE_QUALITY`, then select the best `MAX_TEAM_SAMPLES_PER_TRACK` samples per track.
4. Cluster all selected samples globally and vote one label per track.
5. Leave tracks unassigned when they are under-sampled or ambiguous.

This matters because several older notes in `future_tasks/` describe a more elaborate planned design. The code today is one global clustering pass plus per-track voting, not a staged colour-first / SigLIP-refinement pipeline.

---

## 8. The cost knobs, and what they cost

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

### Why lowering `frame_stride` backfires

Intuition says calling the detector more often should improve tracking. Measured
on the test clip, it does the opposite:

| Config | Detector calls | Distinct IDs (10 players on court) |
|---|---|---|
| stride 15 | 10 | 14 |
| stride 4 | 37 | **32** |
| stride 15 + carry-forward | 10 | **11** |

SAM2 is re-seeded from scratch at every detector call. Seeding it with only the
current detections means a player the detector misses for one frame is erased,
and gets a new ID when found again. More detector calls means more chances to
drop a track. `max_missed_detections` fixes the underlying problem by keeping
unclaimed tracks alive for a couple of calls; stride tuning does not.

Two safety habits worth keeping:

- Set a spend limit in the Modal dashboard. It is the only backstop that works
  when someone leaves a loop running overnight.
- Always `--max-frames` on a new clip before running it whole.

---

## 9. Running without Modal

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
