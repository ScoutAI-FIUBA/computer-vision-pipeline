# computer-vision-pipeline

GPU-first basketball pipeline for SAM3.1 player observations, independent ball evidence, team identification, and conservative shot/possession attribution.

## Branch quick path: `feature/modelv2-sam31-grace-v5`

This branch's current production-style path is `modal_app_v2.py`:

```text
video -> SAM3.1 masks/observations -> stable identity + ReID + Team ID
      -> TrackNet ball evidence -> shot/number/court evidence
      -> V3/V4/V5 possession -> stats and derived events
```

The stats adapter consumes the authoritative SAM3.1 rows. It does not load SAM2,
run a second player tracker, or recreate players through legacy `run_pipeline`.
The legacy path remains available through `modal_app.py` and the local CLI.

### SAM3.1 segmented tracking

The default V2 session is one full-video SAM3.1 session. Set
`SAM31_V2_SEGMENTED=1` for bounded windows (default window `32`, overlap `1`).
Later segments own overlap rows; seams use mask IoU first, then bbox IoU/center
fallback, and ambiguous matches fail closed.

SAM3.1 and downstream contracts remain capped at **16 active objects**. A
segmented window with **17–22** objects receives deterministic grace: IDs are
ranked by confidence, duration, and temporal consistency, then pruned to 16.
Overflow counts, dropped IDs, ranking reasons, and grace use are recorded. **More
than 22 objects is rejected**, so the tracking stage fails and can be retried
instead of silently dropping an unbounded number of players.

### Components and outputs

- **ReID/Team ID:** stable player IDs are resolved from temporal fragments;
  mask-gated appearance evidence and team embeddings are clustered conservatively.
  Under-sampled or contested assignments remain unresolved.
- **TrackNet:** the Modal image reads
  `/tracknet-data/tracknet-finetune/runs/basketball-v1-20ep-r3/checkpoints/TrackNet_best.pt`.
  Ball misses become explicit `lost` rows; geometry is never fabricated.
- **Stats and derived events:** the adapter writes tracks, teams, numbers, shots,
  ball rows, possession, transitions, and optional court-projected artifacts.
  Possession, pass/steal/turnover/lost-possession, and V3/V4/V5 events retain
  evidence and explicit unknown/lost/unassigned states.

### Execution modes

| Mode | Command shape | Durable job |
| --- | --- | --- |
| V2 direct | `modal run modal_app_v2.py --video-path ...` | No |
| V2 staged/full | `--staged --job-id ... --video-path ...` | Yes |
| Tracking retry | `--tracking-only --job-id ... --video-path ...` | Yes |
| Stats retry | `--stats-only --job-id ...` | Yes |
| Possession-only | `--possession-only --job-id ... --possession-engine v4\|v5` | Yes |
| Legacy rollback | `modal run modal_app.py ...` | No |

Modes are mutually exclusive. Stats-only and possession-only consume the existing
job and must not receive `--video-path`.

#### Production-style staged V5 run for `game027`

```bash
modal run modal_app_v2.py \
  --staged \
  --job-id game027-modelv2-v5-test3 \
  --video-path "dataset-tracknet/train/game_27/video/clip_027.mp4" \
  --frame-start 0 \
  --frame-end 654 \
  --segmented \
  --segment-window 150 \
  --segment-overlap 20 \
  --possession-engine v5
```

This is the full staged V5 run: it persists a durable job while processing the
complete frame range, enables bounded SAM3.1 sessions with `--segmented`, uses
150-frame windows, and reuses 20 frames between adjacent windows for seam
matching.

#### Exact V4/V5 possession-only comparison

Run both commands after the full run has produced a tracking generation:

```bash
modal run modal_app_v2.py --possession-only --job-id game027 --possession-engine v4
modal run modal_app_v2.py --possession-only --job-id game027 --possession-engine v5
```

V4 adds diagnostic control/release/acquisition and gate-reason fields. V5 means
the possession scorer/resolver profile from commit
`a4fb4e85979e13d13a3071b25ad8dad25b38f527` over the current V2 tracking, masks,
and ball rows; it is not a new tracker and does not run V4 in parallel. V5 records
`engine: v5` and omits V4-only diagnostics. V2 defaults to V3.

### Durable job Volume, retries, and download/debug

The named `v2-job-data` Modal Volume is mounted at `/v2-job-data`. Each job has
`job.json`, immutable SHA-256-addressed inputs, and immutable tracking, stats, and
possession generations. Manifests contain relative paths, sizes, and hashes;
`current.json` advances atomically. Input publication uses a locked temporary-file
rename, so a preempted worker cannot expose a partial input or overwrite another
video.

Matching completed generations are reused without recomputation. A live lease
rejects duplicate workers; an expired lease can be reclaimed. Failures are recorded
in `job.json`; retry the failed stage rather than rerunning completed stages.

Download a job and replay a selected tracking generation without inference:

```bash
mkdir -p outputs/game027-job
modal volume get v2-job-data game027 outputs/game027-job
python scripts/tracking/debug_v2.py \
  --job-id outputs/game027-job \
  --generation <tracking-generation> \
  --output outputs/game027-job/replay.mp4
```

Direct runs use `--output-dir`; staged runs are read from the downloaded job
generation. Do not replay an incomplete or hash-invalid generation.

### Limits and troubleshooting

- Crossings/occlusions can still switch IDs; ReID, Team ID, OCR, shots, and
  possession can remain unknown. Court outputs require a valid homography.
- Repository tests do not prove live Modal GPU, secret, checkpoint, or Volume
  integration.
- **Preemption/failure:** rerun the same stage; completed matching generations are
  reused.
- **`active lease exists`:** another worker owns the job; wait for the lease or
  stop that worker rather than starting a duplicate.
- **Input fingerprint/collision:** the Volume input is immutable; use the original
  bytes or a new job ID for a different video.
- **Timeout:** V2 class invocations use a 3600-second timeout. Reduce frame range
  or segment/window cost, or split tracking and stats with retryable stages.
- **Nested possession manifest self-hash warning:** a downloaded
  `possession_v3_manifest.json` can report a SHA-256 value for itself that does not
  match the file's actual hash. Treat that as a validation warning, not as a
  successful self-integrity check; the parent stats manifest remains the source
  for the recorded hash of the nested file.

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

`modal_app_v2.py` keeps its SAM3.1 observation artifact at the root of the returned bundle (`tracks.jsonl` and `manifest.json`). Its stats adapter consumes those same observations in `stats/`; it does not load SAM2, run a second player tracker, or call Roboflow to recreate V2 players.

`stats/results.json`, `stats/tracks.jsonl`, `stats/teams.json`, `stats/shot_events.jsonl`, `stats/shot_evidence.jsonl`, `stats/possession_events.jsonl`, `stats/possession_transitions.jsonl`, and `stats/ball_tracks.jsonl` use the legacy contracts and are referenced in the V2 manifest with relative paths, sizes, and hashes. For jersey numbers, V2 samples its authoritative rows, invokes the configured basketball object detector on those frames only, associates its `number` boxes to V2 players, and sends only those number crops to OCR. It never calls legacy `run_pipeline` or legacy player tracking. Number votes are keyed by authoritative `stable_player_id`, while raw IDs remain provenance. The adapter carries through V2 raw/stable IDs, boxes, team IDs, confidence, state, and frame indices. For stats geometry, rows marked `bbox_coordinate_space: "normalized"` are converted to pixel `xyxy` using the probed video dimensions; pixel or unspecified legacy rows are preserved. Ball tracking remains an independent pass: the V2 Modal image enables TrackNet by default from the mounted `tracknet-data` Volume at `/tracknet-data/tracknet-finetune/runs/basketball-v1-20ep-r3/checkpoints/TrackNet_best.pt`; local callers can enable it with `TRACKNET_ENABLED=1` and `TRACKNET_CHECKPOINT`. If TrackNet is disabled, exact V2 frames receive lost ball evidence rather than fabricated data. Shot detection keeps its own model pass, then shot evidence, possession, and transitions join to the supplied V2 player rows.
When `COURT_MODEL_ID` is configured and valid court keypoints are detected, the same V2 rows also produce optional `stats/projected_tracks.jsonl`, `stats/projected_shots.json`, and `stats/projected_shots.png`. `projected_tracks.jsonl` uses pixel foot points and legacy court-meter coordinates while retaining raw `track_id`, `stable_player_id`, and `team_id`; the shot artifacts are built from the existing shot-event and projected-shot helpers. Missing frames, detector failures, insufficient keypoints, and invalid homographies omit these pointers rather than fabricating geometry. The V2 manifest includes each emitted file as a relative path with size and SHA-256 hash.
When a projected shot has a stable shooter identity but ball-control evidence cannot authoritatively confirm possession, `projected_shots.json` selects the shooter from the projected track and exposes its `stable_player_id`. `shot_evidence.jsonl` retains that deduced identity with `status: "inferred"`, `attribution_method: "projected_shot_track"`, and `authority: "non_authoritative"`. This is a shooter attribution, not confirmed possession: possession remains an independent artifact and may be `unknown` or `lost`. Do not interpret the shooter identity as proof that the player controlled the ball. A projected shot without a stable identity leaves shot evidence unknown; raw track IDs are provenance only.
For made shots with raw detector predictions, `projected_shots.json` attributes the shooter by a vote among the detector's shooting-pose boxes (possession, jump-shot, layup/dunk, and shot-block classes) in the 2 seconds before the ball-in-basket frame. A box counts only when a ball detection sits at its hands or overhead; the winner needs a score of at least 1.0 and at least 2x the runner-up. `shooter_attribution_method` is `action_box_vote` for a clear winner, or `ambiguous` with the shooter fields null when the vote is unclear. An ambiguous shot keeps an approximate `shot_origin_court_xy_m` projected from the foot point of the detector's shot box (homography from the projected players, requires at least 4), so it still appears as an unattributed shot rather than being dropped as unprojected. All other shots keep `nearest_player_bbox_to_shot_anchor`.
For a made shot with a clear vote, `start_frame_index`, `start_timestamp_seconds`, `start_image_xy`, and `start_shot_xyxy` are corrected to the first frame of the winner's last shooting-pose run (gap tolerance 0.25s). The detector's own values stay in `detector_start_frame_index` and `detector_start_timestamp_seconds`, and `start_frame_source` is `action_box_release` or `shot_detector`. `shot_events.jsonl` is not rewritten: `shot_evidence.jsonl` windows and `projected_shots.json` use the release start, while `shot_events.jsonl` keeps the detector start. The backend reads the shot time from `projected_shots.json`.
Shot extraction also filters detector noise: an edge-truncated shot box is ignored when every confidently detected ball is more than 2 box-heights away; a ball-in-basket detection is ignored when a ball with confidence >= 0.5 is detected and every such ball is more than 10 basket-widths away; and a `missed` event built from a single detector frame is dropped, so `event_id` values can have gaps. These are heuristics tuned and checked on three clips; they are not validated accuracy claims.

For a real Modal validation run, compare the same event across the artifacts rather
than checking only that a shot exists. At minimum, compare `event_id`,
`shooter_track_id`, and `shooter_stable_player_id` between `projected_shots.json`
and the corresponding `shot_evidence.jsonl` row, then verify the shot-evidence
fields (`status`, `attribution_method`, `authority`). Check possession separately
in its possession artifact; an `unknown`/`lost` possession result is compatible
with inferred shooter evidence and must not be upgraded to `confirmed`.

The V2 Modal image sets `POSSESSION_ENGINE=v3`, so its existing stats adapter also writes the additive V3 artifacts. Run it with:

### Restartable V2 stages

Staged execution is opt-in. These native Modal CLI flags run the deployed methods through
the normal `modal run` runner, so remote logs and failures remain visible:

```bash
# Command 1: persist tracking artifacts/bundles in one durable job
modal run modal_app_v2.py --tracking-only --job-id game-03 \
  --video-path clips/game_03.mp4

# Command 2: run stats from the persisted tracking bundle
modal run modal_app_v2.py --stats-only --job-id game-03
```

The two commands are equivalent to the existing `--staged` one-command flow,
but make the tracking and stats retry boundary explicit.

Tracking and stats use the `v2-job-data` Modal Volume at `/v2-job-data`. Each job stores
`job.json`, the immutable input, and immutable `tracking/generations/<fingerprint>/`,
`stats/generations/<fingerprint>/`, and `possession/generations/<fingerprint>/`
directories. `current.json` is advanced atomically;
incomplete or hash-invalid generations are ignored. Use the local replay command with an
explicit job directory and generation to render saved tracking without inference:

Direct stage retries may call `track_stage(job_id=...)` and `stats_stage(job_id=...)` separately;
active leases reject duplicate workers, and matching completed generations are returned without
recomputation. The local `main` command remains monolithic unless a caller explicitly uses the
`--tracking-only`, `--staged`, `--stats-only`, or `--possession-only` flag. With none of these
flags, the monolithic behavior is unchanged.

Compare possession engines without rerunning tracking or stats on the same durable job:

```bash
modal run modal_app_v2.py --possession-only --job-id game-03 --possession-engine v4
modal run modal_app_v2.py --possession-only --job-id game-03 --possession-engine v5
```

```text
python scripts/tracking/debug_v2.py --job-id /v2-job-data/JOB_ID \
  --generation GENERATION --output replay.mp4
```

The current timeout target remains 3600 seconds per invocation. Confirm GPU seconds and
Volume retention in Modal before enabling staged routing in a caller. To roll back, omit
staged routing and use `process`; no staged artifacts are consumed.

```bash
modal run modal_app_v2.py --video-path clips/game_03.mp4 \
  --frame-start 60 --frame-end 130 --output-dir outputs/v2-game
```

Expected V3 outputs under the returned `stats/` directory are `possession_v3_manifest.json` and generation-scoped `possession_v3/` artifacts for possession rows, transitions, and events. The legacy `modal_app.py` image does not set this variable and keeps the global default unchanged.

### Explicit V4/V5 execution

V2 accepts `v3`, `v4`, or `v5`; the selected value is passed into the Modal
container's `PipelineConfig`, so a local `POSSESSION_ENGINE` cannot silently
select a different remote engine. V4 runs only the V4 scorer, not a parallel V5
scorer. V5 uses the possession scorer/resolver behavior from commit
`a4fb4e85979e13d13a3071b25ad8dad25b38f527`, while reusing the current V2
tracking, masks, ball tracks, and staged stats flow. It does not emit V4-only
diagnostics. The root manifest, `stats/results.json`, and
`stats/possession_v3_manifest.json` record the selected engine. V3 remains the
default for V2; `modal_app.py` is unchanged.

Full segmented V4 command:

```bash
SAM31_V2_SEGMENTED=1 modal run modal_app_v2.py \
  --video-path clips/game_03.mp4 --frame-start 60 --frame-end 130 \
  --segment-window 32 --segment-overlap 1 \
  --possession-engine v4 --output-dir outputs/v2-segmented-v4
```

Expect `stats/possession_v3_manifest.json`, generation-scoped V4 possession
rows/transitions/events, and V4 diagnostics referenced from `stats/results.json`
and the root `manifest.json`.

Run the same persisted tracking generation with V4 and V5 into separate stats
generations (or use separate job IDs when comparing complete jobs):

```bash
modal run modal_app_v2.py --tracking-only --job-id game-03-possession-compare \
  --video-path clips/game_03.mp4
modal run modal_app_v2.py --stats-only --job-id game-03-possession-compare \
  --possession-engine v4
modal run modal_app_v2.py --stats-only --job-id game-03-possession-compare \
  --possession-engine v5
```

The two stats requests keep distinct fingerprints and generation directories;
V4 has its diagnostics artifact, while V5 records `engine: v5` without it.

Rollback V2 with `--possession-engine v3`. Use `legacy` only when invoking the
general local pipeline/config directly, for example
`POSSESSION_ENGINE=legacy python -m pipeline.cli clips/game_03.mp4`; V2's
explicit option does not accept `legacy`. Calibration is offline and is not
automatically run by Modal.

### Conservative temporal BBox sanitizer

Set `BBOX_SANITIZER_ENABLED=1` to run the offline V2 sanitizer before ReID and team identification; leave it at `0` for compatibility runs. Correction precedence is **mask cleanup → temporal interpolation → unresolved**. Mask cleanup uses connected components and conservative temporal evidence, writes optional `clean_mask_ref` artifacts under `masks_clean/`, and leaves the raw mask and `raw_bbox` available for provenance. Component areas are compared in the same units as the mask: normalized player boxes are converted to mask-pixel units before validation. If cleanup cannot produce one defensible component, the existing temporal interpolator may correct only a bounded multi-signal gap; otherwise the giant raw bbox remains unchanged and is marked `bbox_unresolved`. Initial calibration values are `BBOX_HISTORY_SIZE=15`, `BBOX_MAX_INTERPOLATION_GAP=2`, and `BBOX_SIGNAL_THRESHOLD=3.0`; tune them from reported corrected and unresolved counts, not from a single area threshold.

### Bounded SAM3.1 V2 propagation

The default V2 path remains the single full-video session for rollback. Enable
bounded sessions explicitly; each temporary clip is prompted at local frame zero,
overlap is owned by the later clip, and only the deduplicated global observations
reach ReID and Team ID:

```bash
SAM31_V2_SEGMENTED=1 modal run modal_app_v2.py \
  --video-path clips/game_03.mp4 --frame-start 60 --frame-end 130 \
  --segment-window 32 --segment-overlap 1 --output-dir outputs/v2-segmented
```

At each seam, current local IDs receive the previous stitching identity only
after a unique physical match: mask IoU is primary, with bbox IoU/center fallback.
`SAM31_V2_SEAM_MIN_SCORE` (default `0.5`) and
`SAM31_V2_SEAM_AMBIGUITY_MARGIN` (default `0.05`) tune the gate. Missing,
crossing, or ambiguous evidence fails closed, retaining separate fragments
instead of guessing an identity. In segmented output, `raw_track_id` is the
namespaced segment-local SAM ID; `local_raw_track_id` is the original model
local ID, and `raw_observation_key` preserves its original unique observation
key. Seam grouping uses the separate `canonical_raw_track_id`. Mask keys remain
lossless raw observation keys, and the later segment still owns duplicate
overlap rows.
If the later owner has no usable mask, its observation remains the owner but is
marked mask-missing; the earlier mask is discarded for that key rather than
being reused.

`--segment-window` and `--segment-overlap` may also be set with
`SAM31_V2_SEGMENT_WINDOW` and `SAM31_V2_SEGMENT_OVERLAP`. The default overlap is
one frame. SAM3.1 remains configured with `max_num_objects=16`. A segmented
session may temporarily expose 17–22 active IDs; the lowest-ranked IDs are
deterministically pruned before the existing 16-object guard, using confidence,
active frame count, and temporal consistency. More than 22 IDs is rejected.
Overflow counts, dropped raw IDs, ranking reasons, and grace usage are recorded
alongside seam diagnostics. Roll back by omitting `SAM31_V2_SEGMENTED=1` to use
the existing full-video path.

Render possession from a saved V2 bundle with the root SAM3.1 video overlay and the legacy stats events/tracks:

```bash
./.venv/bin/python scripts/tracking/debug_v2.py \
  --video-path path/to/input.mp4 \
  --artifact-dir outputs/v2-bundle \
  --stats-dir outputs/v2-bundle/stats \
  --output outputs/v2-bundle/possession-debug.mp4
```

The renderer matches possession events by exact `frame_index`. It displays only the event's authoritative `stable_player_id`, `team_id`, or explicit `UNKNOWN`/`LOST`/`UNASSIGNED` status; it never infers a holder.

To inspect V3 plus recognized V2 jersey numbers without loading legacy possession
overlays, provide the V3 directory and the stats number summary directly:

```bash
./.venv/bin/python scripts/tracking/debug_v2.py \
  --video-path path/to/input.mp4 \
  --artifact-dir outputs/v2-bundle \
  --v3-dir outputs/v2-bundle/stats/possession_v3/generations/<generation-id> \
  --numbers-json outputs/v2-bundle/stats/numbers.json \
  --output outputs/v2-bundle/v3-number-debug.mp4
```

`--numbers-json` is optional. When `--stats-dir` is used, an existing
`<stats-dir>/numbers.json` is picked up automatically; legacy possession overlays
still load as before. Without a numbers file, rendering remains backward-compatible.

Exact V3-only debug command:

```bash
./.venv/bin/python scripts/tracking/debug_v2.py \
  --video-path clips/game_03.mp4 \
  --artifact-dir outputs/v2-segmented-v4 \
  --v3-dir outputs/v2-segmented-v4/stats/possession_v3/generations/<generation-id> \
  --numbers-json outputs/v2-segmented-v4/stats/numbers.json \
  --output outputs/v2-segmented-v4/v3-number-debug.mp4
```

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
- `possession_transitions.jsonl` with conservative pass, steal, turnover, and lost-possession transitions
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
possession from proximity when candidates are tied, or assigns a holder to a
missing frame.

`possession_transitions.jsonl` is schema version 1 and uses the event vocabulary
`pass`, `steal`, `turnover`, and `lost_possession`. A `pass` requires different
authoritative players with the same non-null team; a `steal` requires different
authoritative players with different non-null teams. Both must be within
`POSSESSION_TRANSITION_MAX_GAP` frames (default `15`) and retain source-row and
evidence references. Same-player, team-null, ambiguous, and out-of-gap changes
are not passes or steals. Loss rows are `uncertain` unless the source explicitly
marks terminal loss metadata; a temporary `unknown` or `lost` row is never, by
itself, proof of a turnover.

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
| `SHOT_RIM_TRAJECTORY_OUTCOME` | `1` | Heuristic rim-trajectory outcome described under "Shot events"; set `0` to disable. |
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
| `NUMBER_EVIDENCE_ENABLED` | `1` | Enable V2 detector-backed jersey-number evidence; disabled skips predictor/detector initialization. |

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

Shot type: `shot_type` is decided when the event is finalized (made, missed or unknown), by a confidence-weighted vote over the last burst of shot boxes. A burst is a run of shot-signal frames with gaps under 0.3 s (9 frames when a timestamp is missing); ties go to `jump_shot`. A single stray layup box among many jump-shot boxes no longer relabels the event, and a putback after an earlier jump burst is still a `layup_dunk`. Whether a frame opens a layup or jump candidate is unchanged.

Rim-trajectory outcome (`SHOT_RIM_TRAJECTORY_OUTCOME`, default on; `0` disables it): a heuristic that upgrades a `missed` (timeout or new shot) or `unknown` event to `made` when the ball is seen above the detected `rim`, then inside the rim zone, then leaves the zone below and centred, using TrackNet `detected` points and detector balls with confidence >= 0.6. It runs after the shot pass, rewrites `shot_events.jsonl` atomically, and adds `outcome_source` (`ball_in_basket`, `rim_trajectory` or `shot_detector`), `rim_pass_frame_index` and `rim_exit_frame_index`; the upgraded event has `resolution_reason: "rim_trajectory"` and keeps `ball_in_basket_frame_index` null. Any ambiguous trajectory keeps the existing outcome (prefer unknown over made). The same flag also checks `made` events resolved by `ball_in_basket`: the detector's `ball-in-basket` class is known to fire on the hoop/net with no ball present, so such an event is kept only if a ball point (TrackNet `detected`, or detector ball with confidence >= 0.5) lies within 2.0 rim widths of the rim centre in the 60 frames before to 10 frames after the basket frame, and gets `ball_near_rim_evidence: true`. Otherwise it becomes `unknown` (not `missed`: absent evidence is not proof of a miss) with `resolution_reason: "ball_in_basket_unconfirmed"` and `ball_near_rim_evidence: false`; `outcome_source` stays `ball_in_basket`. With no rim box around the detection the event is left alone, and shooter attribution treats a downgraded event as non-made. Shooter attribution anchors on `rim_pass_frame_index` for these events. The thresholds were tuned on five local clips only and are not a validated accuracy claim. With the flag off, the written files are unchanged.

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

V2 does not re-run legacy player tracking. On every sampled V2 frame, the configured
basketball detector produces `player` and `number` boxes; the existing geometry
association assigns each number box to one V2 `stable_player_id`; only that box is
sent to OCR. `FRAME_STRIDE` controls both detector and OCR sampling, keeping hosted
inference cost bounded. The dummy backend is safe for dry runs and produces unknown
votes. The V2 Modal image uses the configured `PLAYER_MODEL_ID` detector, the
`NUMBER_PREDICTOR_ROBOFLOW_MODEL_ID` OCR model, and the attached
`scoutai-roboflow` secret. The torso ROI is only a local compatibility fallback
when no detector is supplied, and its provenance is recorded separately.

What it does today:

1. On each sampled frame, call the configured basketball object detector and collect overlapping `number` detections as number evidence.
2. Crop the matched `number` box and send it to the hosted Roboflow OCR model when `NUMBER_PREDICTOR_BACKEND=roboflow-hosted`.
3. Store every OCR attempt in `number_predictions.json` and `number_prediction_cases.jsonl`.
4. Aggregate accepted reads per track at the end of the run and write confirmed/ambiguous results to `numbers.json`.
5. When an annotated video is requested, show confirmed jersey numbers in labels. If no number is confirmed, the overlay stays generic instead of falling back to internal track IDs.

Confirmation is intentionally conservative:

- A track is `confirmed` only when one number wins cleanly with at least two
  accepted reads.
- Conflicting accepted reads keep the track `ambiguous` instead of guessing.
- A single accepted valid read remains ambiguous until a second matching read is available.
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
