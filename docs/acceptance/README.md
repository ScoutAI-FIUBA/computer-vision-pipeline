# V2 acceptance reports

`scripts/ops/acceptance_report.py` turns one staged V2 Modal run into a redacted
acceptance report (GitHub issue #5). It is free and local: it only reads the
`modal run` log and the downloaded job folder. It never calls Modal.

## Usage

```bash
# 1. Capture the run log
modal run modal_app_v2.py ... 2>&1 | tee outputs/<job>/run.log

# 2. Download the job folder (contains job.json, tracking/, stats/, inputs/)

# 3. Generate the report
python scripts/ops/acceptance_report.py \
  --log outputs/<job>/run.log \
  --job outputs/<job>-job/<job> \
  --commit "$(git rev-parse HEAD)" \
  [--replay-output <debug_v2 render dir or .mp4>] \
  [--notes <human findings.md>] [--also-log <other run.log> ...] \
  [--repeat-log <repeat run.log>] [--retry-log <stats-only retry run.log>] \
  [--retention-policy <policy.md>] \
  [--json] [--output <report file>]

# 4. Compare two JSON reports (e.g. before/after a fix); always exits 0
python scripts/ops/acceptance_report.py --diff before.json after.json
```

Markdown is the default; `--json` emits the full machine-readable report
including the SHA-256 of every listed artifact (Markdown lists only the largest
`--max-files` per generation). Exit code is 1 when a downloaded file is missing
or its SHA-256 differs from the manifest.

## What it reports

Job id, commit, input clip metadata (size, recomputed sha256 vs `input_fingerprint`,
frames/fps/resolution from the tracking manifest), GPU seen in the log, tracking and
stats generation ids and job state, log-reported stage durations, artifact manifest
summary, a hash verification pass over the downloaded files (mismatches and missing
files), deduplicated warnings, bytes per generation, and a checklist mirroring the
issue's acceptance criteria with PASS / FAIL / NOT_RUN / NOT_MEASURED and evidence.

## Evidence sections

- **Tracking identity**: `tracking_identity` and `acceptance_reasons` from
  `tracking-manifest.json`, `identity_excluded_observations` / `_keys` (printed as
  `absent` when the manifest predates them, never as 0), and the count of `tracks.jsonl`
  rows with and without `stable_player_id` (stats generation first, tracking generation
  as fallback). Checklist row "Stable player identity present in tracks" is FAIL when
  no row has an id or identity is `rejected`.
- **External integration evidence**: "Jersey number OCR ran" is PASS only when
  `results.json` has a non-null `numbers_file` that exists on disk; otherwise FAIL. A
  configured backend alone proves nothing. The "Configured external model credentials..."
  row is derived per integration from the job artifacts (listed in the `integrations`
  JSON key): hosted shot detector (`shot_raw_predictions.jsonl` rows), court model
  (`projected_tracks.jsonl` rows), TrackNet (non-lost `ball_tracks` rows), team ID
  (`teams.json` assignments), hosted jersey-number OCR (numbers file plus a number
  artifact in the manifest; `unavailable: not run in this job` when the backend is not
  roboflow-hosted) and SAM 3.1 tracking (non-empty tracking package). PASS when none
  failed (unavailable ones are listed explicitly), FAIL when a configured integration
  produced no output. Without any hosted-model artifact (shot or court rows) the row stays
  NOT_MEASURED. Only names and counts are printed, never credential values.
- **Outcome summary**: shot events by `outcome` and `outcome_source`, projected shots by
  `shooter_attribution_method`, shots with no shooter, ball rows lost vs non-lost.
  Facts only; fields a run did not emit show as `absent`.
- **Replay**: an `.mp4` (given directly or found inside the directory) is opened with
  cv2 and its frame count compared with the tracking `frame_bounds` (inclusive). PASS
  only if equal; FAIL shows both numbers. Without `--replay-output` it stays NOT_RUN.
- **Operational findings**: `--notes FILE` is embedded verbatim after redaction; nothing
  is written when it is not given.
- **GPU**: exact GPU strings per log; `--also-log` lists strings from extra logs of the
  same job and flags when variants differ.
- **`--diff A.json B.json`**: table of changed checklist statuses and key numbers
  (identity status, stable-id rows, numbers_file, outcome counts, generation ids).

## Repeat, retry and retention evidence

All optional; absent means the row keeps its NOT_RUN / NOT_MEASURED value.

- `--repeat-log FILE`: `modal run` log of repeating the identical command on the same
  job. PASS only if (a) the final JSON `generation` and `tracking_generation` equal the
  job's current ids, (b) the log has none of `propagate_in_video`, `started new session`,
  `Running add_prompt`, `frame loading (OpenCV)`, and (c) it has no `[v2-timing] stage=`
  line (a pure reuse writes none). Otherwise FAIL naming the failed conditions. Attempts
  from `job.json` are reported, not judged. Evidence also shows the client wall time and
  "preempted and restarted" when the log has `terminated due to preemption`.
- `--retry-log FILE`: log of a `--stats-only` retry. PASS only if `job.json` has a
  recorded error (shown redacted) or the log shows an expired/failed state, the job is
  `complete` with `lease` null, stats attempts >= 2, and the log's final JSON is
  `complete` with the job's `tracking_generation` and no tracking-activity markers.
  Evidence includes the attempts and the retry's `stage=stats` elapsed seconds.
- `--retention-policy PATH`: PASS when the file exists and is non-empty; evidence is its
  file name, size and the per-job published bytes. The content is not embedded.

The JSON `evidence_inputs` object records which of these were given; `--diff` ignores it.

## Performance and cost

The "Performance and cost" section reads what the pipeline records per stage
generation: `stage_metrics.json` (schema `v2-stage-metrics-1`) inside
`tracking/`, `stats/` and `possession/generations/<id>/`, and the `[v2-timing]` lines of
the `modal run` log (`stage=... elapsed=... timeout_margin=... gpu=...` and
`job id=... total_wall_seconds=...`). When several generations exist only the one in
`current.json` is used; the log lines are the fallback and a cross-check. It reports:

- per-stage elapsed and phases, sum of stage elapsed (container-side), client-side total
  wall seconds and their difference (cold start, model load, transfer overhead);
- smallest timeout margin across stages and the percentage of the function timeout used;
- GPU name and memory, model/API call counts per stage and totals (a `null` count is
  printed as `unknown`, never as 0; a total with unknown stages is a lower bound);
- attempts and recorded errors from `job.json` (a stage that reused a generation has no
  new metrics and is listed as such);
- number and bytes of published generations per stage, and the downloaded job size.

Runs recorded before the pipeline wrote these files show `NOT_MEASURED (run predates
stage metrics)`; nothing is invented.

Cost options (all estimates, not the invoice; the Modal dashboard is the source of truth):

| Option | Meaning |
|---|---|
| `--gpu-rate-usd-per-hour R` | No default. Without it cost is `NOT_MEASURED` (pass the rate from the Modal plan). |
| `--video-seconds S` | Video duration. Default: tracking `frame_bounds` (inclusive) / video fps from the tracking manifest, else `NOT_MEASURED`. |
| `--volume-rate-usd-per-gib-month R` | Needed for the storage cost; otherwise `NOT_MEASURED`. |
| `--volume-included-gib N` | Storage included in the plan (default none = 0). Cost is for this job's bytes beyond it, 0 when within it. |
| `--volume-listing-before/after FILE` | Volume listings to compute growth, see below. |

Estimated cost = (sum of stage elapsed seconds / 3600) x rate. This lower figure is GPU
time only: container CPU/memory and cold start are excluded. The upper bound uses the
client-side wall seconds instead. Cost per video minute = cost / (video seconds / 60).

Volume growth: save the job listing before and after the run and pass both files. Accepted
formats: a JSON object `{"path": bytes, ...}`, or a plain text copy of
`modal volume ls v2-job-data <job>` output (table borders are ignored; rows typed `dir`
are skipped; a size is either a human size such as `1.5 MiB`, which makes the result
approximate, or a plain integer of bytes). Only file rows are summed, so list the
directories that hold the files. Unreadable or size-less files make growth
`NOT_MEASURED`; without listings the generation bytes are shown as "published bytes
(as downloaded)".

Worked example with made-up numbers (an example only, not a measurement). The plan rate
for an A100-80GB is 2.50 USD/hour here:

```
tracking 120.0 s + stats 60.5 s -> sum of stage elapsed 180.5 s
client-side total wall           250.0 s  (difference 69.5 s: cold start, model load, transfer)
video 120 s (2 min)
cost (GPU stage time) = 180.5 / 3600 * 2.50 = 0.1253 USD
upper bound (wall)    = 250.0 / 3600 * 2.50 = 0.1736 USD
cost per video minute = 0.0627 USD (lower) .. 0.0868 USD (upper)
```

```bash
python scripts/ops/acceptance_report.py --log run.log --job <job dir> \
  --gpu-rate-usd-per-hour 2.50 --video-seconds 120 \
  --volume-listing-before before.txt --volume-listing-after after.txt
```

`--diff` also compares per-stage elapsed, total elapsed and wall time, smallest timeout
margin, estimated cost, volume growth and published bytes. The JSON report carries all
of this in the `performance` object.

## What the logs do NOT contain

These stay NOT_MEASURED unless the inputs above provide them, with their source:

| Item | Where it comes from |
|---|---|
| Billed GPU runtime / invoice | Modal usage export or dashboard |
| Volume growth | `--volume-listing-before/after` (`modal volume ls` before and after the run) |
| Cost | `stage_metrics.json` or `[v2-timing]` plus `--gpu-rate-usd-per-hour` |
| Model/API call counts | `api_calls` in `stage_metrics.json`, else Roboflow dashboard |
| Modal image version | `modal app list` / deploy output |

Steps never exercised by the provided run stay NOT_RUN: SAM3.1 smoke test (unless
the log shows a result), repeat-stage reuse (unless `--repeat-log`), stats-only retry
(unless `--retry-log`), download and replay (unless `--replay-output` exists).

## Redaction

Applied to all report text: Modal app URLs, token-like strings (`gho_`, `ghp_`,
`github_pat_`, `sk-`, `Bearer ...`), email addresses and absolute home paths. The
report states that redaction was applied and how many replacements were made
(including matches found only in the raw log). Review before committing: redaction
is pattern based and cannot recognise arbitrary secrets.

Generated reports are not committed by the tooling; attach or commit them
deliberately once reviewed.
