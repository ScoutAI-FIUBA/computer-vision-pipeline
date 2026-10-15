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
  configured backend alone proves nothing, so the older "Configured external model
  credentials..." row stays NOT_MEASURED (configuration and indirect evidence) and
  points to the OCR row for the artifact-based verdict. Only env var names are printed.
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

## What the logs do NOT contain

These are always reported as NOT_MEASURED, with their source:

| Item | Where it comes from |
|---|---|
| GPU runtime / billing, cost, cost per video minute | Modal usage export or dashboard |
| Timeout margin | configured function timeout minus Modal function-call duration |
| Total job duration | Modal function-call duration (the CLI log only has stamps for SAM3) |
| Volume growth | `modal volume ls` before and after the run |
| Model/API call counts | Roboflow dashboard |
| Modal image version | `modal app list` / deploy output |

Steps never exercised by the provided run stay NOT_RUN: SAM3.1 smoke test (unless
the log shows a result), repeat-stage reuse, stats-only retry, download and replay
(unless `--replay-output` exists).

## Redaction

Applied to all report text: Modal app URLs, token-like strings (`gho_`, `ghp_`,
`github_pat_`, `sk-`, `Bearer ...`), email addresses and absolute home paths. The
report states that redaction was applied and how many replacements were made
(including matches found only in the raw log). Review before committing: redaction
is pattern based and cannot recognise arbitrary secrets.

Generated reports are not committed by the tooling; attach or commit them
deliberately once reviewed.
