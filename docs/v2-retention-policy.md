# v2-job-data retention policy (draft, issue #6)

Status: DRAFT. Every number below is **proposed, owner to confirm**. Only a read-only
report exists today; nothing deletes data.

## Volume layout (from `modal_app_v2.py`, `pipeline/results.py`)

Per job in `v2-job-data`: `job.json`, `.job.lock`, `inputs/<sha256>.mp4` (immutable),
and per stage `tracking|stats|possession`: `current.json` plus
`generations/<id>/manifest.json` and artifacts. Leftovers from interrupted writes:
`<stage>/.stage-*`, `<stage>/.current.json.tmp`, `.job.json.<uuid>.tmp`, `inputs/.*.tmp`.
A lease is `job.json["lease"] = {"owner", "expires_at"}` (TTL 1800 s, `JOB_LEASE_SECONDS`).

## Proposed defaults (owner to confirm)

| Item | Proposed | Notes |
|---|---|---|
| Generations kept per job/stage | 2 (current counts as one) | `--keep-generations` |
| Failed / abandoned in-progress jobs | 7 days since `job.json` change | `--max-age-days` |
| Unreferenced raw inputs | 30 days | `--input-max-age-days` |
| Orphan tmp/staging entries | 24 h minimum age | `--orphan-min-age-hours` |
| Debug environment | shorter: 3 days jobs, 1 generation, 7 days inputs | production uses the rows above |

Completed, healthy jobs are not expired by age in this draft; only their surplus
generations are. Whether finished jobs should ever expire is open.

## Protected (never candidates)

- the target of every stage `current.json` and the `tracking_generation`, `stats_generation`
  and `possession_generations` values in `job.json`;
- the newest N generations per stage;
- any tracking generation referenced (`tracking_generation`) by a retained stats/possession manifest;
- every input named by `job.json` or a retained generation's `input_fingerprint`;
- whole jobs with `lease.owner` set and `lease.expires_at` in the future;
- jobs in `created|tracking|stats|possession` state younger than the failed-job age;
- files the report cannot classify (`other`), and tmp/staging entries younger than 24 h.

## Eligible

Non-current generations beyond keep-N (and not referenced), generations without a complete
manifest, unreferenced old inputs, aged tmp/staging orphans, and every non-`other` item
of a failed or abandoned job older than the max age (unless it holds an active lease).

## What the report does

`python -m scripts.ops.v2_retention_report --root <tree> [--json] [--now EPOCH|ISO]`
lists every job and generation with bytes, file count, PROTECTED/ELIGIBLE and a reason,
plus totals. Output is deterministic for a given tree and `--now`. Each file belongs to
exactly one item, so totals add up. It contains no delete, move or write code path; a test
checks the tree is byte-identical (content and mtime) after a run. Files inside a
generation that its manifest does not list are noted as `unlisted_files` only.

## What it does not do

No deletion, no Modal access, no Volume locking (it reads a snapshot and can race a live
writer: run it on a copy, or treat a plan as stale once leases change), no metrics emission.
Live storage growth for issue #5 is not measured yet.

## How a future deletion step must work

- Input is an explicit list of job/generation/input identifiers taken from a saved plan,
  never a path glob or "everything eligible".
- Dry-run is the default; deletion needs an explicit flag.
- Re-verify each target immediately before removal under the job lock (`.job.lock`):
  lease still expired, pointer/job.json still not naming it, no retained manifest referencing it.
- Remove a generation as a whole directory; never edit retained manifests.
- Idempotent: a missing target is a no-op, not a failure; reruns produce no further change.
- Log every decision and failure (job, stage, id, files, bytes, reason) and a summary of
  deleted bytes/files/generations; failures do not stop the other targets.
- Recovery: deletion is permanent. Modal Volumes have no per-file undo, so recovery means
  re-running the pipeline from the retained input, which is why inputs outlive generations.
  Owner to decide whether to snapshot/copy before the first destructive run.

## Open for the owner

- Confirm or change every default above, including debug versus production values.
- Does a retained input matter after its generations are gone (reproducibility versus cost)?
- Should completed jobs ever expire, and should a failed job with a valid earlier current
  generation lose it after 7 days?
- Where cleanup runs (scheduled Modal function or manual) and how metrics are surfaced.
- How to snapshot safely given concurrent writers.
