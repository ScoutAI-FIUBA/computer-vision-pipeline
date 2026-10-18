# v2-job-data retention policy (draft, issue #6)

Status: DRAFT. Every number below is **proposed, owner to confirm**.

Principle: keep what is useful, delete what is objectively useless. When in doubt a path
is PROTECTED. `job.json` has no owner field, so deletion decisions belong to whoever
created the job.

## Scope

Only the `v2-job-data` Modal Volume (`<job>/job.json`, `.job.lock`, `inputs/<sha>.mp4`,
`{tracking,stats,possession}/{current.json,generations/<id>}`).

NEVER in scope and never touched: `tracknet-data` (fine-tuning data and checkpoints; the live
pipeline reads its ball model from there), `scoutai-model-cache`, R2 source videos, and any
fine-tuning data, checkpoints, weights or datasets. Code enforces this: any path with a
component matching `tracknet`, `finetune`, `fine-tune`, `fine_tune`, `checkpoint`, `dataset`,
`weights`, `model-cache`, `model_cache`, `runs`, `*.pt`, `*.pth`, `*.ckpt`, `*.onnx` is
PROTECTED (`protected_data_class`), and the tools refuse a `--root` or volume with such a name.

## Classes (`scripts/ops/v2_retention_report.py`, read-only)

| Class | What | Proposed default |
|---|---|---|
| ELIGIBLE (1) | orphan temp files: `.stage-*`, `.*.tmp`, `.current.json.tmp` | older than 7 days (`--orphan-min-age-days`) |
| ELIGIBLE (2) | generations/orphans of a failed or never-completed job, no live lease | job older than 30 days (`--failed-job-min-age-days`) |
| ELIGIBLE (3) | superseded generations: non-current, not referenced by a retained generation | beyond newest 2 (current counts) and older than 30 days (`--keep-generations`, `--generation-min-age-days`) |
| REVIEW | mask-only compaction of complete jobs older than 90 days | NOT implemented, open owner decision |
| PROTECTED | everything else | always |

Complete jobs with a current pointer are PROTECTED at any age. In a failed job, current
pointer targets, `job.json`, `.job.lock` and inputs stay PROTECTED (inputs are never
deleted), so only a small skeleton remains. REVIEW exists but no tool acts on it: dropping
masks would break hash validation, replay and stats re-runs.

## Applying (`scripts/ops/v2_retention_apply.py`, the only deleting code)

Locked to the volume `v2-job-data`; `--only JOB[,JOB]` is mandatory; dry-run by default.
Only ELIGIBLE generations and orphans matching an exact path allowlist can be removed, and
each one is re-validated against the live volume first (skipped if it became current,
referenced, leased, or its failed job resumed).

1. Download a snapshot of the volume (or of the jobs concerned) to a local directory.
2. Run the planner on it with `--json` and save the plan; it contains a plan hash.
3. Review the plan with the creator of each job; pick explicit job ids, no wildcards.
4. Run the apply tool with `--plan`, `--only` and no `--apply`: a dry run printing what would go.
5. Repeat with `--apply --yes --plan-hash <hash from the plan>`.
6. Keep the JSONL log it writes (paths, bytes, files, failures) with the issue.

Who may run it: the creator of the jobs listed in `--only`, or someone they ask. Nobody
runs it for other people's jobs without their agreement.

## What it cannot know

- Not race safe: the plan comes from a snapshot; the re-validation narrows but cannot close the window.
- No ownership data in `job.json`, and age comes from file times (a fresh download resets them;
  prefer snapshot metadata).
- Orphan age is not re-checked on the live volume.
- The Modal adapter is untested against the real volume.

## Open questions for the owner

- Add an `owner` (and `created_at`) field to `job.json`?
- Snapshot cadence for planning?
- Should inputs of failed jobs ever be removed, and mask-only compaction at 90 days?
- Confirm the numbers above (7 / 30 / 2 / 30 days).
