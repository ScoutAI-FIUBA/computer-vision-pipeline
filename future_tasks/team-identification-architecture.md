# Team identification architecture

This file documents the team-identification flow that is actually implemented today in `pipeline/`.

## Current status

Team identification is implemented, end-of-run, and still experimental.

It is already wired into the deployed pipeline:

1. Team evidence is collected during tracking.
2. Final team assignment happens after tracking finishes.
3. `tracks.jsonl` is rewritten with final `team_id`s.
4. `teams.json` stores assignments and diagnostics.
5. `annotated.mp4` is rendered after final assignment when requested.

## Actual pipeline flow

Relevant files:

- `pipeline/runner.py`
- `pipeline/team_embedder.py`
- `pipeline/team_evidence.py`
- `pipeline/team_id.py`

Run flow:

1. Detect players on the first frame of each segment.
2. Match detections to active tracks.
3. Propagate tracks with SAM2 through the segment.
4. Collect team evidence on the detector frame.
5. Collect additional team evidence on propagated frames every `team_sample_stride` offsets.
6. Reject bad samples, then keep only the top `max_team_samples_per_track` accepted samples per track.
7. Cluster all selected samples globally.
8. Vote one team label per track.
9. Leave the track unassigned when it is under-sampled, tied, or too close to the runner-up cluster.

## What each backend really does

| Backend | Current behavior |
|---|---|
| `none` | Produces no team features; every track stays unassigned. |
| `colour` | Builds a jersey-focused colour histogram only. |
| `siglip` | Builds the same jersey-focused colour histogram and also computes a SigLIP embedding from the same crop. |

Important: the current code does not run a staged "colour-first, then SigLIP refinement" decision tree.

`identify_teams()` clusters one vector per selected sample:

- it uses `embedding` when present;
- otherwise it falls back to `color_signature`.

So with the default `siglip` backend, clustering is embedding-led, not a two-pass hybrid classifier.

## Current decision logic

`pipeline/team_id.py` currently does this:

1. Optionally reduce higher-dimensional sample vectors with UMAP when there are enough samples.
2. Run deterministic K-means over the selected sample vectors.
3. Count cluster votes per track.
4. Assign a team only when:
   - the track has at least `min_team_samples_per_track` selected samples;
   - one label has a strict majority;
   - the vote margin is greater than `team_assignment_margin`.

Current rejection reasons from voting:

- `under_sampled`
- `no_majority`
- `ambiguous_vote`

## Current configuration knobs

| Knob | Meaning |
|---|---|
| `team_embedder_backend` | `siglip`, `colour`, or `none` |
| `team_embedder_model_id` | SigLIP model id when using the `siglip` backend |
| `team_count` | Cluster count |
| `min_team_samples_per_track` | Minimum selected samples required before assignment |
| `team_sample_stride` | Propagated-frame evidence sampling interval |
| `min_team_sample_quality` | Minimum quality score before a sample counts |
| `max_team_samples_per_track` | Top-K cap per track before clustering |
| `team_assignment_margin` | Minimum winning vote margin |
| `team_debug_enabled` | Whether to write crop/mask debug artifacts |

## Outputs

| File | What it contains |
|---|---|
| `tracks.jsonl` | Final per-frame rows, rewritten with `team_id` after assignment |
| `teams.json` | Assignments, per-track sample counts, diagnostics, and unassigned tracks |
| `team_debug.json` | Track-level evidence summary |
| `team_debug/` | Saved crops and masks for manual inspection |

## Main limitations today

- Team evidence quality is still sensitive to occlusion, crop contamination, and mask quality.
- The default `siglip` path still depends on one jersey-focused crop, not a richer multi-region representation.
- The current result has no explicit per-track confidence beyond vote diagnostics.
- The implementation is wired and test-covered, but not yet reliable enough to treat as production-ready.

## Next useful work

1. Improve evidence quality before adding more classifier logic.
2. Validate on multiple clips instead of tuning one clip at a time.
3. Add a clearer confidence story only if the current vote diagnostics prove insufficient.
