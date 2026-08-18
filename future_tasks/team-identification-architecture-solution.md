# Team identification solution notes

This file records the current minimal solution that exists in code, plus the smallest coherent next step.

## What exists today

The deployed path already does all of this:

1. Extract team evidence during tracking.
2. Score and reject bad samples.
3. Keep the best accepted samples per track.
4. Cluster selected samples globally.
5. Vote one `team_id` per track.
6. Leave ambiguous tracks unassigned.

That means the next work is not "wire team ID into the pipeline." That part is done.

## Smallest accurate mental model

Use this model when discussing the code:

- `pipeline/team_embedder.py` builds features from a jersey-focused crop.
- `pipeline/team_evidence.py` filters and ranks those samples.
- `pipeline/team_id.py` clusters selected samples and votes per track.
- `pipeline/runner.py` rewrites outputs after final assignment.

## What is intentionally not true anymore

Do not assume any of these older design claims:

- colour-first assignment followed by SigLIP refinement;
- explicit prototype-building from anchor tracks;
- online team assignment during tracking;
- confidence already being part of the output contract.

Those may still be useful ideas, but they are not the current implementation.

## Recommended next solution step

If team-ID quality must improve, start with evidence quality rather than new classifier stages:

1. inspect rejected and low-quality accepted samples in `team_debug/`;
2. tighten crop or mask behavior only where the debug artifacts prove contamination;
3. validate the change on more than one clip before changing the default thresholds.
