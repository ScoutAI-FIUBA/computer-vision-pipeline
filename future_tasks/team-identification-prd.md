# Team identification PRD

This file replaces the older proposal with a current-state product note.

## Goal

Improve the accuracy of per-track team assignment in the deployed `pipeline/` path without destabilizing tracking or changing the current output contract.

## Current implemented baseline

Already implemented:

1. End-of-run team assignment in `pipeline/runner.py`.
2. Jersey-focused sample extraction in `pipeline/team_embedder.py`.
3. Sample quality scoring and rejection.
4. Per-track top-K evidence selection.
5. Global clustering plus per-track voting.
6. `teams.json`, rewritten `tracks.jsonl`, and optional team debug artifacts.

Not implemented yet:

1. Explicit per-track confidence beyond vote diagnostics.
2. A richer multi-region uniform representation.
3. Cross-clip validation tooling for team-ID quality.
4. A stronger strategy for recovering from contaminated or identity-mixed tracks.

## Problem statement

The biggest current problem is not missing plumbing. It is evidence quality.

Observed failure modes still include:

- visible players left unassigned;
- wrong-team assignments after noisy crops or occlusion;
- brittle tuning that helps one clip but does not generalize.

## Product constraints

- Keep the main path in `pipeline/` plus `modal_app.py`.
- Preserve `tracks.jsonl`, `teams.json`, and optional `annotated.mp4`.
- Prefer small, explainable changes over more stages or more dependencies.
- Use multiple clips for validation before trusting threshold changes.

## Current output contract

| File | Contract |
|---|---|
| `tracks.jsonl` | Per-frame rows with final `team_id` written after assignment |
| `teams.json` | Team assignments, selected-sample counts, diagnostics, and unassigned tracks |
| `results.json` | Run summary plus the resolved config |
| `team_debug.json` / `team_debug/` | Optional evidence-inspection artifacts |

## Next product tasks

1. Improve crop cleanliness and sample reliability before changing the classifier shape.
2. Add evaluation over multiple clips with the same metrics and review path.
3. Add a confidence story only if the existing vote diagnostics are not sufficient for downstream use.
