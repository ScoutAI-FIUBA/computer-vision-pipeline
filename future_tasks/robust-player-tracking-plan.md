# Robust Player Tracking Plan

Practical implementation plan for turning the current local tracking path into a more reliable basketball player tracking pipeline. This is the working reference for the tracking rework, not a generic MOT writeup.

## Objective

Improve player identity consistency in `scripts/tracking/video_pipeline.py` / `scripts/tracking/track_players_video.py` so the local pipeline can survive common basketball crossings, partial occlusions, and short full occlusions without treating `--frame-stride 1` as the default answer.

## In Scope

- Tracking inside the current local video pipeline.
- Tracker API replacement or wrapper work if `sv.ByteTrack()` is the wrong base.
- Stronger identity handling for crossings and occlusions.
- Evaluation clips, scoring workflow, and track-quality success gates.
- Keeping existing JSONL/debug outputs useful for review.

## Out of Scope

- Team identification, jersey OCR, or event logic.
- Ball detection experiments, TrackNet training, or ball-only dataset work. That work lives under `scripts/ball/`.
- Full API integration or deployment decisions.
- SAM 2 or other heavy segmentation-first tracking unless lighter tracker/re-ID changes are proven insufficient.
- Perfect long-range re-identification across scene cuts or broadcast replay edits.

## Current Repo Gap vs Target Architecture

| Area | Current repo state | Target state |
| --- | --- | --- |
| Detection cadence | Sampled Roboflow detections with cost controls. | Keep sampled workflow, but make identity handoff robust enough that sparse inference is still usable. |
| Tracker core | Current local path still builds `sv.ByteTrack()` and already shows a deprecation warning. | Stable tracker API choice with repo-owned integration points for tuning and evaluation. |
| Identity recovery | Short-lived predicted persistence plus lightweight torso-color reassignment. | Measured identity recovery that handles crossings/occlusions better than the current heuristic baseline. |
| Evaluation | Failure modes are described, but there is no small repeatable benchmark clip set yet. | Small, repeatable evaluation set with manual scorekeeping for ID switches, full losses, and recoveries. |
| Acceptance gate | Tracking is known to be improved but not yet judged against explicit pass/fail targets. | Tracking changes land only after they beat the current baseline on the same clips. |

## Prioritized Phases

### Phase 1: Lock the evaluation loop

1. Create a small clip set with crossings, partial occlusions, and short full occlusions.
2. Reuse current outputs (`tracks.jsonl`, annotated MP4, projected-track debug when useful) to review failures.
3. Score baseline metrics manually: ID switches, full track losses, successful recoveries.

Exit condition: the repo has a repeatable baseline that can reject fake improvements.

### Phase 2: Replace unstable tracker wiring before heavier modeling

1. Recheck the current recommended Supervision tracker API.
2. Replace or wrap the deprecated `sv.ByteTrack()` construction behind one repo-owned integration point.
3. Rerun the same clip set and compare against the baseline before adding new identity logic.

Exit condition: the repo is no longer building on a deprecated tracker path, and the new path is at least baseline-equal on target clips.

### Phase 3: Strengthen short-occlusion identity handling

1. Tune persistence and track-lifecycle rules against the evaluation set.
2. Keep the current appearance heuristic only if it measurably helps.
3. If tracker/API cleanup is still not enough, add one stronger appearance embedding path for re-association, but keep it optional and evaluation-driven.

Exit condition: the pipeline reduces ID switches and full losses on target clips without hiding the problem behind denser inference.

### Phase 4: Stabilize outputs for downstream stages

1. Keep track JSONL fields stable enough for later team/OCR work.
2. Confirm projected-track export still works with the updated tracker path.
3. Document any new tracking flags only if they are needed to reproduce measured gains.

Exit condition: downstream tasks can consume track outputs without depending on one-off debug behavior.

## Measurable Success Criteria

- A fixed evaluation clip set exists and is referenced by the tracking backlog work.
- Every tracker change is compared on the same clips against the current baseline.
- On the target clip set, the chosen tracking path shows fewer ID switches and fewer full track losses than the current baseline.
- `--frame-stride 1` is not required to claim the improvement.
- The pipeline still emits usable annotated video and track JSONL for review.

## Immediate Next Slice

1. Build the first evaluation clip set for Task 3.3.
2. Define one lightweight review sheet for three counts only: ID switches, full losses, successful recoveries.
3. After the baseline is recorded, swap the deprecated tracker API path and rerun the same clips before considering stronger re-ID.

## Working References

- `future_tasks/basketball_pipeline_tasks.md`
- `TRACKING_ISSUES.md`
- `README.md`
- `scripts/ball/FINETUNING.md` (ball-only experimental reference, not part of the tracking rework)
