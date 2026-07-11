# Tracking Issues

Short status doc for the unresolved tracking failures in the current local pipeline.

## Scope

Applies to `scripts/tracking/video_pipeline.py` and `scripts/tracking/track_players_video.py`.

Implemented now:

- Shared annotation helpers are consolidated.
- Short-lived predicted track persistence is enabled by default.
- A lightweight torso-color appearance reassignment baseline exists.
- Optional court-keypoint + projected-track export is wired into the local video pipeline.
- Homography debug now rejects obviously bad keypoint geometry before fitting.

## Unresolved Issues

| Issue | Symptom | Why current fixes are insufficient | Likely next action |
| --- | --- | --- | --- |
| ID switches during crossings/occlusions | Two nearby players swap `track_id` after overlap or partial hide. | Smoothing and short-lived persistence stabilize boxes, but they do not provide strong identity evidence once detections overlap. The torso-color heuristic is too weak when uniforms are similar, lighting shifts, or the visible crop changes. | Build a small crossing/occlusion evaluation set, measure switch frequency, and compare the current `sv.ByteTrack()` path with the current recommended Supervision tracker API before adding heavier models. |
| Full track loss during harder occlusions | A player disappears from tracking entirely and comes back with a new ID or no recovered ID. | Predicted persistence only covers very short sampled gaps (`--max-missing-sampled-frames` defaults to `1`). The current appearance reassignment only looks like a recovery heuristic, not durable re-identification. | Test a slightly longer persistence window on evaluation clips, then decide whether to strengthen re-ID or swap tracker strategy based on measured recovery vs false-merge behavior. |
| `--frame-stride 1` is not a real fix | Runtime/cost increases sharply, but identity consistency still breaks in difficult clips. | More frequent detections help recall, but they do not solve ambiguous identity handoff during overlap. It is the wrong default answer for both cost and quality. | Keep the default sampled workflow. Use stride experiments only for evaluation data, not as the baseline product decision. |
| Appearance re-ID is only a cheap baseline | Recovered IDs sometimes work, but they are not dependable across lighting, pose, blur, or same-color jerseys. | Current re-ID uses lightweight torso-color appearance only. That is useful as a small heuristic, not as robust identity modeling. | Keep it as a baseline and document it honestly. Only invest in stronger appearance embeddings if evaluation shows tracker/API changes are still insufficient. |
| ByteTrack API choice is under review | Current code still constructs `sv.ByteTrack()`, and the library now emits a deprecation warning. | Even if behavior is acceptable today, a deprecated API is not a stable foundation for more tracker-specific work. | Recheck the current Supervision tracker recommendation, update the pipeline to that API, then rerun the same failure clips before making bigger architecture decisions. |

## Recommended Next Steps

1. Create a small evaluation clip set focused on crossings, partial occlusions, and full occlusions.
2. Score three things manually: ID switches, full track losses, and successful recoveries, then summarize them with `scripts/tracking/evaluate_tracking.py`.
3. Replace or wrap the deprecated `sv.ByteTrack()` path with the current recommended Supervision tracker API.
4. Re-run the same clips before adding a heavier identity solution.
5. Only if tracker/API changes are still not enough, evaluate stronger appearance embeddings or a different tracker.

## Non-Goals For Now

- Do not claim tracking is solved because projected-track export and top-down visualization now exist.
- Do not treat `--frame-stride 1` as the default product answer.
- Do not treat the current torso-color baseline as production-grade re-identification.
