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

## Measured findings, 2026-07-26 (Modal `pipeline/` path)

First numbers from running the SAM2 path on a full clip rather than a short one.
Test clip: WNBA broadcast, 1280x720 @ 60fps, 583 frames (9.7s), ~10 players on court.

**The tracking was never verified past 5 seconds.** `notebooks/sam2_colab.ipynb`
sets `MAX_FRAMES = 300`, which at 60fps is 5.0 seconds. Measured on that same
clip, the notebook's configuration produces **20 distinct track IDs for 10
players within its own 5-second window**, and 30 across the full 9.7 seconds.

**Root cause of most ID churn is not drift — it is drop-on-miss.** SAM2 is
re-seeded from scratch at every detector call using only that call's detections.
A player the detector misses for a single frame is erased from SAM2 state, and
gets a new ID when found again. Counter-intuitive consequence, measured:
*lowering* `--frame-stride` makes identity **worse**, because more detector calls
means more chances to permanently drop a track.

| Config (same clip, 583 frames) | Detector calls | Distinct IDs | Extra IDs per call |
| --- | --- | --- | --- |
| Colab notebook config (60fps, hiera-large) | 39 | 30 | 0.51 |
| Same + carry-forward + osnet re-ID | 39 | 16 | 0.15 |
| 15fps + hiera-small + both fixes | 10 | 11 | 0.10 |

Raw ID counts are not comparable across frame rates; the last column is.

Other measurements worth keeping:

- **`hiera-large` costs ~7% more than `hiera-small`**, not 2x. The image encoder
  is not the only bottleneck. Checkpoint choice is nearly free; **sampling rate
  is the only significant cost lever** (~4x between 60fps and 15fps).
- **Torso-colour appearance is not usable for identity.** Self-similarity on
  correct matches ranged 0.63-0.95, and most "best other match" hits were
  teammates in the same kit. Replaced with osnet re-ID embeddings.
- **osnet is better but not decisive.** Self-similarity on correct matches only
  reaches ~0.63-0.76, so the swap detector built on it is a weak signal, useful
  for comparing configs but not as an oracle.
- **`distinct_track_ids` cannot measure swaps.** Swapping two labels neither
  creates nor destroys an ID, so a clip where all ten players trade numbers still
  scores exactly ten. `suspected_swaps` in `results.json` is the (imperfect)
  swap signal.
- **The detector only works on the wide court shot.** At frame 464 the broadcast
  cuts to a close-up and detections drop to zero. Real games contain a lot of
  replays, close-ups and crowd shots that cost GPU time and yield nothing.

Still unresolved after all of the above: **players still swap IDs during
crossings.** Reviewed visually and rejected as not good enough. The fixes reduce
lost-and-renumbered tracks; they do not solve identity handoff during overlap.

Suggested next investigation, in order:

1. Run the existing BoT-SORT / DeepOCSORT path (`video_pipeline.py`, which
   already uses osnet re-ID) head-to-head on this same clip. It is purpose-built
   for identity through occlusion; the SAM2 path may simply be the wrong tool.
2. Build the crossing/occlusion evaluation set below, so quality stops depending
   on someone watching a video.
3. Only then consider retroactively relabelling frames during an occlusion once
   a swap is detected on recovery.

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
