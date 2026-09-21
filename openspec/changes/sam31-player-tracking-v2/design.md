# Design: SAM 3.1 Mask-Authoritative ReID, Stable Identity, and Team Phase

## Technical Approach

Keep SAM 3.1/session and artifact packaging unchanged, but make `RunnerV2` the single
mask-authoritative boundary. It validates each full-frame PNG mask, creates a zeroed
frame plus bbox ROI for valid observations, and invokes the existing ReID and team
modules through adapters. Invalid masks are ineligible—there is no bbox fallback.
Stable identity resolves before team clustering; statistics remain deferred.

## Architecture Decisions

| Decision | Choice | Alternatives / rationale |
|---|---|---|
| Mask authority | One `MaskAppearanceAdapter` at the runner boundary validates shape/non-empty pixels, zeroes outside the mask, and passes only masked ROI evidence. | Changing `reid.py` or duplicating team extraction would fork tested contracts. |
| Identity | Implement the explicit resolver behind `RunnerV2Seams.identity`: deterministic stable IDs, `similarity_threshold=0.90`, `max_frame_gap=5`, and ambiguous cases fragment. If the seam is absent, identity remains unvalidated/rejected; no resolver is synthesized. Duplicate observation keys reject before authority invocation. | Geometry-only continuity can contaminate another player, violating fragmentation-first safety. |
| Team order | Run `TeamEmbedder`/`identify_teams` only for valid, stably identified rows; aggregate labels by stable ID and commit only when each ID has one label. | Raw-track voting allows SAM fragmentation to create conflicting assignments. |
| Rejection | Treat identity/team validation as one transaction; scrub `stable_player_id`, `reid_embedding`, and `team_id` on rejection while retaining raw rows and artifacts. | Partial derived output is unsafe for downstream joins. |

## Data Flow

```text
SAM rows + full-frame PNG masks
  → normalize/validate → MaskAppearanceAdapter (zero outside mask, bbox as ROI)
  → existing ReID seam → conservative stable resolver
  → existing mask-aware TeamEmbedder + team_id → stable-ID conflict check
  → accepted rows/artifacts OR scrubbed rejected rows
```

`raw_track_id` is preserved as SAM provenance; `stable_player_id` and `team_id` are
derived fields. `results.package_artifacts` continues to write lossless PNG masks,
deterministic `mask_ref`s, `tracks.jsonl`, and manifest acceptance/rejection reasons.
`debug_v2.py` must read those references and render mask/bbox/raw/stable/team labels;
it must not infer identity from pixels or replace missing masks.

## File Changes

| File | Action | Description |
|---|---|---|
| `pipeline/runner_v2.py` | Modify | Add the mask adapter, valid-row gating, conservative resolver, stable-first ordering, team aggregation, and atomic scrubbing. |
| `modal_app_v2.py` | Modify | Pass each full-frame PNG mask alongside normalized rows into `RunnerV2`; preserve existing artifact return contract. |
| `pipeline/sam31_tracking.py` | Modify | Preserve full-frame mask references/data when normalizing SAM output; never convert masks into bbox-only evidence. |
| `tests/test_runner_v2.py` | Modify | Cover mask validation, zeroing, no fallback, resolver gates, ordering, stable/team conflicts, and rejection scrubbing. |
| `tests/test_modal_app_v2.py`, `tests/test_results_artifacts.py`, `tests/test_debug_v2.py` | Modify | Verify mask/artifact provenance and debug rendering. |
| `pipeline/reid.py`, `pipeline/team_embedder.py`, `pipeline/team_id.py`, `modal_app.py` | Unchanged | Reuse existing contracts; retain legacy rollback path. |

## Interfaces / Contracts

```python
class MaskAppearanceAdapter(Protocol):
    def reid(self, frame, bbox_xyxy, full_mask) -> list[float] | None: ...
    def team(self, frame, sample, full_mask) -> TeamFeatures: ...

class IdentityResolver(Protocol):
    def __call__(self, rows: list[dict]) -> Mapping[str, str]: ...
```

The adapter accepts only a 2-D mask matching `frame.shape[:2]` with nonzero pixels.
It supplies a crop-local mask to the existing team contract and a masked frame to the
existing ReID contract. Any invalid mask returns no feature. Accepted records retain
`raw_track_id`, `stable_player_id`, and `team_id`; rejected records retain raw fields,
`bbox_xyxy`, and `mask_ref` only. `statistics: deferred` remains explicit.

## Testing Strategy

| Layer | What to test | Approach |
|---|---|---|
| Unit | Mask gates, resolver threshold/gap, fragmentation, atomic conflicts | NumPy frames and fake ReID/team seams; assert received pixels and fields. |
| Integration | SAM rows through runner and artifact manifest | Temporary PNGs, fake session/models, verify ordering and deterministic refs. |
| Validation | `game_03`, approved prompts, frames 60–130 | Explicit Modal/GPU run; inspect masks, acceptance, and debug output. |

## Migration / Rollout

No legacy migration. Roll out behind the existing V2 entrypoint, first on the validation
slice. Roll back to unchanged `modal_app.py` or the prior V2 validation-only runner;
lossless masks remain readable. No statistics or semantic home/away labels are enabled.

## Open Questions

- [x] Confirm `similarity_threshold=0.90`, `max_frame_gap=5`, and `player-NNN` stable-ID format.
- [ ] Confirm the final SAM checkpoint and mask dtype/encoding at the runner boundary.
