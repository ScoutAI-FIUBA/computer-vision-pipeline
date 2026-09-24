# Possession Engine V3

V3 is an additive, deterministic diagnostic path. Enable it with
`POSSESSION_ENGINE=v3`; the default `legacy` mode writes the existing files and
does not create V3 artifacts.

## Outputs

Enabled runs publish one generation under `possession_v3/generations/<id>/` and
atomically replace `possession_v3_manifest.json`. Consumers must resolve JSONL
paths from that manifest; the pointer is the only current-generation marker, so
failed runs cannot expose a partial or mixed generation. If publication fails,
the pointer may continue to reference the valid last-known-good generation; that
complete previous generation is the intentional rollback/current fallback, not
an incomplete result from the failed run.

The writer retains the current generation and one rollback generation, cleans
stale staging directories, and keeps the last-known-good pointer if a later V3
build or publication fails.

| File | Contents |
|---|---|
| `possession_v3.jsonl` | Per-frame holder roles, state, candidates, scores, thresholds, grace, clear reason, audit windows, and provenance |
| `possession_primitives_v3.jsonl` | Release, transit, reception, and loose-ball primitives |
| `ball_transitions_v3.jsonl` | Source/target transition windows and evidence |
| `possession_events_v3.jsonl` | Transfer, PASS, STEAL, loss, turnover, or UNKNOWN labels |

If V3 building or publication fails, the writer preserves the last-known-good
pointer and generation. It records best-effort failure observability separately;
failure reporting itself cannot make the run fail. Legacy output remains
authoritative and is never interrupted by V3 diagnostics.

Every record declares the V3 schema version and frame range. Timestamp ranges
are included when source timestamps exist and are null only when they are not
available. Records retain profile, thresholds, candidate contributions,
rejected reasons, and provenance. Debug rendering shows holder state, candidate
scores/contributions, rejected evidence, transition start/end windows, and
final event labels.

## Continuity fields

V3 keeps schema version `1` and adds optional `context` and `audit` objects.
`context` separates `effective_holder_id`, `observed_candidate_id`,
`challenger_id`, team, and grace fields. `audit` records `last_holder_id/team`,
`clear_reason`, source quality/authority, and frame/time windows. Existing
consumers can ignore these fields; missing objects use deterministic defaults.

The debug overlay renders `INFERRED`, `CONTESTED`, `TRANSITIONING`, and
`LOST_NO_DETECTION`, plus all three holder roles, clear reason, grace remaining,
and switch/clear audit windows. Transition and event windows remain event-level
facts, not per-frame semantic claims.

## Rollback

Set `POSSESSION_ENGINE=legacy` or stop consuming the V3 pointers. Existing
possession and transition artifacts remain authoritative; no migration is
required. Disabled/legacy runs omit V3 pointers and files; re-enabling V3
produces additive continuity records without upstream model or semantic-event
changes.
