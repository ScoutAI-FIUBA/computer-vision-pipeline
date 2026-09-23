# Possession Engine V3

V3 is an additive, deterministic diagnostic path. Enable it with
`POSSESSION_ENGINE=v3`; the default `legacy` mode writes the existing files and
does not create V3 artifacts.

## Outputs

Enabled runs add these JSONL files and `possession_v3_manifest.json`:

| File | Contents |
|---|---|
| `possession_v3.jsonl` | Per-frame holder, state, candidates, scores, thresholds, and provenance |
| `possession_primitives_v3.jsonl` | Release, transit, reception, and loose-ball primitives |
| `ball_transitions_v3.jsonl` | Source/target transition windows and evidence |
| `possession_events_v3.jsonl` | Transfer, PASS, STEAL, loss, turnover, or UNKNOWN labels |

Every record declares the V3 schema version and frame range. Timestamp ranges
are included when source timestamps exist and are null only when they are not
available. Records retain profile, thresholds, candidate contributions,
rejected reasons, and provenance. Debug rendering shows holder state, candidate
scores/contributions, rejected evidence, transition start/end windows, and
final event labels.

## Rollback

Set `POSSESSION_ENGINE=legacy` or stop consuming the V3 pointers. Existing
possession and transition artifacts remain authoritative; no migration is
required.
