# V3 Calibration Benchmark
The harness is offline and deterministic. It reads JSONL fixtures, does not
start Modal, and does not load a model.
## Quick path

1. Copy `tests/fixtures/possession_v3/calibration-template.jsonl`.
2. Supply **10–30 representative clips** as one JSON object per JSONL line.
3. Label holder frames and events, including explicit `"UNKNOWN"` ground truth.
4. Run:

```bash
python -m pipeline.calibration labels.jsonl report.json --profile pilot
```

The report includes raw and benchmark-input SHA-256 values. The latter hashes clips plus profile, thresholds, feature flag, fixture reference, clip bounds, calibration mode, and configuration, so reruns are reproducible and traceable. Without labeled frames/events, metrics are `null`; the report is descriptive only and must not be presented as benchmark quality. A single clip is always non-authoritative and cannot promote.

## Fixture fields

Each clip requires a unique `clip_id`, has `frames`, and may declare `difficulty` as `easy`, `medium`, or `hard`. Clip boundaries are mandatory so recovery metrics cannot join adjacent clips. Optional primitive/semantic event lists are supported.
Frame fields are `frame_index`, `predicted_holder`, `ground_truth_holder`, and
optional `predicted_state`. Event matching uses `(clip_id, event_type, frame_index)`;
events from different clips never match.
The report provides coverage, contested rate, holder accuracy, false-holder/switch rates, fragmentation, UNKNOWN rate, UNKNOWN/grace recovery, event counts, fixture classification, primitive/semantic precision/recall/F1 hooks, a `PROMOTE`/`REJECT` gate, and a legacy/V2 comparison hook. Gate thresholds, feature-flag state, profile, fixture IDs, and configuration are recorded under `traceability`.

## Rollout and rollback

- Keep `POSSESSION_ENGINE=legacy` as the default; V3 remains opt-in.
- Run legacy and V3 on the **same saved inputs** and compare reports/artifacts.
- Acceptance gates: labeled 10–30 clip set (or an explicit configured minimum of at least 10), deterministic replay, easy/medium/hard coverage, required precision/recall and event evidence, non-empty legacy/V2 comparison evidence, no raw-output mutation, acceptable false-holder/switch rates, and reviewed UNKNOWN behavior. Missing metrics fail the gate; do not claim quality from unlabeled fixtures.
- Exploratory calibration may set `calibration_mode="non_authoritative"` with smaller custom bounds. It can produce descriptive reports but can never promote V3.
- Roll back by setting `POSSESSION_ENGINE=legacy` and stop consuming V3
  pointers/artifacts. Legacy outputs remain authoritative; no migration is
  required.
- `OUT_OF_BOUNDS` remains UNKNOWN/deferred until court geometry is available.
