# Derived Data Exports

These two CSV files are convenience views derived without changing the packaged raw JSONL records. They describe the completed `dynamic_scale` diagnostic/resource probe at code revision `67c1a8c40c6b94de4add2f6d0a6d2dadbdddd3e7`. They are **not** results from the pending full multi-seed pilot and must not be presented as confirmatory performance evidence.

## `dynamic_scale_timeline.csv`

Source: `dynamic_scale/seed_42/attempt_001/result/steps.jsonl`.

Filter: `phase == "evaluation"`, `arm == "switching"`, and `episode == 0`. The export contains exactly 100 source steps. `raw_step` preserves the zero-based source index; `display_step = raw_step + 1` is provided only for figure axes.

The three context fields map exactly to the implementation in [`dynamic_experiment/environment.py`](../../03_code_comments/source/dynamic_experiment/environment.py):

- `normalized_active_user_fraction = active_users / pool_users`, unitless and clipped by construction to the configured population range.
- `normalized_edge_backlog = clip(mean edge-computation backlog / 12, 0, 1)`, unitless. It is a normalized controller context, not raw queue bytes or task count.
- `normalized_access_pressure = clip(1 - mean best normalized AP-channel rate, 0, 1)`, unitless. It is not a measured packet-loss rate.

`action_seconds`, `context_selector_seconds`, and `controller_seconds` are seconds per recorded step. `controller_seconds` is the sum of the other two fields in the source record. `completion_now` is not exported; the exact source field is `completed_now`. The source has no per-step QoE, delay, energy, or drop value, so this timeline must not be used to invent per-step performance curves.

## `dynamic_scale_evaluation.csv`

Source: `dynamic_scale/seed_42/attempt_001/result/episodes.jsonl`.

Filter: `phase == "evaluation"` and `episode == 0`. The four rows are `fixed`, `adaptive`, `contextual`, and `switching`, all evaluated on the same held-out `trace_hash`. Source metrics are copied exactly. `completion_rate` remains a fraction in [0, 1]; `completion_rate_percent` is the sole derived metric and equals `100 * completion_rate`.

This is one seed and one evaluation episode from a scale/diagnostic run. It is suitable for checking the export and drafting figure structure, not for claiming statistical superiority or replacing the full paired multi-seed evaluation.

## Reproduce

From the handoff package root:

```sh
python3 tools/export_review.py
```

The script uses only the Python standard library, checks the expected phase/arm/episode filters and row counts, confirms the four evaluation arms share one trace, then reads both CSV files back and compares every exported source value with its JSONL record.
