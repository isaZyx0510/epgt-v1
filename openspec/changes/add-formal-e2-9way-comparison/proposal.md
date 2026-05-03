# Add Formal E2 Nine-Way Comparison

## Why

E2 is the effective-path compression experiment. The older E2 entry point could
run a focused subset of hybrid scenarios, but it did not package enough metadata
for thesis-ready comparison across model families.

The project now needs one reproducible E2 command that compares oracle, direct-H,
hybrid, query, uncertainty, and EPGT routes under the same `L_eff` sweep and
writes artifacts that can be inspected or copied into reports.

## What

- Promote `scripts/e2/run_e2_training_scenarios.py` to the formal E2 comparison
  entry point.
- Compare nine scenarios by default:
  `oracle`, `hybrid`, `uncertainty_traditional`, `uncertainty_weighted`, `query`,
  `query_weighted`, `epgt`, `epgt_weighted`, and `direct_h`.
- Record run metadata, dataset shapes, training hyperparameters, model
  parameter counts, partial results, final results, summary tables, and plots.
- Use `experiments/e2_effective_paths/formal_9way/` as the default output root.
- Keep H reconstruction loss as the default hybrid/EPGT training objective.

## Non-Goals

- Do not merge E2 with E3/E4/E5 scripts.
- Do not change the meaning of existing E0-E6 data configs.
- Do not make direct-H use LS; it remains a full-grid H regression baseline.
