# Design: Formal E2 Nine-Way Comparison

## Scenario Matrix

| Scenario | Route | LS mode | Purpose |
| --- | --- | --- | --- |
| `oracle` | oracle labels | `traditional_ls` | Lower-bound LS reference for each `L_eff`. |
| `hybrid` | `original_v1` hybrid | `traditional_ls` | Plain hybrid Transformer baseline. |
| `uncertainty_traditional` | `uncertainty_v1` | `traditional_ls` | Tests uncertainty-head architecture without weighted LS. |
| `uncertainty_weighted` | `uncertainty_v1` | `learnable_weighted_ls` | Mean-pooling uncertainty weighted LS baseline. |
| `query` | `query_v1` | `traditional_ls` | Query decoder without weighted LS. |
| `query_weighted` | `query_v1` | `learnable_weighted_ls` | Query decoder with uncertainty-derived LS weights. |
| `epgt` | `epgt_v1_full` | `traditional_ls` | Physics-guided Transformer baseline. |
| `epgt_weighted` | `epgt_v1_uncertainty_ls` | `learnable_weighted_ls` | EPGT with path uncertainty and weighted LS. |
| `direct_h` | `original_v1` direct-H | none | Direct full-grid H regression baseline. |

## Output Package

The default output root is:

```text
experiments/e2_effective_paths/formal_9way/
```

The package contains:

```text
run_metadata.json
results_partial.json
results.json
summary_scientific.csv
summary_scientific.md
plots/
```

`run_metadata.json` records the scenario order, `L_eff` values, training
hyperparameters, model hyperparameters, serialized base config, dataset shapes,
token feature shape, and seed policy. `results_partial.json` is updated after
each scenario so long runs still leave recoverable progress.

## Training Defaults

Formal defaults are intentionally larger than smoke settings:

```text
steps = 80
lr = 1e-4
eval_interval = 1
train_batches = 32
val_batches = 8
batch_size = config-defined
loss_mode = reconstruction
device = auto
```

Hybrid routes use `train_hybrid_quick`; direct-H uses `train_direct_h_quick`;
oracle uses `run_oracle_reference`.
