# E2 Training Scenarios

E2 focuses on effective-path compression. The main question is how reconstruction
quality changes when `L_eff` is smaller than the true path count.

## Recommended Entry Point

Use:

```powershell
uv run --extra dev python scripts\e2\run_e2_training_scenarios.py
```

Default sweep:

```text
L_eff = 2, 4, 6, 8, 12
```

Default objective:

```text
loss_mode = reconstruction
```

This matches the current project direction:

```text
tokens -> model parameters -> differentiable LS -> H_hat -> NMSE(H_hat, H_true)
```

## Formal 9-Way Scenario Matrix

| Scenario | Architecture | LS mode | Purpose |
| --- | --- | --- | --- |
| `oracle` | oracle labels | traditional LS | Lower-bound reference for each `L_eff`. |
| `hybrid` | `original_v1` | traditional LS | Plain Hybrid Transformer baseline. |
| `uncertainty_traditional` | `uncertainty_v1` | traditional LS | Tests the uncertainty head without weighted LS. |
| `uncertainty_weighted` | `uncertainty_v1` | learnable weighted LS | Mean-pooling uncertainty weighted LS. |
| `query` | `query_v1` | traditional LS | Query decoder without weighted LS. |
| `query_weighted` | `query_v1` | learnable weighted LS | Tests whether uncertainty-derived `W` helps. |
| `epgt` | `epgt_v1_full` | traditional LS | Physics-guided Transformer baseline. |
| `epgt_weighted` | `epgt_v1_uncertainty_ls` | learnable weighted LS | EPGT plus uncertainty-derived LS weights. |
| `direct_h` | `original_v1` direct-H | none | Direct full-grid H regression baseline. |

## Main Comparisons

1. Compression sensitivity:

```text
oracle vs hybrid vs query vs epgt
across L_eff = 2, 4, 6, 8, 12
```

2. Query benefit:

```text
hybrid vs query
```

This isolates whether dedicated global/path/RX queries improve reconstruction.

3. Weighted LS benefit:

```text
query vs query_weighted
epgt vs epgt_weighted
```

This isolates whether model-predicted delay/Doppler uncertainty produces a
useful observation weight matrix `W`.

4. Physics guidance benefit:

```text
query vs epgt
query_weighted vs epgt_weighted
```

This isolates whether EPGT physics priors help beyond the query decoder and
weighted LS.

## Useful Commands

Quick smoke run:

```powershell
uv run --extra dev python scripts\e2\run_e2_training_scenarios.py --values 4 --steps 2 --eval-interval 1 --train-batches 1 --val-batches 1 --d-model 32 --num-layers 1 --nhead 4 --dim-feedforward 64 --dropout 0.0
```

Main E2 comparison:

```powershell
uv run --extra dev python scripts\e2\run_e2_training_scenarios.py
```

Formal defaults:

```text
steps = 80
lr = 1e-4
eval_interval = 1
train_batches = 32
val_batches = 8
batch_size = 16
loss_mode = reconstruction
device = auto
```

Only uncertainty-weighted comparison:

```powershell
uv run --extra dev python scripts\e2\run_e2_training_scenarios.py --scenarios query query_weighted epgt epgt_weighted --steps 80 --eval-interval 20
```

Parameter-supervised diagnostic:

```powershell
uv run --extra dev python scripts\e2\run_e2_training_scenarios.py --scenarios hybrid query epgt --loss-mode param --steps 40 --eval-interval 10
```

## Output

Default output:

```text
experiments/e2_effective_paths/formal_9way/
```

The result folder includes:

```text
run_metadata.json
results.json
results_partial.json
summary_scientific.csv
summary_scientific.md
plots/
```

The most important metrics are:

```text
channel_nmse_db
observed_symbol_nmse_db
weights_source
weights_min / weights_max
```

`weights_source` appears when the weighted LS plugin is used during evaluation.
