# E2 Formal 9-Way Comparison Plan

## Purpose

Compare 9 model/LS combinations under identical E2 effective-path compression
conditions.

## Command

```powershell
$env:PYTHONPATH='F:\HUAWEI_Theise\Thesis Transformer version1\src'
python scripts\e2\run_e2_training_scenarios.py
```

## Fixed E2 Condition

```text
data config = configs/data/e2_effective_paths.yaml
L_eff = 2, 4, 6, 8, 12
max_doppler_hz = 0
cfo_range_hz = [0, 0]
observation symbols = [6, 7]
symbol_error_rate = 0
```

## Scenario Matrix

```text
1. oracle                         + traditional_ls
2. hybrid original_v1             + traditional_ls
3. uncertainty_v1                 + traditional_ls
4. uncertainty_v1                 + learnable_weighted_ls
5. query_v1                       + traditional_ls
6. query_v1                       + learnable_weighted_ls
7. epgt_v1_full                   + traditional_ls
8. epgt_v1_uncertainty_ls         + learnable_weighted_ls
9. direct_h original_v1           + no LS
```

## Training Hyperparameters

```text
steps = 80
lr = 1e-4
optimizer = AdamW
eval_interval = 1
train_batches = 32
val_batches = 8
batch_size = 16
loss_mode = reconstruction
reconstruction_weight = 1.0
warmup_steps = 0
finetune_loss_mode = param_plus_reconstruction
uncertainty_regularization_weight = 1e-4
device = auto
```

This gives:

```text
train samples per L_eff/scenario = 512
validation samples per L_eff/scenario = 128
```

## Model Hyperparameters

```text
d_model = 96
num_layers = 2
nhead = 4
dim_feedforward = 192
dropout = 0.05
baseline model config = configs/model/hybrid_transformer.yaml
EPGT full config = configs/model/pgt/epgt_v1_full.yaml
EPGT weighted config = configs/model/pgt/epgt_v1_uncertainty_ls.yaml
```

## Expected Outputs

```text
run_metadata.json
results_partial.json
results.json
summary_scientific.csv
summary_scientific.md
plots/train_loss_l_eff_*.png
plots/validation_loss_l_eff_*.png
plots/final_validation_nmse_vs_l_eff.png
```

All scalar summary values are written in scientific notation in
`summary_scientific.csv` and `summary_scientific.md`.
