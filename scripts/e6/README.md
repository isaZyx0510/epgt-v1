# E6 Scripts

EPGT-specific entry points live here.

```text
train_pgt_h_loss.py                  Explicit H-loss EPGT training
run_e6_pgt_comparison.py             E6 baseline-vs-EPGT comparison
run_oracle_replacement_ablation.py   Oracle-parameter replacement diagnostics
```

Use `configs/model/pgt/epgt_v1_uncertainty_ls.yaml` with
`--ls-mode learnable_weighted_ls` when testing the path-uncertainty weighted LS
variant. This is separate from the reliability-mask ablation: uncertainty is a
model output, while reliability is an observation-token annotation.

Shared training and evaluation logic belongs in `src/thesis_transformer_v1/experiments/`.
