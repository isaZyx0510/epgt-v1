# PGT Scripts

EPGT-specific entry points live here while the interface is still changing.

```text
train_pgt.py                         Default EPGT training, now H-loss by default
train_pgt_h_loss.py                  Explicit H-loss training alias
run_e6_pgt_comparison.py             E6 baseline-vs-EPGT comparison
run_e2_e3_pgt_comparison.py          E2/E3 robustness comparisons
run_pgt_ablation.py                  Physics-guidance ablations
run_oracle_replacement_ablation.py   Oracle-parameter replacement diagnostics
```

Shared training and evaluation logic belongs in `src/thesis_transformer_v1/experiments/`.
