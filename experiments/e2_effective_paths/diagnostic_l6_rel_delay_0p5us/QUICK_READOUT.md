# E2 L=6 Low-Cost Diagnostic: max_rel_delay_s = 0.5 us

## Completed Matrix

```text
rows = 7
L_eff = [6]
scenarios = [
  hybrid,
  uncertainty_traditional,
  uncertainty_weighted,
  query,
  query_weighted,
  epgt,
  epgt_weighted,
]
loss_mode = two_stage
warmup_steps = 80
finetune_loss_mode = param_plus_reconstruction
total_steps = 160
reconstruction_weight = 0.05
max_rel_delay_s = 5.0e-7
device = cuda
```

## Best By Validation H NMSE

| rank | variant | validation H NMSE | val normalized MSE sum | val delay loss |
| ---: | --- | ---: | ---: | ---: |
| 1 | `uncertainty_v1_weighted_ls` | 1.029182e-02 | 8.844367e-02 | 4.061e-03 |
| 2 | `query_v1_weighted_ls` | 1.807e-02 | 6.241e-02 | 1.107e-02 |
| 3 | `epgt_v1_uncertainty_weighted_ls` | 2.385e-02 | 1.730e-01 | 1.918e-03 |
| 4 | `hybrid_original_v1_traditional_ls` | 6.358e-02 | 1.357e-01 | 5.198e-03 |
| 5 | `uncertainty_v1_traditional_ls` | 3.601e-01 | 1.849e-01 | 5.907e-03 |
| 6 | `epgt_v1_full_traditional_ls` | 6.329e-01 | 2.449e-01 | 4.072e-03 |
| 7 | `query_v1_traditional_ls` | 4.975e+00 | 9.651e-02 | 9.636e-03 |

## Comparison At L_eff = 6

| run | best variant | best validation H NMSE |
| --- | --- | ---: |
| param-only | `epgt_v1_uncertainty_weighted_ls` | 2.069003e-02 |
| two-stage original max_rel_delay_s=3.0us | `epgt_v1_full_traditional_ls` | 2.093953e-01 |
| two-stage diagnostic max_rel_delay_s=0.5us | `uncertainty_v1_weighted_ls` | 1.029182e-02 |
| oracle LS no AI | `oracle_ls` | 1.780487e-10 |

## Interpretation

- Tightening `max_rel_delay_s` strongly improves the L=6 two-stage result.
- This supports the hypothesis that the original delay normalization scale was too wide, making delay errors underweighted during parameter supervision.
- Weighted LS variants benefit most in this diagnostic. The best result comes from `uncertainty_v1_weighted_ls`.
- The oracle gap is still large, so this fixes a training-scale bottleneck but does not solve the full learned-parameter precision problem.

## Artifacts

- `run_metadata.json`
- `results.json`
- `summary_scientific.csv`
- `summary_scientific.md`
- `plots/final_validation_nmse_vs_l_eff.png` includes `oracle_ls_no_ai`
- `plots/validation_channel_nmse_l_eff_6.png` includes `oracle_ls_no_ai`
- `plots/train_param_loss_l_eff_6.png`
- `plots/validation_param_loss_l_eff_6.png`
- `plots/train_normalized_mse_sum_l_eff_6.png`
- `plots/validation_normalized_mse_sum_l_eff_6.png`
