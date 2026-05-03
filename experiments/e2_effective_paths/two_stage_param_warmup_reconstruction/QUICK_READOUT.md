# E2 Two-Stage Param Warmup + Reconstruction Quick Readout

## Completed Matrix

```text
rows = 35
L_eff = [2, 4, 6, 8, 12]
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
device = cuda
```

## Best By Validation H NMSE

| L_eff | best variant | validation H NMSE | val normalized MSE sum | val param loss |
| ---: | --- | ---: | ---: | ---: |
| 2 | `epgt_v1_full_traditional_ls` | 4.766314e-03 | 2.987839e-02 | 2.987839e-02 |
| 4 | `epgt_v1_uncertainty_weighted_ls` | 2.969648e-03 | 4.401504e-02 | -1.654074e-01 |
| 6 | `epgt_v1_full_traditional_ls` | 2.093953e-01 | 2.445635e-01 | 2.445635e-01 |
| 8 | `query_v1_weighted_ls` | 1.025567e-02 | 4.231594e-02 | -2.929957e-01 |
| 12 | `query_v1_weighted_ls` | 1.776852e-03 | 4.080624e-02 | -3.018506e-01 |

## Best By Pure Normalized Physical MSE

| L_eff | best variant | val normalized MSE sum | validation H NMSE |
| ---: | --- | ---: | ---: |
| 2 | `epgt_v1_uncertainty_weighted_ls` | 2.867282e-02 | 6.459337e-03 |
| 4 | `epgt_v1_uncertainty_weighted_ls` | 4.401504e-02 | 2.969648e-03 |
| 6 | `epgt_v1_full_traditional_ls` | 2.445635e-01 | 2.093953e-01 |
| 8 | `query_v1_weighted_ls` | 4.231594e-02 | 1.025567e-02 |
| 12 | `query_v1_weighted_ls` | 4.080624e-02 | 1.776852e-03 |

## Interpretation Notes

- Two-stage improves the best H NMSE at L_eff = 2, 4, 8, and 12 compared with the param-only run.
- L_eff = 6 is the exception: the best H NMSE is worse than the param-only best, suggesting the reconstruction fine-tune can still destabilize some path geometries.
- Weighted variants remain strongest for high L_eff. `query_v1_weighted_ls` is best at L_eff = 8 and 12.
- `val_param_loss` can be negative for uncertainty-output models because it includes delay/doppler NLL terms. Use `val_normalized_mse_sum` for cross-model physical-parameter comparison.
- H NMSE plots include `oracle_ls_no_ai`; param-loss plots do not include oracle because oracle has no trainable model.

## Artifacts

- `run_metadata.json`
- `results.json`
- `summary_scientific.csv`
- `summary_scientific.md`
- `plots/final_validation_nmse_vs_l_eff.png` includes `oracle_ls_no_ai`
- `plots/final_validation_param_loss_vs_l_eff.png`
- `plots/final_validation_normalized_mse_vs_l_eff.png`
- `plots/train_param_loss_l_eff_*.png`
- `plots/validation_param_loss_l_eff_*.png`
- `plots/train_normalized_mse_sum_l_eff_*.png`
- `plots/validation_normalized_mse_sum_l_eff_*.png`
- `plots/validation_channel_nmse_l_eff_*.png` includes `oracle_ls_no_ai`
