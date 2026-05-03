# E2 Param-Only Physical-Model Quick Readout

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
loss_mode = param
device = cuda
```

## Best By Pure Normalized Physical MSE

This ranks models by the sum of the five normalized MSE terms:
tau0, CFO, relative delay, Doppler, and RX time offset.

| L_eff | best variant | val normalized MSE sum | validation H NMSE |
| ---: | --- | ---: | ---: |
| 2 | `epgt_v1_full_traditional_ls` | 3.679786e-02 | 2.974496e-01 |
| 4 | `query_v1_weighted_ls` | 3.753250e-02 | 2.853359e-02 |
| 6 | `query_v1_traditional_ls` | 3.911727e-02 | 3.262893e+01 |
| 8 | `query_v1_traditional_ls` | 3.764582e-02 | 6.048482e+00 |
| 12 | `query_v1_traditional_ls` | 4.025917e-02 | 2.006596e+01 |

## Best By Validation H NMSE

This ranks models by the final validation channel reconstruction NMSE, even though training used param-only loss.

| L_eff | best variant | validation H NMSE | val normalized MSE sum |
| ---: | --- | ---: | ---: |
| 2 | `epgt_v1_uncertainty_weighted_ls` | 2.735756e-02 | 4.173907e-02 |
| 4 | `uncertainty_v1_weighted_ls` | 1.837577e-02 | 6.218546e-02 |
| 6 | `epgt_v1_uncertainty_weighted_ls` | 2.069003e-02 | 5.791449e-02 |
| 8 | `query_v1_weighted_ls` | 1.221546e-02 | 3.977150e-02 |
| 12 | `query_v1_weighted_ls` | 2.906267e-03 | 4.273860e-02 |

## Interpretation Notes

- `val_param_loss` can be negative for uncertainty-output models because it includes delay/doppler NLL terms with log variance.
- For cross-model physical-parameter comparison, prefer `val_normalized_mse_sum`.
- Param-only training is much more stable than the previous H-only reconstruction training.
- Some models obtain good physical MSE but poor H NMSE, which suggests the LS/reconstruction stage remains sensitive to path geometry, weighting, and conditioning.

## Artifacts

- `run_metadata.json`
- `results.json`
- `summary_scientific.csv`
- `summary_scientific.md`
- `plots/final_validation_param_loss_vs_l_eff.png`
- `plots/final_validation_nmse_vs_l_eff.png` includes `oracle_ls_no_ai`
- `plots/final_validation_normalized_mse_vs_l_eff.png`
- `plots/train_param_loss_l_eff_*.png`
- `plots/validation_param_loss_l_eff_*.png`
- `plots/train_normalized_mse_sum_l_eff_*.png`
- `plots/validation_normalized_mse_sum_l_eff_*.png`
- `plots/validation_channel_nmse_l_eff_*.png` includes `oracle_ls_no_ai`

Oracle LS is not shown in param-loss plots because it has no trainable model and no param loss.
