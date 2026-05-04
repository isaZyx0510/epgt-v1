# E2 L=6 Reconstruction Weight and Finetune Length Sweep

## Best Result Per Run

| rank | run | best variant | H NMSE | H NMSE dB | val normalized MSE sum | val delay loss |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `oracle_ls_no_ai` | `oracle_ls` | 1.780487e-10 | -9.747029e+01 |  |  |
| 2 | `two_stage_0p5us_rw0p02_ft20` | `uncertainty_v1_weighted_ls` | 2.783713e-04 | -3.785178e+01 | 3.870734e-02 | 3.460838e-03 |
| 3 | `two_stage_0p5us_rw0p01_ft20` | `epgt_v1_uncertainty_weighted_ls` | 4.607480e-04 | -3.501403e+01 | 4.027131e-02 | 8.644880e-04 |
| 4 | `two_stage_0p5us_rw0p01_ft40` | `uncertainty_v1_weighted_ls` | 5.308669e-04 | -3.396683e+01 | 4.315809e-02 | 1.096582e-03 |
| 5 | `two_stage_0p5us_rw0p02_ft40` | `query_v1_weighted_ls` | 5.402334e-04 | -3.320102e+01 | 4.035495e-02 | 3.377649e-03 |
| 6 | `two_stage_0p5us_rw0p05_ft80` | `uncertainty_v1_weighted_ls` | 1.029182e-02 | -2.229658e+01 | 8.844367e-02 | 4.061423e-03 |
| 7 | `param_only` | `epgt_v1_uncertainty_weighted_ls` | 2.069003e-02 | -1.731739e+01 | 5.791449e-02 | 1.950206e-02 |
| 8 | `two_stage_original_3us_rw0p05_ft80` | `epgt_v1_full_traditional_ls` | 2.093953e-01 | -8.141297e+00 | 2.445635e-01 | 1.671729e-02 |

## Interpretation

- The best learned result is `two_stage_0p5us_rw0p02_ft20` with H NMSE `2.783713e-04`.
- Short finetuning is clearly better than the previous 80-step finetune at reconstruction_weight=0.05.
- Lower reconstruction weights prevent the reconstruction term from pulling the warmed-up physical parameters into unstable LS regions.
- The oracle LS baseline is still far lower, so the learned model remains precision-limited even after the training-scale fix.

## Artifacts

- `summary_scientific.csv`
- `all_variants_scientific.csv`
- `plots/best_h_nmse_by_run.png`
- `plots/all_variant_h_nmse_by_run.png`
