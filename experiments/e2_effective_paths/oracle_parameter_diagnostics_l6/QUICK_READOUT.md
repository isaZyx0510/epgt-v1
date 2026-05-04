# E2 Oracle Perturbation and Replace-One Diagnostics

## Settings

```text
L_eff = 6
architecture = uncertainty_v1
ls_mode = learnable_weighted_ls
max_rel_delay_s = 5.000000e-07
warmup_steps = 80
finetune_steps = 20
reconstruction_weight = 0.02
train_batches = 32
val_batches = 8
```

## Replace-One Ranking

| rank | variant | H NMSE | H NMSE dB | weights min | weights max |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `oracle_ls_no_ai` | 1.752024e-10 | -9.754948e+01 |  |  |
| 2 | `oracle_all_physical_keep_model_uncertainty` | 4.435265e-06 | -5.353680e+01 | 5.000000e-01 | 1.653714e+00 |
| 3 | `oracle_shared_phase` | 5.739573e-06 | -5.257980e+01 | 5.000000e-01 | 1.653714e+00 |
| 4 | `oracle_total_delay` | 1.094044e-05 | -4.975946e+01 | 5.000000e-01 | 1.653714e+00 |
| 5 | `oracle_rx_offsets` | 3.643346e-05 | -4.476786e+01 | 5.000000e-01 | 1.653714e+00 |
| 6 | `oracle_cfo` | 4.259268e-05 | -4.395156e+01 | 5.000000e-01 | 1.653714e+00 |
| 7 | `oracle_doppler` | 4.501444e-05 | -4.430718e+01 | 5.000000e-01 | 1.653714e+00 |
| 8 | `oracle_path_phase` | 6.863017e-05 | -4.185433e+01 | 5.000000e-01 | 1.653714e+00 |
| 9 | `model_all` | 1.676899e-03 | -4.004533e+01 | 5.000000e-01 | 1.653714e+00 |
| 10 | `oracle_rel_delay` | 1.781607e-03 | -3.871734e+01 | 5.000000e-01 | 1.653714e+00 |

## Perturbation Takeaway

The perturbation CSV records how quickly oracle LS degrades when one physical parameter is artificially noised.
Compare the curves to the model-all H NMSE to estimate how precise the learned parameter must be.

## Initial Interpretation

`model_all` H NMSE is `1.676899e-03`.
If replacing one parameter with oracle produces a large drop in H NMSE, that parameter is the likely bottleneck.

## Artifacts

- `run_metadata.json`
- `trained_result.json`
- `oracle_perturbation.csv`
- `replace_one_parameter.csv`
- `plots/oracle_perturbation_*.png`
- `plots/replace_one_parameter_h_nmse.png`
