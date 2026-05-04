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
| 2 | `oracle_all_physical_keep_model_uncertainty` | 4.396195e-06 | -5.357700e+01 | 5.000000e-01 | 1.653714e+00 |
| 3 | `oracle_shared_phase` | 9.332557e-06 | -5.044926e+01 | 5.000000e-01 | 1.653714e+00 |
| 4 | `oracle_cfo` | 4.146745e-05 | -4.426647e+01 | 5.000000e-01 | 1.653714e+00 |
| 5 | `oracle_rx_offsets` | 4.522338e-05 | -4.422238e+01 | 5.000000e-01 | 1.653714e+00 |
| 6 | `oracle_doppler` | 4.727943e-05 | -4.427095e+01 | 5.000000e-01 | 1.653714e+00 |
| 7 | `oracle_path_phase` | 6.701064e-05 | -4.195551e+01 | 5.000000e-01 | 1.653714e+00 |
| 8 | `oracle_total_delay` | 1.752699e-04 | -4.632844e+01 | 5.000000e-01 | 1.653714e+00 |
| 9 | `oracle_rel_delay` | 1.553209e-03 | -3.895543e+01 | 5.000000e-01 | 1.653714e+00 |
| 10 | `model_all` | 1.852403e-03 | -4.005133e+01 | 5.000000e-01 | 1.653714e+00 |

## Perturbation Takeaway

The perturbation CSV records how quickly oracle LS degrades when one physical parameter is artificially noised.
Compare the curves to the model-all H NMSE to estimate how precise the learned parameter must be.

## Initial Interpretation

`model_all` H NMSE is `1.852403e-03`.
If replacing one parameter with oracle produces a large drop in H NMSE, that parameter is the likely bottleneck.

The strongest improvement comes from replacing the shared phase parameters
(`total_delay_s`, `cfo_hz`, and `rx_time_offsets_s`) with oracle values:

```text
model_all H NMSE:          1.852403e-03
oracle_shared_phase H NMSE:9.332557e-06
oracle_all physical H NMSE:4.396195e-06
oracle_ls_no_ai H NMSE:    1.752024e-10
```

The learned shared-delay parameters are compressed toward the mean:

```text
total_delay target range: 6.743e-10 to 1.984e-07
total_delay pred range:   9.265e-08 to 1.059e-07
total_delay MAE:          4.860e-08

rx_offset target range:   -4.973e-08 to 4.984e-08
rx_offset pred range:     -6.818e-09 to 6.064e-09
rx_offset MAE:            1.984e-08
```

This points to shared phase parameter collapse as the current dominant
bottleneck, more than relative path-delay precision.

## Artifacts

- `run_metadata.json`
- `trained_result.json`
- `oracle_perturbation.csv`
- `replace_one_parameter.csv`
- `parameter_errors.csv`
- `plots/oracle_perturbation_*.png`
- `plots/replace_one_parameter_h_nmse.png`
