# Shared Phase Setting: normalization_only

## Result

- `model_all` H NMSE: `1.721505e-04`
- `oracle_shared_phase` H NMSE: `6.795696e-05`
- total delay MAE: `4.780372e-08`
- RX offset MAE: `1.990408e-08`

## Replace-One Ranking

| rank | variant | H NMSE |
| ---: | --- | ---: |
| 1 | `oracle_ls_no_ai` | 1.752024e-10 |
| 2 | `oracle_all_physical_keep_model_uncertainty` | 4.493256e-06 |
| 3 | `oracle_rel_delay` | 6.636324e-05 |
| 4 | `oracle_path_phase` | 6.734506e-05 |
| 5 | `oracle_shared_phase` | 6.795696e-05 |
| 6 | `oracle_total_delay` | 8.175870e-05 |
| 7 | `oracle_cfo` | 1.696211e-04 |
| 8 | `model_all` | 1.721505e-04 |
| 9 | `oracle_doppler` | 1.721912e-04 |
| 10 | `oracle_rx_offsets` | 1.726361e-04 |

## Parameter Error

| parameter | pred min | pred max | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| `total_delay_s` | 9.033649e-08 | 1.030935e-07 | 4.780372e-08 | 5.554367e-08 |
| `rx_time_offsets_s` | -5.722862e-09 | 4.494370e-09 | 1.990408e-08 | 2.603260e-08 |
