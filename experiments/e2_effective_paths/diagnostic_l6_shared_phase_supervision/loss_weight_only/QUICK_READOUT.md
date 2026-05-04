# Shared Phase Setting: loss_weight_only

## Result

- `model_all` H NMSE: `4.494386e-05`
- `oracle_shared_phase` H NMSE: `8.944284e-06`
- total delay MAE: `4.853420e-08`
- RX offset MAE: `2.001560e-08`

## Replace-One Ranking

| rank | variant | H NMSE |
| ---: | --- | ---: |
| 1 | `oracle_ls_no_ai` | 1.752024e-10 |
| 2 | `oracle_all_physical_keep_model_uncertainty` | 4.440078e-06 |
| 3 | `oracle_shared_phase` | 8.944284e-06 |
| 4 | `oracle_total_delay` | 1.099088e-05 |
| 5 | `oracle_rx_offsets` | 4.100970e-05 |
| 6 | `oracle_doppler` | 4.490682e-05 |
| 7 | `model_all` | 4.494386e-05 |
| 8 | `oracle_cfo` | 4.586657e-05 |
| 9 | `oracle_path_phase` | 6.315515e-05 |
| 10 | `oracle_rel_delay` | 6.754050e-05 |

## Parameter Error

| parameter | pred min | pred max | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| `total_delay_s` | 9.722575e-08 | 1.091475e-07 | 4.853420e-08 | 5.650103e-08 |
| `rx_time_offsets_s` | -2.185906e-08 | 1.788878e-08 | 2.001560e-08 | 2.659466e-08 |
