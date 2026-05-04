# Shared Phase Setting: combined

## Result

- `model_all` H NMSE: `9.512926e-05`
- `oracle_shared_phase` H NMSE: `3.520246e-05`
- total delay MAE: `4.820149e-08`
- RX offset MAE: `1.995057e-08`

## Replace-One Ranking

| rank | variant | H NMSE |
| ---: | --- | ---: |
| 1 | `oracle_ls_no_ai` | 1.752024e-10 |
| 2 | `oracle_all_physical_keep_model_uncertainty` | 4.647700e-06 |
| 3 | `oracle_shared_phase` | 3.520246e-05 |
| 4 | `oracle_total_delay` | 4.381204e-05 |
| 5 | `oracle_rel_delay` | 6.805546e-05 |
| 6 | `oracle_rx_offsets` | 8.541910e-05 |
| 7 | `oracle_path_phase` | 8.871875e-05 |
| 8 | `model_all` | 9.512926e-05 |
| 9 | `oracle_doppler` | 9.616516e-05 |
| 10 | `oracle_cfo` | 1.192280e-02 |

## Parameter Error

| parameter | pred min | pred max | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| `total_delay_s` | 9.028358e-08 | 1.038891e-07 | 4.820149e-08 | 5.587261e-08 |
| `rx_time_offsets_s` | -5.976287e-09 | 5.665186e-09 | 1.995057e-08 | 2.612147e-08 |
