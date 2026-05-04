# Shared Phase Setting: baseline_repeat

## Result

- `model_all` H NMSE: `1.071753e-03`
- `oracle_shared_phase` H NMSE: `4.213264e-07`
- total delay MAE: `5.024383e-08`
- RX offset MAE: `2.014959e-08`

## Replace-One Ranking

| rank | variant | H NMSE |
| ---: | --- | ---: |
| 1 | `oracle_ls_no_ai` | 1.752024e-10 |
| 2 | `oracle_shared_phase` | 4.213264e-07 |
| 3 | `oracle_total_delay` | 1.062052e-06 |
| 4 | `oracle_rx_offsets` | 3.969817e-06 |
| 5 | `oracle_all_physical_keep_model_uncertainty` | 4.393298e-06 |
| 6 | `oracle_cfo` | 5.766071e-06 |
| 7 | `oracle_doppler` | 5.916519e-06 |
| 8 | `oracle_rel_delay` | 6.870474e-05 |
| 9 | `oracle_path_phase` | 6.874693e-05 |
| 10 | `model_all` | 1.071753e-03 |

## Parameter Error

| parameter | pred min | pred max | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| `total_delay_s` | 1.046162e-07 | 1.191773e-07 | 5.024383e-08 | 5.899220e-08 |
| `rx_time_offsets_s` | -2.442296e-08 | 1.758314e-08 | 2.014959e-08 | 2.677740e-08 |
