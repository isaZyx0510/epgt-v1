# E2 L=6 Shared Phase Supervision Sweep

| setting | model_all H NMSE | oracle_shared_phase H NMSE | total_delay MAE | rx_offset MAE |
| --- | ---: | ---: | ---: | ---: |
| `loss_weight_only` | 4.494386e-05 | 8.944284e-06 | 4.853420e-08 | 2.001560e-08 |
| `combined` | 9.512926e-05 | 3.520246e-05 | 4.820149e-08 | 1.995057e-08 |
| `normalization_only` | 1.721505e-04 | 6.795696e-05 | 4.780372e-08 | 1.990408e-08 |
| `baseline_repeat` | 1.071753e-03 | 4.213264e-07 | 5.024383e-08 | 2.014959e-08 |

## Interpretation

- `model_all` is the learned model with no oracle replacement.
- A smaller gap between `model_all` and `oracle_shared_phase` means the shared phase bottleneck is reduced.
- Compare total delay and RX offset MAE to detect whether prediction collapse improved.

## Main Finding

The best setting is `loss_weight_only`:

```text
tau0_loss_weight = 4
rx_offset_loss_weight = 4
max_total_delay_s = 1.0e-6
max_rx_time_offset_s = 2.0e-7
model_all H NMSE = 4.494386e-05
```

This improves over the previous L=6 best result:

```text
previous best: 2.783713e-04
new best:      4.494386e-05
```

The `combined` setting also beats the previous best:

```text
combined H NMSE = 9.512926e-05
```

The shared-phase gap shrinks most clearly for `loss_weight_only`:

```text
model_all H NMSE:           4.494386e-05
oracle_shared_phase H NMSE: 8.944284e-06
gap: about 5x
```

Parameter MAE changes only mildly across settings, so the improvement is not
just a simple average-error reduction. The stronger shared-phase loss appears
to move the learned parameters into a region that is much better aligned with
the LS reconstruction geometry.

## Artifacts

- `summary_scientific.csv`
- `all_variants_scientific.csv`
- `plots/best_h_nmse_by_shared_phase_setting.png`
- `plots/shared_phase_parameter_error.png`
