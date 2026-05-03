# E2 Formal 9-Way Quick Readout

## Completed Matrix

```text
rows = 45
L_eff = [2, 4, 6, 8, 12]
scenarios = ['direct_h', 'epgt', 'epgt_weighted', 'hybrid', 'oracle', 'query', 'query_weighted', 'uncertainty_traditional', 'uncertainty_weighted']
```

## Best Non-Oracle Result Per L_eff

| L_eff | best variant | channel NMSE | channel NMSE dB |
| ---: | --- | ---: | ---: |
| 2 | `uncertainty_v1_traditional_ls` | 1.333790e-02 | -1.887428e+01 |
| 4 | `epgt_v1_uncertainty_weighted_ls` | 2.422317e-01 | -6.179774e+00 |
| 6 | `direct_h_original_v1` | 9.909610e-01 | -3.943656e-02 |
| 8 | `direct_h_original_v1` | 9.897798e-01 | -4.462437e-02 |
| 12 | `direct_h_original_v1` | 9.913016e-01 | -3.794625e-02 |

## Oracle Reference

| L_eff | oracle channel NMSE | oracle channel NMSE dB |
| ---: | ---: | ---: |
| 2 | 4.080275e-06 | -5.389310e+01 |
| 4 | 1.719429e-10 | -9.762097e+01 |
| 6 | 1.780487e-10 | -9.747029e+01 |
| 8 | 1.441342e-14 | -1.199379e+02 |
| 12 | 1.751449e-14 | -1.199246e+02 |

## Weighted LS Rows

| L_eff | variant | channel NMSE | weights min | weights max |
| ---: | --- | ---: | ---: | ---: |
| 2 | `epgt_v1_uncertainty_weighted_ls` | 1.049753e+00 | 5.000000e-01 | 1.586275e+00 |
| 2 | `query_v1_weighted_ls` | 2.934607e-01 | 5.000000e-01 | 1.626909e+00 |
| 2 | `uncertainty_v1_weighted_ls` | 1.059490e+00 | 5.000000e-01 | 1.625424e+00 |
| 4 | `epgt_v1_uncertainty_weighted_ls` | 2.422317e-01 | 5.000000e-01 | 1.632369e+00 |
| 4 | `query_v1_weighted_ls` | 8.399338e-01 | 5.000000e-01 | 1.635226e+00 |
| 4 | `uncertainty_v1_weighted_ls` | 1.232132e+00 | 5.000000e-01 | 1.628598e+00 |
| 6 | `epgt_v1_uncertainty_weighted_ls` | 4.148972e+01 | 5.000000e-01 | 1.632670e+00 |
| 6 | `query_v1_weighted_ls` | 8.032419e+01 | 5.000000e-01 | 1.628959e+00 |
| 6 | `uncertainty_v1_weighted_ls` | 2.500476e+02 | 5.000000e-01 | 1.631644e+00 |
| 8 | `epgt_v1_uncertainty_weighted_ls` | 9.426072e+02 | 5.000000e-01 | 1.631925e+00 |
| 8 | `query_v1_weighted_ls` | 3.724326e+02 | 5.000000e-01 | 1.635094e+00 |
| 8 | `uncertainty_v1_weighted_ls` | 3.973257e+01 | 5.000000e-01 | 1.630526e+00 |
| 12 | `epgt_v1_uncertainty_weighted_ls` | 2.171701e+01 | 5.000000e-01 | 1.630071e+00 |
| 12 | `query_v1_weighted_ls` | 1.001095e+02 | 5.000000e-01 | 1.622106e+00 |
| 12 | `uncertainty_v1_weighted_ls` | 1.673422e+02 | 5.000000e-01 | 1.632472e+00 |

## Artifacts

- `run_metadata.json`
- `results.json`
- `summary_scientific.csv`
- `summary_scientific.md`
- `plots/final_validation_nmse_vs_l_eff.png`
