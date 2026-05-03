# Script Map

Scripts are grouped by experiment stage. One experiment folder should not run a
combined multi-E pipeline.

## Experiment Scripts

| Folder | Script | Purpose | Default config |
| --- | --- | --- | --- |
| `e0/` | `run_oracle_ls.py` | Oracle LS sanity check. | `configs/data/e0_oracle_clean.yaml` |
| `e1/` | `run_e1_comparison.py` | Clean comparison of oracle LS, hybrid, and direct-H. | `configs/data/e1_clean_transformer.yaml` |
| `e2/` | `run_e2_training_scenarios.py` | Effective-path compression scenarios across `L_eff`. | `configs/data/e2_effective_paths.yaml`, `configs/model/hybrid_transformer.yaml`, optional `configs/model/pgt/*.yaml` |
| `e3/` | `run_snr_sweep.py` | AWGN robustness sweep over SNR values. | `configs/data/base_common_delay.yaml` |
| `e4/` | `run_ser_sweep.py` | Symbol-error robustness sweep over SER values. | `configs/data/base_common_delay.yaml` |
| `e5/` | `run_full_stress.py` | Full-stress comparison. | `configs/data/e5_full_stress.yaml` |
| `e6/` | `run_e6_pgt_comparison.py` | EPGT-v1 comparison. | `configs/data/e1_clean_transformer.yaml`, `configs/model/hybrid_transformer.yaml`, `configs/model/pgt/*.yaml` |
| `e6/` | `train_pgt_h_loss.py` | Single EPGT H-loss training run. | `configs/data/e1_clean_transformer.yaml`, `configs/model/pgt/epgt_v1_full.yaml` |
| `e6/` | `run_oracle_replacement_ablation.py` | Replace predicted parameter groups with oracle values for diagnostics. | `configs/data/e1_clean_transformer.yaml`, `configs/model/pgt/epgt_v1_full.yaml` |

## Generic Scripts

| Folder | Script | Purpose | Default config |
| --- | --- | --- | --- |
| `train/` | `train_hybrid.py` | Generic hybrid training entry point. | `configs/data/e1_clean_transformer.yaml` |
| `train/` | `train_direct_h.py` | Generic direct-H baseline training entry point. | `configs/data/e1_clean_transformer.yaml` |
| `tools/` | `build_dataset.py` | Inspect/generate dataset tensors. | `configs/data/base_common_delay.yaml` |
| `tools/` | `plot_results.py` | Plot metrics JSON as comparison or sweep figures. | metrics JSON path supplied by CLI |

## Config Organization

Current config layout is already mostly classified:

```text
configs/data/       E0-E6 data/experiment settings
configs/model/      generic model settings
configs/model/pgt/  EPGT model and physics-guidance variants
configs/physics/    reusable physics-prior fragments
```

Recommended next cleanup, if needed: add `configs/data/e0/` ... `configs/data/e6/`
subfolders only after updating all script defaults and docs. For now, the
filename prefixes (`e0_`, `e1_`, ...) are enough and avoid breaking paths.
