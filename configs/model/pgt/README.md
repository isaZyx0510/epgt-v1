# PGT Model Configs

Physics-Guided Transformer variants live here. All files extend the same
EPGT-v1 model family and change only the physics-guidance or uncertainty knobs,
so they can be compared under the same data scenarios.

| Config | Purpose |
| --- | --- |
| `epgt_v1_base.yaml` | Shared EPGT-v1 architecture and default physics settings. |
| `epgt_v1_bias_only.yaml` | Enables delay-Doppler cross-attention bias only. |
| `epgt_v1_mask_only.yaml` | Enables reliability masking without cross-attention bias. |
| `epgt_v1_loss_only.yaml` | Keeps physics heads/loss regularization without bias or mask. |
| `epgt_v1_full.yaml` | Enables the main full EPGT setting: bias, reliability mask, and sparsity loss. |
| `epgt_v1_uncertainty_ls.yaml` | Adds path-level delay/Doppler uncertainty outputs so EPGT can drive `learnable_weighted_ls`. |
