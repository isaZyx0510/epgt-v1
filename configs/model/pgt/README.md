# PGT Model Configs

Put Physics-Guided Transformer model variants here. Use these configs to switch
between bias-only, mask-only, loss-only, and full physics-guided attention
ablations while reusing the same data scenarios.

- `epgt_v1_uncertainty_ls.yaml`: enables path-level delay/Doppler uncertainty
  outputs so EPGT can drive `learnable_weighted_ls`.
