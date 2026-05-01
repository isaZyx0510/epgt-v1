# Experiments

This directory keeps saved metric JSON files and generated plots from named
experiment runs.

Current retained groups:

```text
e0_oracle_clean/                 Oracle LS sanity check
e1_clean_transformer/            Clean baseline comparison
e2_effective_paths/              L_eff sweep
e3_awgn/                         AWGN robustness sweep
e4_symbol_error/                 Symbol-error robustness sweep
e5_full_stress/                  Combined stress setting
e6_physics_guided_attention/     EPGT-v1 comparisons, ablations, diagnostics
query_reconstruction_comparison/ Query/uncertainty model comparison
two_stage_reconstruction_sweep/  Historical two-stage reconstruction sweep
```

Temporary smoke-test outputs, ad hoc quick runs, and test artifacts are ignored
by `.gitignore` and can be regenerated from scripts.
