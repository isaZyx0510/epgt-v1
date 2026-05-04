# Design: E2 Training Diagnostics

## Diagnostic Trail

The E2 debugging sequence is organized as a set of immutable result packages:

```text
experiments/e2_effective_paths/param_only_physical_models/
experiments/e2_effective_paths/two_stage_param_warmup_reconstruction/
experiments/e2_effective_paths/diagnostic_l6_rel_delay_0p5us/
experiments/e2_effective_paths/diag_l6_rel_delay_0p5us_rw0p01_ft20/
experiments/e2_effective_paths/diag_l6_rel_delay_0p5us_rw0p01_ft40/
experiments/e2_effective_paths/diag_l6_rel_delay_0p5us_rw0p02_ft20/
experiments/e2_effective_paths/diag_l6_rel_delay_0p5us_rw0p02_ft40/
experiments/e2_effective_paths/diagnostic_l6_reconstruction_weight_finetune_sweep/
```

Each package writes `run_metadata.json`, `results.json`, summary tables, plots,
and a `QUICK_READOUT.md` when the result is used for interpretation.

## Findings

The param-only run showed that learned physical-parameter models are not
fundamentally unable to fit E2 labels. Pure normalized physical-parameter MSE
stayed in a stable range for many variants, which points away from model
capacity as the primary bottleneck.

The first two-stage run improved most `L_eff` values, but L_eff=6 regressed:

```text
param-only L=6 best H NMSE:        2.069003e-02
two-stage original L=6 best H NMSE:2.093953e-01
```

The L=6 normalization diagnostic tightened `max_rel_delay_s` from `3.0e-6` to
`5.0e-7` and improved best H NMSE to:

```text
1.029182e-02
```

The low-cost sweep then showed that shorter and weaker reconstruction
fine-tuning is much more stable:

| Setting | Best variant | L=6 H NMSE |
| --- | --- | ---: |
| `max_rel_delay_s=0.5us`, `rw=0.01`, `finetune=20` | `epgt_v1_uncertainty_weighted_ls` | 4.607480e-04 |
| `max_rel_delay_s=0.5us`, `rw=0.01`, `finetune=40` | `uncertainty_v1_weighted_ls` | 5.308669e-04 |
| `max_rel_delay_s=0.5us`, `rw=0.02`, `finetune=20` | `uncertainty_v1_weighted_ls` | 2.783713e-04 |
| `max_rel_delay_s=0.5us`, `rw=0.02`, `finetune=40` | `query_v1_weighted_ls` | 5.402334e-04 |

The current recommended diagnostic setting is:

```text
max_rel_delay_s = 5.0e-7
loss_mode = two_stage
warmup_steps = 80
finetune_steps = 20
reconstruction_weight = 0.02
weighted LS variants prioritized
```

## Oracle Parameter Diagnostics

The follow-up oracle diagnostics are stored in:

```text
experiments/e2_effective_paths/oracle_parameter_diagnostics_l6/
experiments/e2_effective_paths/oracle_parameter_diagnostics_l6_with_errors/
```

They train `uncertainty_v1_weighted_ls` at the recommended L=6 setting and run
two probes:

1. oracle perturbation sensitivity: add synthetic noise to one oracle parameter
   family at a time;
2. replace-one ablation: evaluate model predictions while replacing one
   physical parameter family with oracle values.

The replace-one ablation identifies shared phase parameters as the current
dominant bottleneck:

```text
model_all H NMSE:           1.852403e-03
oracle_shared_phase H NMSE: 9.332557e-06
oracle_all physical H NMSE: 4.396195e-06
oracle_ls_no_ai H NMSE:     1.752024e-10
```

The parameter error file shows that the learned shared-delay parameters are
compressed toward their means:

```text
total_delay target range: 6.743e-10 to 1.984e-07
total_delay pred range:   9.265e-08 to 1.059e-07
total_delay MAE:          4.860e-08

rx_offset target range:   -4.973e-08 to 4.984e-08
rx_offset pred range:     -6.818e-09 to 6.064e-09
rx_offset MAE:            1.984e-08
```

This shifts the next optimization target from relative path-delay precision to
shared phase supervision: total delay and RX time offsets need stronger,
less-collapsed learning signals before path-matching loss is prioritized.

## Shared Phase Supervision Follow-Up

The shared phase follow-up is stored in:

```text
experiments/e2_effective_paths/diagnostic_l6_shared_phase_supervision/
```

It evaluates four L=6 settings with the same two-stage schedule:

```text
max_rel_delay_s = 5.0e-7
warmup_steps = 80
finetune_steps = 20
reconstruction_weight = 0.02
architecture = uncertainty_v1
ls_mode = learnable_weighted_ls
```

| Setting | Change | model_all H NMSE | oracle_shared_phase H NMSE |
| --- | --- | ---: | ---: |
| `baseline_repeat` | original shared-phase weights and bounds | 1.071753e-03 | 4.213264e-07 |
| `normalization_only` | tighter total-delay/RX-offset bounds | 1.721505e-04 | 6.795696e-05 |
| `loss_weight_only` | `tau0_weight=4`, `rx_offset_weight=4` | 4.494386e-05 | 8.944284e-06 |
| `combined` | tighter bounds plus stronger weights | 9.512926e-05 | 3.520246e-05 |

The best setting is `loss_weight_only`, which improves the previous L=6 best
H NMSE from `2.783713e-04` to `4.494386e-05`. The combined setting also improves
over the previous best, but it is not the winner.

Parameter MAE changes only mildly across these settings, so the improvement is
not simply a reduction in average parameter error. The stronger shared-phase
loss appears to guide the learned parameters into a region that is better
aligned with LS reconstruction geometry.

## Runner Interface

`scripts/e2/run_e2_training_scenarios.py` accepts optional model-scale
overrides:

```text
--max-rel-delay-s
--max-doppler-hz
--max-total-delay-s
--max-cfo-hz
--max-rx-time-offset-s
--tau0-loss-weight
--cfo-loss-weight
--delay-loss-weight
--doppler-loss-weight
--rx-offset-loss-weight
```

These overrides are passed through `apply_cli_model_overrides` and recorded in
`run_metadata.json`. They allow scale diagnostics without editing YAML configs.

## Rollback Strategy

There are two rollback levels:

1. Code rollback:
   use Git to revert the runner/training helper edits if a diagnostic change
   should not remain in the working tree.
2. Experiment rollback:
   leave completed result directories intact for auditability, or explicitly
   delete only the affected `experiments/e2_effective_paths/...` result folder
   when the run is known to be invalid.

Because result directories are report artifacts, prefer moving or renaming an
invalid run over deleting it unless disk space or clarity requires deletion.
