# Task List

## 1. Diagnostics

- [x] Run param-only physical-parameter diagnostics.
- [x] Run two-stage param warmup plus reconstruction fine-tuning.
- [x] Run L_eff=6 `max_rel_delay_s=0.5us` diagnostic.
- [x] Run L_eff=6 reconstruction weight and fine-tune length sweep.
- [x] Run oracle perturbation sensitivity diagnostics.
- [x] Run replace-one-parameter ablation diagnostics.
- [x] Add parameter error statistics for model-vs-oracle labels.
- [x] Run L_eff=6 shared phase supervision follow-up.

## 2. Output Artifacts

- [x] Save result packages under `experiments/e2_effective_paths/`.
- [x] Write `QUICK_READOUT.md` for interpreted diagnostic packages.
- [x] Generate train and validation curves.
- [x] Generate oracle-comparison H NMSE plots where relevant.
- [x] Generate a combined L_eff=6 sweep summary package.
- [x] Save oracle perturbation, replace-one, and parameter error CSVs.
- [x] Save shared-phase supervision summary, per-setting artifacts, and plots.

## 3. Runner Support

- [x] Add CLI model-scale overrides to the E2 runner.
- [x] Add CLI physical-parameter loss weights to the E2 runner.
- [x] Record model-scale overrides in run metadata.
- [x] Record physical-parameter loss weights in run metadata and summaries.
- [x] Preserve existing formal E2 defaults when overrides are omitted.

## 4. Follow-Up

- [ ] Run a full E2 sweep with the recommended diagnostic setting.
- [ ] Decide whether `max_rel_delay_s=0.5us` should become a default config.
- [x] Strengthen total-delay and RX-offset supervision to reduce shared phase
      parameter collapse.
- [ ] Run a full E2 sweep with `tau0_weight=4` and `rx_offset_weight=4`.
- [ ] Consider path-matching loss only after training-scale diagnostics are
      exhausted.
