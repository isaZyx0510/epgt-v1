# Add E2 Training Diagnostics

## Why

The formal E2 nine-way comparison showed that pure H reconstruction training can
be unstable for learned physical-parameter models, especially as `L_eff`
increases. The oracle LS reference remained extremely strong, so the failure was
not caused by LS being inherently unable to reconstruct the channel when
physical parameters are correct.

The project needs a recorded diagnostic trail that explains which parts of the
training pipeline were limiting performance, which low-cost changes helped, and
which settings should be used for the next E2 comparison runs.

## What

- Record the param-only diagnostic experiment showing that models can learn
  physical parameters under normalized supervision.
- Record the two-stage training experiment:
  param-only warmup followed by `param_plus_reconstruction` fine-tuning.
- Record the L_eff=6 diagnosis that tightened `max_rel_delay_s` from `3.0e-6`
  to `5.0e-7`.
- Record the L_eff=6 sweep over reconstruction weight and fine-tune length.
- Add runner support for model normalization overrides such as
  `--max-rel-delay-s`.
- Use result directories under `experiments/e2_effective_paths/` as the source
  of truth for generated tables, curves, and quick readouts.

## Non-Goals

- Do not make the L_eff=6 diagnostic setting the global default yet.
- Do not change the E2 data-generation config.
- Do not introduce path-matching loss in this change.
- Do not remove the oracle LS or formal nine-way comparison artifacts.
