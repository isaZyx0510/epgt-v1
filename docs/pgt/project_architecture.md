# New Project Folder Architecture

This project keeps one thesis workspace and adds a new OpenSpec-managed
Physics-Guided Transformer research line.

## Top-Level Layout

```text
Thesis Transformer version1/
  pyproject.toml
  README.md

  openspec/
    config.yaml
    changes/
      build-thesis-transformer-v1/
      build-experiment-pipeline-v1/
      build-physics-guided-transformer/

  configs/
    data/
    model/
    physics/

  docs/
    pgt/

  scripts/
    pgt/

  src/
    thesis_transformer_v1/
      data/
      estimation/
      experiments/
      metrics/
      models/
      physics/
      tdlc/

  tests/

  experiments/
    e0_oracle_clean/
    e1_clean_transformer/
    e2_effective_paths/
    e3_awgn/
    e4_symbol_error/
    e5_full_stress/
    e6_physics_guided_attention/
```

## OpenSpec Layer

```text
openspec/
  config.yaml
  changes/
    build-physics-guided-transformer/
      .openspec.yaml
      proposal.md
      design.md
      tasks.md
```

Use this folder as the development control plane.

- `proposal.md`: why the new architecture exists, what it includes, and what it
  deliberately excludes.
- `design.md`: technical design, folder boundaries, attention formulation,
  configs, experiments, and acceptance criteria.
- `tasks.md`: implementation checklist from documentation to physics modules,
  model code, configs, scripts, tests, and thesis artifacts.

## Runtime Source Layer

```text
src/thesis_transformer_v1/
  tdlc/          # copied and adapted TDL-C generation utilities
  data/          # dataset configs, tokenization, labels, SER corruption
  estimation/    # LS recovery and channel reconstruction
  experiments/   # reusable training/evaluation/plotting utilities
  metrics/       # NMSE and related metrics
  models/        # baseline models plus PGT model family
  physics/       # physical priors, masks, losses, diagnostics
```

The key rule is:

- `physics/` describes the communication-system constraints.
- `models/pgt/` uses those constraints inside neural attention.

## Model Layer

```text
src/thesis_transformer_v1/models/
  transformer.py             # current baseline Transformer
  original_v1_transformer.py # frozen earlier baseline
  factory.py                 # architecture selection
  pgt/
    config.py
    attention.py
    encoder.py
    heads.py
    hybrid.py
    direct_h.py
```

PGT should be added as a new architecture option rather than replacing
`current`.

## Physics Layer

```text
src/thesis_transformer_v1/physics/
  priors.py
  attention_bias.py
  masks.py
  losses.py
  diagnostics.py
```

This layer should be independent of a specific neural architecture where
possible. That makes it easier to test the physics assumptions and reuse them in
future models.

## Config Layer

```text
configs/
  data/
    e0_oracle_clean.yaml
    e1_clean_transformer.yaml
    e2_effective_paths.yaml
    e3_awgn.yaml
    e4_symbol_error.yaml
    e5_full_stress.yaml

  model/
    direct_h_baseline.yaml
    hybrid_transformer.yaml
    pgt/
      epgt_v1_base.yaml
      epgt_v1_bias_only.yaml
      epgt_v1_mask_only.yaml
      epgt_v1_loss_only.yaml
      epgt_v1_full.yaml

  physics/
    base_ofdm_priors.yaml
    delay_doppler_bias.yaml
    pilot_reliability_mask.yaml
    impairment_constraints.yaml
```

Data scenarios stay in `configs/data/`; PGT-specific physical assumptions stay
in `configs/physics/`; architecture switches stay in `configs/model/pgt/`.

## Experiment Layer

```text
experiments/e6_physics_guided_attention/
  README.md
  baseline_current/
  epgt_v1_bias_only/
  epgt_v1_mask_only/
  epgt_v1_loss_only/
  epgt_v1_full/
  ablations/
  attention_diagnostics/
```

The first PGT paper/thesis comparison should be:

1. current hybrid Transformer baseline
2. PGT bias-only
3. PGT mask-only
4. PGT loss-only
5. full PGT

## Script Layer

```text
scripts/pgt/
  train_pgt.py
  train_pgt_h_loss.py
  run_e6_pgt_comparison.py
  run_e2_e3_pgt_comparison.py
  run_pgt_ablation.py
  run_oracle_replacement_ablation.py
```

Keep PGT scripts separate until the interface stabilizes. Shared logic can later
move into `src/thesis_transformer_v1/experiments/`.

## Test Layer

```text
tests/
  test_physics_priors.py
  test_pgt_attention.py
  test_pgt_model_shapes.py
  test_pgt_losses.py
```

The first tests should protect shapes, masks, output contracts, and the
physics-disabled attention fallback.
