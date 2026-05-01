# Thesis Transformer version1

Transformer-based MIMO-OFDM channel-estimation experiments with compressed
effective paths, receiver timing impairment, differentiable LS recovery, and
EPGT physics-guided attention.

## Current Focus

The current development line trains hybrid / EPGT models with full-grid channel
reconstruction loss:

```text
observation tokens -> model -> nonlinear effective-path parameters
-> differentiable LS -> H_hat -> NMSE(H_hat, H_true)
```

Parameter-supervised training is still available for simulation diagnostics by
passing `--loss-mode param`, but it is no longer the default.

## Project Layout

```text
configs/
  data/          Experiment data settings: E0-E5 and shared common-delay config
  model/         Baseline model configs
  model/pgt/     EPGT-v1 model and physics-guidance configs
  physics/       Reusable physics-prior fragments

src/thesis_transformer_v1/
  data/          Dataset generation, tokenization, labels for diagnostics
  estimation/    LS, differentiable LS, weighted LS plugins
  experiments/   Training, evaluation, sweeps, metrics IO
  models/        Baselines, query models, uncertainty models, EPGT
  physics/       Priors, attention bias, masks, losses, diagnostics
  tdlc/          TDL-C channel and OFDM utilities

scripts/
  train_hybrid.py           Generic hybrid training entry point
  train_direct_h.py         Direct-H baseline training
  run_*_sweep.py            Standard E1-E5 experiment scripts
  pgt/                      EPGT-specific training, ablation, diagnostics

docs/
  pgt/                      EPGT architecture notes, diagrams, slide assets

experiments/
  e0_* ... e6_*             Saved metrics from named experiment runs

openspec/
  changes/                  Chinese OpenSpec plans/designs/tasks

tests/
  Unit and smoke tests for configs, models, physics helpers, and training paths
```

## Common Commands

EPGT H-loss quick run:

```powershell
uv run --extra dev python scripts\pgt\train_pgt.py --steps 25 --eval-interval 5 --train-batches 2 --val-batches 1 --d-model 32 --num-layers 1 --nhead 4 --dim-feedforward 64 --dropout 0.0
```

E6 comparison:

```powershell
uv run --extra dev python scripts\pgt\run_e6_pgt_comparison.py --steps 25 --eval-interval 5 --train-batches 2 --val-batches 1 --d-model 32 --num-layers 1 --nhead 4 --dim-feedforward 64 --dropout 0.0
```

Tests:

```powershell
uv run --extra dev pytest
```

Lint:

```powershell
uv run --extra dev ruff check src tests scripts
```

## Key Docs

- `docs/pgt/epgt_v1_architecture_diagram.md`
- `docs/pgt/project_architecture.md`
- `docs/pgt/folder_plan.md`
- `docs/ls_plugin_notes.md`
- `docs/query_reconstruction_notes.md`
- `openspec/changes/build-physics-guided-transformer/`
