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

The active EPGT branch also supports an uncertainty-weighted LS variant:

```text
EPGT path head -> rel_delay_log_var / doppler_log_var
-> observation weights for learnable_weighted_ls
-> weighted LS -> H_hat -> H reconstruction loss
```

This is separate from the reliability mask. Path uncertainty is a learnable
model output that describes delay/Doppler confidence per effective path;
reliability is an observation-token annotation that marks whether an observed
RE is trustworthy, for example under symbol errors.

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
uv run --extra dev python scripts\pgt\train_pgt_h_loss.py --config configs\data\e6_h_loss_l5.yaml --model-config configs\model\pgt\epgt_v1_bias_only.yaml --steps 60 --eval-interval 20 --train-batches 4 --val-batches 2 --loss-mode reconstruction
```

EPGT uncertainty-weighted LS run:

```powershell
uv run --extra dev python scripts\pgt\train_pgt_h_loss.py --config configs\data\e6_h_loss_l5.yaml --model-config configs\model\pgt\epgt_v1_uncertainty_ls.yaml --steps 200 --lr 1e-3 --eval-interval 20 --train-batches 4 --val-batches 2 --ls-mode learnable_weighted_ls --loss-mode reconstruction --uncertainty-regularization-weight 1e-4
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
