# Physics-Guided Transformer Folder Plan

This research line introduces a physics-guided attention mechanism for the
MIMO-OFDM channel estimation setting already used by this project. The key idea
is to keep the current baselines intact, then add a new PGT family whose
attention logits, masks, auxiliary losses, and diagnostics can be ablated
independently.

## Naming

- Short name: `epgt_v1`
- Full name: `effective_path_guided_transformer_v1`
- First experiment stage: `e6_physics_guided_attention`

Use `epgt_v1_*` for configs, metrics, and plots so results can be separated
from the existing `hybrid_transformer` and `direct_h` baselines.

## Source Layout

```text
src/thesis_transformer_v1/
  physics/
    __init__.py
    priors.py              # Delay, Doppler, CFO, pilot/data reliability priors.
    attention_bias.py      # Convert physical priors into attention-logit bias.
    masks.py               # Hard/soft masks over OFDM symbols, subcarriers, antennas.
    losses.py              # Physics consistency and regularization losses.
    diagnostics.py         # Attention entropy, prior alignment, constraint violation.

  models/
    pgt/
      __init__.py
      config.py            # PGT-specific dataclass/config validation.
      attention.py         # Physics-guided multi-head attention layer.
      encoder.py           # Stack of PGT encoder blocks.
      heads.py             # Hybrid parameter head and optional direct-H head.
      hybrid.py            # Main PhysicsGuidedHybridTransformer.
      decoder.py           # Parameterized channel reconstruction helpers.
```

Recommended design boundary:

- `physics/` should contain domain rules that are independent of the neural
  architecture.
- `models/pgt/` should contain PyTorch modules that consume those rules.
- Existing `models/transformer.py` should remain the baseline reference.

## Config Layout

```text
configs/
  physics/
    base_ofdm_priors.yaml
    delay_doppler_bias.yaml
    pilot_reliability_mask.yaml
    impairment_constraints.yaml

  model/
    pgt/
      epgt_v1_base.yaml
      epgt_v1_bias_only.yaml
      epgt_v1_mask_only.yaml
      epgt_v1_loss_only.yaml
      epgt_v1_full.yaml
```

Config split:

- `configs/physics/`: physical assumptions and constraint weights.
- `configs/model/pgt/`: neural architecture size, attention variant, and which
  physics modules are enabled.
- Existing `configs/data/`: keep scenario generation and robustness conditions.

## Experiment Layout

```text
experiments/
  e6_physics_guided_attention/
    README.md
    baseline_current/
    epgt_v1_bias_only/
    epgt_v1_mask_only/
    epgt_v1_loss_only/
    epgt_v1_full/
    ablations/
    attention_diagnostics/
```

Suggested first comparisons:

1. Current hybrid transformer.
2. PGT with attention-logit bias only.
3. PGT with soft physical masks only.
4. PGT with physics regularization loss only.
5. Full PGT: bias + mask + loss.

## Scripts

```text
scripts/
  pgt/
    train_pgt.py
    train_pgt_h_loss.py
    run_e6_pgt_comparison.py
    run_e2_e3_pgt_comparison.py
    run_pgt_ablation.py
    run_oracle_replacement_ablation.py
```

Keep PGT scripts separate at first. After the interface stabilizes, common
training/evaluation logic can be folded back into `src/thesis_transformer_v1/experiments/`.

## Tests

Add these tests when implementation starts:

```text
tests/
  test_physics_priors.py
  test_pgt_attention.py
  test_pgt_model_shapes.py
  test_pgt_losses.py
```

Minimum invariants:

- Attention bias shape is broadcastable to `[batch, heads, tokens, tokens]`.
- Physical masks never remove all valid observations for a sample.
- PGT output contract matches the current `HybridTransformer` output keys.
- Setting all physics weights to zero recovers ordinary self-attention behavior.
