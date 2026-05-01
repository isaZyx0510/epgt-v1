# Design: query_v1 + differentiable reconstruction training

## Architecture

`query_v1` 使用两段结构：

1. `SequenceTokenEncoder`
   - 输入 observation tokens。
   - 经过 `Linear + LayerNorm + GELU + TransformerEncoder`。
   - 输出完整 token sequence memory，不做 mean pooling。

2. `TransformerDecoder`
   - `global_query`: 输出 `total_delay_s` 和 `cfo_hz`。
   - `path_queries`: 每个 effective path 一个 query，输出 `rel_delay_s`、`doppler_hz`、`rel_delay_log_var`、`doppler_log_var`。
   - `rx_queries`: 每个 RX antenna 一个 query，输出 `rx_time_offsets_s`，第一个 RX 固定为 0 作为 reference。

这样 path-level、global-level、rx-level 参数有独立 query state。

## Differentiable LS

新增 `estimation/differentiable_ls.py`：

- `phase_terms_torch`
- `recover_path_gains_lstsq_torch`
- `reconstruct_channel_torch`
- `reconstruct_via_differentiable_ls`
- `complex_nmse_loss`

LS 使用 `torch.linalg.lstsq`，因此 reconstruction loss 可以反传到 Transformer 输出的 nonlinear parameters。

## Loss Modes

`train_hybrid_quick` 新增：

- `loss_mode="param"`: 原始 parameter-supervised training。
- `loss_mode="reconstruction"`: 只优化 full-grid `H_hat` NMSE。
- `loss_mode="param_plus_reconstruction"`: parameter warm-up 和 reconstruction objective 混合。

`ls_mode` 继续控制 LS backend：

- `traditional_ls`: uniform LS。
- `learnable_weighted_ls`: 使用 uncertainty-derived observation weights。

## Recommended Experiments

Baseline:

```powershell
uv run python scripts/train_hybrid.py --architecture original_v1 --loss-mode param
```

Query reconstruction:

```powershell
uv run python scripts/train_hybrid.py --architecture query_v1 --loss-mode reconstruction
```

Mixed training:

```powershell
uv run python scripts/train_hybrid.py --architecture query_v1 --loss-mode param_plus_reconstruction
```

Weighted LS:

```powershell
uv run python scripts/train_hybrid.py --architecture query_v1 --loss-mode param_plus_reconstruction --ls-mode learnable_weighted_ls
```

