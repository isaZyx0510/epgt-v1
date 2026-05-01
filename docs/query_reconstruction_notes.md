# Query Reconstruction Notes

## 为什么做这个版本

`original_v1` 是 baseline：它把所有 encoded tokens 做 mean pooling，然后预测 nonlinear parameters。这个设计简单稳定，但会压缩掉 subcarrier/time 上的 phase slope 信息。

`query_v1` 的目标是保留 token-level structure，并让训练目标直接对齐最终的 channel reconstruction：

```text
tokens
  -> query_v1
  -> delay / Doppler / total delay / CFO / rx time offset / uncertainty
  -> differentiable LS
  -> path gains
  -> H_hat
  -> NMSE(H_hat, H_true)
```

## 输入输出

输入仍然是最后两个 OFDM symbols 的 observation tokens。

输出包括：

- `total_delay_s`
- `cfo_hz`
- `rel_delay_s [B, L_eff]`
- `doppler_hz [B, L_eff]`
- `rx_time_offsets_s [B, N_r]`
- `rel_delay_log_var [B, L_eff]`
- `doppler_log_var [B, L_eff]`

## 三种训练目标

`param`:

只学习 pseudo-label propagation parameters。适合作为 baseline 或 warm-up。

`reconstruction`:

只学习 `H_hat` 和 `H_true` 的 full-grid NMSE。适合解决 `L_eff < L_true` 时没有严格 effective path label 的问题。

`param_plus_reconstruction`:

同时使用 parameter supervision 和 reconstruction loss。适合作为更稳定的中间方案。

## 推荐命令

原始 baseline：

```powershell
uv run python scripts/train_hybrid.py --architecture original_v1 --loss-mode param
```

改进版 reconstruction training：

```powershell
uv run python scripts/train_hybrid.py --architecture query_v1 --loss-mode reconstruction --steps 100 --lr 1e-4
```

使用 uncertainty weighted LS：

```powershell
uv run python scripts/train_hybrid.py --architecture query_v1 --loss-mode param_plus_reconstruction --ls-mode learnable_weighted_ls --steps 100 --lr 1e-4
```

Two-stage training：

```powershell
uv run python scripts/train_hybrid.py --architecture query_v1 --loss-mode two_stage --warmup-steps 80 --steps 160 --reconstruction-weight 0.05 --lr 1e-4
```

Reconstruction weight sweep：

```powershell
uv run python scripts/run_two_stage_reconstruction_sweep.py --weights 0.01 0.05 0.1 0.5 --warmup-steps 80 --finetune-steps 80 --lr 1e-4
```

## 当前 E1 clean 结果

在 `E1 clean`、`traditional_ls`、`80 warmup + 80 finetune`、`train_batches=8`、`val_batches=3` 下：

| Method | Reconstruction weight | Final channel NMSE |
|---|---:|---:|
| `original_v1 + param loss`, 160 steps | - | `1.252e-08` |
| `query_v1 + two_stage` | `0.01` | `2.717e-09` |
| `query_v1 + two_stage` | `0.05` | `1.538e-09` |
| `query_v1 + two_stage` | `0.1` | `2.136e-09` |
| `query_v1 + two_stage` | `0.5` | `5.796e-09` |

当前最好的是 `reconstruction_weight=0.05`。这说明 reconstruction loss 适合在参数 warm-up 之后作为 fine-tuning objective 使用，而不是从随机初始化开始单独使用。

## 注意

`learnable_weighted_ls` 是可选 plugin。当前它已经能跑通并使用 uncertainty weights，但是否优于 `traditional_ls` 需要通过更长训练和 calibration ablation 验证。
