# E1/E2 稳定训练设置与汇报说明

本文档记录当前更适合导师汇报的 E1/E2 tuned setting。中文说明为主，关键术语保留 English。

## 1. 为什么要调整 E1/E2

最早的 quick mode 只是在单个 batch 上 overfit，主要用于验证 pipeline。它的问题是：

- 数据量太小，曲线受 random seed 影响明显。
- `learning rate=1e-3` 对 nonlinear parameters 较激进。
- `CFO` 和 `Doppler` 会强烈影响从 `n=6,7` 到 `n=0..7` 的 time extrapolation。
- 如果把所有困难因素都放进 E1/E2，第一组图不够清楚。

因此当前 E1/E2 tuned setting 采用分层策略：

```text
E1/E2: clean/static teaching setting
  no AWGN
  no SER
  Doppler = 0
  CFO = 0

E3/E4/E5:
  再逐步加入 AWGN, SER, CFO/Doppler/time extrapolation 等复杂因素
```

这样第一组图专注讲清楚：

- `effective path compression`
- `Transformer nonlinear parameter estimation`
- `LS recovery`
- hybrid method 相比 direct-H baseline 的优势

## 2. 当前推荐超参数

E1 稳定设置：

```text
steps = 120
learning rate = 1e-4
train_batches = 16
val_batches = 4
d_model = 128
num_layers = 3
dim_feedforward = 256
dropout = 0.05
```

E2 sweep 稳定设置：

```text
steps = 80
learning rate = 1e-4
train_batches = 8
val_batches = 3
d_model = 96
num_layers = 2
dim_feedforward = 192
dropout = 0.05
L_eff = 2, 4, 6, 8, 12
```

## 3. 关键代码修正

为了让 clean/static setting 更稳定，`TransformerConfig` 的 output scale 现在会根据 experiment config 自动设置：

- 如果 `max_doppler_hz = 0`，则 `Doppler` 输出范围近似冻结到 0。
- 如果 `cfo_range_hz = [0, 0]`，则 `CFO` 输出范围近似冻结到 0。
- `total delay` 和 `rx_time_offset` 的输出范围也从 config 的 range 自动推导。

这避免了一个问题：即使 normalized parameter loss 很小，固定的大输出范围也会让 `CFO/Doppler` 的物理误差破坏 full-grid reconstruction。

## 4. E1 推荐命令

```powershell
$env:PYTHONPATH='F:\HUAWEI_Theise\Thesis Transformer version1\src'
python scripts\run_e1_comparison.py `
  --config configs\data\e1_clean_transformer.yaml `
  --steps 120 `
  --lr 1e-4 `
  --eval-interval 30 `
  --train-batches 16 `
  --val-batches 4 `
  --d-model 128 `
  --num-layers 3 `
  --dim-feedforward 256 `
  --dropout 0.05 `
  --output experiments\e1_clean_transformer\stable_linear_comparison_metrics.json
```

画图：

```powershell
python scripts\plot_results.py `
  --metrics experiments\e1_clean_transformer\stable_linear_comparison_metrics.json `
  --output experiments\e1_clean_transformer\stable_linear_comparison_nmse.png `
  --kind comparison `
  --metric channel_nmse
```

当前结果：

| Method | Full-grid `NMSE` |
| --- | ---: |
| `oracle_ls` | `1.770e-14` |
| `hybrid` | `4.031e-07` |
| `direct_h` | `9.377e-01` |

解释：

- `oracle_ls` 是上限，说明 signal model 和 LS recovery 是正确的。
- `hybrid` 接近很低的 NMSE，说明 `Transformer params + LS` 能有效重建 full-grid channel。
- `direct_h` 在相同 limited observation 下很弱，说明直接回归完整 channel 不如 physics-inspired hybrid 稳定。

## 5. E2 推荐命令

```powershell
$env:PYTHONPATH='F:\HUAWEI_Theise\Thesis Transformer version1\src'
python scripts\run_l_eff_sweep.py `
  --config configs\data\e2_effective_paths.yaml `
  --values 2 4 6 8 12 `
  --methods oracle_ls hybrid `
  --steps 80 `
  --lr 1e-4 `
  --eval-interval 40 `
  --train-batches 8 `
  --val-batches 3 `
  --d-model 96 `
  --num-layers 2 `
  --dim-feedforward 192 `
  --dropout 0.05 `
  --output experiments\e2_effective_paths\stable_linear_sweep_metrics.json
```

画图：

```powershell
python scripts\plot_results.py `
  --metrics experiments\e2_effective_paths\stable_linear_sweep_metrics.json `
  --output experiments\e2_effective_paths\stable_linear_nmse_vs_l_eff.png `
  --kind sweep `
  --metric channel_nmse
```

当前结果：

| `L_eff` | `oracle_ls` NMSE | `hybrid` NMSE |
| ---: | ---: | ---: |
| 2 | `4.080e-06` | `6.611e-03` |
| 4 | `1.728e-10` | `3.744e-05` |
| 6 | `2.038e-10` | `9.652e-06` |
| 8 | `1.516e-14` | `4.201e-05` |
| 12 | `1.770e-14` | `3.752e-06` |

解释：

- `oracle_ls` 展示了 effective path compression 的理论上限。
- 从 `L_eff=2` 到 `L_eff=4`，性能显著提升。
- `L_eff >= 6` 后 hybrid curve 进入相对饱和区。
- 这可以作为向导师说明的核心点：不一定需要使用全部真实路径，少量 `effective paths` 已经能逼近主要 channel structure。

## 6. 关于 dynamic setting 的说明

当开启 nonzero `CFO` 或 `Doppler` 时，任务会从 “frequency/path reconstruction” 变成 “time extrapolation”：

```text
input: n = 6, 7
output: n = 0..7
```

这时 `CFO/Doppler` 的微小误差会在未观测的 `n=0..5` 上放大。  
因此建议汇报时先展示 E1/E2 static clean 结果，再把 dynamic setting 放到 E3-E5 或 future work 中讨论。
