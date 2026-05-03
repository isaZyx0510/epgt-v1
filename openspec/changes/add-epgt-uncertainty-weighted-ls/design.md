# 设计说明：EPGT uncertainty-weighted LS

## 概念区分

`uncertainty` 和 `reliability` 是两个不同概念。

```text
uncertainty:
  网络输出，描述模型对 path 参数预测的置信度。
  当前维度为 [B, L_eff, 2]，分别对应 delay 和 Doppler。

reliability:
  数据侧 annotation，描述 observed RE/token 本身是否可信。
  当前维度为 [B, N_obs]，例如标记符号错误。
```

因此 EPGT reliability mask 使用 token 中的 RE-level reliability；而
uncertainty weighted LS 使用 path-level delay/Doppler variance，经物理公式传播到
RE-level LS weights。

## 输出契约

当 `predict_path_uncertainty: true` 时，EPGT 输出增加：

```text
rel_delay_log_var: [B, L_eff]
doppler_log_var:  [B, L_eff]
```

完整 hybrid 输出为：

```text
{
  total_delay_s: [B],
  cfo_hz: [B],
  rx_time_offsets_s: [B, N_r],
  path_gates: [B, L_eff],
  rel_delay_s: [B, L_eff],
  doppler_hz: [B, L_eff],
  rel_delay_log_var: [B, L_eff],
  doppler_log_var: [B, L_eff],
}
```

## 从 path uncertainty 到 LS weights

对观测 RE `i=(n_i,k_i)`，使用一阶相位误差传播：

```text
phase_var_{l,i}
= (2 pi f_{k_i})^2 sigma^2_{d,l}
  + (2 pi t_{n_i})^2 sigma^2_{nu,l}
```

聚合 path 维度：

```text
obs_var_i = mean_l phase_var_{l,i}
```

转成 normalized observation weight：

```text
w_i = 1 / (1 + obs_var_i / mean_i(obs_var_i))
```

实现中权重会：

```text
normalize to mean 1
clamp to [0.5, 2.0]
```

然后进入 weighted LS：

```text
G_hat = argmin_G sum_i w_i |y_i - A_i(theta)G|^2 + ridge ||G||^2
```

## 防止投机权重

只用 H-loss 训练时，uncertainty 可能被模型用来操纵 LS 权重，而不再代表真实
path 参数不确定性。因此训练入口提供轻量正则：

```text
--uncertainty-regularization-weight
```

正则对象是 normalized log variance，目标是避免 log variance 变成极端值。
推荐初始值：

```text
1.0e-4
```

## 推荐命令

```powershell
uv run --extra dev python scripts\e6\train_pgt_h_loss.py `
  --config configs\data\e6_h_loss_l5.yaml `
  --model-config configs\model\pgt\epgt_v1_uncertainty_ls.yaml `
  --steps 200 `
  --lr 1e-3 `
  --eval-interval 20 `
  --train-batches 4 `
  --val-batches 2 `
  --ls-mode learnable_weighted_ls `
  --loss-mode reconstruction `
  --uncertainty-regularization-weight 1e-4
```
