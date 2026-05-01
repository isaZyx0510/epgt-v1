# 设计文档：Learnable Weighted LS Plugin

## 数学形式

当前 observation model 可以写成：

```text
y = A(theta, x) g + noise
```

其中：

- `theta`: nonlinear parameters。
- `x`: observed transmit symbols。
- `g`: linear path gains。
- `A`: common-delay dictionary。

传统 ridge LS：

```text
g = (A^H A + lambda I)^(-1) A^H y
```

weighted ridge LS：

```text
g = (A^H W A + lambda I)^(-1) A^H W y
```

这里 `W` 是 observation-level diagonal weights。

## Uncertainty 权重

`uncertainty_v1` 输出：

```text
rel_delay_log_var: [B, L_eff]
doppler_log_var: [B, L_eff]
```

先使用 diagonal approximation：

```text
phase_var_l(n,k)
  = (2 pi f_k)^2 sigma_delay_l^2
    + (2 pi time_n)^2 sigma_doppler_l^2
```

然后把 path-level uncertainty 聚合到 observation-level：

```text
obs_var(n,k) = mean_l phase_var_l(n,k)
weight(n,k) = 1 / (obs_var(n,k) + eps)
```

为了避免极端权重，weights 会 normalize 到 mean=1，并 clamp 到 `[min_weight, max_weight]`。

## Plugin API

新增模块：

```text
src/thesis_transformer_v1/estimation/ls_plugins.py
```

核心接口：

```python
recover_with_ls_plugin(
    mode,
    rx_grid,
    tx_grid_observed,
    observation_indices,
    params,
    freq_hz,
    time_s,
    ridge,
) -> (path_gains, h_hat, metrics_extra)
```

其中 `mode`：

```text
traditional_ls
learnable_weighted_ls
```

## Architecture API

新增：

```text
src/thesis_transformer_v1/models/uncertainty_transformer.py
```

并在 factory 注册：

```text
uncertainty_v1
```

输出 dict 包含：

```text
total_delay_s
cfo_hz
rel_delay_s
doppler_hz
rx_time_offsets_s
rel_delay_log_var
doppler_log_var
```

如果使用 `learnable_weighted_ls` 但模型没有输出 uncertainty，则 plugin 自动使用 uniform weights，并记录 warning 字段。

## Quick Comparison

新增脚本：

```text
scripts/run_ls_plugin_comparison.py
```

默认比较：

```text
original_v1 + traditional_ls
uncertainty_v1 + traditional_ls
uncertainty_v1 + learnable_weighted_ls
```

输出：

```text
experiments/ls_plugin_comparison/metrics.json
experiments/ls_plugin_comparison/comparison_nmse.png
```

