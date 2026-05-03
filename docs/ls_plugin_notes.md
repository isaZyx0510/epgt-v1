# Learnable Weighted LS Plugin 说明

本文档说明新增的 `LS plugin` 和 uncertainty 输出。

## 1. 可选 LS 模式

当前 hybrid evaluation 支持：

```text
traditional_ls
learnable_weighted_ls
```

命令行使用：

```powershell
--ls-mode traditional_ls
--ls-mode learnable_weighted_ls
```

## 2. 可选 Architecture

新增：

```text
uncertainty_v1
```

它保留 version1 的 mean-pooling Transformer 编码器，但输出额外 uncertainty：

```text
rel_delay_log_var
doppler_log_var
```

调用示例：

```powershell
--architecture uncertainty_v1 --ls-mode learnable_weighted_ls
```

## 3. Weighted LS 原理

传统 LS：

```text
g = argmin ||A g - y||^2
```

weighted LS：

```text
g = argmin ||sqrt(W) (A g - y)||^2
```

其中 `W` 来自 predicted phase uncertainty：

```text
phase_var_l(n,k)
  = (2 pi f_k)^2 sigma_delay_l^2
    + (2 pi time_n)^2 sigma_doppler_l^2
```

当前实现先将 path-level uncertainty 聚合成 observation-level weight，并做温和限制：

```text
weights normalized to mean 1
weights clamped to [0.5, 2.0]
```

这样可以避免训练早期 uncertainty 不准时过度降权。

## 4. 重要注意

`learnable_weighted_ls` 已经是可插拔模块，但它不一定立刻优于 `traditional_ls`。

原因：

- uncertainty head 是新增输出，需要足够训练才能 calibration。
- 目前 training loss 只是初步加入 heteroscedastic delay/Doppler loss。
- 如果 uncertainty 没学好，weighted LS 可能会给错误样本过高或过低权重。

所以当前最合理的使用方式是：

```text
traditional_ls 作为稳定 baseline
learnable_weighted_ls 作为实验插件
```

后续可以继续优化：

- 加更长训练。
- 加 observed reconstruction loss。
- 让网络输出 ridge lambda。
- 加 covariance term `Cov(mu, rho)`。
- 对 weights 做 curriculum，从 uniform 逐渐切到 uncertainty weights。

## 5. Quick Comparison 命令

```powershell
$env:PYTHONPATH='F:\HUAWEI_Theise\Thesis Transformer version1\src'
python scripts\e2\run_e2_training_scenarios.py `
  --scenarios uncertainty query_weighted epgt_weighted `
  --steps 40 `
  --eval-interval 20 `
  --train-batches 4 `
  --val-batches 2 `
  --d-model 64 `
  --num-layers 2 `
  --dim-feedforward 128 `
  --output experiments\e2_effective_paths\weighted_ls_comparison.json
```

默认比较：

```text
original_v1 + traditional_ls
uncertainty_v1 + traditional_ls
uncertainty_v1 + learnable_weighted_ls
```

## 6. E3-E5 使用 weighted LS

```powershell
python scripts\e3\run_snr_sweep.py `
  --architecture uncertainty_v1 `
  --ls-mode learnable_weighted_ls `
  --steps 80 `
  --lr 1e-4 `
  --train-batches 8 `
  --val-batches 3 `
  --output experiments\e3_awgn\weighted_ls_sweep_metrics.json
```
