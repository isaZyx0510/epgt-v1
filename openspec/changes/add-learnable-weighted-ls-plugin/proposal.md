# 增加 Learnable Weighted LS Plugin

## 为什么要做

当前 `HybridTransformer` 的核心流程是：

```text
Transformer -> nonlinear parameters -> traditional LS -> H_hat
```

传统 `LS recovery` 对 nonlinear parameters 的误差比较敏感。尤其是 `rel_delay`、`Doppler`、`total delay`、`CFO`、`rx_time_offset` 的小偏差会通过 complex phase 放大，导致 path gains `G` 不稳定。

因此需要增加一个可插拔的 LS 模块：

```text
traditional_ls
learnable_weighted_ls
```

同时让新的 Transformer architecture 能输出传播参数的不确定性，例如：

```text
sigma_rel_delay_l^2
sigma_doppler_l^2
```

再根据 phase uncertainty 构造 weighted LS，提高 `G` 的稳定性。

## 做什么

- 保留 `original_v1` 架构不动。
- 新增 `uncertainty_v1` architecture：
  - 继承 version1 的 mean-pooling Transformer 结构。
  - 输出原 nonlinear parameters。
  - 额外输出 `rel_delay_log_var` 和 `doppler_log_var`。
- 新增 LS plugin 模块：
  - `traditional_ls`: 调用现有 NumPy LS。
  - `learnable_weighted_ls`: PyTorch differentiable LS layer。
  - 支持 ridge parameter `lambda`，可固定也可 learnable。
  - 根据 uncertainty 构造 observation weights。
- 增加 CLI 参数：
  - `--architecture uncertainty_v1`
  - `--ls-mode traditional_ls`
  - `--ls-mode learnable_weighted_ls`
- 增加 comparison script，快速比较不同 LS plugin 的 `channel_nmse`。

## 不做什么

- 本阶段不做完整大规模训练。
- 本阶段不把 differentiable LS loss 全量并入长期训练，只先建立插件和 quick comparison。
- 本阶段不实现完整 covariance `Cov(mu,rho)`，先使用 diagonal uncertainty。

## 影响

完成后，项目可以手动切换 LS recovery 方法，并能测试：

```text
original_v1 + traditional_ls
uncertainty_v1 + traditional_ls
uncertainty_v1 + learnable_weighted_ls
```

这为后续把 `H_hat NMSE` 或 observed reconstruction loss 接入 training graph 打基础。

