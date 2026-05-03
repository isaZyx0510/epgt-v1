# 为 EPGT-v1 增加 uncertainty-weighted LS

## 为什么要做

当前 EPGT-v1 通过 hybrid 路径训练：

```text
observation tokens
-> EPGT nonlinear physical parameters
-> differentiable LS recovery of path gains
-> H_hat
-> H reconstruction loss
```

如果使用 `learnable_weighted_ls`，LS 层需要 observation-level weights。旧的
`uncertainty_v1` 可以输出 path-level delay/Doppler 方差，再传播成每个观测 RE
的 LS 权重；但 EPGT-v1 原本没有 uncertainty 输出，因此 weighted LS 对 EPGT
无法真正发挥作用。

这个 change 的目标是增加一个可选 EPGT 变体：

```text
epgt_v1_uncertainty_ls
```

它在 effective path head 中额外输出：

```text
rel_delay_log_var: [B, L_eff]
doppler_log_var:  [B, L_eff]
```

然后让 `--ls-mode learnable_weighted_ls` 可以基于这些 uncertainty 生成 LS 权重。

## 做什么

- 在 `TransformerConfig` 中加入 `predict_path_uncertainty` 开关。
- 在 EPGT `EffectivePathHead` 中可选输出 path-level log variance。
- 新增模型配置 `configs/model/pgt/epgt_v1_uncertainty_ls.yaml`。
- H-loss 训练入口支持 `--uncertainty-regularization-weight`。
- weighted LS 继续复用现有机制：如果模型输出 uncertainty，就生成权重；否则退化为 uniform weights。
- 增加测试确认 EPGT uncertainty 输出维度为 `[B, L_eff]`。

## 不做什么

- 不把 reliability mask 和 uncertainty 混为一个概念。
- 不把 per-token reliability 改成网络预测项。
- 不改变 baseline/current/original_v1 的默认输出契约。
- 不把 physical parameter supervised loss 重新设为默认训练方式。

## 影响

这个 change 让 EPGT 可以比较两种 LS 路线：

```text
EPGT + traditional_ls
EPGT + learnable_weighted_ls
```

其中 uncertainty weighted LS 的物理意义是：当模型认为某些 path 的
delay/Doppler 预测不确定时，对相位误差更敏感的观测 RE 在 LS 中应降低权重。
