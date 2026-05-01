# Proposal: add-query-reconstruction-transformer

## 目标

在保留 `original_v1` 作为 baseline 的前提下，新增一个改进版 hybrid Transformer：

- 使用 `query-based path decoder` 代替 mean pooling。
- 支持 `differentiable LS reconstruction loss`，让训练目标直接对齐 `H_hat` 和 `H_true`。
- 保留 `traditional_ls` / `learnable_weighted_ls` 的 plugin 选择。
- 让 uncertainty 输出可以用于 weighted LS，但不强制替代传统 LS。

## 动机

`original_v1` 的主要问题是：网络用 mean-pooled token representation 预测 nonlinear parameters，然后在训练中只看 parameter label loss。这个目标和最终的 channel reconstruction NMSE 不完全一致，尤其在 `L_eff < L_true` 时，effective path label 只是 pseudo-label。

新方案把训练路径改成：

```text
tokens -> query_v1 -> nonlinear params + uncertainty
       -> differentiable LS -> path gains
       -> H_hat
       -> reconstruction loss(H_hat, H_true)
```

这样模型会更关注“重建出来的 channel 对不对”。

## 范围

- 新增 `query_v1` hybrid architecture。
- 新增 PyTorch differentiable LS/reconstruction layer。
- 在 training helper 和 CLI 中加入 `--loss-mode`。
- 支持 `param`、`reconstruction`、`param_plus_reconstruction` 三种训练目标。
- 增加 smoke tests，确认 forward、backward、CLI pipeline 可运行。

## 非目标

- 不修改 `original_v1_transformer.py`。
- 不删除已有 `uncertainty_v1` 和 `epgt_v1`。
- 不承诺 weighted LS 立刻优于 traditional LS；它是可选实验模块，需要进一步调 uncertainty calibration。

