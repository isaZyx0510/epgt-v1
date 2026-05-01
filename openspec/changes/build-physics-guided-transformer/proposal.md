# 构建 EPGT-v1 物理约束 Transformer

## 为什么要做

当前 thesis project 已经有一条可复现的 Transformer baseline，用于
compressed effective-path MIMO-OFDM 信道估计。下一步需要设计一个具体的
Physics Transformer 版本，使 attention 不只依赖通用 token 相似度，而是由
effective-path delay-Doppler 结构引导。

目标贡献是 EPGT-v1：Effective-Path Guided Transformer。它从稀疏观测中估计
可解释的物理参数，并通过模型驱动的参数化解码和 cross-attention bias 重建完整
MIMO-OFDM 信道张量。

## 做什么

新增一个 `pgt` model family，但不替换现有 baseline models。该版本遵循以下方法定义：

```text
H[r,t,n,k] =
  exp(-j 2 pi f_k (tau0 + epsilon_r))
  * exp(j 2 pi cfo * t_n)
  * sum_l G[l,r,t]
      exp(j 2 pi nu_l * t_n)
      exp(-j 2 pi f_k d_l)
```

v1 中，模型应遵守如下规范：

- TDL-C `common_delay` 作为固定或强约束的全局 delay convention。
- 估计相对 RX offsets，并固定 `epsilon_0 = 0`。
- 对 residual Doppler 做中心化，降低 CFO/Doppler 混淆。
- 使用 `L_eff` 个 gated effective paths。

核心目标：

- 新增 `physics/` 模块，用于 priors、masks、attention bias、physics losses
  和 diagnostics。
- 新增 `models/pgt/` package，用于 EPGT-v1 layers 和 hybrid variant。
- 保留现有 `HybridTransformer` 和 `DirectHTransformer` 作为 baseline。
- 引入实验阶段 `E6 Physics-Guided Attention`。
- 对比 baseline、bias-only、mask-only、loss-only、full EPGT variants。
- 记录 attention diagnostics，用于论文图和方法解释。

## 不做什么

- 不重写现有 E0-E5 pipeline。
- 不删除或重命名当前 baseline Transformer 文件。
- 不在本 change 中引入 AoA/AoD 或 array-response 估计。
- 在 synthetic TDL-C EPGT pipeline 稳定前，不扩展到真实测量数据。
- EPGT-v1 不依赖 direct black-box full-grid `H` regression。
- 第一版 scaffold 不直接学习 `G[l,r,t]`，先保持与现有 LS recovery 路径兼容；
  后续再把 parameterized decoder 推进到训练图中。

## 影响

这个 change 会在同一 thesis project 中创建第二条研究线：

- E0-E5 保持为 baseline 和 robustness pipeline。
- E6 成为 EPGT-v1 architecture-contribution pipeline。
- PGT/EPGT artifacts 统一用 `pgt` 或 `epgt_v1` 命名，便于与 `current` 和
  `original_v1` architectures 做干净对比。

