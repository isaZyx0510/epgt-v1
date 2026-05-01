# EPGT-v1 物理约束 Transformer 规格

## ADDED Requirements

### Requirement: 使用 OpenSpec 管理 EPGT-v1 开发

项目 SHALL 通过 OpenSpec change `build-physics-guided-transformer` 管理
Physics-Guided Transformer 研究线。

#### Scenario: 开发者检查 EPGT-v1 change 状态

- **WHEN** 开发者运行 `build-physics-guided-transformer` 的 OpenSpec status
- **THEN** 该 change 暴露 proposal、design、specs 和 tasks artifacts。

### Requirement: 分离物理层和神经网络架构层

实现 SHALL 将通信系统物理约束与 PyTorch architecture code 分离。

#### Scenario: 实现物理先验

- **WHEN** 需要 delay、Doppler/CFO、reliability 或 OFDM coordinate priors
- **THEN** 它们应实现于 `src/thesis_transformer_v1/physics/`。
- **AND** PGT neural layers 从 `models/pgt/` 消费这些 priors，而不是在本地重复定义。

### Requirement: EPGT-v1 signal model

EPGT-v1 SHALL 将 channel 建模为 global timing/CFO terms 乘以 gated effective-path
delay-Doppler atoms 之和。

#### Scenario: 预测 nonlinear parameters

- **WHEN** EPGT-v1 接收 sparse observation tokens
- **THEN** 它预测 CFO、relative RX timing offsets、path gates、relative delays
  和 residual Dopplers。
- **AND** `rx_time_offsets_s[:, 0]` 被固定为零。
- **AND** 开启 centering 时，residual Doppler values 被中心化。

### Requirement: Physics-guided cross-attention variants

EPGT-v1 model family SHALL 支持受控 cross-attention variants，用于 ablation。

#### Scenario: 运行 architecture ablations

- **WHEN** 配置 E6 experiment
- **THEN** 项目可以选择 baseline、bias-only、mask-only、loss-only 和 full
  effective-path-guided attention variants。

### Requirement: Effective-path attention bias

EPGT-v1 SHALL 根据 path gates、relative delays 和 residual Dopplers 计算
cross-attention bias。

#### Scenario: 构建 attention bias

- **WHEN** 给定 query grid coordinates 和 context observation coordinates
- **THEN** 模型构建 `Gamma_qi = lambda * log(eps + K(q,i))`。
- **AND** `K(q,i)` 是 gated delay-Doppler correlation sum 的 magnitude。

### Requirement: Baseline compatibility

PGT hybrid model SHALL 保持与现有 `HybridTransformer` 相同的 output contract。

#### Scenario: 训练或评估 PGT hybrid model

- **WHEN** PGT hybrid model 接收 observation tokens
- **THEN** 它返回与现有 LS recovery pipeline 兼容的 nonlinear parameter predictions。
- **AND** 现有 evaluation code 可以比较其 reconstructed full-grid channel 和
  current hybrid baseline。

### Requirement: Reproducible E6 experiment layout

项目 SHALL 将 PGT experiment outputs 保存到
`experiments/e6_physics_guided_attention/`。

#### Scenario: 保存 PGT comparison results

- **WHEN** E6 comparison 或 ablation run 完成
- **THEN** metrics、plots、checkpoints 和 attention diagnostics 保存到
  variant-specific subfolder。

### Requirement: Attention diagnostics

PGT experiments SHALL 导出 diagnostics，用于解释 attention 是否遵循预期物理先验。

#### Scenario: 检查训练后的 PGT attention

- **WHEN** 运行 attention inspection
- **THEN** 项目保存 attention entropy、prior-alignment 和 constraint violation summaries。

