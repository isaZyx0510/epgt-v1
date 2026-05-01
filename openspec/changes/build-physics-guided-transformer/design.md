# 设计文档：EPGT-v1 物理约束 Transformer

## 项目边界

EPGT-v1 是当前 thesis project 的扩展，不是替代。它应尽量复用已有的数据生成、
tokenization、LS recovery、evaluation 和 plotting utilities。

新架构分成两层：

- `physics/`：可复用的物理假设和约束。
- `models/pgt/`：消费这些物理信号的 PyTorch 模块。

这个拆分让科学假设可检查，也方便消融：同一个模型可以关闭或打开 bias、mask、
loss、diagnostics。

## Signal Model

对 receive antenna `r`、transmit antenna `t`、OFDM symbol `n`、subcarrier
`k`，EPGT-v1 使用：

```text
H[r,t,n,k] =
  A_glob[r,n,k]
  * sum_l G[l,r,t] P_l[n,k]
```

其中：

```text
A_glob[r,n,k] =
  exp(-j 2 pi f_k (tau0 + epsilon_r))
  * exp(j 2 pi cfo * t_n)

P_l[n,k] =
  exp(j 2 pi nu_l * t_n)
  * exp(-j 2 pi f_k d_l)
```

v1 中：

- `tau0` 遵循固定或强约束的 TDL-C `common_delay` convention。
- `epsilon_0 = 0` 固定 RX timing reference。
- `nu_l` 是 residual Doppler，预测后进行中心化。
- `alpha_l` gate 每条 effective path。
- 第一版 scaffold 中，`G[l,r,t]` 仍由现有 LS path 恢复。

## 可辨识性规则

实现必须编码以下规范：

- Delay ambiguity：避免自由平移 `tau0` 和所有 `d_l`。
- CFO/Doppler ambiguity：对 predicted residual Doppler 做中心化，使 global
  CFO 表示公共时间相位旋转。
- RX timing ambiguity：设置 `epsilon_0 = 0`，只估计相对 RX offsets。

## 物理信号

当前问题设定中已有的物理信号：

- OFDM coordinates：symbol index `n`、subcarrier index `k`。
- pilot/data indicator 和 optional reliability flag。
- delay-related structure：在有界 delay 下，邻近 subcarriers 应具有相干相位行为。
- Doppler/CFO-related structure：在有界 frequency offset 下，邻近 OFDM symbols
  应具有相干时间相位行为。
- RX timing reference：`rx_time_offsets_s[0] = 0`。
- effective-path ordering：path labels 按 relative delay 排序。

EPGT-v1 应优先使用 token 和 labels 中已有的信息。只有当某个 attention 机制明确
需要时，才新增数据字段。

## EPGT-v1 架构

EPGT-v1 包含三段：

1. Observation encoder：编码稀疏观测 tokens 和一个 global token。
2. Parameter heads：预测 global CFO、relative RX timing offsets、path gates、
   relative delays 和 residual Dopplers。
3. Cross-attention refinement：用 physics-guided attention bias `Gamma`，让
   full OFDM grid queries 从 observed context 中读取信息。

v1 输出契约保持与当前 hybrid model 兼容：

```text
{
  "total_delay_s": [B],
  "cfo_hz": [B],
  "rel_delay_s": [B, L_eff],
  "doppler_hz": [B, L_eff],
  "rx_time_offsets_s": [B, N_rx],
  "path_gates": [B, L_eff]
}
```

现有 LS recovery path 使用这些 nonlinear parameters 恢复 `G`。

## Cross-Attention Bias

对于 query grid point `q=(n_q,k_q)` 和 context observation `i=(n_i,k_i)`：

```text
Delta t_qi = t_nq - t_ni
Delta f_qi = f_kq - f_ki
```

effective-path correlation kernel 为：

```text
K(q,i) =
  abs(sum_l alpha_l
      exp(j 2 pi nu_l Delta t_qi)
      exp(-j 2 pi d_l Delta f_qi))
```

cross-attention logit bias 为：

```text
Gamma_qi = lambda_gamma * log(eps + K(q,i))
```

guided cross-attention 为：

```text
softmax(Q_q K_ctx^T / sqrt(d) + Gamma_qi + M_qi) V_ctx
```

第一版实现支持以下 variants：

- `baseline`: 不使用 physical bias、不使用 physical mask、不使用 physics loss。
- `bias_only`: 只加入 `B_phys`。
- `mask_only`: 只加入 `M_phys`。
- `loss_only`: 普通 attention 加 physics-consistency losses。
- `full`: bias + mask + physics-consistency losses。

## Source Layout

```text
src/thesis_transformer_v1/
  physics/
    __init__.py
    priors.py
    attention_bias.py
    masks.py
    losses.py
    diagnostics.py

  models/
    pgt/
      __init__.py
      config.py
      attention.py
      encoder.py
      heads.py
      hybrid.py
      decoder.py
```

职责划分：

- `physics/priors.py`: 构造 delay/Doppler/CFO/reliability prior tensors。
- `physics/attention_bias.py`: 将 prior tensors 转成 attention-logit bias。
- `physics/masks.py`: 基于 observation validity 和 reliability 构造 attention masks。
- `physics/losses.py`: 计算 consistency 和 regularization losses。
- `physics/diagnostics.py`: 衡量 attention entropy、prior alignment 和 constraint violation。
- `models/pgt/attention.py`: 实现 guided multi-head cross-attention。
- `models/pgt/encoder.py`: 实现 observation encoder 和 full-grid query embeddings。
- `models/pgt/heads.py`: 实现 global 和 effective-path heads。
- `models/pgt/hybrid.py`: 预测用于 LS recovery 的 nonlinear parameters。
- `models/pgt/decoder.py`: 显式参数化 channel reconstruction helper。

## Config Layout

```text
configs/
  physics/
    base_ofdm_priors.yaml
    delay_doppler_bias.yaml
    pilot_reliability_mask.yaml
    impairment_constraints.yaml

  model/
    pgt/
      epgt_v1_base.yaml
      epgt_v1_bias_only.yaml
      epgt_v1_mask_only.yaml
      epgt_v1_loss_only.yaml
      epgt_v1_full.yaml
```

data scenarios 继续放在 `configs/data/`。PGT model configs 应引用这些 data
scenarios，而不是复制它们。

## Experiment Layout

```text
experiments/
  e6_physics_guided_attention/
    README.md
    baseline_current/
    epgt_v1_bias_only/
    epgt_v1_mask_only/
    epgt_v1_loss_only/
    epgt_v1_full/
    ablations/
    attention_diagnostics/
```

主要输出：

- `comparison_metrics.json`
- `nmse_vs_snr.png`
- `nmse_vs_ser.png`
- `attention_prior_alignment.png`
- `attention_entropy_by_layer.png`
- `constraint_violation_summary.json`

## Acceptance Criteria

- 现有 E0-E5 tests 仍然通过。
- EPGT-v1 forward pass 返回与 `HybridTransformer` 兼容的输出契约。
- physics-disabled EPGT 可以退化为普通 cross-attention。
- `path_gates` 被限制在 `[0, 1]`。
- 开启 residual-Doppler centering 时，`doppler_hz` 被中心化。
- 每个 EPGT variant 都可以从 config 选择。
- E6 能输出与 current hybrid baseline 的 comparison metrics。
- attention diagnostics 按可复现的文件夹结构保存。

