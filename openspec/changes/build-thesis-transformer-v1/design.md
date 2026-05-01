# 设计文档：Thesis Transformer v1

## 项目结构

目标工程目录：

```text
F:\HUAWEI_Theise\Thesis Transformer version1
|-- pyproject.toml
|-- README.md
|-- configs
|   |-- data
|   `-- model
|-- scripts
|-- src
|   `-- thesis_transformer_v1
|       |-- tdlc
|       |-- data
|       |-- models
|       |-- estimation
|       |-- experiments
|       `-- metrics
|-- tests
|-- experiments
`-- openspec
```

其中 `src/thesis_transformer_v1/tdlc` 用来放从 `TDL_C` 复制并适配过来的
数据生成代码。论文自己的 estimator、model、experiment logic 不写回
`tdlc`，而是放在 `data`、`models`、`estimation`、`experiments`、
`metrics` 等模块中。

## Signal Model

v1 固定使用 TDL-C `common_delay`。真实数据由 `L_true=12` 条路径生成，
估计器使用 `L_eff` 条 `effective paths` 近似，`L_eff` 是实验超参数。

对 receive antenna `r`、transmit antenna `t`、OFDM symbol `n`、
subcarrier `k`，信道模型为：

```text
H[r,t,n,k] =
  exp(-j 2 pi f_k (tau0 + eps_r))
  * exp(j 2 pi cfo * time_n)
  * sum_l G[l,r,t] exp(j 2 pi doppler_l * time_n) exp(-j 2 pi f_k rel_delay_l)
```

参数含义：

- `tau0`: global `total delay`，所有 path 和 antenna 共享。
- `cfo`: global `CFO`，所有 path 和 antenna 共享。
- `rel_delay_l`: 第 `l` 条 effective path 的 relative delay。
- `doppler_l`: 第 `l` 条 effective path 的 Doppler。
- `eps_r`: 第 `r` 根 RX antenna 的 sampling `time offset`，样本内固定。
- `G[l,r,t]`: complex linear effective-path gain，由 `LS recovery` 恢复。

## 可辨识性约束

`total delay` 和 per-RX `time offset` 都会带来 frequency-domain phase
ramp，因此需要加约束避免 ambiguity：

- 固定 `eps_0 = 0`，把第一根 RX antenna 作为 timing reference。
- `rel_delay_l` 表示相对于 `tau0` 的 non-negative excess delay。
- 监督标签和结果报告中，effective paths 按 `rel_delay_l` 排序。
- 不能复用旧 `TDL_C` 里的 `rx_offsets` 字段，因为它表示 array spatial
  offset，单位是 wavelength；这里新增字段必须命名为 `rx_time_offsets_s`。

## Transformer 输入

输入只使用 `N_sym=8` 中最后两个 OFDM symbols，即 `n=6,7`。
每个可用 resource element 构造成一个 token：

```text
[
  Re/Im y_r(n,k) for all N_r,
  Re/Im x_t(n,k) for all N_t,
  normalized k,
  normalized n,
  pilot_or_data_flag,
  optional reliability flag
]
```

在 symbol-error 实验中：

- pilot RE 的 `x` 永远保持真实 pilot symbol。
- data-aided RE 的 `x` 按配置的 `SER` 替换为错误 decision symbol。

## Transformer 输出

hybrid model 的 `Transformer` 不直接输出完整 channel，也不输出 path gains。
输出结构为：

```text
global_params:
  tau0_s
  cfo_hz
path_params:
  [L_eff, 2], columns = rel_delay_s, doppler_hz
rx_time_offsets_s:
  [N_r], with element 0 fixed or overwritten to 0
path_logits:
  [L_eff], optional for future masking
```

`path_gains G` 必须在 nonlinear parameters 固定后，通过 complex
`LS recovery` 求解。

## LS Recovery

给定 oracle 或 predicted nonlinear parameters 后，在观测到的 `n=6,7`
resource elements 上构造 common-delay dictionary。对每个 sample、RX
antenna、TX antenna pair，求解：

```text
y_observed = A(theta, tau0, cfo, eps_r, x_observed) g + noise
```

其中 `g` 是该 `(r,t)` 天线对上的 `L_eff` 个 complex gains。

实现要求：

- E0 使用 `numpy.linalg.lstsq`。
- noisy experiments 可以加入 small ridge regularization。
- 得到 `G_hat` 后，重建完整 `H_hat`，覆盖所有 `n=0..7` 和 `k=0..47`。

## 递进实验

- **E0 Oracle LS sanity**：无 AWGN、无 symbol error，使用 oracle nonlinear
  parameters，扫描 `L_eff`，验证 `LS recovery` 和 full-grid reconstruction。
- **E1 Clean Transformer**：无 AWGN、无 symbol error，`Transformer` 估计
  nonlinear parameters，`LS recovery` 恢复 gains。
- **E2 Effective path ablation**：扫描 `L_eff=2,4,6,8,12`，输出
  `NMSE vs L_eff`。
- **E3 AWGN robustness**：无 symbol error，扫描 `SNR=30,20,10,5 dB`，
  对比 hybrid method 和 direct-H baseline。
- **E4 Symbol error robustness**：扫描 `SER=0,0.01,0.05,0.10,0.20`，
  只污染 data-aided symbols，不污染 pilot symbols。
- **E5 Full stress**：同时存在 sparse observation、AWGN、SER、compressed
  `L_eff`，作为论文主复杂场景。

## Metrics

核心 metrics：

- `channel_nmse_db`: full-grid `H_freq` 上的 NMSE。
- `observed_symbol_nmse_db`: 只在输入使用的 `n=6,7` 上的 NMSE。
- `param_mse`: nonlinear parameters 的 supervised loss。
- `gain_nmse_db`: oracle path alignment 可用时的 path gain NMSE。

