# 任务列表

## 1. Project Scaffold

- [x] 创建 `src/thesis_transformer_v1` Python package。
- [x] 添加 `pyproject.toml`，依赖包括 `numpy`、`pyyaml`、`torch`、`pytest`，
      可选 `matplotlib`。
- [x] 添加 `README.md`，包含 quick-start commands 和 OpenSpec 文档入口。
- [x] 创建 `configs`、`scripts`、`tests`、`experiments` 目录。

## 2. 复制并适配 TDLC 数据生成

- [x] 从 `TDL_C` 复制成熟模块：config、constants、modulation、pilots、
      OFDM、impairments、channel generation、NPZ IO。
- [x] 设置 thesis base config：
  - `N_sc=48`
  - `N_sym=8`
  - modulation=`16QAM`
  - `mimo_mode=common_delay`
  - `L_true=12`
- [x] 在加入 thesis-specific 扩展前，先保证复制后的 TDLC 基础行为可运行。

## 3. 加入接收端 timing 和 global impairments

- [x] 添加配置字段：
  - `tau0_range_s`
  - `cfo_range_hz`
  - `rx_time_offset_range_s`
- [x] 生成 labels：
  - `channel_labels.total_delay_s`
  - `channel_labels.cfo_hz`
  - `channel_labels.rx_time_offsets_s`
- [x] 固定 `rx_time_offsets_s[:, 0] = 0`，将第一根 RX antenna 作为 reference。
- [x] 在 `H_freq` 中加入 frequency-domain phase ramp：
      `exp(-j 2 pi f_k (tau0 + eps_r))`。
- [x] 在 `H_freq` 中加入 shared CFO phase：
      `exp(j 2 pi cfo time_n)`。

## 4. Tokenization 和 SER corruption

- [x] token builder 只使用 OFDM symbols `n=6,7`。
- [x] token 包含 received complex values、transmit complex values、
      normalized coordinates、pilot/data flag、optional reliability flag。
- [x] 实现 SER corruption，只污染 data-aided symbols。
- [x] 确保 pilot symbols 永远不被污染。

## 5. Nonlinear labels 和 effective paths

- [x] 构建 nonlinear labels：
  - `tau0_s`
  - `cfo_hz`
  - `rel_delay_s`
  - `doppler_hz`
  - `rx_time_offsets_s`
- [x] path labels 按 `rel_delay_s` 排序。
- [x] 支持 `L_eff=2,4,6,8,12`。
- [x] 在 E0 oracle compression 中，使用 strongest paths 规则选择 `L_eff`
      条 effective paths，并在文档和代码中保持一致。

## 6. LS Recovery 和 Reconstruction

- [x] 实现 common-delay dictionary construction。
- [x] 对每个 `(sample, rx)` 联合所有 `(path, tx)` 实现 complex `LS recovery`。
- [x] 为 noisy settings 添加 optional ridge regularization。
- [x] 重建 full-grid `H_hat`，覆盖 8 个 OFDM symbols 和 48 个 subcarriers。
- [x] 实现 linear NMSE 和 dB NMSE metrics。

## 7. Models

- [x] 实现 `HybridTransformer`，只输出 nonlinear parameters。
- [x] 实现 direct-H Transformer baseline。
- [x] 添加 nonlinear parameter supervised loss 和 direct-H full-channel
      supervised loss。
- [x] 为每个 model 添加 one-batch forward smoke test。

## 8. Experiment Scripts 和 Configs

- [x] 在 `configs/data` 下添加 E0-E5 configs。
- [x] 实现 `scripts/run_oracle_ls.py`，用于 E0。
- [x] 实现 `scripts/train_hybrid.py`，用于 E1-E5 hybrid runs 的最小训练入口。
- [x] 实现 `scripts/train_direct_h.py`，用于 baseline runs 的最小训练入口。
- [x] 实现 `scripts/evaluate.py` 和 `scripts/plot_results.py`。
- [x] 将 metrics 和 plots 保存到 `experiments/<experiment_name>`。

## 9. Tests 和 Acceptance

- [x] 测试 `rx_time_offsets_s` shape 和 reference constraint。
- [x] 测试 `total delay` 和 RX `time offset` 产生符合预期的 subcarrier
      phase ramp。
- [x] 测试 tokenization 只使用 `n=6,7`。
- [x] 测试 SER corruption 只改 data REs，不改变 pilot REs。
- [x] 测试 clean model 中，当 `L_eff=L_true` 且使用 oracle nonlinear
      parameters 时，oracle LS 可以达到 near-zero NMSE。
- [x] 在任何 neural-network training 前，先跑通 E0 并记录结果。

