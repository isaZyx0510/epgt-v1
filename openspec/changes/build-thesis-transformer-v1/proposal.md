# 构建 Thesis Transformer v1

## 为什么要做

毕设需要一个独立、可复现、可交付的主工程，用来实现基于 `Transformer`
的 hybrid MIMO-OFDM 信道估计。已有 `TDL_C` 目录包含较成熟的
MIMO-OFDM 数据生成代码，但新的论文方案还需要加入：

- 接收端硬件导致的 per-RX sampling `time offset`
- 全局 `total delay`
- 全局 `CFO`
- 使用 `effective paths` 近似真实多径，即 `L_eff < L_true`
- 从简单场景到复杂场景的递进实验路线

因此，新建 `Thesis Transformer version1` 作为主工程，并把 `TDL_C`
中的成熟模块复制进来，在此基础上扩展。

## 做什么

- 新建 `Thesis Transformer version1` 作为毕设主工程。
- 复制并复用 `TDL_C` 中的 OFDM、pilot、modulation、TDL-C channel
  generation、NPZ IO、sparse tokenization 等成熟模块。
- 第一版固定研究设定：
  - channel model: TDL-C `common_delay`
  - `N_sc=48`
  - `N_sym=8`
  - modulation: `16QAM`
  - true path number: `L_true=12`
- `Transformer` 输入只使用最后两个 OFDM symbols，即 `n=6,7`。
- 评估时重建完整 full-grid channel：`n=0..7`, `k=0..47`。
- `Transformer` 估计 nonlinear parameters：
  - global `total delay`
  - global `CFO`
  - effective-path `delay / Doppler`
  - per-RX sampling `time offset`
- `LS recovery` 只恢复 linear path gains `G_l`。
- 建立 E0-E5 递进实验：
  - oracle LS
  - clean Transformer
  - effective path ablation
  - AWGN robustness
  - symbol-error robustness
  - full stress setting

## 不做什么

- v1 不做 `array_response`。
- v1 不估计 `AoA/AoD`。
- v1 不接入真实测量数据。
- v1 不做 modulation sweep，统一固定为 `16QAM`。
- 在 E0-E1 跑通之前，不引入复杂 training framework。

## 影响

这个 change 会创建一个新的自包含 thesis project，不修改原始 `TDL_C`
参考目录。实现完成后，工程应能输出论文所需的核心实验结果、metrics 和
plots，包括：

- `NMSE vs L_eff`
- `NMSE vs SNR`
- `NMSE vs SER`
- hybrid method vs direct-H baseline

