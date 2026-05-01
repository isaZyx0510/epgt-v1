# E3-E5 Robustness Pipeline 说明

本文档说明如何运行 E3-E5 训练流程，以及如何保留并复现原始 `Transformer` architecture。

## 1. Architecture 版本

当前项目支持两个 architecture 名称：

```text
original_v1
current
```

### `original_v1`

冻结副本：

```text
src/thesis_transformer_v1/models/original_v1_transformer.py
```

这是当前 mean-pooling Transformer baseline 的稳定副本。后续你修改主架构时，不要改这个文件。

### `current`

当前工作文件：

```text
src/thesis_transformer_v1/models/transformer.py
```

后续可以在这里改成 query-based decoder、CLS token、path queries 等新架构。

所有训练脚本都支持：

```powershell
--architecture original_v1
--architecture current
```

默认使用 `original_v1`，保证已有结果可复现。

## 2. 一键运行 E3-E5

smoke test 命令：

```powershell
$env:PYTHONPATH='F:\HUAWEI_Theise\Thesis Transformer version1\src'
python scripts\run_e3_e5_pipeline.py `
  --snr-values 20 `
  --ser-values 0.05 `
  --steps 1 `
  --eval-interval 1 `
  --train-batches 1 `
  --val-batches 1 `
  --d-model 32 `
  --num-layers 1 `
  --dim-feedforward 64 `
  --architecture original_v1 `
  --output-root experiments\robustness_pipeline_smoke
```

完整版本建议命令：

```powershell
$env:PYTHONPATH='F:\HUAWEI_Theise\Thesis Transformer version1\src'
python scripts\run_e3_e5_pipeline.py `
  --snr-values 30 20 10 5 `
  --ser-values 0 0.01 0.05 0.10 0.20 `
  --steps 80 `
  --lr 1e-4 `
  --eval-interval 20 `
  --train-batches 8 `
  --val-batches 3 `
  --d-model 96 `
  --num-layers 2 `
  --dim-feedforward 192 `
  --dropout 0.05 `
  --architecture original_v1 `
  --output-root experiments\robustness_pipeline_original_v1
```

## 3. 输出文件

pipeline 会生成：

```text
experiments/<output-root>/
  e3_awgn_metrics.json
  e4_symbol_error_metrics.json
  e5_full_stress_metrics.json
  summary.json
  e3_nmse_vs_snr.png
  e4_nmse_vs_ser.png
  e5_full_stress_comparison.png
```

其中：

- E3: `NMSE vs SNR`
- E4: `NMSE vs SER`
- E5: full stress method comparison

## 4. 当前 E3-E5 的含义

### E3 AWGN Robustness

固定 data-aided symbol 没有错误，扫描：

```text
SNR = 30, 20, 10, 5 dB
```

目标：看 noise 对 `hybrid` 和 `direct_h` 的影响。

### E4 Symbol Error Robustness

固定 channel/noise setting，扫描：

```text
SER = 0, 0.01, 0.05, 0.10, 0.20
```

目标：看 data-aided `x` 的 symbol decision error 如何传播到 channel estimation。

### E5 Full Stress

同时加入：

```text
AWGN
SER
compressed L_eff
```

目标：作为复杂场景对比。

## 5. 后续修改新架构的建议

后续如果要改 `Transformer` 架构：

1. 保持 `original_v1_transformer.py` 不变。
2. 修改 `transformer.py` 或新增新架构文件。
3. 如果新增架构，在 `models/factory.py` 注册新名字。
4. 用同样命令分别跑：

```powershell
--architecture original_v1
--architecture current
```

这样可以公平比较新旧架构。

