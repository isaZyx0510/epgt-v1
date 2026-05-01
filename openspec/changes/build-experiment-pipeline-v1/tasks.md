# 任务列表

## 1. 统一 Metrics 和 Evaluation API

- [x] 定义统一 `ExperimentMetrics` JSON schema。
- [x] 添加 metrics writer/reader 工具。
- [x] 添加 method names：`oracle_ls`、`hybrid`、`direct_h`。
- [x] 确保所有脚本输出到 `experiments/<experiment_name>/<method>/metrics.json`。

## 2. 升级 Hybrid Training

- [x] 保留 supervised nonlinear-parameter loss 作为 training loss。
- [x] 每个 epoch 或固定 interval 运行 predicted params 的 `LS recovery`。
- [x] 记录 full-grid `channel_nmse_db`。
- [x] 记录 observed-symbol `observed_symbol_nmse_db`。
- [x] 保存 loss curve 和 NMSE curve。
- [x] 输出 final metrics，便于和 oracle/direct-H 对比。

## 3. 升级 Direct-H Baseline

- [x] 使用统一 metrics schema。
- [x] 记录 direct-H full-grid `NMSE`。
- [x] 保存 train loss curve。
- [x] 确保使用同一套 config/data setting，与 hybrid 可比较。

## 4. 实现 E1 Clean Transformer

- [x] 运行 clean setting 下的 hybrid training。
- [x] 运行 clean setting 下的 direct-H training。
- [x] 跑 oracle LS reference。
- [x] 生成 E1 method comparison table。
- [x] 生成 E1 method comparison plot。

## 5. 实现 E2 Effective Path Ablation

- [x] 编写 `run_l_eff_sweep.py`。
- [x] Sweep `L_eff=2,4,6,8,12`。
- [x] 对每个 `L_eff` 跑 oracle LS。
- [x] 对每个 `L_eff` 跑 hybrid training/evaluation。
- [x] 生成 `NMSE vs L_eff` 曲线。

## 6. 实现 E3 AWGN Robustness

- [x] 编写 `run_snr_sweep.py`。
- [x] Sweep `SNR=30,20,10,5 dB`。
- [x] 对每个 SNR 跑 hybrid 和 direct-H。
- [x] 生成 `NMSE vs SNR` 曲线。

## 7. 实现 E4 Symbol Error Robustness

- [x] 编写 `run_ser_sweep.py`。
- [x] Sweep `SER=0,0.01,0.05,0.10,0.20`。
- [x] 对每个 SER 跑 hybrid 和 direct-H。
- [x] 确认 pilot symbols 不被污染。
- [x] 生成 `NMSE vs SER` 曲线。

## 8. 实现 E5 Full Stress

- [x] 固定一个 compressed `L_eff`，例如 `L_eff=6`。
- [x] 同时开启 AWGN 和 SER。
- [x] 跑 hybrid、direct-H、oracle reference。
- [x] 输出 final comparison table。

## 9. Plotting 和导师汇报图

- [x] 扩展 `plot_results.py` 支持 method comparison。
- [x] 支持 `NMSE vs L_eff`、`NMSE vs SNR`、`NMSE vs SER`。
- [x] 从 `docs/visual_overview.md` 整理 presentation-ready diagrams。
- [x] 输出至少四张导师汇报图。

## 10. Verification

- [x] 添加测试：metrics schema 可以 round-trip。
- [x] 添加测试：hybrid evaluation 能从 predicted params 走到 `H_hat`。
- [x] 添加测试：sweep runner 能生成所有 sweep values 的 metrics。
- [x] 运行 E1 quick mode，确认 hybrid loss 下降并生成 metrics 文件。

