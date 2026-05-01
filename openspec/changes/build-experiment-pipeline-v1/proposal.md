# 构建 Experiment Pipeline v1

## 为什么要做

`build-thesis-transformer-v1` 已经完成了主工程 scaffold、数据生成、
`LS recovery`、`HybridTransformer`、`Direct-H baseline`、E0 oracle
sanity check 和最小训练脚本。

下一阶段要把当前 prototype 变成真正能支撑论文实验和导师汇报的
experiment pipeline。核心目标不是再证明代码能跑，而是系统地产生可解释、
可对比、可画图的实验结果：

- `Transformer params + LS` 是否真的能重建 full-grid channel？
- `HybridTransformer` 和 `Direct-H Transformer` 谁更稳？
- `L_eff < L_true` 时，effective path compression 损失多少？
- AWGN 和 symbol error 对方法的影响如何？

## 做什么

- 升级 `train_hybrid.py`：
  - 每个 epoch 或固定 interval 记录 nonlinear parameter loss。
  - 定期执行 `Transformer output -> LS recovery -> H_hat -> NMSE`。
  - 输出 full-grid `NMSE` 和 observed-symbol `NMSE`。
- 升级 `train_direct_h.py`：
  - 输出 direct-H full-grid `NMSE`。
  - 与 hybrid pipeline 使用一致的 dataset/config/metrics 格式。
- 增加统一 evaluation runner：
  - 支持 oracle LS、hybrid、direct-H 三类方法。
  - 保存统一 `metrics.json`。
- 实现 E1-E5 自动实验：
  - E1 clean Transformer
  - E2 `L_eff` sweep
  - E3 SNR sweep
  - E4 SER sweep
  - E5 full stress setting
- 增加 plotting scripts：
  - `NMSE vs L_eff`
  - `NMSE vs SNR`
  - `NMSE vs SER`
  - oracle / hybrid / direct-H 对比图
- 整理导师汇报材料：
  - 从 `docs/visual_overview.md` 中导出或复用流程图。
  - 配合 E0-E2 初步结果形成 presentation-ready figures。

## 不做什么

- 不在这一阶段引入 `array_response`。
- 不改变 v1 的主 signal model。
- 不做大规模 hyperparameter search。
- 不追求最终论文级最优性能，先追求实验闭环和可解释结果。

## 影响

完成后，项目会从“核心模块可运行”升级为“实验可复现”。导师汇报时可以展示：

- 物理模型中的 data generation 流程。
- `HybridTransformer` 的输入输出。
- `LS recovery` 如何重建 full-grid channel。
- E0/E1/E2 的初步结果曲线。
- 后续 E3/E4/E5 robustness 实验计划与初步结果。

