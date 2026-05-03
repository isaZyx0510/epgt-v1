# 任务列表

## 1. OpenSpec 和文档

- [x] 创建 OpenSpec change `build-physics-guided-transformer`。
- [x] 在 `docs/pgt/` 下添加 PGT folder plan。
- [x] 在 `docs/pgt/` 下添加 project architecture overview。
- [x] 将该 change 收敛为具体的 EPGT-v1 方法。
- [ ] 添加面向论文的参数化解码和 cross-attention bias 图示。

## 2. Physics Modules

- [x] 实现 `physics/priors.py`，用于 OFDM coordinate、delay、Doppler/CFO
      和 reliability priors。
- [x] 实现 `physics/attention_bias.py`，用 EPGT-v1 effective-path kernel
      将 priors 转成 attention logit bias tensors。
- [x] 实现 `physics/masks.py`，用于 pilot/data/reliability-aware attention masks。
- [x] 实现 `physics/losses.py`，用于 parameter-range 和 reconstruction
      consistency regularization。
- [x] 实现 `physics/diagnostics.py`，用于 attention entropy、prior alignment
      和 constraint violation reports。

## 3. PGT Model Family

- [x] 添加 `models/pgt/config.py`，用于 architecture 和 physics-control config。
- [x] 添加 `models/pgt/attention.py`，实现 guided cross-attention。
- [x] 添加 `models/pgt/encoder.py`，实现 observation encoder 和 full-grid query embeddings。
- [x] 添加 `models/pgt/heads.py`，实现 global 和 effective-path heads。
- [x] 添加 `models/pgt/hybrid.py`，返回与现有 `HybridTransformer` 相同的
      nonlinear-parameter output contract。
- [x] 添加 `models/pgt/decoder.py`，用于显式参数化 channel reconstruction helpers。
- [x] 在 model factory 中注册 `epgt_v1` architecture。

## 4. Configs

- [x] 添加 `configs/physics/base_ofdm_priors.yaml`。
- [x] 添加 `configs/physics/delay_doppler_bias.yaml`。
- [x] 添加 `configs/physics/pilot_reliability_mask.yaml`。
- [x] 添加 `configs/physics/impairment_constraints.yaml`。
- [x] 添加 `configs/model/pgt/epgt_v1_base.yaml`。
- [x] 添加 `configs/model/pgt/epgt_v1_bias_only.yaml`。
- [x] 添加 `configs/model/pgt/epgt_v1_mask_only.yaml`。
- [x] 添加 `configs/model/pgt/epgt_v1_loss_only.yaml`。
- [x] 添加 `configs/model/pgt/epgt_v1_full.yaml`。

## 5. Scripts 和 E6 Pipeline

- [x] 添加 `scripts/e6/train_pgt_h_loss.py`。
- [x] 添加 `scripts/e6/run_e6_pgt_comparison.py`。
- [x] 添加 PGT ablation entry point（历史路径为 `scripts/pgt/run_pgt_ablation.py`）。
- [x] 添加 `scripts/e6/run_oracle_replacement_ablation.py`，用于定位 full-grid
      外推误差的物理参数瓶颈。
- [x] 添加 E2/E3 PGT comparison entry point（现由 `scripts/e2/`、`scripts/e3/`
  分阶段维护），用于在 E2/E3 场景比较
      baseline hybrid 与 EPGT variants。
- [x] 添加 attention inspection/diagnostic support。
- [ ] 添加 EPGT diagnostics plotting entry point。
- [x] 将 outputs 保存到 `experiments/e6_physics_guided_attention/<variant>/`。

## 6. Tests

- [x] 添加 physical-prior tensor shapes 和 ranges 测试。
- [x] 添加 attention bias 可 broadcast 到 `[batch, heads, tokens, tokens]` 的测试。
- [ ] 添加 masks 不会移除所有 valid observations 的测试。
- [x] 添加 PGT hybrid output keys 与现有 hybrid baseline 匹配的测试。
- [ ] 添加所有 physics weights 为零时退化为 ordinary attention behavior 的测试。

## 7. Acceptance 和论文 artifacts

- [ ] 为 current hybrid model 跑 baseline E1/E3/E4 settings。
- [ ] 跑 E6 EPGT bias-only、mask-only、loss-only 和 full variants。
- [ ] 导出 comparison metrics 和 plots。
- [ ] 导出用于论文解释的 attention diagnostic figures。
- [ ] 脚本接口稳定后更新 README 中的最终命令。
