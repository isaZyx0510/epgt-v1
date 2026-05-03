# 任务列表

## 1. 模型输出

- [x] 在 `TransformerConfig` 中增加 `predict_path_uncertainty`。
- [x] 在 EPGT `EffectivePathHead` 中可选输出 `rel_delay_log_var`。
- [x] 在 EPGT `EffectivePathHead` 中可选输出 `doppler_log_var`。
- [x] 保持默认 EPGT 输出不变，只有配置打开时才增加 uncertainty。

## 2. 配置和训练入口

- [x] 新增 `configs/model/pgt/epgt_v1_uncertainty_ls.yaml`。
- [x] 让 `train_pgt_h_loss.py` 支持 `--uncertainty-regularization-weight`。
- [x] 让 `run_e6_pgt_comparison.py` 支持记录 uncertainty regularization 配置。

## 3. LS 和训练约束

- [x] 复用现有 `learnable_weighted_ls`，当 EPGT 输出 uncertainty 时生成 observation weights。
- [x] 保持权重 mean normalization。
- [x] 保持权重 clamp 到 `[0.5, 2.0]`。
- [x] 增加轻量 log variance regularization，降低 H-loss 下的投机权重风险。

## 4. 验证

- [x] 增加 EPGT uncertainty 输出 shape 测试。
- [x] 运行 targeted ruff。
- [x] 运行相关 pytest。
- [x] 运行 smoke train，确认 `weights_source = uncertainty`。

## 5. 后续实验

- [ ] 比较 `epgt_v1_bias_only + traditional_ls` 与 `epgt_v1_uncertainty_ls + learnable_weighted_ls`。
- [ ] sweep `uncertainty_regularization_weight = 0, 1e-5, 1e-4, 1e-3`。
- [ ] 在 E2/E3 环境下检查 weighted LS 是否比 traditional LS 稳定。
