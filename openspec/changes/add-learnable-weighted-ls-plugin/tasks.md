# 任务列表

## 1. Architecture

- [x] 新增 `uncertainty_v1` Transformer architecture。
- [x] 输出 `rel_delay_log_var` 和 `doppler_log_var`。
- [x] 在 `models/factory.py` 注册 `uncertainty_v1`。
- [x] 保持 `original_v1_transformer.py` 不变。

## 2. LS Plugin

- [x] 新增 `estimation/ls_plugins.py`。
- [x] 实现 `traditional_ls` plugin。
- [x] 实现 PyTorch `learnable_weighted_ls` plugin。
- [x] 支持 ridge `lambda`。
- [x] 支持 uncertainty-based observation weights。
- [x] 无 uncertainty 输出时 fallback 到 uniform weights。

## 3. Evaluation Wiring

- [x] 在 hybrid evaluation 中支持 `ls_mode`。
- [x] 在 training helpers 中传入 `ls_mode`。
- [x] 在 CLI 中增加 `--ls-mode`。
- [x] 确保旧默认行为仍是 `traditional_ls`。

## 4. Comparison Script

- [x] 新增 `scripts/run_ls_plugin_comparison.py`。
- [x] 比较 `original_v1 + traditional_ls`。
- [x] 比较 `uncertainty_v1 + traditional_ls`。
- [x] 比较 `uncertainty_v1 + learnable_weighted_ls`。
- [x] 输出 metrics 和 comparison plot。

## 5. Tests

- [x] 测试 `uncertainty_v1` forward shapes。
- [x] 测试 `learnable_weighted_ls` 在 uniform weights 下能返回合法 `H_hat`。
- [x] 测试 comparison script quick mode 可运行。
