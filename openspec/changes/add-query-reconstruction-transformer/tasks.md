# Tasks

## 1. Model

- [x] 新增 `query_transformer.py`。
- [x] 实现 token sequence encoder，避免 mean pooling。
- [x] 实现 global/path/RX learned queries。
- [x] path query 输出 propagation parameters 和 uncertainty。
- [x] 在 `models/factory.py` 注册 `query_v1`。
- [x] 保持 `original_v1_transformer.py` 不变。

## 2. Differentiable LS

- [x] 新增 `estimation/differentiable_ls.py`。
- [x] 实现 torch complex phase dictionary。
- [x] 实现 differentiable complex LS。
- [x] 实现 full-grid torch reconstruction。
- [x] 实现 `complex_nmse_loss`。
- [x] 支持 optional uncertainty weights。

## 3. Training

- [x] `train_hybrid_quick` 支持 `loss_mode`。
- [x] 支持 `param` loss。
- [x] 支持 `reconstruction` loss。
- [x] 支持 `param_plus_reconstruction` loss。
- [x] CLI 增加 `--loss-mode` 和 `--reconstruction-weight`。
- [x] E1/E2/E3/E4/E5 脚本暴露 `query_v1` 和 reconstruction loss 参数。

## 4. Tests

- [x] 测试 `query_v1` forward shape。
- [x] 测试 differentiable LS 可以 backward 到 nonlinear parameters。
- [x] 跑通 `query_v1 + reconstruction + traditional_ls` smoke。
- [x] 跑通 `query_v1 + param_plus_reconstruction + learnable_weighted_ls` smoke。

