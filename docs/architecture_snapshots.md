# Architecture Snapshots

本文档记录可复现的 `Transformer` architecture 版本。

## `original_v1`

文件：

```text
src/thesis_transformer_v1/models/original_v1_transformer.py
```

用途：

- 保留最初的 mean-pooling Transformer baseline。
- 后续修改 `current` architecture 时，仍然可以通过 `--architecture original_v1` 复现实验。
- E3-E5 robustness pipeline 默认使用 `original_v1`。

结构：

```text
tokens
 -> Linear(input_dim, d_model)
 -> LayerNorm
 -> GELU
 -> TransformerEncoder x num_layers
 -> mean pooling over tokens
 -> MLP head
 -> nonlinear params or direct-H output
```

调用示例：

```powershell
python scripts\run_e3_e5_pipeline.py --architecture original_v1
```

## `current`

文件：

```text
src/thesis_transformer_v1/models/transformer.py
```

用途：

- 当前工作架构。
- 后续可以改为 query-based decoder、CLS token pooling、path queries 等新结构。
- 修改后仍可与 `original_v1` 做公平对比。

