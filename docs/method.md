# 模型、数据与训练 · Model, Data and Training

## 完整工程流程

```text
原始预训练语料
    ↓ prepare_data.py
数据抽样检查
    ↓ inspect_data.py
Tokenizer 训练与验收
    ↓ train_tokenizer.py
全量 uint16 token 分片
    ↓ tokenize_data.py
混合来源 DataLoader
    ↓ data_loader.py
200M Base 预训练
    ↓ train.py + evaluate.py
best_pretrain.pt
    ↓
SFT 原始数据下载
    ↓ prepare_sft_data.py
格式统一、清洗、去重、划分
    ↓ build_sft_dataset.py
定长 token、loss mask、length 分片
    ↓ tokenize_sft_data.py
Assistant-only SFT
    ↓ train_sft.py
best_sft.pt
    ↓
generate_sft.py / app.py
```

预训练和 SFT 之所以拥有不同的数据构建脚本，是因为两者的学习目标不同：预训练只需要连续正文和文档边界；SFT 需要明确的 `user` / `assistant` 消息结构、截断策略以及 assistant-only loss mask。


## 模型架构

XiangLM 使用 GPT 风格的 pre-norm Decoder-only Transformer，但采用了现代小模型中常见的若干组件。

| 配置 | 数值 |
|---|---:|
| 参数量 | 200,308,992 |
| Vocabulary | 32,768 |
| Context length | 1,024 |
| `d_model` | 768 |
| Transformer layers | 25 |
| Attention heads | 12 |
| Head dimension | 64 |
| `d_ff` | 2,016 |
| Normalization | RMSNorm，`eps=1e-5` |
| Position encoding | RoPE，`theta=10,000` |
| Feed-forward | SwiGLU |
| Attention | PyTorch scaled dot-product causal attention |
| Bias | 不使用 |
| Dropout | 0.0 |
| Embedding / LM head | 权重绑定 |

每个 block 的计算顺序为：

```text
x = x + CausalSelfAttention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

训练时使用 activation checkpointing 降低显存占用。词嵌入与输出 LM head 共享参数，在 32K 词表下显著节省了模型规模。模型没有使用 KV cache；当前生成脚本会在每个新 token 上重新计算保留的上下文，因此实现清晰但长回答的生成速度仍有优化空间。


## Tokenizer

Tokenizer 基于 Hugging Face `tokenizers` 库训练，但词表和 merges 来自本项目自己的混合语料。

- 类型：Byte-level BPE。
- 词表：32,768。
- 训练抽样：128 MiB 文本。
- 抽样比例：60% 中文网页、25% 中文教育、10% 英文教育、5% 代码。
- 普通正文编码时禁用特殊 token 自动匹配，避免原始网页文本伪造控制标记。
- 预训练文档之间插入 `<|endoftext|>`。
- SFT 使用 `<|im_start|>` / `<|im_end|>` 表示消息边界。

特殊 token：

| Token | ID |
|---|---:|
| `<|pad|>` | 0 |
| `<|unk|>` | 1 |
| `<|bos|>` | 2 |
| `<|eos|>` | 3 |
| `<|endoftext|>` | 4 |
| `<|im_start|>` | 5 |
| `<|im_end|>` | 6 |

抽样验收结果：

| 来源 | 文档 | 字符 / token | UTF-8 字节 / token | UNK | 回译失败 |
|---|---:|---:|---:|---:|---:|
| 中文网页 | 100 | 1.536 | 4.143 | 0 | 0 |
| 中文教育 | 100 | 1.614 | 4.583 | 0 | 0 |
| 英文教育 | 100 | 3.757 | 3.774 | 0 | 0 |
| 代码 | 100 | 2.959 | 2.964 | 0 | 0 |

完整报告保存在 [`data/tokenizer/xianglm-32k/training_report.json`](../data/tokenizer/xianglm-32k/training_report.json)。


## 预训练数据

原始下载计划以中文为主，同时保留适量教育英文和代码：

| 类别 | 数据集 | 计划目标 | 实际选择 | 训练采样权重 | 上游许可摘要 |
|---|---|---:|---:|---:|---|
| 中文网页 | [FineWeb 2 Mandarin Chinese](https://modelscope.cn/datasets/HuggingFaceFW/fineweb-2) | 36 GiB | 39.64 GiB | 60% | ODC-By 1.0 |
| 中文教育 | [Chinese FineWeb Edu V2](https://modelscope.cn/datasets/opencsg/chinese-fineweb-edu-v2) | 16 GiB | 16.29 GiB | 25% | OpenCSG Community License + Apache-2.0 |
| 英文教育 | [FineWeb Edu English](https://modelscope.cn/datasets/HuggingFaceFW/fineweb-edu) | 5 GiB | 6.01 GiB | 10% | ODC-By 1.0 |
| 代码 | [Dolma v1.7 StarCoder](https://modelscope.cn/datasets/modelscope/dolma) | 3 GiB | 3.93 GiB | 5% | ODC-By 1.0 与源文件许可 |
| 合计 | — | 60 GiB | 65.88 GiB | 100% | — |

全量 tokenize 与校验得到：

| 项目 | 数值 |
|---|---:|
| 文档数 | 27,538,101 |
| Token 数 | 33,500,442,523 |
| 二进制分片 | 339 |
| 二进制大小 | 62.40 GiB |
| 存储类型 | little-endian `uint16` |
| 单分片目标 | 100,000,000 tokens |
| Validation 划分 | 基于文档 ID 的稳定 hash，约 0.5% |

完整 tokenized 语料远大于本轮训练实际使用量。正式模型从中按来源概率随机取样 4.00B tokens，因此剩余语料可以继续用于未来的 continued pretraining，而无需重新训练 Tokenizer。


## Base 预训练

主要训练配置：

| 配置 | 数值 |
|---|---:|
| 目标 tokens | 4,000,000,000 |
| 实际 tokens | 4,000,022,528 |
| Optimizer steps | 122,071 |
| Micro batch | 2 |
| Gradient accumulation | 16 |
| Global batch | 32,768 tokens |
| Precision | BF16 |
| Max / min learning rate | `2e-4` / `2e-5` |
| Warmup | 50,000,000 tokens |
| Optimizer | AdamW，betas `(0.9, 0.95)` |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Activation checkpointing | 开启 |
| Seed | 42 |

Validation loss 从 step 100 的 `9.387295` 下降到 step 122,071 的 `3.231346`。最终各来源 loss 为：

| 来源 | Final validation loss |
|---|---:|
| 加权总 loss | 3.231346 |
| 中文网页 | 3.404679 |
| 中文教育 | 3.334161 |
| 英文教育 | 2.988205 |
| 代码 | 1.123570 |

曲线数据与图表保存在 [`docs/loss_curves_pretrain.xlsx`](loss_curves_pretrain.xlsx)。


## 中文 SFT

第一阶段 SFT 使用五类公开中文指令数据：

- [Magpie-Qwen2-Pro-200K-Chinese](https://modelscope.cn/datasets/AI-ModelScope/Magpie-Qwen2-Pro-200K-Chinese)：通用中文指令遵循。
- [Fineweb-Edu-Chinese-V2.2 Pure QA](https://modelscope.cn/datasets/opencsg/Fineweb-Edu-Chinese-V2.2)：教育、科普和解释型问答。
- [SmolTalk-Chinese](https://modelscope.cn/datasets/opencsg/smoltalk-chinese)：聊天、写作、总结、规划、代码、数学和角色扮演等多任务数据。
- [Chinese Qwen3 235B Thinking Distill 100K](https://modelscope.cn/datasets/Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k)：复杂任务的最终答案；构建阶段删除长 reasoning / CoT。
- [Chinese DeepSeek R1 Distill 110K](https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k)：精选推理、数学和 STEM 最终答案；按质量字段筛选并删除长 reasoning。

统一记录采用：

```json
{
  "messages": [
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "助手回答"}
  ]
}
```

构建阶段执行 Unicode 与空白规范化、角色映射、长度过滤、重复行过滤、teacher 身份过滤、稳定去重和基于记录 ID 的 hash 划分。tokenize 阶段将每条记录保存为 1,025 个 token 槽位，对应 1,024-token 输入与 next-token label，并单独保存 loss mask；用户文本、消息前缀和 padding 不参与监督损失。

正式 SFT checkpoint 的训练统计：

| 项目 | 数值 |
|---|---:|
| Train samples | 244,218 |
| Epochs | 1 |
| Steps | 7,632 |
| Effective tokens | 134,897,353 |
| Supervised assistant tokens | 121,248,119 |
| Micro batch / accumulation | 2 / 16 |
| Global batch | 32 samples |
| Max / min learning rate | `2e-5` / `2e-6` |
| Warmup | 200 steps |
| Best validation loss | 1.983117 |

SFT validation loss 从 step 100 的 `2.461950` 下降到 step 7,632 的 `1.983117`。曲线保存在 [`docs/loss_curve_sft.xlsx`](loss_curve_sft.xlsx)。

> 预训练 loss 与 SFT loss 都是 token-level cross entropy，但不能直接横向比较。预训练对连续语料中的全部目标 token 计算损失；SFT 只对 assistant 回答 token 计算损失，而且验证集分布也不同。


## 从原始数据复现

预训练数据链路：

```powershell
py -u -m src.prepare_data
py -u -m src.inspect_data
py -u -m src.train_tokenizer
py -u -m src.tokenize_data
py -u -m src.tokenize_data --verify-only
py -u -m src.train
```

`src.train` 默认自动读取 `latest_pretrain.pt` 续训；使用 `--no-resume` 可禁止自动恢复，使用 `--max-steps N` 可进行短程测试。

SFT 数据链路：

```powershell
py -u -m src.prepare_sft_data
py -u -m src.build_sft_dataset
py -u -m src.tokenize_sft_data
py -u -m src.tokenize_sft_data --verify-only
py -u -m src.train_sft
```

下载和处理脚本会生成 manifest、文件指纹和阶段 checkpoint；中断后重新运行即可从已完成文件或分片继续。上游数据集可能更新，因此未来重新构建得到的精确记录数和哈希不保证与本次已训练 checkpoint 完全相同。


## Checkpoint 与可恢复性

预训练 checkpoint 保存：

- 模型参数。
- AdamW 优化器状态。
- AMP GradScaler 状态（如适用）。
- Python、NumPy、PyTorch CPU / CUDA RNG 状态。
- DataLoader 随机采样状态和各来源累计样本数。
- 当前 step、已处理 token 数和最佳 validation loss。
- 完整实验配置。

写入采用临时文件加原子替换，尽量避免中断时损坏已有 checkpoint。正常训练会维护 `latest_pretrain.pt` / `latest_sft.pt`，而发布前保留的正式归档为 `best_pretrain.pt` / `best_sft.pt`。


## Tokenizer

The tokenizer is trained with the Hugging Face `tokenizers` library, while its vocabulary and merges are learned entirely from this project's mixed corpus.

- Type: Byte-level BPE.
- Vocabulary: 32,768.
- Training sample: 128 MiB of text.
- Sampling mixture: 60% Chinese web, 25% Chinese educational, 10% English educational, and 5% code.
- Special-token matching is disabled for ordinary source text to prevent raw web pages from injecting control markers.
- `<|endoftext|>` separates pretraining documents.
- `<|im_start|>` and `<|im_end|>` delimit SFT messages.

| Token | ID |
|---|---:|
| `<|pad|>` | 0 |
| `<|unk|>` | 1 |
| `<|bos|>` | 2 |
| `<|eos|>` | 3 |
| `<|endoftext|>` | 4 |
| `<|im_start|>` | 5 |
| `<|im_end|>` | 6 |

Sampled acceptance results:

| Source | Documents | Characters / token | UTF-8 bytes / token | UNK | Round-trip failures |
|---|---:|---:|---:|---:|---:|
| Chinese web | 100 | 1.536 | 4.143 | 0 | 0 |
| Chinese educational | 100 | 1.614 | 4.583 | 0 | 0 |
| English educational | 100 | 3.757 | 3.774 | 0 | 0 |
| Code | 100 | 2.959 | 2.964 | 0 | 0 |

The complete report is available at [`data/tokenizer/xianglm-32k/training_report.json`](../data/tokenizer/xianglm-32k/training_report.json).


## Pretraining Data

The raw download plan prioritizes Chinese text while retaining educational English and code:

| Category | Dataset | Requested | Selected | Training weight | Upstream license summary |
|---|---|---:|---:|---:|---|
| Chinese web | [FineWeb 2 Mandarin Chinese](https://modelscope.cn/datasets/HuggingFaceFW/fineweb-2) | 36 GiB | 39.64 GiB | 60% | ODC-By 1.0 |
| Chinese educational | [Chinese FineWeb Edu V2](https://modelscope.cn/datasets/opencsg/chinese-fineweb-edu-v2) | 16 GiB | 16.29 GiB | 25% | OpenCSG Community License + Apache-2.0 |
| English educational | [FineWeb Edu English](https://modelscope.cn/datasets/HuggingFaceFW/fineweb-edu) | 5 GiB | 6.01 GiB | 10% | ODC-By 1.0 |
| Code | [Dolma v1.7 StarCoder](https://modelscope.cn/datasets/modelscope/dolma) | 3 GiB | 3.93 GiB | 5% | ODC-By 1.0 and source-file licenses |
| Total | — | 60 GiB | 65.88 GiB | 100% | — |

Full tokenization and validation produced:

| Item | Value |
|---|---:|
| Documents | 27,538,101 |
| Tokens | 33,500,442,523 |
| Binary shards | 339 |
| Binary size | 62.40 GiB |
| Storage dtype | little-endian `uint16` |
| Target shard size | 100,000,000 tokens |
| Validation split | Stable document-ID hash, approximately 0.5% |

The tokenized corpus is much larger than the amount consumed by this training run. The official model sampled 4.00B tokens according to source probabilities, leaving the remaining corpus available for future continued pretraining without retraining the tokenizer.


## Base Pretraining

| Configuration | Value |
|---|---:|
| Target tokens | 4,000,000,000 |
| Actual tokens | 4,000,022,528 |
| Optimizer steps | 122,071 |
| Micro batch | 2 |
| Gradient accumulation | 16 |
| Global batch | 32,768 tokens |
| Precision | BF16 |
| Max / min learning rate | `2e-4` / `2e-5` |
| Warmup | 50,000,000 tokens |
| Optimizer | AdamW, betas `(0.9, 0.95)` |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Activation checkpointing | Enabled |
| Seed | 42 |

Validation loss decreased from `9.387295` at step 100 to `3.231346` at step 122,071. Final source-specific values were:

| Source | Final validation loss |
|---|---:|
| Weighted overall | 3.231346 |
| Chinese web | 3.404679 |
| Chinese educational | 3.334161 |
| English educational | 2.988205 |
| Code | 1.123570 |

Curve data and charts are stored in [`docs/loss_curves_pretrain.xlsx`](loss_curves_pretrain.xlsx).


## Chinese SFT

The first-stage instruction mixture uses five public Chinese datasets:

- [Magpie-Qwen2-Pro-200K-Chinese](https://modelscope.cn/datasets/AI-ModelScope/Magpie-Qwen2-Pro-200K-Chinese) for general Chinese instruction following.
- [Fineweb-Edu-Chinese-V2.2 Pure QA](https://modelscope.cn/datasets/opencsg/Fineweb-Edu-Chinese-V2.2) for educational, scientific, and explanatory answers.
- [SmolTalk-Chinese](https://modelscope.cn/datasets/opencsg/smoltalk-chinese) for chat, writing, summarization, planning, coding, mathematics, and role-play tasks.
- [Chinese Qwen3 235B Thinking Distill 100K](https://modelscope.cn/datasets/Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k) for complex final answers, with long reasoning / CoT removed during dataset construction.
- [Chinese DeepSeek R1 Distill 110K](https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k) for selected reasoning, mathematics, and STEM final answers, filtered by quality metadata and stripped of long reasoning.

All records are normalized to:

```json
{
  "messages": [
    {"role": "user", "content": "User question"},
    {"role": "assistant", "content": "Assistant answer"}
  ]
}
```

Dataset construction performs Unicode and whitespace normalization, role mapping, length filtering, repeated-line filtering, teacher-identity filtering, stable deduplication, and record-ID hash splitting. Tokenization stores 1,025 token slots per record, corresponding to a 1,024-token input and next-token labels, together with a separate loss mask. User text, control prefixes, and padding do not contribute to supervised loss.

Official SFT checkpoint statistics:

| Item | Value |
|---|---:|
| Training examples | 244,218 |
| Epochs | 1 |
| Steps | 7,632 |
| Effective tokens | 134,897,353 |
| Supervised assistant tokens | 121,248,119 |
| Micro batch / accumulation | 2 / 16 |
| Global batch | 32 examples |
| Max / min learning rate | `2e-5` / `2e-6` |
| Warmup | 200 steps |
| Best validation loss | 1.983117 |

SFT validation loss decreased from `2.461950` at step 100 to `1.983117` at step 7,632. The curve is stored in [`docs/loss_curve_sft.xlsx`](loss_curve_sft.xlsx).

> Pretraining and SFT losses are both token-level cross-entropies, but they are not directly comparable. Pretraining supervises every target token in continuous text, whereas SFT supervises only assistant-answer tokens on a different validation distribution.
