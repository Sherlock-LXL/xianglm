# XiangLM

[中文](#中文) · [English](#english) · [下载](https://github.com/Sherlock-LXL/xianglm/releases)

## 中文

从随机初始化训练的 200M 中文 Decoder-only Transformer，覆盖语料下载、Byte-level BPE、全量 tokenize、混合预训练、assistant-only SFT 与流式单轮聊天。

### 运行

源码主要在 Windows、Python 3.12 与 CUDA 环境开发：

```bash
python -m pip install -r requirements.txt
python -u app.py
python -m unittest discover -s tests -v
```

聊天需要 `data/checkpoints/best_sft.pt` 与 `data/tokenizer/xianglm-32k/tokenizer.json`；Base 测试还需 `best_pretrain.pt`。完整 checkpoint 含优化器与恢复状态，每份约 2.24 GiB，不随普通 Git 发布。

Windows 桌面预览版下载 `XiangLM-1.0.0.zip`，完整解压后运行 `XiangLM.exe`，保留同目录 `_internal`。要求 Windows 10/11 x64、约 2 GB 空间，建议至少 8 GB RAM，本地 CPU 推理。

### 技术与结果

| 配置/结果 | 数值 |
|---|---:|
| 参数量 | 200,308,992 |
| Layers / hidden / FFN / heads | 25 / 768 / 2,016 / 12 |
| Context / vocabulary | 1,024 / 32,768 |
| 预训练 tokens / steps | 4,000,022,528 / 122,071 |
| 最佳预训练 validation loss | 3.231346 |
| SFT 样本 / steps / epochs | 244,218 / 7,632 / 1 |
| Assistant 监督 tokens | 121,248,119 |
| 最佳 SFT validation loss | 1.983117 |
| 训练设备 | RTX 5060 Laptop，8 GB VRAM |

手工实现 RMSNorm、RoPE、因果注意力、SwiGLU、残差与 embedding/LM-head 权重绑定。BF16、micro batch 2、梯度累积 16 与 activation checkpointing 支持笔记本训练；checkpoint 原子保存模型、优化器、RNG 与 loader 状态。

语料编码为 **33,500,442,523** 个 `uint16` tokens、339 个分片（62.40 GiB），训练按中文网页/中文教育/英文教育/代码 **60/25/10/5%** 混合采样。Tokenizer 抽样验收 UNK 与回译失败均为 0。预训练和 SFT loss 使用不同数据与监督目标，不能直接横向比较。

### 聊天与复现

| 终端模式 | Temperature | Top-k | Top-p | Repetition penalty | 最大 tokens |
|---|---:|---:|---:|---:|---:|
| 稳定 | 0.30 | 20 | 0.85 | 1.12 | 400 |
| 平衡 | 0.65 | 40 | 0.90 | 1.10 | 600 |
| 激进 | 0.90 | 60 | 0.95 | 1.10 | 700 |

`/mode` 切换模式，`/exit` 退出，`Ctrl+C` 停止当前生成。每轮独立，不保留聊天历史；生成未使用 KV cache。

`src/` 包含数据与模型工程，`tests/` 检查因果性、梯度、数据移位与精确恢复。结构测试无需完整语料或正式 checkpoint；正式训练需要支持 BF16 的 NVIDIA GPU。

[完整参数、数据来源与复现命令](docs/method.md) · [预训练曲线](docs/loss_curves_pretrain.xlsx) · [SFT 曲线](docs/loss_curve_sft.xlsx)

原始语料与训练权重不随代码仓库提供；上游分别适用 ODC-By、Apache-2.0、OpenCSG Community License 和源文件许可。仓库尚未附独立代码许可证。

这是实验模型，可能出现事实、代码、计算错误、重复和截断；用于学习、研究与娱乐，不用于专业决策。作者执行数据与训练实验、判断结果并决定方案；Codex/ChatGPT 协助工程、调试与文档。

## English

A 200M Chinese decoder-only Transformer trained from random initialization, covering corpus downloads, Byte-level BPE, full tokenization, mixed-source pretraining, assistant-only SFT and streaming single-turn chat.

### Run

Source development primarily used Windows, Python 3.12 and CUDA:

```bash
python -m pip install -r requirements.txt
python -u app.py
python -m unittest discover -s tests -v
```

Chat requires `data/checkpoints/best_sft.pt` and `data/tokenizer/xianglm-32k/tokenizer.json`; Base tests also need `best_pretrain.pt`. Full resumable checkpoints include optimizer/state data, about 2.24 GiB each, and are excluded from ordinary Git.

Download the Windows desktop preview `XiangLM-1.0.0.zip`, extract it completely and run `XiangLM.exe` with `_internal` beside it. Requires Windows 10/11 x64, about 2 GB storage and preferably at least 8 GB RAM; inference runs locally on CPU.

### Technology and results

| Configuration/result | Value |
|---|---:|
| Parameters | 200,308,992 |
| Layers / hidden / FFN / heads | 25 / 768 / 2,016 / 12 |
| Context / vocabulary | 1,024 / 32,768 |
| Pretraining tokens / steps | 4,000,022,528 / 122,071 |
| Best pretraining validation loss | 3.231346 |
| SFT samples / steps / epochs | 244,218 / 7,632 / 1 |
| Supervised assistant tokens | 121,248,119 |
| Best SFT validation loss | 1.983117 |
| Training hardware | RTX 5060 Laptop, 8 GB VRAM |

RMSNorm, RoPE, causal attention, SwiGLU, residuals and tied embedding/LM-head weights are implemented manually. BF16, micro batch 2, accumulation 16 and activation checkpointing support laptop training. Atomic checkpoints preserve model, optimizer, RNG and loader state.

The encoded corpus contains **33,500,442,523** `uint16` tokens in 339 shards (62.40 GiB). Chinese web/Chinese education/English education/code sampling weights are **60/25/10/5%**. Tokenizer sample checks had zero UNK and round-trip failures. Pretraining and SFT losses use different data and supervision objectives and are not directly comparable.

### Chat and reproduction

| Terminal mode | Temperature | Top-k | Top-p | Repetition penalty | Max tokens |
|---|---:|---:|---:|---:|---:|
| Stable | 0.30 | 20 | 0.85 | 1.12 | 400 |
| Balanced | 0.65 | 40 | 0.90 | 1.10 | 600 |
| Adventurous | 0.90 | 60 | 0.95 | 1.10 | 700 |

`/mode` changes sampling, `/exit` quits and `Ctrl+C` stops the current generation. Each turn is independent; no history is retained and generation has no KV cache.

`src/` contains data/model engineering. `tests/` checks causality, gradients, shifted data and exact recovery without the full corpus or production checkpoint. Full training requires a BF16-capable NVIDIA GPU.

[Parameters, data sources and reproduction commands](docs/method.md) · [Pretraining curves](docs/loss_curves_pretrain.xlsx) · [SFT curves](docs/loss_curve_sft.xlsx)

Raw corpora and training weights are excluded. Sources retain ODC-By, Apache-2.0, OpenCSG Community License and source-file terms. No independent code license is included.

This experimental model can produce factual, coding and arithmetic errors, repetition or truncation. It is for learning, research and entertainment rather than professional decisions. The author executes data/training experiments and decides from results; Codex/ChatGPT assisted with engineering, debugging and documentation.
