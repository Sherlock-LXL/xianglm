# XiangLM — 从零训练的 200M 中文 Transformer / A 200M Chinese Transformer Trained from Scratch

XiangLM 是一个面向学习与工程实践的中文语言模型项目：从原始语料下载、数据检查、Byte-level BPE Tokenizer 训练、全量 tokenize、混合数据加载、Decoder-only Transformer 预训练，到中文指令微调与流式终端聊天，整条链路均在本项目中实现并完成了实际训练。

XiangLM is an educational and engineering-oriented Chinese language model project. It implements and runs the complete pipeline—from raw corpus acquisition, inspection, Byte-level BPE tokenizer training, full-corpus tokenization, and mixed-source loading to Decoder-only Transformer pretraining, Chinese supervised fine-tuning, and streaming terminal chat.

[中文](#中文) · [English](#english)

---

## 中文

### 项目概览

这个项目的目标不是调用一个现成的 Transformer 模型完成微调，而是亲自走完一次小型语言模型的完整生命周期，并把每个重要环节保留为可理解、可检查、可恢复的工程模块。

当前正式模型具有以下状态：

| 项目 | 结果 |
|---|---:|
| 模型类型 | Decoder-only Transformer Language Model |
| 参数量 | 200,308,992（约 200.3M） |
| Transformer Blocks | 25 |
| 注意力头 | 12 |
| 上下文长度 | 1,024 tokens |
| Tokenizer | Byte-level BPE，32,768 词表 |
| 预训练 | 4,000,022,528 tokens，122,071 steps |
| 最佳预训练 validation loss | 3.231346 |
| SFT | 244,218 条训练样本，7,632 steps，1 epoch |
| SFT 监督 tokens | 121,248,119 |
| 最佳 SFT validation loss | 1.983117 |
| 主要训练设备 | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB VRAM |

项目先用约 12M 和 30M 参数模型验证工程管线，再将同一套训练流程扩展到正式的 200M 模型。最终 Base 模型从随机初始化开始训练，不依赖外部预训练权重；Chat 模型则由该 Base checkpoint 继续进行中文 SFT 得到。

### 项目亮点

- 从随机初始化训练约 200M 参数的中文语言模型，而不是套用 Hugging Face Transformers 中的现成模型类。
- 使用 PyTorch 手工实现 RMSNorm、RoPE、Causal Self-Attention、SwiGLU、残差连接、权重绑定和完整语言模型损失。
- 在 8 GB 笔记本显存上通过 BF16、micro batch、梯度累积和 activation checkpointing 完成 4B-token 预训练。
- 构建 32,768 词表的 Byte-level BPE Tokenizer；抽样验收中 `UNK = 0`，文本回译失败数为 `0`。
- 将 33.50B tokens 的本地语料编码为可随机访问的 `uint16` 二进制分片，训练时通过 memory mapping 按需读取。
- 支持按中文网页、中文教育、英文教育和代码四类来源进行概率混合采样，并分别计算 validation loss。
- checkpoint 同时保存模型、优化器、随机数状态和 DataLoader 状态，支持原位置恢复训练。
- SFT 数据统一为对话消息格式，只在 assistant 回答 token 上计算监督损失。
- 提供稳定、平衡、激进三种采样模式，以及逐 token 输出的终端聊天程序。
- 保留独立的模型、DataLoader 和 checkpoint 单元测试，便于验证因果性、梯度、数据移位和精确恢复。

### 终端聊天

项目根目录下的 `app.py` 是最直接的体验入口：

```powershell
py -u app.py
```

启动后可选择三种生成模式：

| 模式 | Temperature | Top-k | Top-p | Repetition penalty | 最大生成 tokens | 适用场景 |
|---|---:|---:|---:|---:|---:|---|
| 稳定 | 0.30 | 20 | 0.85 | 1.12 | 400 | 更确定、更保守的回答 |
| 平衡 | 0.65 | 40 | 0.90 | 1.10 | 600 | 默认聊天与通用测试 |
| 激进 | 0.90 | 60 | 0.95 | 1.10 | 700 | 创作与高随机性观察 |

运行中的命令：

- `/mode`：切换采样模式。
- `/exit`：退出程序。
- `Ctrl+C`：只停止当前一轮生成，保留程序与模型。

回答会随着 token 生成实时输出。当前 App 将每个问题视为独立单轮对话，不保存跨轮聊天历史；这能减少 1,024-token 上下文下的角色混淆，也更适合观察模型本身的单轮能力。

### 完整工程流程

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

### 项目结构与文件作用

```text
xianglm/
├── app.py
├── requirements.txt
├── README.md
├── docs/
│   ├── loss_curves_pretrain.xlsx
│   └── loss_curve_sft.xlsx
├── data/
│   ├── checkpoints/
│   │   ├── best_pretrain.pt
│   │   └── best_sft.pt
│   └── tokenizer/
│       └── xianglm-32k/
│           ├── tokenizer.json
│           ├── vocab.json
│           ├── merges.txt
│           └── training_report.json
├── src/
│   ├── config.py
│   ├── model.py
│   ├── checkpoint.py
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train_tokenizer.py
│   ├── tokenize_data.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── prepare_sft_data.py
│   ├── build_sft_dataset.py
│   ├── tokenize_sft_data.py
│   ├── train_sft.py
│   └── generate_sft.py
└── tests/
    ├── support.py
    ├── test_model.py
    ├── test_data_loader.py
    └── test_checkpoint.py
```

| 文件 | 作用 |
|---|---|
| `app.py` | 加载正式 SFT checkpoint，选择采样模式并逐 token 输出单轮回答。 |
| `src/config.py` | 模型、训练、数据混合、路径和 checkpoint 名称的统一配置。 |
| `src/model.py` | 从零实现 Decoder-only Transformer 与语言模型交叉熵。 |
| `src/checkpoint.py` | 原子保存和加载模型、优化器、随机状态与 DataLoader 状态。 |
| `src/prepare_data.py` | 从 ModelScope 规划、下载、断点续传并验收四类预训练原始数据。 |
| `src/inspect_data.py` | 统一读取 Parquet / gzip JSON 数据，检查字段并展示正文样例。它同时为 Tokenizer 和 tokenize 阶段提供共享数据读取逻辑。 |
| `src/train_tokenizer.py` | 按训练混合比例抽样 128 MiB 文本，训练 32K Byte-level BPE 并完成回译验收。 |
| `src/tokenize_data.py` | 全量编码预训练语料，稳定划分 train / validation，写入 100M-token `uint16` 分片并生成 manifest。 |
| `src/data_loader.py` | 用 `numpy.memmap` 随机读取 token 窗口，按来源权重组成自回归训练 batch。 |
| `src/train.py` | 完成 Base 模型训练、梯度累积、学习率调度、验证、日志、定期保存和中断恢复。 |
| `src/evaluate.py` | 独立或训练中评估 Base checkpoint；分别计算四类来源 loss，再按训练权重得到加权总 loss。 |
| `src/generate.py` | 对 Base 模型执行 continuation 测试，支持 temperature、top-k、top-p 和 repetition penalty。 |
| `src/prepare_sft_data.py` | 下载并验收五类公开中文指令数据，保存来源与文件 manifest。 |
| `src/build_sft_dataset.py` | 适配不同上游 schema，规范为 messages，过滤异常样本、删除长 CoT、去重、分层抽样并稳定划分数据集。 |
| `src/tokenize_sft_data.py` | 按 ChatML 风格编码消息，生成固定长度 token、loss mask 与有效长度分片；只监督 assistant 内容。 |
| `src/train_sft.py` | 从 `best_pretrain.pt` 开始进行 assistant-only SFT，支持恢复、验证与最佳 checkpoint 保存。SFT 专用 loader 位于该文件内。 |
| `src/generate_sft.py` | 将普通问题自动包装为 user / assistant prompt，执行单轮或交互式 SFT 模型测试。 |
| `tests/test_model.py` | 检查参数量、embedding 权重绑定、前后向传播和 causal attention。 |
| `tests/test_data_loader.py` | 检查 batch shift、来源比例和 DataLoader 状态的精确恢复。 |
| `tests/test_checkpoint.py` | 检查 checkpoint 保存、模型恢复、优化器恢复、loader 恢复与 RNG 状态。 |

### 模型架构

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

### Tokenizer

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

完整报告保存在 [`data/tokenizer/xianglm-32k/training_report.json`](data/tokenizer/xianglm-32k/training_report.json)。

### 预训练数据

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

### Base 预训练

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

曲线数据与图表保存在 [`docs/loss_curves_pretrain.xlsx`](docs/loss_curves_pretrain.xlsx)。

### 中文 SFT

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

SFT validation loss 从 step 100 的 `2.461950` 下降到 step 7,632 的 `1.983117`。曲线保存在 [`docs/loss_curve_sft.xlsx`](docs/loss_curve_sft.xlsx)。

> 预训练 loss 与 SFT loss 都是 token-level cross entropy，但不能直接横向比较。预训练对连续语料中的全部目标 token 计算损失；SFT 只对 assistant 回答 token 计算损失，而且验证集分布也不同。

### 安装

项目主要在 Windows、Python 3.12 与 CUDA 环境下开发。建议在独立虚拟环境中安装：

```powershell
cd xianglm

py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

依赖：

- `torch`
- `tokenizers`
- `numpy`
- `pyarrow`
- `modelscope-hub`

训练 200M 模型需要支持 BF16 的 NVIDIA GPU。CPU 可以运行结构测试和小规模推理，但正式训练和长文本生成会非常慢。

### 运行所需文件

仓库通过 `.gitignore` 排除了大体积 checkpoint、原始数据、tokenized 数据、日志和 manifest。仅运行最终聊天程序至少需要：

```text
data/
├── checkpoints/
│   └── best_sft.pt
└── tokenizer/
    └── xianglm-32k/
        └── tokenizer.json
```

如需测试 Base 模型，还需要：

```text
data/checkpoints/best_pretrain.pt
```

当前 `.pt` 文件是可恢复训练的完整 checkpoint，其中包含模型、优化器、随机状态和 loader 状态，因此每个文件约 2.24 GiB，并非压缩后的 inference-only 权重。

### 常用命令

检查模型配置：

```powershell
py -u -m src.config
```

运行最终聊天 App：

```powershell
py -u app.py
```

单轮测试 SFT 模型：

```powershell
py -u -m src.generate_sft `
    --prompt "请简单介绍一下人工智能。" `
    --max-new-tokens 200 `
    --temperature 0.7 `
    --top-k 40 `
    --top-p 0.9 `
    --repetition-penalty 1.10
```

测试 Base 模型续写：

```powershell
py -u -m src.generate `
    --checkpoint "data\checkpoints\best_pretrain.pt" `
    --prompt "人工智能是计算机科学的一个分支，主要研究" `
    --max-new-tokens 150
```

独立评估 Base 模型：

```powershell
py -u -m src.evaluate `
    --checkpoint "data\checkpoints\best_pretrain.pt"
```

### 从原始数据复现

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

### 测试

```powershell
py -m unittest discover -s tests -v
```

测试使用临时构造的小型 token 分片，不需要下载完整 60+ GiB 数据，也不需要正式 checkpoint。CUDA 专属 RNG 映射测试只在 CUDA 可用时执行。

### Checkpoint 与可恢复性

预训练 checkpoint 保存：

- 模型参数。
- AdamW 优化器状态。
- AMP GradScaler 状态（如适用）。
- Python、NumPy、PyTorch CPU / CUDA RNG 状态。
- DataLoader 随机采样状态和各来源累计样本数。
- 当前 step、已处理 token 数和最佳 validation loss。
- 完整实验配置。

写入采用临时文件加原子替换，尽量避免中断时损坏已有 checkpoint。正常训练会维护 `latest_pretrain.pt` / `latest_sft.pt`，而发布前保留的正式归档为 `best_pretrain.pt` / `best_sft.pt`。

### 数据、许可与发布说明

数据不随本仓库发布。每个上游数据集仍受其数据集卡、原始来源与许可条款约束，包括但不限于 ODC-By、Apache-2.0、OpenCSG Community License 以及原始代码文件的各自许可。使用者应在下载、再分发数据或发布衍生权重前自行核对最新条款。

本仓库目前尚未附带独立的 `LICENSE` 文件。在公开发布或接受外部复用前，应先选择代码许可证，并分别确认训练权重与各上游数据条款的兼容性。README 中的实验结果不构成对模型正确性、安全性或适用性的保证。

### 已知限制

- 200M 参数规模足以学习基本中文表达和指令格式，但远小于现代通用大模型。
- 模型会产生事实错误、概念混淆、无效代码、算术错误、重复和虚构信息。
- 1,024-token 上下文限制了长文档理解和多轮聊天。
- 当前 App 是独立单轮对话，不具备长期记忆。
- 当前生成没有 KV cache，长回答速度仍可明显优化。
- SFT 采用公开合成数据，可能继承 teacher 模型的偏差、措辞习惯和错误。
- 未进行系统性安全对齐、DPO/RLHF、事实性评测或生产环境压力测试。
- 不应将该模型用于医疗、法律、金融、安全决策或其他高风险场景。

### 后续方向

- 为生成加入 KV cache 和更精确的流式解码。
- 建立固定的中文问答、知识、代码、数学、写作和重复率评测集。
- 对 Base 模型继续使用尚未消费的 tokenized 语料进行 continued pretraining。
- 改进 SFT 数据质量评分、短答案比例和事实性过滤。
- 在保留通用能力的前提下探索领域 SFT，例如法律、教育或游戏社区风格。
- 尝试 LoRA、DPO 或轻量 preference optimization，并与完整参数 SFT 对比。
- 导出只包含模型权重的 inference checkpoint，降低发布体积。
- 增加多轮上下文、系统提示和可配置的结束策略。

### AI 协作说明

项目的学习路线、数据规模、模型配置、训练决策、运行验证和结果判断由项目作者持续参与并执行。Codex / ChatGPT 用于协助讨论方案、编写与整理部分工程代码、定位下载和 checkpoint 问题、分析训练日志以及完善文档。最终训练均在作者自己的本地环境中实际运行并验证。

---

## English

### Overview

XiangLM is not a wrapper around a pretrained Transformer. Its purpose is to complete and document the full lifecycle of a small language model while keeping every major stage understandable, inspectable, and recoverable.

The current release state is summarized below:

| Item | Result |
|---|---:|
| Model type | Decoder-only Transformer Language Model |
| Parameters | 200,308,992 (~200.3M) |
| Transformer blocks | 25 |
| Attention heads | 12 |
| Context length | 1,024 tokens |
| Tokenizer | 32,768-vocabulary Byte-level BPE |
| Pretraining | 4,000,022,528 tokens, 122,071 steps |
| Best pretraining validation loss | 3.231346 |
| SFT | 244,218 training examples, 7,632 steps, 1 epoch |
| SFT supervised tokens | 121,248,119 |
| Best SFT validation loss | 1.983117 |
| Primary training device | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM |

The engineering pipeline was first validated with approximately 12M- and 30M-parameter experiments and then scaled to the official 200M model. The Base model was pretrained from random initialization without external pretrained weights. The Chat model was obtained by supervised fine-tuning the resulting Base checkpoint.

### Highlights

- Pretrains a roughly 200M-parameter Chinese language model from random initialization.
- Implements RMSNorm, RoPE, causal self-attention, SwiGLU, residual connections, tied embeddings, and LM cross-entropy directly in PyTorch.
- Completes 4B-token pretraining on an 8 GB laptop GPU using BF16, micro-batching, gradient accumulation, and activation checkpointing.
- Trains a 32,768-vocabulary Byte-level BPE tokenizer with zero sampled UNK tokens and zero round-trip failures.
- Encodes 33.50B local tokens into randomly accessible `uint16` binary shards and reads them with memory mapping.
- Samples Chinese web, Chinese educational, English educational, and code data with explicit mixture weights and reports per-source validation loss.
- Stores model, optimizer, RNG, and DataLoader states for exact in-place training continuation.
- Normalizes heterogeneous instruction datasets into a message format and computes SFT loss only on assistant tokens.
- Includes stable, balanced, and aggressive sampling modes with token-by-token terminal streaming.
- Provides unit tests for causality, gradients, shifted labels, source sampling, checkpointing, and deterministic restoration.

### Terminal Chat

Run the interactive application from the repository root:

```powershell
py -u app.py
```

Three decoding modes are available:

| Mode | Temperature | Top-k | Top-p | Repetition penalty | Max new tokens | Intended use |
|---|---:|---:|---:|---:|---:|---|
| Stable | 0.30 | 20 | 0.85 | 1.12 | 400 | More deterministic and conservative output |
| Balanced | 0.65 | 40 | 0.90 | 1.10 | Default chat and general testing |
| Aggressive | 0.90 | 60 | 0.95 | 1.10 | Creative and high-variance sampling |

Runtime commands:

- `/mode`: switch decoding mode.
- `/exit`: leave the application.
- `Ctrl+C`: stop only the current generation while keeping the model loaded.

Tokens are printed as soon as they are decoded. The current application treats every prompt as an independent single-turn conversation. This reduces role drift within the 1,024-token context and makes the model's single-turn behavior easier to inspect.

### End-to-End Pipeline

```text
Raw pretraining corpora
    ↓ prepare_data.py
Data inspection
    ↓ inspect_data.py
Tokenizer training and validation
    ↓ train_tokenizer.py
Full uint16 token shards
    ↓ tokenize_data.py
Mixed-source DataLoader
    ↓ data_loader.py
200M Base pretraining
    ↓ train.py + evaluate.py
best_pretrain.pt
    ↓
Raw SFT dataset download
    ↓ prepare_sft_data.py
Normalization, filtering, deduplication, splitting
    ↓ build_sft_dataset.py
Fixed-length tokens, loss masks, lengths
    ↓ tokenize_sft_data.py
Assistant-only supervised fine-tuning
    ↓ train_sft.py
best_sft.pt
    ↓
generate_sft.py / app.py
```

Pretraining and SFT use separate dataset builders because their objectives differ. Pretraining needs continuous text plus document boundaries. SFT additionally needs explicit `user` / `assistant` structure, conversation-aware truncation, and assistant-only loss masks.

### Repository Layout and File Guide

```text
xianglm/
├── app.py
├── requirements.txt
├── README.md
├── docs/
│   ├── loss_curves_pretrain.xlsx
│   └── loss_curve_sft.xlsx
├── data/
│   ├── checkpoints/
│   │   ├── best_pretrain.pt
│   │   └── best_sft.pt
│   └── tokenizer/
│       └── xianglm-32k/
│           ├── tokenizer.json
│           ├── vocab.json
│           ├── merges.txt
│           └── training_report.json
├── src/
│   ├── config.py
│   ├── model.py
│   ├── checkpoint.py
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train_tokenizer.py
│   ├── tokenize_data.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── prepare_sft_data.py
│   ├── build_sft_dataset.py
│   ├── tokenize_sft_data.py
│   ├── train_sft.py
│   └── generate_sft.py
└── tests/
    ├── support.py
    ├── test_model.py
    ├── test_data_loader.py
    └── test_checkpoint.py
```

| File | Purpose |
|---|---|
| `app.py` | Loads the official SFT checkpoint, selects a decoding mode, and streams single-turn answers token by token. |
| `src/config.py` | Central definition of model, training, data mixture, paths, and checkpoint names. |
| `src/model.py` | Implements the Decoder-only Transformer and language-model cross-entropy from scratch. |
| `src/checkpoint.py` | Atomically saves and restores model, optimizer, RNG, and DataLoader states. |
| `src/prepare_data.py` | Plans, downloads, resumes, and validates four pretraining source groups from ModelScope. |
| `src/inspect_data.py` | Provides shared Parquet / gzip JSON readers, schema inspection, and text previews. |
| `src/train_tokenizer.py` | Samples 128 MiB according to the training mixture, trains the 32K Byte-level BPE tokenizer, and validates round trips. |
| `src/tokenize_data.py` | Encodes the complete pretraining corpus, creates stable train / validation splits, writes 100M-token `uint16` shards, and builds a manifest. |
| `src/data_loader.py` | Reads random token windows through `numpy.memmap` and assembles autoregressive batches according to source weights. |
| `src/train.py` | Runs Base training, gradient accumulation, LR scheduling, validation, logging, checkpointing, and resume logic. |
| `src/evaluate.py` | Evaluates Base checkpoints overall and by source, both independently and during training. |
| `src/generate.py` | Tests Base-model continuation with temperature, top-k, top-p, and repetition penalty. |
| `src/prepare_sft_data.py` | Downloads and validates five public Chinese instruction datasets and records source manifests. |
| `src/build_sft_dataset.py` | Adapts upstream schemas, normalizes messages, filters anomalies, removes long CoT, deduplicates, samples, and splits records. |
| `src/tokenize_sft_data.py` | Applies ChatML-style message encoding and writes fixed-length token, loss-mask, and length shards. |
| `src/train_sft.py` | Fine-tunes `best_pretrain.pt` with assistant-only loss and contains the dedicated SFT dataset loader. |
| `src/generate_sft.py` | Wraps plain questions in user / assistant control tokens and supports one-shot or interactive testing. |
| `tests/test_model.py` | Tests parameter accounting, tied embeddings, forward/backward propagation, and causality. |
| `tests/test_data_loader.py` | Tests shifted targets, source ratios, and exact DataLoader restoration. |
| `tests/test_checkpoint.py` | Tests checkpoint writes and restoration of models, optimizers, loaders, and RNG state. |

### Architecture

XiangLM is a GPT-style pre-norm Decoder-only Transformer with several components commonly used in modern compact language models.

| Configuration | Value |
|---|---:|
| Parameters | 200,308,992 |
| Vocabulary | 32,768 |
| Context length | 1,024 |
| `d_model` | 768 |
| Transformer layers | 25 |
| Attention heads | 12 |
| Head dimension | 64 |
| `d_ff` | 2,016 |
| Normalization | RMSNorm, `eps=1e-5` |
| Position encoding | RoPE, `theta=10,000` |
| Feed-forward | SwiGLU |
| Attention | PyTorch scaled dot-product causal attention |
| Bias | Disabled |
| Dropout | 0.0 |
| Embedding / LM head | Tied weights |

Each block follows:

```text
x = x + CausalSelfAttention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

Activation checkpointing reduces training memory. Sharing the token embedding and LM head saves a substantial number of parameters with a 32K vocabulary. Generation currently does not use a KV cache, so it recomputes the retained context for every new token; this keeps the implementation simple but leaves clear room for inference optimization.

### Tokenizer

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

The complete report is available at [`data/tokenizer/xianglm-32k/training_report.json`](data/tokenizer/xianglm-32k/training_report.json).

### Pretraining Data

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

### Base Pretraining

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

Curve data and charts are stored in [`docs/loss_curves_pretrain.xlsx`](docs/loss_curves_pretrain.xlsx).

### Chinese SFT

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

SFT validation loss decreased from `2.461950` at step 100 to `1.983117` at step 7,632. The curve is stored in [`docs/loss_curve_sft.xlsx`](docs/loss_curve_sft.xlsx).

> Pretraining and SFT losses are both token-level cross-entropies, but they are not directly comparable. Pretraining supervises every target token in continuous text, whereas SFT supervises only assistant-answer tokens on a different validation distribution.

### Installation

The project was primarily developed on Windows with Python 3.12 and CUDA. A dedicated virtual environment is recommended:

```powershell
cd xianglm

py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Dependencies:

- `torch`
- `tokenizers`
- `numpy`
- `pyarrow`
- `modelscope-hub`

Training the 200M model requires a BF16-capable NVIDIA GPU. CPU execution is sufficient for structural tests and small inference checks, but full training and long-form generation will be very slow.

### Required Runtime Files

Large checkpoints, raw data, tokenized data, logs, and manifests are excluded through `.gitignore`. Running the final chat application requires at least:

```text
data/
├── checkpoints/
│   └── best_sft.pt
└── tokenizer/
    └── xianglm-32k/
        └── tokenizer.json
```

Base-model testing additionally requires:

```text
data/checkpoints/best_pretrain.pt
```

The current `.pt` files are full resumable training checkpoints containing model, optimizer, RNG, and loader state. Each is therefore approximately 2.24 GiB and is not an inference-only compressed weight file.

### Common Commands

Inspect the configuration:

```powershell
py -u -m src.config
```

Run the final chat application:

```powershell
py -u app.py
```

Test the SFT model with one prompt:

```powershell
py -u -m src.generate_sft `
    --prompt "请简单介绍一下人工智能。" `
    --max-new-tokens 200 `
    --temperature 0.7 `
    --top-k 40 `
    --top-p 0.9 `
    --repetition-penalty 1.10
```

Test Base-model continuation:

```powershell
py -u -m src.generate `
    --checkpoint "data\checkpoints\best_pretrain.pt" `
    --prompt "人工智能是计算机科学的一个分支，主要研究" `
    --max-new-tokens 150
```

Evaluate the Base model independently:

```powershell
py -u -m src.evaluate `
    --checkpoint "data\checkpoints\best_pretrain.pt"
```

### Rebuilding from Raw Data

Pretraining pipeline:

```powershell
py -u -m src.prepare_data
py -u -m src.inspect_data
py -u -m src.train_tokenizer
py -u -m src.tokenize_data
py -u -m src.tokenize_data --verify-only
py -u -m src.train
```

`src.train` automatically resumes from `latest_pretrain.pt` by default. Use `--no-resume` to disable automatic restoration or `--max-steps N` for a short engineering run.

SFT pipeline:

```powershell
py -u -m src.prepare_sft_data
py -u -m src.build_sft_dataset
py -u -m src.tokenize_sft_data
py -u -m src.tokenize_sft_data --verify-only
py -u -m src.train_sft
```

Download and processing stages maintain manifests, fingerprints, and stage checkpoints. Re-running after interruption resumes from completed files or shards. Upstream datasets may change, so a future rebuild is not guaranteed to reproduce the exact record count or hash of the already-trained checkpoint.

### Tests

```powershell
py -m unittest discover -s tests -v
```

Tests generate tiny temporary token fixtures, so they do not require the full 60+ GiB corpus or official checkpoints. CUDA-specific RNG mapping is executed only when CUDA is available.

### Checkpoints and Resumability

Pretraining checkpoints contain:

- Model parameters.
- AdamW optimizer state.
- AMP GradScaler state when applicable.
- Python, NumPy, PyTorch CPU, and CUDA RNG states.
- DataLoader sampling state and per-source counters.
- Current step, processed-token count, and best validation loss.
- The complete experiment configuration.

Writes use a temporary file followed by atomic replacement to reduce the chance of corrupting a valid checkpoint during interruption. Active training maintains `latest_pretrain.pt` / `latest_sft.pt`, while the retained release archives are `best_pretrain.pt` / `best_sft.pt`.

### Data, Licensing, and Release Notes

Training data is not distributed with this repository. Every upstream dataset remains subject to its dataset card, original source terms, and licenses, including ODC-By, Apache-2.0, the OpenCSG Community License, and source-specific licenses for code files. Users must verify the latest terms before downloading, redistributing data, or publishing derived weights.

This repository does not currently include a standalone `LICENSE` file. A code license should be selected before public reuse is invited, and compatibility between model-weight distribution and all upstream dataset terms should be reviewed separately. The reported results are not a warranty of correctness, safety, or fitness for any purpose.

### Known Limitations

- A 200M-parameter model can learn basic Chinese expression and instruction formats, but it is far smaller than modern general-purpose LLMs.
- It can produce factual errors, conceptual confusion, invalid code, arithmetic mistakes, repetition, and fabricated information.
- The 1,024-token context limits long-document understanding and multi-turn chat.
- The current application is single-turn and has no persistent memory.
- Generation has no KV cache and can be substantially optimized for long answers.
- Public synthetic SFT data may transfer teacher-model biases, phrasing habits, and errors.
- No systematic safety alignment, DPO/RLHF, factual benchmark suite, or production stress test has been completed.
- The model should not be used for medical, legal, financial, safety-critical, or other high-stakes decisions.

### Future Work

- Add KV caching and more precise streaming decode behavior.
- Build fixed Chinese evaluation suites for QA, knowledge, code, mathematics, writing, and repetition.
- Continue pretraining the Base model on the remaining tokenized corpus.
- Improve SFT quality scoring, short-answer balance, and factual filtering.
- Explore domain SFT for legal, educational, or game-community use while preserving general capability.
- Compare LoRA, DPO, and lightweight preference optimization against full-parameter SFT.
- Export inference-only model weights to reduce release size.
- Add multi-turn context, system prompts, and configurable stopping policies.

### AI Collaboration Disclosure

The project author remained responsible for the learning path, data scale, model configuration, training decisions, execution, verification, and interpretation of results. Codex / ChatGPT assisted with design discussions, parts of the engineering implementation and cleanup, download and checkpoint debugging, training-log analysis, and documentation. All reported training runs were executed and validated in the author's own local environment.
