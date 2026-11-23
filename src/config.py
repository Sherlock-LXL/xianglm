from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
TOKENIZER_PATH: Final = PROJECT_ROOT / "data" / "tokenizer" / "xianglm-32k" / "tokenizer.json"
TOKENIZED_DATA_ROOT: Final = PROJECT_ROOT / "data" / "tokenized" / "xianglm-32k"
DATA_MANIFEST_PATH: Final = TOKENIZED_DATA_ROOT / "manifest.json"
CHECKPOINT_ROOT: Final = PROJECT_ROOT / "data" / "checkpoints"
PRETRAIN_BEST_CHECKPOINT_PATH: Final = CHECKPOINT_ROOT / "best_pretrain.pt"
PRETRAIN_LATEST_CHECKPOINT_PATH: Final = CHECKPOINT_ROOT / "latest_pretrain.pt"
SFT_BEST_CHECKPOINT_PATH: Final = CHECKPOINT_ROOT / "best_sft.pt"
SFT_LATEST_CHECKPOINT_PATH: Final = CHECKPOINT_ROOT / "latest_sft.pt"
LOG_ROOT: Final = PROJECT_ROOT / "data" / "logs"

SOURCE_WEIGHTS: Final = (
    ("chinese_web", 0.60),
    ("chinese_education", 0.25),
    ("english_education", 0.10),
    ("code", 0.05),
)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True
    use_bias: bool = False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size 必须为正数")
        if self.vocab_size > 65_536:
            raise ValueError("当前 uint16 数据格式要求 vocab_size <= 65,536")
        if self.context_length <= 0:
            raise ValueError("context_length 必须为正数")
        if self.d_model <= 0:
            raise ValueError("d_model 必须为正数")
        if self.n_layers <= 0:
            raise ValueError("n_layers 必须为正数")
        if self.n_heads <= 0:
            raise ValueError("n_heads 必须为正数")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if self.head_dim % 8 != 0:
            raise ValueError("head_dim 建议为 8 的倍数，以便使用高效注意力内核")
        if self.d_ff <= 0:
            raise ValueError("d_ff 必须为正数")
        if self.d_ff % 32 != 0:
            raise ValueError("d_ff 必须为 32 的倍数")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps 必须为正数")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta 必须为正数")


@dataclass(frozen=True)
class TrainConfig:
    target_tokens: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_learning_rate: float
    min_learning_rate: float
    warmup_tokens: int
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    precision: str = "bfloat16"
    log_interval_steps: int = 10
    eval_interval_steps: int = 500
    eval_batches: int = 50
    checkpoint_interval_steps: int = 1_000
    gradient_checkpointing: bool = False
    seed: int = 42

    def global_batch_tokens(self, model: ModelConfig) -> int:
        return self.micro_batch_size * model.context_length * self.gradient_accumulation_steps

    def estimated_steps(self, model: ModelConfig) -> int:
        return ceil(self.target_tokens / self.global_batch_tokens(model))

    def validate(self, model: ModelConfig) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens 必须为正数")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size 必须为正数")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps 必须为正数")
        if self.max_learning_rate <= 0:
            raise ValueError("max_learning_rate 必须为正数")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate 不能为负数")
        if self.min_learning_rate > self.max_learning_rate:
            raise ValueError("min_learning_rate 不能高于 max_learning_rate")
        if not 0 <= self.warmup_tokens:
            raise ValueError("warmup_tokens 不能为负数")
        if self.warmup_tokens >= self.target_tokens:
            raise ValueError("warmup_tokens 必须小于 target_tokens")
        if self.weight_decay < 0:
            raise ValueError("weight_decay 不能为负数")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm 必须为正数")
        if self.precision not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"不支持的精度：{self.precision}")
        if self.log_interval_steps <= 0:
            raise ValueError("log_interval_steps 必须为正数")
        if self.eval_interval_steps <= 0:
            raise ValueError("eval_interval_steps 必须为正数")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches 必须为正数")
        if self.checkpoint_interval_steps <= 0:
            raise ValueError("checkpoint_interval_steps 必须为正数")
        if self.global_batch_tokens(model) <= 0:
            raise ValueError("global_batch_tokens 必须为正数")


@dataclass(frozen=True)
class DataConfig:
    manifest_path: Path
    source_weights: tuple[tuple[str, float], ...]
    train_split: str = "train"
    validation_split: str = "validation"
    pin_memory: bool = True
    seed: int = 42

    @property
    def source_weight_dict(self) -> dict[str, float]:
        return dict(self.source_weights)

    def validate(self, require_manifest: bool = True) -> None:
        if require_manifest and not self.manifest_path.exists():
            raise FileNotFoundError(f"没有找到 tokenized manifest：{self.manifest_path}")
        names = [name for name, _ in self.source_weights]
        if len(names) != len(set(names)):
            raise ValueError("source_weights 中存在重复来源")
        for name, weight in self.source_weights:
            if not name:
                raise ValueError("来源名称不能为空")
            if weight <= 0:
                raise ValueError(f"{name} 的权重必须为正数")
        total_weight = sum(weight for _, weight in self.source_weights)
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError(f"source_weights 之和必须为 1，当前为 {total_weight}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    model: ModelConfig
    train: TrainConfig
    data: DataConfig

    @property
    def checkpoint_directory(self) -> Path:
        return CHECKPOINT_ROOT

    def validate(self, require_data_manifest: bool = True) -> None:
        if not self.name:
            raise ValueError("实验名称不能为空")
        self.model.validate()
        self.train.validate(self.model)
        self.data.validate(require_manifest=require_data_manifest)


def estimate_parameter_count(config: ModelConfig) -> int:
    embedding_parameters = config.vocab_size * config.d_model
    attention_parameters_per_layer = 4 * config.d_model * config.d_model
    swiglu_parameters_per_layer = 3 * config.d_model * config.d_ff
    norm_parameters_per_layer = 2 * config.d_model
    block_parameters = config.n_layers * (
        attention_parameters_per_layer + swiglu_parameters_per_layer + norm_parameters_per_layer
    )
    final_norm_parameters = config.d_model
    lm_head_parameters = 0 if config.tie_embeddings else config.vocab_size * config.d_model
    return embedding_parameters + block_parameters + final_norm_parameters + lm_head_parameters


COMMON_DATA_CONFIG = DataConfig(
    manifest_path=DATA_MANIFEST_PATH,
    source_weights=SOURCE_WEIGHTS,
    train_split="train",
    validation_split="validation",
    pin_memory=True,
    seed=42,
)

LOCAL_200M = ExperimentConfig(
    name="local_200m",
    description="约 200.3M 参数的本地正式模型；计划训练约 4B tokens，需要梯度检查点和长时间训练。",
    model=ModelConfig(
        vocab_size=32_768,
        context_length=1_024,
        d_model=768,
        n_layers=25,
        n_heads=12,
        d_ff=2_016,
        dropout=0.0,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        tie_embeddings=True,
        use_bias=False,
    ),
    train=TrainConfig(
        target_tokens=4_000_000_000,
        micro_batch_size=2,
        gradient_accumulation_steps=16,
        max_learning_rate=2e-4,
        min_learning_rate=2e-5,
        warmup_tokens=50_000_000,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        grad_clip_norm=1.0,
        precision="bfloat16",
        log_interval_steps=10,
        eval_interval_steps=1_000,
        eval_batches=50,
        checkpoint_interval_steps=2_000,
        gradient_checkpointing=True,
        seed=42,
    ),
    data=COMMON_DATA_CONFIG,
)


def get_experiment_config(require_data_manifest: bool = True) -> ExperimentConfig:
    LOCAL_200M.validate(require_data_manifest=require_data_manifest)
    return LOCAL_200M


def print_config_summary(config: ExperimentConfig) -> None:
    parameter_count = estimate_parameter_count(config.model)
    global_batch_tokens = config.train.global_batch_tokens(config.model)
    estimated_steps = config.train.estimated_steps(config.model)
    print("=" * 72)
    print(f"配置：{config.name}")
    print(f"说明：{config.description}")
    print()
    print("模型")
    print(f"参数量：{parameter_count:,} ({parameter_count / 1e6:.3f}M)")
    print(f"词表：{config.model.vocab_size:,}")
    print(f"上下文长度：{config.model.context_length:,}")
    print(f"d_model：{config.model.d_model}")
    print(f"层数：{config.model.n_layers}")
    print(f"注意力头数：{config.model.n_heads}")
    print(f"head_dim：{config.model.head_dim}")
    print(f"d_ff：{config.model.d_ff}")
    print()
    print("训练")
    print(f"目标 tokens：{config.train.target_tokens:,}")
    print(f"Micro batch：{config.train.micro_batch_size}")
    print(f"梯度累积：{config.train.gradient_accumulation_steps}")
    print(f"全局 batch tokens：{global_batch_tokens:,}")
    print(f"预计 optimizer steps：{estimated_steps:,}")
    print(f"最高学习率：{config.train.max_learning_rate:g}")
    print(f"梯度检查点：{config.train.gradient_checkpointing}")
    print()
    print("数据比例")
    for source, weight in config.data.source_weights:
        print(f"- {source}: {weight:.0%}")
    print()


def main() -> None:
    config = get_experiment_config()
    print_config_summary(config)


if __name__ == "__main__":
    main()
