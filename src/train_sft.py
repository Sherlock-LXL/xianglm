from __future__ import annotations
import argparse
import bisect
import gc
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from src.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    copy_checkpoint_atomic,
    restore_rng_state,
)
from src.config import (
    CHECKPOINT_ROOT,
    LOG_ROOT,
    PRETRAIN_BEST_CHECKPOINT_PATH,
    PRETRAIN_LATEST_CHECKPOINT_PATH,
    PROJECT_ROOT,
    SFT_BEST_CHECKPOINT_PATH,
    SFT_LATEST_CHECKPOINT_PATH,
    get_experiment_config,
)
from src.evaluate import autocast_context
from src.model import TransformerLM

DATA_ROOT = PROJECT_ROOT / "data" / "post_training" / "tokenized" / "sft" / "xianglm-32k"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
OUTPUT_ROOT = CHECKPOINT_ROOT
LOG_PATH = LOG_ROOT / "train_local_200m_sft.jsonl"
LATEST_PATH = SFT_LATEST_CHECKPOINT_PATH
BEST_PATH = SFT_BEST_CHECKPOINT_PATH
STAGE_NAME = "sft"
SFT_TITLE = "XiangLM 200M SFT"
DEFAULT_BASE_CHECKPOINT_PATHS = (PRETRAIN_BEST_CHECKPOINT_PATH, PRETRAIN_LATEST_CHECKPOINT_PATH)
DEFAULT_EPOCHS = 1
DEFAULT_MAX_LEARNING_RATE = 2e-5
DEFAULT_MIN_LEARNING_RATE = 2e-6
DEFAULT_WARMUP_STEPS = 200
DEFAULT_EVAL_INTERVAL_STEPS = 500
DEFAULT_CHECKPOINT_INTERVAL_STEPS = 1_000
ARCHIVE_LATEST_ON_COMPLETE = False


@dataclass(frozen=True)
class SFTConfig:
    epochs: int = 1
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_learning_rate: float = 2e-5
    min_learning_rate: float = 2e-6
    warmup_steps: int = 200
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    precision: str = "bfloat16"
    log_interval_steps: int = 10
    eval_interval_steps: int = 500
    eval_examples: int = 256
    checkpoint_interval_steps: int = 1_000
    gradient_checkpointing: bool = True
    seed: int = 42

    @property
    def global_batch_examples(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def steps_per_epoch(self, records: int) -> int:
        return math.ceil(records / self.global_batch_examples)

    def total_steps(self, records: int) -> int:
        return self.epochs * self.steps_per_epoch(records)

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs必须为正数")
        if self.micro_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch参数必须为正数")
        if self.max_learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("学习率设置无效")
        if self.min_learning_rate > self.max_learning_rate:
            raise ValueError("min_learning_rate不能高于max_learning_rate")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps不能为负数")
        if self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("优化器参数无效")
        if self.precision not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"不支持的精度：{self.precision}")
        if (
            min(
                self.log_interval_steps,
                self.eval_interval_steps,
                self.eval_examples,
                self.checkpoint_interval_steps,
            )
            <= 0
        ):
            raise ValueError("日志、验证和checkpoint参数必须为正数")


@dataclass(frozen=True)
class ShardInfo:
    tokens_path: Path
    loss_mask_path: Path
    lengths_path: Path
    records: int


@dataclass
class SFTBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    examples: int
    effective_tokens: int
    supervised_tokens: int


class SFTDataset:
    def __init__(self, manifest_path: Path, split: str) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"没有找到SFT manifest：{manifest_path}")
        self.manifest_path = manifest_path
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.split = split
        self.record_tokens = int(self.manifest["storage"]["record_tokens"])
        self.context_length = int(self.manifest["storage"]["model_context_length"])
        self.tokens_dtype = np.dtype(self.manifest["storage"]["tokens_dtype"])
        self.loss_mask_dtype = np.dtype(self.manifest["storage"]["loss_mask_dtype"])
        self.lengths_dtype = np.dtype(self.manifest["storage"]["lengths_dtype"])
        if self.record_tokens != self.context_length + 1:
            raise RuntimeError("record_tokens必须等于模型上下文长度加一")
        if self.tokens_dtype != np.dtype("<u2") or self.loss_mask_dtype != np.dtype("|u1"):
            raise RuntimeError("SFT分片数据类型与当前加载器不兼容")
        split_entry = self.manifest["splits"][split]
        self.records = int(split_entry["stats"]["records"])
        self.shards: list[ShardInfo] = []
        self.cumulative_records: list[int] = []
        self.maps: dict[int, tuple[np.memmap, np.memmap, np.memmap]] = {}
        cumulative = 0
        for entry in split_entry["shards"]:
            records = int(entry["records"])
            shard = ShardInfo(
                tokens_path=self.root / entry["tokens_path"],
                loss_mask_path=self.root / entry["loss_mask_path"],
                lengths_path=self.root / entry["lengths_path"],
                records=records,
            )
            self._validate_shard(shard)
            self.shards.append(shard)
            cumulative += records
            self.cumulative_records.append(cumulative)
        if cumulative != self.records:
            raise RuntimeError(f"{split}分片记录数不一致：{cumulative} != {self.records}")

    def _validate_shard(self, shard: ShardInfo) -> None:
        for path in (shard.tokens_path, shard.loss_mask_path, shard.lengths_path):
            if not path.exists():
                raise FileNotFoundError(f"SFT分片不存在：{path}")
        expected_tokens_bytes = shard.records * self.record_tokens * self.tokens_dtype.itemsize
        expected_mask_bytes = shard.records * self.record_tokens * self.loss_mask_dtype.itemsize
        expected_lengths_bytes = shard.records * self.lengths_dtype.itemsize
        if shard.tokens_path.stat().st_size != expected_tokens_bytes:
            raise RuntimeError(f"token分片大小错误：{shard.tokens_path}")
        if shard.loss_mask_path.stat().st_size != expected_mask_bytes:
            raise RuntimeError(f"loss mask分片大小错误：{shard.loss_mask_path}")
        if shard.lengths_path.stat().st_size != expected_lengths_bytes:
            raise RuntimeError(f"lengths分片大小错误：{shard.lengths_path}")

    def _get_maps(self, shard_index: int) -> tuple[np.memmap, np.memmap, np.memmap]:
        cached = self.maps.get(shard_index)
        if cached is not None:
            return cached
        shard = self.shards[shard_index]
        tokens = np.memmap(
            shard.tokens_path,
            mode="r",
            dtype=self.tokens_dtype,
            shape=(shard.records, self.record_tokens),
        )
        loss_mask = np.memmap(
            shard.loss_mask_path,
            mode="r",
            dtype=self.loss_mask_dtype,
            shape=(shard.records, self.record_tokens),
        )
        lengths = np.memmap(
            shard.lengths_path, mode="r", dtype=self.lengths_dtype, shape=(shard.records,)
        )
        self.maps[shard_index] = (tokens, loss_mask, lengths)
        return self.maps[shard_index]

    def get_batch(
        self, indices: np.ndarray, device: torch.device, pin_memory: bool = True
    ) -> SFTBatch:
        batch_size = len(indices)
        if batch_size <= 0:
            raise ValueError("batch不能为空")
        tokens = np.empty((batch_size, self.record_tokens), dtype=np.int64)
        loss_mask = np.empty((batch_size, self.record_tokens), dtype=np.bool_)
        lengths = np.empty(batch_size, dtype=np.int64)
        for output_index, global_index_value in enumerate(indices):
            global_index = int(global_index_value)
            if not 0 <= global_index < self.records:
                raise IndexError(f"样本索引越界：{global_index}")
            shard_index = bisect.bisect_right(self.cumulative_records, global_index)
            previous = 0 if shard_index == 0 else self.cumulative_records[shard_index - 1]
            local_index = global_index - previous
            token_map, mask_map, length_map = self._get_maps(shard_index)
            tokens[output_index] = token_map[local_index]
            loss_mask[output_index] = mask_map[local_index]
            lengths[output_index] = length_map[local_index]
        token_tensor = torch.from_numpy(tokens)
        mask_tensor = torch.from_numpy(loss_mask)
        input_ids = token_tensor[:, :-1]
        labels = token_tensor[:, 1:].clone()
        target_mask = mask_tensor[:, 1:]
        labels.masked_fill_(~target_mask, -100)
        supervised_tokens = int(target_mask.sum().item())
        effective_tokens = int(np.maximum(lengths - 1, 0).sum())
        if supervised_tokens <= 0:
            raise RuntimeError("batch没有任何assistant监督token")
        if pin_memory and device.type == "cuda":
            input_ids = input_ids.pin_memory()
            labels = labels.pin_memory()
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        return SFTBatch(
            input_ids=input_ids,
            labels=labels,
            examples=batch_size,
            effective_tokens=effective_tokens,
            supervised_tokens=supervised_tokens,
        )

    def close(self) -> None:
        self.maps.clear()


class ShuffledSFTLoader:
    def __init__(self, dataset: SFTDataset, seed: int, examples_processed: int = 0) -> None:
        self.dataset = dataset
        self.seed = seed
        self.examples_processed = examples_processed
        self._permutation_epoch = -1
        self._permutation: np.ndarray | None = None
        self.load_state_dict({"examples_processed": examples_processed})

    @property
    def epoch(self) -> int:
        return self.examples_processed // self.dataset.records

    @property
    def position(self) -> int:
        return self.examples_processed % self.dataset.records

    def _epoch_permutation(self, epoch: int) -> np.ndarray:
        if self._permutation is None or self._permutation_epoch != epoch:
            generator = np.random.default_rng(self.seed + epoch)
            self._permutation = generator.permutation(self.dataset.records)
            self._permutation_epoch = epoch
        return self._permutation

    def next_indices(self, batch_size: int, total_examples: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size必须为正数")
        if self.examples_processed >= total_examples:
            raise StopIteration
        epoch = self.epoch
        position = self.position
        remaining_epoch = self.dataset.records - position
        remaining_total = total_examples - self.examples_processed
        count = min(batch_size, remaining_epoch, remaining_total)
        indices = self._epoch_permutation(epoch)[position : position + count]
        self.examples_processed += count
        return indices

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "examples_processed": self.examples_processed}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        restored_seed = int(state.get("seed", self.seed))
        restored_examples = int(state["examples_processed"])
        if restored_seed != self.seed:
            raise RuntimeError(f"SFT loader seed不一致：{restored_seed} != {self.seed}")
        if not 0 <= restored_examples:
            raise RuntimeError("examples_processed不能为负数")
        self.examples_processed = restored_examples
        self._permutation_epoch = -1
        self._permutation = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_json_log(entry: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_optimizer(
    model: nn.Module, config: SFTConfig, device: torch.device
) -> torch.optim.Optimizer:
    decay_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    ]
    no_decay_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim < 2
    ]
    parameter_groups = [
        {"params": decay_parameters, "weight_decay": config.weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = {
        "params": parameter_groups,
        "lr": config.max_learning_rate,
        "betas": (config.adam_beta1, config.adam_beta2),
        "eps": config.adam_eps,
    }
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(**kwargs, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(**kwargs)


def build_grad_scaler(config: SFTConfig, device: torch.device) -> Any:
    if device.type != "cuda" or config.precision != "float16":
        return None
    return torch.amp.GradScaler("cuda", enabled=True)


def learning_rate_for_step(step: int, total_steps: int, config: SFTConfig) -> float:
    warmup_steps = min(config.warmup_steps, max(0, total_steps - 1))
    if warmup_steps > 0 and step <= warmup_steps:
        return config.max_learning_rate * step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (config.max_learning_rate - config.min_learning_rate)


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def resolve_base_checkpoint(requested: str | None) -> Path:
    if requested is not None:
        path = Path(requested).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Base checkpoint不存在：{path}")
        return path
    for path in DEFAULT_BASE_CHECKPOINT_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(f"没有找到200M Base checkpoint：{CHECKPOINT_ROOT}")


def load_base_model(model: TransformerLM, path: Path) -> dict[str, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_name") != "local_200m":
        raise RuntimeError(f"Base checkpoint配置错误：{checkpoint.get('experiment_name')}")
    model.load_state_dict(checkpoint["model"], strict=True)
    metadata = {
        "step": int(checkpoint.get("step", 0)),
        "tokens_processed": int(checkpoint.get("tokens_processed", 0)),
    }
    del checkpoint
    gc.collect()
    return metadata


def save_sft_checkpoint(
    *,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    loader: ShuffledSFTLoader,
    config: SFTConfig,
    step: int,
    effective_tokens: int,
    supervised_tokens: int,
    best_validation_loss: float,
    base_checkpoint: Path,
    base_metadata: dict[str, int],
    is_best: bool,
) -> Path:
    payload = {
        "format_version": 1,
        "experiment_name": "local_200m",
        "training_stage": STAGE_NAME,
        "data_manifest_sha256": sha256_file(MANIFEST_PATH),
        "sft_config": asdict(config),
        "step": step,
        "tokens_processed": supervised_tokens,
        "sft_step": step,
        "sft_examples_processed": loader.examples_processed,
        "sft_effective_tokens_processed": effective_tokens,
        "sft_supervised_tokens_processed": supervised_tokens,
        "best_validation_loss": best_validation_loss,
        "base_checkpoint": str(base_checkpoint),
        "base_step": base_metadata["step"],
        "base_tokens_processed": base_metadata["tokens_processed"],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "train_loader": loader.state_dict(),
        "rng_state": capture_rng_state(),
    }
    atomic_torch_save(payload, LATEST_PATH)
    if is_best:
        copy_checkpoint_atomic(LATEST_PATH, BEST_PATH)
    return LATEST_PATH


def load_sft_checkpoint(
    *,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    loader: ShuffledSFTLoader,
    config: SFTConfig,
) -> tuple[int, int, int, float, Path, dict[str, int]]:
    checkpoint = torch.load(LATEST_PATH, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("experiment_name") != "local_200m"
        or checkpoint.get("training_stage") != STAGE_NAME
    ):
        raise RuntimeError(f"{LATEST_PATH.name}不是当前SFT任务的checkpoint")
    if checkpoint.get("data_manifest_sha256") != sha256_file(MANIFEST_PATH):
        raise RuntimeError("SFT数据manifest与checkpoint不一致")
    if checkpoint.get("sft_config") != asdict(config):
        raise RuntimeError("SFT配置与checkpoint不一致")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler_state = checkpoint.get("scaler")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    loader.load_state_dict(checkpoint["train_loader"])
    restore_rng_state(checkpoint["rng_state"])
    step = int(checkpoint["sft_step"])
    effective_tokens = int(checkpoint["sft_effective_tokens_processed"])
    supervised_tokens = int(checkpoint["sft_supervised_tokens_processed"])
    best_validation_loss = float(checkpoint["best_validation_loss"])
    base_checkpoint = Path(checkpoint["base_checkpoint"])
    base_metadata = {
        "step": int(checkpoint["base_step"]),
        "tokens_processed": int(checkpoint["base_tokens_processed"]),
    }
    del checkpoint
    gc.collect()
    return (
        step,
        effective_tokens,
        supervised_tokens,
        best_validation_loss,
        base_checkpoint,
        base_metadata,
    )


@torch.no_grad()
def evaluate(
    model: TransformerLM, dataset: SFTDataset, config: SFTConfig, device: torch.device
) -> tuple[float, float, int, int]:
    model.eval()
    count = min(config.eval_examples, dataset.records)
    permutation = np.random.default_rng(config.seed + 10_000).permutation(dataset.records)[:count]
    loss_sum = 0.0
    supervised_tokens = 0
    effective_tokens = 0
    for start in range(0, count, config.micro_batch_size):
        batch = dataset.get_batch(permutation[start : start + config.micro_batch_size], device)
        with autocast_context(device, config.precision):
            output = model(input_ids=batch.input_ids, labels=batch.labels)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("validation出现无效loss")
        loss_sum += float(output.loss.item()) * batch.supervised_tokens
        supervised_tokens += batch.supervised_tokens
        effective_tokens += batch.effective_tokens
    loss = loss_sum / max(1, supervised_tokens)
    perplexity = math.exp(min(20.0, loss))
    model.train()
    return loss, perplexity, effective_tokens, supervised_tokens


def print_summary(
    config: SFTConfig,
    train_dataset: SFTDataset,
    validation_dataset: SFTDataset,
    base_checkpoint: Path,
    total_steps: int,
) -> None:
    model_config = get_experiment_config(require_data_manifest=False).model
    print("=" * 72)
    print(SFT_TITLE)
    print("=" * 72)
    print(f"Base checkpoint：{base_checkpoint}")
    print(f"训练数据：{MANIFEST_PATH}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"训练样本：{train_dataset.records:,}")
    print(
        f"验证样本：{validation_dataset.records:,}，每次固定评估{min(config.eval_examples, validation_dataset.records):,}条"
    )
    print(f"Epochs：{config.epochs}")
    print(f"Micro batch：{config.micro_batch_size}")
    print(f"梯度累积：{config.gradient_accumulation_steps}")
    print(f"全局batch样本：{config.global_batch_examples}")
    print(f"每epoch steps：{config.steps_per_epoch(train_dataset.records):,}")
    print(f"总steps：{total_steps:,}")
    print(f"学习率：{config.max_learning_rate:g} -> {config.min_learning_rate:g}")
    print(f"Warmup steps：{min(config.warmup_steps, max(0, total_steps - 1)):,}")
    print(f"上下文长度：{model_config.context_length:,}")
    print(f"Loss：仅assistant回答token")
    print()


def train(
    *,
    config: SFTConfig,
    device: torch.device,
    base_checkpoint_argument: str | None,
    resume: bool,
    max_steps: int | None,
    inspect_only: bool,
) -> None:
    config.validate()
    experiment = get_experiment_config(require_data_manifest=False)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch无法使用CUDA")
        if config.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前GPU或PyTorch不支持BF16")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("high")
    set_global_seed(config.seed)
    train_dataset = SFTDataset(MANIFEST_PATH, "train")
    validation_dataset = SFTDataset(MANIFEST_PATH, "validation")
    if train_dataset.context_length != experiment.model.context_length:
        raise RuntimeError(
            f"数据与模型上下文不一致：{train_dataset.context_length} != {experiment.model.context_length}"
        )
    if int(train_dataset.manifest["tokenizer"]["vocab_size"]) != experiment.model.vocab_size:
        raise RuntimeError("SFT tokenizer词表与模型不一致")
    loader = ShuffledSFTLoader(train_dataset, config.seed)
    total_examples = train_dataset.records * config.epochs
    total_steps = config.total_steps(train_dataset.records)
    target_steps = total_steps if max_steps is None else min(total_steps, max_steps)
    base_checkpoint = resolve_base_checkpoint(base_checkpoint_argument)
    print_summary(config, train_dataset, validation_dataset, base_checkpoint, total_steps)
    print(f"设备：{device}")
    if inspect_only:
        train_dataset.close()
        validation_dataset.close()
        return
    if not LATEST_PATH.exists() and BEST_PATH.exists():
        raise RuntimeError(f"检测到已归档SFT模型：{BEST_PATH}。如需重新SFT，请先移动该文件。")
    model = TransformerLM(experiment.model, gradient_checkpointing=config.gradient_checkpointing)
    base_metadata = {"step": 0, "tokens_processed": 0}
    if not LATEST_PATH.exists():
        base_metadata = load_base_model(model, base_checkpoint)
    model = model.to(device)
    optimizer = build_optimizer(model, config, device)
    scaler = build_grad_scaler(config, device)
    step = 0
    effective_tokens = 0
    supervised_tokens = 0
    best_validation_loss = math.inf
    if LATEST_PATH.exists():
        if not resume:
            raise RuntimeError(f"检测到已有SFT checkpoint：{LATEST_PATH}")
        (
            step,
            effective_tokens,
            supervised_tokens,
            best_validation_loss,
            base_checkpoint,
            base_metadata,
        ) = load_sft_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loader=loader,
            config=config,
        )
        print(f"已恢复：{LATEST_PATH}")
        print(f"恢复step：{step:,}")
        print(f"恢复样本：{loader.examples_processed:,}/{total_examples:,}")
        print(f"恢复监督tokens：{supervised_tokens:,}")
    elif not resume:
        print("未检测到SFT checkpoint，从Base开始新训练")
    print(f"Base step：{base_metadata['step']:,}")
    print(f"Base tokens：{base_metadata['tokens_processed']:,}")
    if step >= target_steps or loader.examples_processed >= total_examples:
        print("当前checkpoint已经达到本次目标，无需继续训练")
        train_dataset.close()
        validation_dataset.close()
        return
    model.train()
    optimizer.zero_grad(set_to_none=True)
    interval_loss_sum = 0.0
    interval_supervised_tokens = 0
    interval_effective_tokens = 0
    interval_examples = 0
    interval_start = time.perf_counter()
    in_step = False
    step_start_loader_state: dict[str, int] | None = None
    step_start_rng_state: dict[str, Any] | None = None
    try:
        while step < target_steps and loader.examples_processed < total_examples:
            in_step = True
            step_start_loader_state = loader.state_dict()
            step_start_rng_state = capture_rng_state()
            current_epoch = loader.epoch
            batches: list[SFTBatch] = []
            for _ in range(config.gradient_accumulation_steps):
                if loader.examples_processed >= total_examples or loader.epoch != current_epoch:
                    break
                indices = loader.next_indices(config.micro_batch_size, total_examples)
                batches.append(train_dataset.get_batch(indices, device))
            if not batches:
                continue
            step_supervised_tokens = sum(batch.supervised_tokens for batch in batches)
            step_effective_tokens = sum(batch.effective_tokens for batch in batches)
            step_examples = sum(batch.examples for batch in batches)
            next_step = step + 1
            learning_rate = learning_rate_for_step(next_step, total_steps, config)
            set_learning_rate(optimizer, learning_rate)
            optimizer.zero_grad(set_to_none=True)
            step_loss_sum = 0.0
            for batch in batches:
                with autocast_context(device, config.precision):
                    output = model(input_ids=batch.input_ids, labels=batch.labels)
                    if output.loss is None:
                        raise RuntimeError("训练时模型没有返回loss")
                    micro_loss = output.loss
                    scaled_loss = micro_loss * (batch.supervised_tokens / step_supervised_tokens)
                if not torch.isfinite(micro_loss):
                    raise RuntimeError("训练出现NaN或Inf loss")
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                step_loss_sum += float(micro_loss.item()) * batch.supervised_tokens
            if scaler is not None:
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_norm
            )
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            step = next_step
            effective_tokens += step_effective_tokens
            supervised_tokens += step_supervised_tokens
            in_step = False
            interval_loss_sum += step_loss_sum
            interval_supervised_tokens += step_supervised_tokens
            interval_effective_tokens += step_effective_tokens
            interval_examples += step_examples
            is_final = step >= target_steps or loader.examples_processed >= total_examples
            should_log = step % config.log_interval_steps == 0 or is_final
            if should_log:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - interval_start
                train_loss = interval_loss_sum / max(1, interval_supervised_tokens)
                compute_tokens = interval_examples * experiment.model.context_length
                tokens_per_second = compute_tokens / max(elapsed, 1e-9)
                completion = loader.examples_processed / total_examples
                remaining_compute_tokens = (
                    total_examples - loader.examples_processed
                ) * experiment.model.context_length
                eta_hours = remaining_compute_tokens / max(tokens_per_second, 1e-9) / 3600
                memory_gib = (
                    torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
                )
                epoch_value = loader.examples_processed / train_dataset.records
                print(
                    f"step={step:,}/{total_steps:,} epoch={epoch_value:.3f}/{config.epochs} 样本={loader.examples_processed:,}/{total_examples:,} 完成={completion:.2%} "
                    f"loss={train_loss:.4f} lr={learning_rate:.3e} grad={float(gradient_norm):.3f} tok/s={tokens_per_second:,.0f} ETA={eta_hours:.2f}h 显存={memory_gib:.2f}GiB"
                )
                append_json_log(
                    {
                        "event": "train",
                        "step": step,
                        "epoch": epoch_value,
                        "examples_processed": loader.examples_processed,
                        "effective_tokens_processed": effective_tokens,
                        "supervised_tokens_processed": supervised_tokens,
                        "loss": train_loss,
                        "learning_rate": learning_rate,
                        "gradient_norm": float(gradient_norm),
                        "tokens_per_second": tokens_per_second,
                        "eta_hours": eta_hours,
                        "peak_memory_gib": memory_gib,
                        "interval_examples": interval_examples,
                        "interval_effective_tokens": interval_effective_tokens,
                        "interval_supervised_tokens": interval_supervised_tokens,
                    }
                )
                interval_loss_sum = 0.0
                interval_supervised_tokens = 0
                interval_effective_tokens = 0
                interval_examples = 0
                interval_start = time.perf_counter()
            should_evaluate = step % config.eval_interval_steps == 0 or is_final
            is_best = False
            if should_evaluate:
                validation_loss, perplexity, validation_effective, validation_supervised = evaluate(
                    model, validation_dataset, config, device
                )
                print(
                    f"validation：loss={validation_loss:.6f}, ppl={perplexity:.3f}, 样本={min(config.eval_examples, validation_dataset.records):,}, "
                    f"有效tokens={validation_effective:,}, 监督tokens={validation_supervised:,}"
                )
                append_json_log(
                    {
                        "event": "validation",
                        "step": step,
                        "epoch": loader.examples_processed / train_dataset.records,
                        "examples_processed": loader.examples_processed,
                        "loss": validation_loss,
                        "perplexity": perplexity,
                        "effective_tokens": validation_effective,
                        "supervised_tokens": validation_supervised,
                    }
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    is_best = True
            should_checkpoint = step % config.checkpoint_interval_steps == 0 or is_best or is_final
            if should_checkpoint:
                path = save_sft_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    loader=loader,
                    config=config,
                    step=step,
                    effective_tokens=effective_tokens,
                    supervised_tokens=supervised_tokens,
                    best_validation_loss=best_validation_loss,
                    base_checkpoint=base_checkpoint,
                    base_metadata=base_metadata,
                    is_best=is_best,
                )
                print(f"Checkpoint：{path}")
            if should_evaluate or should_checkpoint:
                interval_start = time.perf_counter()
    except KeyboardInterrupt:
        print()
        print("收到中断，正在保存可续训状态……")
        if in_step and step_start_loader_state is not None and step_start_rng_state is not None:
            loader.load_state_dict(step_start_loader_state)
            restore_rng_state(step_start_rng_state)
            optimizer.zero_grad(set_to_none=True)
        path = save_sft_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loader=loader,
            config=config,
            step=step,
            effective_tokens=effective_tokens,
            supervised_tokens=supervised_tokens,
            best_validation_loss=best_validation_loss,
            base_checkpoint=base_checkpoint,
            base_metadata=base_metadata,
            is_best=False,
        )
        print(f"中断checkpoint：{path}")
        return
    finally:
        train_dataset.close()
        validation_dataset.close()
    print("=" * 72)
    print("SFT训练阶段完成")
    print(f"Step：{step:,}")
    print(f"样本：{loader.examples_processed:,}")
    print(f"有效tokens：{effective_tokens:,}")
    print(f"监督tokens：{supervised_tokens:,}")
    print(f"最佳validation loss：{best_validation_loss:.6f}")
    training_complete = step >= total_steps or loader.examples_processed >= total_examples
    if ARCHIVE_LATEST_ON_COMPLETE and training_complete and LATEST_PATH.exists():
        LATEST_PATH.unlink()
        print(f"已清理续训checkpoint：{LATEST_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"训练{SFT_TITLE}")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_MAX_LEARNING_RATE)
    parser.add_argument("--min-learning-rate", type=float, default=DEFAULT_MIN_LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--eval-interval-steps", type=int, default=DEFAULT_EVAL_INTERVAL_STEPS)
    parser.add_argument(
        "--checkpoint-interval-steps", type=int, default=DEFAULT_CHECKPOINT_INTERVAL_STEPS
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps必须为正数")
    config = SFTConfig(
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        eval_interval_steps=args.eval_interval_steps,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
    )
    train(
        config=config,
        device=torch.device(args.device),
        base_checkpoint_argument=args.base_checkpoint,
        resume=not args.no_resume,
        max_steps=args.max_steps,
        inspect_only=args.inspect_only,
    )


if __name__ == "__main__":
    main()
