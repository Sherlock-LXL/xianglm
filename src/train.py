from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from src.checkpoint import BEST_CHECKPOINT_NAME, LATEST_CHECKPOINT_NAME, capture_rng_state, latest_checkpoint_path, load_training_checkpoint, restore_rng_state, save_training_checkpoint
from src.config import LOG_ROOT, ExperimentConfig, get_experiment_config, print_config_summary
from src.data_loader import MixedPretrainingLoader
from src.evaluate import EvaluationResult, autocast_context, evaluate_model
from src.model import TransformerLM

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def learning_rate_for_tokens(*, tokens_processed: int, experiment: ExperimentConfig) -> float:
    train = experiment.train
    if tokens_processed < train.warmup_tokens:
        return train.max_learning_rate * tokens_processed / max(1, train.warmup_tokens)
    decay_tokens = train.target_tokens - train.warmup_tokens
    progress = (tokens_processed - train.warmup_tokens) / max(1, decay_tokens)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train.min_learning_rate + cosine * (train.max_learning_rate - train.min_learning_rate)

def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

def build_optimizer(model: nn.Module, experiment: ExperimentConfig, device: torch.device) -> torch.optim.Optimizer:
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)
    parameter_groups = [{"params": decay_parameters, "weight_decay": experiment.train.weight_decay}, {"params": no_decay_parameters, "weight_decay": 0.0}]
    optimizer_kwargs: dict[str, Any] = {
        "params": parameter_groups,
        "lr": experiment.train.max_learning_rate,
        "betas": (experiment.train.adam_beta1, experiment.train.adam_beta2),
        "eps": experiment.train.adam_eps,
    }
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(**optimizer_kwargs, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(**optimizer_kwargs)

def build_grad_scaler(experiment: ExperimentConfig, device: torch.device) -> Any:
    enabled = device.type == "cuda" and experiment.train.precision == "float16"
    if not enabled:
        return None
    return torch.amp.GradScaler("cuda", enabled=True)

def append_json_log(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False))
        file.write("\n")

def format_source_counts(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    if total == 0:
        return ""
    return ", ".join(f"{name}={count / total:.1%}" for name, count in counts.items())

def validation_log_entry(*, step: int, tokens_processed: int, result: EvaluationResult) -> dict[str, Any]:
    return {
        "event": "validation",
        "step": step,
        "tokens_processed": tokens_processed,
        "overall_loss": result.overall_loss,
        "overall_perplexity": result.overall_perplexity,
        "source_losses": result.source_losses,
        "source_perplexities": result.source_perplexities,
    }

def train(*, experiment: ExperimentConfig, device: torch.device, resume: bool, max_steps: int | None) -> None:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch无法使用CUDA")
        if experiment.train.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前GPU/PyTorch不支持BF16")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("high")
    set_global_seed(experiment.train.seed)
    print_config_summary(experiment)
    print(f"设备：{device}")
    train_loader = MixedPretrainingLoader(data_config=experiment.data, context_length=experiment.model.context_length, split=experiment.data.train_split, seed=experiment.train.seed)
    model = TransformerLM(experiment.model, gradient_checkpointing=experiment.train.gradient_checkpointing).to(device)
    optimizer = build_optimizer(model, experiment, device)
    scaler = build_grad_scaler(experiment, device)
    step = 0
    tokens_processed = 0
    best_validation_loss = math.inf
    latest = latest_checkpoint_path(experiment.checkpoint_directory)
    best = experiment.checkpoint_directory / BEST_CHECKPOINT_NAME
    if latest is None and best.exists():
        raise RuntimeError(f"检测到已归档预训练模型：{best}。如需重新预训练，请先移动该文件。")
    if latest is not None:
        if not resume:
            raise RuntimeError(f"检测到已有{LATEST_CHECKPOINT_NAME}。请启用续训，或手动移动旧checkpoint。")
        restored = load_training_checkpoint(path=latest, experiment=experiment, model=model, optimizer=optimizer, scaler=scaler, train_loader=train_loader, map_location=device)
        step = restored.step
        tokens_processed = restored.tokens_processed
        best_validation_loss = restored.best_validation_loss
        print(f"已恢复：{latest}")
        print(f"恢复step：{step:,}")
        print(f"恢复tokens：{tokens_processed:,}")
    log_path = LOG_ROOT / f"train_{experiment.name}.jsonl"
    global_batch_tokens = experiment.train.global_batch_tokens(experiment.model)
    target_steps = experiment.train.estimated_steps(experiment.model)
    if max_steps is not None:
        target_steps = min(target_steps, max_steps)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    interval_loss_sum = 0.0
    interval_step_count = 0
    interval_source_counts = {name: 0 for name, _ in experiment.data.source_weights}
    interval_start = time.perf_counter()
    interval_start_tokens = tokens_processed
    in_step = False
    step_start_loader_state = None
    step_start_rng_state = None
    try:
        while tokens_processed < experiment.train.target_tokens and step < target_steps:
            in_step = True
            step_start_loader_state = train_loader.state_dict()
            step_start_rng_state = capture_rng_state()
            next_token_count = min(experiment.train.target_tokens, tokens_processed + global_batch_tokens)
            learning_rate = learning_rate_for_tokens(tokens_processed=next_token_count, experiment=experiment)
            set_optimizer_learning_rate(optimizer, learning_rate)
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(experiment.train.gradient_accumulation_steps):
                batch = train_loader.next_batch(experiment.train.micro_batch_size, device=device)
                for name, count in batch.source_counts.items():
                    interval_source_counts[name] += count
                with autocast_context(device, experiment.train.precision):
                    output = model(input_ids=batch.input_ids, labels=batch.labels)
                    if output.loss is None:
                        raise RuntimeError("训练时模型没有返回loss")
                    micro_loss = output.loss
                    scaled_loss = micro_loss / experiment.train.gradient_accumulation_steps
                if not torch.isfinite(micro_loss):
                    raise RuntimeError("发现NaN或Inf loss")
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                step_loss += float(micro_loss.item())
            if scaler is not None:
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), experiment.train.grad_clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            step += 1
            tokens_processed += global_batch_tokens
            in_step = False
            average_step_loss = step_loss / experiment.train.gradient_accumulation_steps
            interval_loss_sum += average_step_loss
            interval_step_count += 1
            should_log = step % experiment.train.log_interval_steps == 0 or step == target_steps
            if should_log:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - interval_start
                interval_tokens = tokens_processed - interval_start_tokens
                tokens_per_second = interval_tokens / max(elapsed, 1e-9)
                average_loss = interval_loss_sum / max(interval_step_count, 1)
                completion = min(1.0, tokens_processed / experiment.train.target_tokens)
                remaining_tokens = max(0, experiment.train.target_tokens - tokens_processed)
                eta_hours = remaining_tokens / max(tokens_per_second, 1e-9) / 3600
                memory_gib = 0.0
                if device.type == "cuda":
                    memory_gib = torch.cuda.max_memory_allocated() / 1024**3
                source_summary = format_source_counts(interval_source_counts)
                print(
                    f"step={step:,}/{target_steps:,} tokens={tokens_processed:,} 完成={completion:.2%} loss={average_loss:.4f} lr={learning_rate:.3e} "
                    f"grad={float(gradient_norm):.3f} tok/s={tokens_per_second:,.0f} ETA={eta_hours:.2f}h 显存={memory_gib:.2f}GiB 来源[{source_summary}]"
                )
                append_json_log(
                    log_path,
                    {
                        "event": "train",
                        "step": step,
                        "tokens_processed": tokens_processed,
                        "completion": completion,
                        "loss": average_loss,
                        "learning_rate": learning_rate,
                        "gradient_norm": float(gradient_norm),
                        "tokens_per_second": tokens_per_second,
                        "eta_hours": eta_hours,
                        "peak_memory_gib": memory_gib,
                        "source_counts": dict(interval_source_counts),
                    },
                )
                interval_loss_sum = 0.0
                interval_step_count = 0
                interval_source_counts = {name: 0 for name, _ in experiment.data.source_weights}
                interval_start = time.perf_counter()
                interval_start_tokens = tokens_processed
            is_final_step = tokens_processed >= experiment.train.target_tokens or step >= target_steps
            should_evaluate = step % experiment.train.eval_interval_steps == 0 or is_final_step
            is_best = False
            if should_evaluate:
                result = evaluate_model(model=model, experiment=experiment, device=device)
                print(f"validation：loss={result.overall_loss:.6f}, ppl={result.overall_perplexity:.3f}, sources={result.source_losses}")
                append_json_log(log_path, validation_log_entry(step=step, tokens_processed=tokens_processed, result=result))
                if result.overall_loss < best_validation_loss:
                    best_validation_loss = result.overall_loss
                    is_best = True
            should_checkpoint = step % experiment.train.checkpoint_interval_steps == 0 or is_best or is_final_step
            if should_checkpoint:
                path = save_training_checkpoint(
                    directory=experiment.checkpoint_directory,
                    experiment=experiment,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    train_loader=train_loader,
                    step=step,
                    tokens_processed=tokens_processed,
                    best_validation_loss=best_validation_loss,
                    is_best=is_best,
                )
                print(f"Checkpoint：{path}")
    except KeyboardInterrupt:
        print()
        print("收到中断，正在保存可续训状态……")
        if in_step and step_start_loader_state is not None and step_start_rng_state is not None:
            train_loader.load_state_dict(step_start_loader_state)
            restore_rng_state(step_start_rng_state)
            optimizer.zero_grad(set_to_none=True)
        path = save_training_checkpoint(
            directory=experiment.checkpoint_directory,
            experiment=experiment,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_loader=train_loader,
            step=step,
            tokens_processed=tokens_processed,
            best_validation_loss=best_validation_loss,
            is_best=False,
        )
        print(f"中断checkpoint：{path}")
        return
    finally:
        train_loader.close()
    print("=" * 72)
    print("训练阶段完成")
    print(f"Step：{step:,}")
    print(f"已训练tokens：{tokens_processed:,}")
    print(f"最佳validation loss：{best_validation_loss:.6f}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练XiangLM")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-resume", action="store_true", help=f"禁止自动读取已有{LATEST_CHECKPOINT_NAME}")
    parser.add_argument("--max-steps", type=int, default=None, help="仅用于短程测试的step上限")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps必须为正数")
    experiment = get_experiment_config()
    train(experiment=experiment, device=torch.device(args.device), resume=not args.no_resume, max_steps=args.max_steps)

if __name__ == "__main__":
    main()