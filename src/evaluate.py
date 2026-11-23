from __future__ import annotations
import argparse
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager
import torch

from src.checkpoint import BEST_CHECKPOINT_NAME, LATEST_CHECKPOINT_NAME, load_model_checkpoint
from src.config import ExperimentConfig, get_experiment_config
from src.data_loader import MixedPretrainingLoader
from src.model import TransformerLM


@dataclass(frozen=True)
class EvaluationResult:
    overall_loss: float
    overall_perplexity: float
    source_losses: dict[str, float]
    source_perplexities: dict[str, float]
    batches_per_source: int


def autocast_context(device: torch.device, precision: str) -> ContextManager[object]:
    if precision == "float32":
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    try:
        dtype = dtype_map[precision]
    except KeyError as error:
        raise ValueError(f"未知精度：{precision}") from error
    return torch.autocast(device_type="cuda", dtype=dtype)


def safe_perplexity(loss: float) -> float:
    return math.exp(min(loss, 50.0))


@torch.no_grad()
def evaluate_model(
    *,
    model: TransformerLM,
    experiment: ExperimentConfig,
    device: torch.device,
    total_batches: int | None = None,
) -> EvaluationResult:
    if total_batches is None:
        total_batches = experiment.train.eval_batches
    source_names = [name for name, _ in experiment.data.source_weights]
    batches_per_source = max(1, total_batches // len(source_names))
    loader = MixedPretrainingLoader(
        data_config=experiment.data,
        context_length=experiment.model.context_length,
        split=experiment.data.validation_split,
        seed=experiment.data.seed + 100_000,
    )
    was_training = model.training
    model.eval()
    source_losses: dict[str, float] = {}
    try:
        for source_name in source_names:
            loss_sum = 0.0
            for _ in range(batches_per_source):
                batch = loader.next_batch(
                    experiment.train.micro_batch_size, device=device, source_name=source_name
                )
                with autocast_context(device, experiment.train.precision):
                    output = model(input_ids=batch.input_ids, labels=batch.labels)
                if output.loss is None:
                    raise RuntimeError("评估时模型没有返回loss")
                loss_sum += float(output.loss.item())
            source_losses[source_name] = loss_sum / batches_per_source
    finally:
        loader.close()
        if was_training:
            model.train()
    weight_dict = experiment.data.source_weight_dict
    overall_loss = sum(source_losses[name] * weight_dict[name] for name in source_names)
    return EvaluationResult(
        overall_loss=overall_loss,
        overall_perplexity=safe_perplexity(overall_loss),
        source_losses=source_losses,
        source_perplexities={name: safe_perplexity(loss) for name, loss in source_losses.items()},
        batches_per_source=batches_per_source,
    )


def print_evaluation_result(result: EvaluationResult) -> None:
    print("=" * 72)
    print("XiangLM Validation")
    print("=" * 72)
    print(f"每来源batch数：{result.batches_per_source}")
    print(f"加权总loss：{result.overall_loss:.6f}")
    print(f"加权perplexity：{result.overall_perplexity:.3f}")
    print()
    for source_name, loss in result.source_losses.items():
        perplexity = result.source_perplexities[source_name]
        print(f"{source_name}: loss={loss:.6f}, ppl={perplexity:.3f}")


def resolve_checkpoint_path(experiment: ExperimentConfig, requested: str | None) -> Path:
    if requested is not None:
        path = Path(requested).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint不存在：{path}")
        return path
    best = experiment.checkpoint_directory / BEST_CHECKPOINT_NAME
    if best.exists():
        return best
    latest = experiment.checkpoint_directory / LATEST_CHECKPOINT_NAME
    if latest.exists():
        return latest
    raise FileNotFoundError(
        f"没有找到{BEST_CHECKPOINT_NAME}或{LATEST_CHECKPOINT_NAME}：{experiment.checkpoint_directory}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立评估XiangLM checkpoint")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batches", type=int, default=None, help="全部来源合计的近似batch预算")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = get_experiment_config()
    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint_path(experiment, args.checkpoint)
    model = TransformerLM(experiment.model, gradient_checkpointing=False).to(device)
    checkpoint = load_model_checkpoint(
        path=checkpoint_path,
        model=model,
        expected_experiment_name=experiment.name,
        map_location=device,
    )
    print(f"Checkpoint：{checkpoint_path}")
    print(f"Step：{int(checkpoint['step']):,}")
    print(f"已训练tokens：{int(checkpoint['tokens_processed']):,}")
    result = evaluate_model(
        model=model, experiment=experiment, device=device, total_batches=args.batches
    )
    print_evaluation_result(result)


if __name__ == "__main__":
    main()
