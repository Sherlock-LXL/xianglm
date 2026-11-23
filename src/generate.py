from __future__ import annotations
import argparse
import random
import numpy as np
import torch
from tokenizers import Tokenizer

from src.checkpoint import load_model_checkpoint
from src.config import TOKENIZER_PATH, get_experiment_config
from src.evaluate import autocast_context, resolve_checkpoint_path
from src.model import TransformerLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_repetition_penalty(
    logits: torch.Tensor, token_ids: list[int], penalty: float
) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    adjusted = logits.clone()
    for token_id in set(token_ids):
        score = adjusted[token_id]
        if score < 0:
            adjusted[token_id] = score * penalty
        else:
            adjusted[token_id] = score / penalty
    return adjusted


def sample_next_token(logits: torch.Tensor, *, temperature: float, top_k: int, top_p: float) -> int:
    if temperature == 0.0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k > 0:
        effective_top_k = min(top_k, logits.numel())
        threshold = torch.topk(logits, effective_top_k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(0, sorted_indices, sorted_logits)
        logits = filtered
    probabilities = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


@torch.no_grad()
def generate_tokens(
    *,
    model: TransformerLM,
    prompt_ids: list[int],
    device: torch.device,
    precision: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stop_token_ids: set[int],
) -> list[int]:
    generated = list(prompt_ids)
    if not generated:
        raise ValueError("prompt编码后不能为空")
    model.eval()
    for _ in range(max_new_tokens):
        context_ids = generated[-model.config.context_length :]
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        with autocast_context(device, precision):
            output = model(input_ids=input_ids)
        next_logits = output.logits[0, -1].float()
        next_logits = apply_repetition_penalty(next_logits, generated, repetition_penalty)
        next_token = sample_next_token(
            next_logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        if next_token in stop_token_ids:
            break
        generated.append(next_token)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用XiangLM checkpoint生成文本")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens必须为正数")
    if args.temperature < 0:
        raise ValueError("--temperature不能为负数")
    if args.top_k < 0:
        raise ValueError("--top-k不能为负数")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p必须位于(0, 1]")
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty不能小于1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    experiment = get_experiment_config(require_data_manifest=False)
    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint_path(experiment, args.checkpoint)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not prompt_ids:
        bos_id = tokenizer.token_to_id("<|bos|>")
        if bos_id is None:
            raise RuntimeError("空prompt且tokenizer没有BOS")
        prompt_ids = [bos_id]
    model = TransformerLM(experiment.model, gradient_checkpointing=False).to(device)
    checkpoint = load_model_checkpoint(
        path=checkpoint_path,
        model=model,
        expected_experiment_name=experiment.name,
        map_location=device,
    )
    stop_ids = {
        token_id
        for token_id in (tokenizer.token_to_id("<|endoftext|>"), tokenizer.token_to_id("<|eos|>"))
        if token_id is not None
    }
    generated_ids = generate_tokens(
        model=model,
        prompt_ids=prompt_ids,
        device=device,
        precision=experiment.train.precision,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=stop_ids,
    )
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print("=" * 72)
    print(f"Checkpoint：{checkpoint_path}")
    print(f"Step：{int(checkpoint['step']):,}")
    print(f"已训练tokens：{int(checkpoint['tokens_processed']):,}")
    print("=" * 72)
    print(text)


if __name__ == "__main__":
    main()
