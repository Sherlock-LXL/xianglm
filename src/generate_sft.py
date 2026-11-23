from __future__ import annotations
import argparse
import gc
from pathlib import Path
import torch
from tokenizers import Tokenizer

from src.config import SFT_BEST_CHECKPOINT_PATH, SFT_LATEST_CHECKPOINT_PATH, get_experiment_config
from src.evaluate import autocast_context
from src.generate import apply_repetition_penalty, sample_next_token, set_seed
from src.model import TransformerLM
from src.tokenize_sft_data import encode_text, load_tokenizers, truncate_middle

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", "退出"}
DEFAULT_CHECKPOINT_PATHS = (SFT_BEST_CHECKPOINT_PATH, SFT_LATEST_CHECKPOINT_PATH)
EXPECTED_STAGE = "sft"
SFT_LABEL = "SFT"


def resolve_checkpoint(requested: str | None) -> Path:
    if requested is not None:
        path = Path(requested).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"SFT checkpoint不存在：{path}")
        return path
    for path in DEFAULT_CHECKPOINT_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(f"没有找到{SFT_LABEL} checkpoint")


def load_sft_model(path: Path, device: torch.device) -> tuple[TransformerLM, dict[str, int]]:
    experiment = get_experiment_config(require_data_manifest=False)
    model = TransformerLM(experiment.model, gradient_checkpointing=False)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_name") != "local_200m":
        raise RuntimeError(f"Checkpoint模型配置错误：{checkpoint.get('experiment_name')}")
    if checkpoint.get("training_stage") != EXPECTED_STAGE:
        raise RuntimeError(f"该checkpoint不是XiangLM {SFT_LABEL}模型")
    model.load_state_dict(checkpoint["model"], strict=True)
    metadata = {
        "sft_step": int(checkpoint.get("sft_step", checkpoint.get("step", 0))),
        "sft_examples": int(checkpoint.get("sft_examples_processed", 0)),
        "sft_supervised_tokens": int(
            checkpoint.get("sft_supervised_tokens_processed", checkpoint.get("tokens_processed", 0))
        ),
        "base_step": int(checkpoint.get("base_step", 0)),
        "base_tokens": int(checkpoint.get("base_tokens_processed", 0)),
    }
    del checkpoint
    gc.collect()
    model = model.to(device)
    model.eval()
    return model, metadata


def build_prompt_ids(
    question: str, raw_tokenizer: Tokenizer, special_ids: dict[str, int], context_length: int
) -> list[int]:
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")
    reserved_ids = set(special_ids.values())
    user_prefix = [special_ids["<|bos|>"], special_ids["<|im_start|>"]] + encode_text(
        raw_tokenizer, "user\n", reserved_ids
    )
    user_suffix = [special_ids["<|im_end|>"]] + encode_text(raw_tokenizer, "\n", reserved_ids)
    assistant_prefix = [special_ids["<|im_start|>"]] + encode_text(
        raw_tokenizer, "assistant\n", reserved_ids
    )
    fixed_ids = user_prefix + user_suffix + assistant_prefix
    question_budget = context_length - len(fixed_ids)
    if question_budget <= 0:
        raise RuntimeError("聊天模板固定开销超过模型上下文")
    question_ids = encode_text(raw_tokenizer, question, reserved_ids)
    separator_ids = encode_text(raw_tokenizer, "\n[…]\n", reserved_ids)
    question_ids = truncate_middle(question_ids, question_budget, separator_ids)
    return user_prefix + question_ids + user_suffix + assistant_prefix


@torch.no_grad()
def generate_answer(
    *,
    model: TransformerLM,
    raw_tokenizer: Tokenizer,
    control_tokenizer: Tokenizer,
    special_ids: dict[str, int],
    question: str,
    device: torch.device,
    precision: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int,
) -> str:
    prompt_ids = build_prompt_ids(question, raw_tokenizer, special_ids, model.config.context_length)
    all_ids = list(prompt_ids)
    answer_ids: list[int] = []
    stop_ids = {
        special_ids["<|im_end|>"],
        special_ids["<|im_start|>"],
        special_ids["<|eos|>"],
        special_ids["<|endoftext|>"],
    }
    for _ in range(max_new_tokens):
        context_ids = all_ids[-model.config.context_length :]
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        with autocast_context(device, precision):
            output = model(input_ids=input_ids)
        logits = output.logits[0, -1].float()
        recent_answer_ids = answer_ids[-repetition_window:] if repetition_window > 0 else answer_ids
        logits = apply_repetition_penalty(logits, recent_answer_ids, repetition_penalty)
        next_token = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        if next_token in stop_ids:
            break
        answer_ids.append(next_token)
        all_ids.append(next_token)
    answer = control_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return answer if answer else "[模型立即结束了回答]"


def ask_once(
    *,
    question: str,
    model: TransformerLM,
    raw_tokenizer: Tokenizer,
    control_tokenizer: Tokenizer,
    special_ids: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    precision: str,
) -> str:
    return generate_answer(
        model=model,
        raw_tokenizer=raw_tokenizer,
        control_tokenizer=control_tokenizer,
        special_ids=special_ids,
        question=question,
        device=device,
        precision=precision,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
    )


def interactive_loop(
    *,
    model: TransformerLM,
    raw_tokenizer: Tokenizer,
    control_tokenizer: Tokenizer,
    special_ids: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    precision: str,
) -> None:
    print("交互测试已启动。每个问题都是独立单轮对话；输入 /exit 退出。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in EXIT_COMMANDS:
            return
        if not question:
            continue
        answer = ask_once(
            question=question,
            model=model,
            raw_tokenizer=raw_tokenizer,
            control_tokenizer=control_tokenizer,
            special_ids=special_ids,
            args=args,
            device=device,
            precision=precision,
        )
        print(f"XiangLM：{answer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"直接测试XiangLM {SFT_LABEL}模型")
    parser.add_argument("--prompt", default=None, help="单轮问题；省略后进入交互模式")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.10)
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens必须为正数")
    if args.temperature < 0 or args.top_k < 0 or args.repetition_window < 0:
        raise ValueError("采样参数不能为负数")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p必须位于(0, 1]")
    if args.repetition_penalty < 1:
        raise ValueError("--repetition-penalty不能小于1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    experiment = get_experiment_config(require_data_manifest=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch无法使用CUDA")
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    control_tokenizer, raw_tokenizer, special_ids = load_tokenizers()
    model, metadata = load_sft_model(checkpoint_path, device)
    print("=" * 72)
    print(f"Checkpoint：{checkpoint_path}")
    print(f"SFT step：{metadata['sft_step']:,}")
    print(f"SFT样本：{metadata['sft_examples']:,}")
    print(f"SFT监督tokens：{metadata['sft_supervised_tokens']:,}")
    print(f"Base step：{metadata['base_step']:,}")
    print(f"Base tokens：{metadata['base_tokens']:,}")
    print("=" * 72)
    if args.prompt is None:
        interactive_loop(
            model=model,
            raw_tokenizer=raw_tokenizer,
            control_tokenizer=control_tokenizer,
            special_ids=special_ids,
            args=args,
            device=device,
            precision=experiment.train.precision,
        )
        return
    answer = ask_once(
        question=args.prompt,
        model=model,
        raw_tokenizer=raw_tokenizer,
        control_tokenizer=control_tokenizer,
        special_ids=special_ids,
        args=args,
        device=device,
        precision=experiment.train.precision,
    )
    print(answer)


if __name__ == "__main__":
    main()
