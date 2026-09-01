from __future__ import annotations
from dataclasses import dataclass
import torch

from src.config import get_experiment_config
from src.evaluate import autocast_context
from src.generate import apply_repetition_penalty, sample_next_token, set_seed
from src.generate_sft import build_prompt_ids, load_sft_model, resolve_checkpoint
from src.tokenize_sft_data import load_tokenizers

LINE = "=" * 72
EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", "退出"}

@dataclass(frozen=True)
class ChatMode:
    name: str
    description: str
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    repetition_window: int
    max_new_tokens: int

MODES = {
    "1": ChatMode("稳定", "回答更稳定，随机性较低", 0.30, 20, 0.85, 1.12, 128, 400),
    "2": ChatMode("平衡", "兼顾稳定性与创造力", 0.65, 40, 0.90, 1.10, 128, 600),
    "3": ChatMode("激进", "随机性更高，更容易出现意外表达", 0.90, 60, 0.95, 1.10, 128, 700)
}

def choose_mode() -> ChatMode:
    print()
    for key, mode in MODES.items():
        print(f"{key}. {mode.name}：{mode.description}")
    while True:
        try:
            choice = input("\n请选择模式 [默认 2]：").strip() or "2"
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)
        if choice in MODES:
            return MODES[choice]
        print("请输入 1、2 或 3。")

@torch.no_grad()
def stream_answer(*, model, raw_tokenizer, control_tokenizer, special_ids: dict[str, int], question: str, mode: ChatMode, device: torch.device, precision: str) -> None:
    prompt_ids = build_prompt_ids(question, raw_tokenizer, special_ids, model.config.context_length)
    all_ids = list(prompt_ids)
    answer_ids: list[int] = []
    printed_text = ""
    stop_ids = {special_ids["<|im_end|>"], special_ids["<|im_start|>"], special_ids["<|eos|>"], special_ids["<|endoftext|>"]}
    for _ in range(mode.max_new_tokens):
        context_ids = all_ids[-model.config.context_length:]
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        with autocast_context(device, precision):
            output = model(input_ids=input_ids)
        logits = output.logits[0, -1].float()
        recent_ids = answer_ids[-mode.repetition_window:] if mode.repetition_window > 0 else answer_ids
        logits = apply_repetition_penalty(logits, recent_ids, mode.repetition_penalty)
        next_token = sample_next_token(logits, temperature=mode.temperature, top_k=mode.top_k, top_p=mode.top_p)
        if next_token in stop_ids:
            break
        answer_ids.append(next_token)
        all_ids.append(next_token)
        decoded = control_tokenizer.decode(answer_ids, skip_special_tokens=True)
        if decoded.startswith(printed_text):
            addition = decoded[len(printed_text):]
            if addition and "�" not in addition:
                print(addition, end="", flush=True)
                printed_text = decoded
    final_text = control_tokenizer.decode(answer_ids, skip_special_tokens=True)
    if final_text.startswith(printed_text):
        print(final_text[len(printed_text):], end="", flush=True)
    if not final_text.strip():
        print("[模型立即结束了回答]", end="", flush=True)
    print()

def chat_loop(*, model, raw_tokenizer, control_tokenizer, special_ids: dict[str, int], mode: ChatMode, device: torch.device, precision: str) -> None:
    print(f"\n已进入{mode.name}模式。输入 /mode 切换模式，输入 /exit 退出。")
    print("每个问题都是独立单轮对话。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in EXIT_COMMANDS:
            return
        if question.lower() == "/mode":
            mode = choose_mode()
            print(f"\n已切换到{mode.name}模式。")
            continue
        if not question:
            continue
        print("XiangLM：", end="", flush=True)
        try:
            stream_answer(model=model, raw_tokenizer=raw_tokenizer, control_tokenizer=control_tokenizer, special_ids=special_ids, question=question, mode=mode, device=device, precision=precision)
        except KeyboardInterrupt:
            print("\n[已停止生成]")

def main() -> None:
    print(LINE)
    print("XiangLM 200M Chat")
    print(LINE)
    mode = choose_mode()
    print("\n正在加载模型……")
    set_seed(42)
    experiment = get_experiment_config(require_data_manifest=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint(None)
    control_tokenizer, raw_tokenizer, special_ids = load_tokenizers()
    model, _ = load_sft_model(checkpoint_path, device)
    print("模型已就绪。")
    chat_loop(model=model, raw_tokenizer=raw_tokenizer, control_tokenizer=control_tokenizer, special_ids=special_ids, mode=mode, device=device, precision=experiment.train.precision)

if __name__ == "__main__":
    main()