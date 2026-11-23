from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
import argparse
import json
import os
import shutil
import numpy as np
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "data" / "post_training" / "processed" / "sft"
INPUT_MANIFEST_PATH = INPUT_ROOT / "manifest.json"
TOKENIZER_PATH = PROJECT_ROOT / "data" / "tokenizer" / "xianglm-32k" / "tokenizer.json"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "post_training" / "tokenized" / "sft" / "xianglm-32k"
CHECKPOINT_PATH = OUTPUT_ROOT / "checkpoint.json"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"

FORMAT_VERSION = 1
MODEL_CONTEXT_LENGTH = 1_024
RECORD_TOKENS = MODEL_CONTEXT_LENGTH + 1
RECORDS_PER_SHARD = 5_000
MAX_LAST_USER_TOKENS = 384
MIN_LAST_ASSISTANT_TOKENS = 128
MIN_FREE_DISK_BYTES = 10 * 1024**3
TOKEN_DTYPE = np.dtype("<u2")
MASK_DTYPE = np.dtype("u1")
LENGTH_DTYPE = np.dtype("<u2")
SPECIAL_TOKEN_STRINGS = (
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
)
SPLITS = ("train", "validation", "test")


@dataclass
class EncodedRecord:
    tokens: np.ndarray
    loss_mask: np.ndarray
    length: int
    source: str
    original_tokens: int
    supervised_tokens: int
    truncated: bool
    dropped_turns: int


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def sha256_file(path: Path, chunk_size: int = 16 * 1024**2) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, path)


def relative_to_output(path: Path) -> str:
    return path.resolve().relative_to(OUTPUT_ROOT.resolve()).as_posix()


def ensure_free_disk_space() -> None:
    free_bytes = shutil.disk_usage(OUTPUT_ROOT).free
    if free_bytes < MIN_FREE_DISK_BYTES:
        raise RuntimeError(f"磁盘剩余空间低于安全线：{format_bytes(free_bytes)}")


def load_tokenizers() -> tuple[Tokenizer, Tokenizer, dict[str, int]]:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"缺少 tokenizer：{TOKENIZER_PATH}")
    control_tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    special_token_ids = {}
    for token in SPECIAL_TOKEN_STRINGS:
        token_id = control_tokenizer.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f"Tokenizer 缺少特殊 token：{token}")
        special_token_ids[token] = token_id
    if control_tokenizer.get_vocab_size() > 65_536:
        raise RuntimeError("词表超过 uint16 范围")
    payload = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    payload["added_tokens"] = []
    raw_tokenizer = Tokenizer.from_buffer(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    raw_tokenizer.no_padding()
    raw_tokenizer.no_truncation()
    probe = "正文<|im_start|>不会成为控制标记<|im_end|>"
    probe_encoding = raw_tokenizer.encode(probe, add_special_tokens=False)
    if set(probe_encoding.ids).intersection(special_token_ids.values()):
        raise RuntimeError("正文 tokenizer 产生了保留 token ID")
    if raw_tokenizer.decode(probe_encoding.ids, skip_special_tokens=False) != probe:
        raise RuntimeError("正文 tokenizer 回译失败")
    return control_tokenizer, raw_tokenizer, special_token_ids


def encode_text(raw_tokenizer: Tokenizer, text: str, reserved_ids: set[int]) -> list[int]:
    token_ids = raw_tokenizer.encode(text, add_special_tokens=False).ids
    accidental = reserved_ids.intersection(token_ids)
    if accidental:
        raise RuntimeError(f"正文中产生保留 token ID：{sorted(accidental)}")
    return token_ids


def truncate_middle(token_ids: list[int], budget: int, separator_ids: list[int]) -> list[int]:
    if len(token_ids) <= budget:
        return token_ids
    if budget <= len(separator_ids) + 2:
        return token_ids[:budget]
    remaining = budget - len(separator_ids)
    head = remaining * 2 // 3
    tail = remaining - head
    return token_ids[:head] + separator_ids + token_ids[-tail:]


def encode_messages(
    messages: list[dict[str, str]],
    raw_tokenizer: Tokenizer,
    special_ids: dict[str, int],
    reserved_ids: set[int],
) -> tuple[list[int], list[int]]:
    tokens = [special_ids["<|bos|>"]]
    loss_mask = [0]
    newline_ids = encode_text(raw_tokenizer, "\n", reserved_ids)
    for message in messages:
        role = message["role"]
        prefix = [special_ids["<|im_start|>"]] + encode_text(
            raw_tokenizer, f"{role}\n", reserved_ids
        )
        content = encode_text(raw_tokenizer, message["content"], reserved_ids)
        suffix = [special_ids["<|im_end|>"]] + newline_ids
        assistant = role == "assistant"
        tokens.extend(prefix + content + suffix)
        loss_mask.extend(
            [0] * len(prefix)
            + [1 if assistant else 0] * len(content)
            + [1 if assistant else 0]
            + [0] * len(newline_ids)
        )
    tokens.append(special_ids["<|eos|>"])
    loss_mask.append(1)
    return tokens, loss_mask


def last_pair_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if messages[-1]["role"] != "assistant":
        raise RuntimeError("对话没有以 assistant 结束")
    for index in range(len(messages) - 2, -1, -1):
        if messages[index]["role"] == "user":
            return [messages[index], messages[-1]]
    raise RuntimeError("对话缺少 user")


def encode_last_pair(
    messages: list[dict[str, str]],
    raw_tokenizer: Tokenizer,
    special_ids: dict[str, int],
    reserved_ids: set[int],
) -> tuple[list[int], list[int]]:
    pair = last_pair_messages(messages)
    user_ids = encode_text(raw_tokenizer, pair[0]["content"], reserved_ids)
    assistant_ids = encode_text(raw_tokenizer, pair[1]["content"], reserved_ids)
    separator_ids = encode_text(raw_tokenizer, "\n[…]\n", reserved_ids)
    empty_pair = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
    fixed_tokens, _ = encode_messages(empty_pair, raw_tokenizer, special_ids, reserved_ids)
    available = RECORD_TOKENS - len(fixed_tokens)
    if available < 2:
        raise RuntimeError("聊天模板固定开销超过记录长度")
    user_budget = min(len(user_ids), MAX_LAST_USER_TOKENS)
    assistant_budget = min(len(assistant_ids), available - user_budget)
    required_assistant = min(len(assistant_ids), MIN_LAST_ASSISTANT_TOKENS)
    if assistant_budget < required_assistant:
        user_budget = max(1, available - required_assistant)
        assistant_budget = available - user_budget
    remaining = available - user_budget - assistant_budget
    if remaining > 0:
        extra_assistant = min(remaining, len(assistant_ids) - assistant_budget)
        assistant_budget += extra_assistant
        remaining -= extra_assistant
        user_budget += min(remaining, len(user_ids) - user_budget)
    user_content_ids = truncate_middle(user_ids, user_budget, separator_ids)
    assistant_content_ids = assistant_ids[:assistant_budget]
    tokens = [special_ids["<|bos|>"]]
    loss_mask = [0]
    newline_ids = encode_text(raw_tokenizer, "\n", reserved_ids)
    for role, content_ids in (("user", user_content_ids), ("assistant", assistant_content_ids)):
        prefix = [special_ids["<|im_start|>"]] + encode_text(
            raw_tokenizer, f"{role}\n", reserved_ids
        )
        suffix = [special_ids["<|im_end|>"]] + newline_ids
        assistant = role == "assistant"
        tokens.extend(prefix + content_ids + suffix)
        loss_mask.extend(
            [0] * len(prefix)
            + [1 if assistant else 0] * len(content_ids)
            + [1 if assistant else 0]
            + [0] * len(newline_ids)
        )
    tokens.append(special_ids["<|eos|>"])
    loss_mask.append(1)
    if len(tokens) > RECORD_TOKENS:
        raise RuntimeError(f"截断后仍超过记录长度：{len(tokens)}")
    return tokens, loss_mask


def drop_oldest_turn(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    start = 1 if messages and messages[0]["role"] == "system" else 0
    if len(messages) - start <= 2:
        return messages
    return messages[:start] + messages[start + 2 :]


def encode_record(
    record: dict[str, Any], raw_tokenizer: Tokenizer, special_ids: dict[str, int]
) -> EncodedRecord:
    messages = record["messages"]
    reserved_ids = set(special_ids.values())
    original_tokens, original_loss_mask = encode_messages(
        messages, raw_tokenizer, special_ids, reserved_ids
    )
    fitted_messages = messages
    dropped_turns = 0
    tokens = original_tokens
    loss_mask = original_loss_mask
    while len(tokens) > RECORD_TOKENS:
        reduced = drop_oldest_turn(fitted_messages)
        if reduced == fitted_messages:
            break
        fitted_messages = reduced
        dropped_turns += 1
        tokens, loss_mask = encode_messages(
            fitted_messages, raw_tokenizer, special_ids, reserved_ids
        )
    if len(tokens) > RECORD_TOKENS:
        tokens, loss_mask = encode_last_pair(
            fitted_messages, raw_tokenizer, special_ids, reserved_ids
        )
    length = len(tokens)
    if length < 3 or length > RECORD_TOKENS or len(tokens) != len(loss_mask):
        raise RuntimeError(f"编码长度异常：{record.get('id')}")
    supervised_tokens = sum(loss_mask)
    if supervised_tokens < 2:
        raise RuntimeError(f"监督 token 过少：{record.get('id')}")
    padded_tokens = np.full(RECORD_TOKENS, special_ids["<|pad|>"], dtype=TOKEN_DTYPE)
    padded_mask = np.zeros(RECORD_TOKENS, dtype=MASK_DTYPE)
    padded_tokens[:length] = tokens
    padded_mask[:length] = loss_mask
    return EncodedRecord(
        padded_tokens,
        padded_mask,
        length,
        record["source"],
        len(original_tokens),
        supervised_tokens,
        len(original_tokens) > RECORD_TOKENS,
        dropped_turns,
    )


def initial_stats() -> dict[str, Any]:
    return {
        "records": 0,
        "stored_tokens": 0,
        "original_tokens": 0,
        "supervised_tokens": 0,
        "truncated_records": 0,
        "dropped_turns": 0,
        "sources": {},
    }


def update_stats(stats: dict[str, Any], records: list[EncodedRecord]) -> None:
    source_stats = {source: dict(values) for source, values in stats.get("sources", {}).items()}
    for record in records:
        stats["records"] += 1
        stats["stored_tokens"] += record.length
        stats["original_tokens"] += record.original_tokens
        stats["supervised_tokens"] += record.supervised_tokens
        stats["truncated_records"] += int(record.truncated)
        stats["dropped_turns"] += record.dropped_turns
        entry = source_stats.setdefault(
            record.source, {"records": 0, "effective_tokens": 0, "supervised_tokens": 0}
        )
        entry["records"] += 1
        entry["effective_tokens"] += record.length
        entry["supervised_tokens"] += record.supervised_tokens
    stats["sources"] = source_stats


def write_binary(path: Path, array: np.ndarray) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        array.tofile(file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def write_shard(split: str, shard_index: int, records: list[EncodedRecord]) -> dict[str, Any]:
    prefix = f"{split}-{shard_index:05d}"
    token_path = OUTPUT_ROOT / f"{prefix}.tokens.bin"
    mask_path = OUTPUT_ROOT / f"{prefix}.loss_mask.bin"
    length_path = OUTPUT_ROOT / f"{prefix}.lengths.bin"
    tokens = np.stack([record.tokens for record in records])
    masks = np.stack([record.loss_mask for record in records])
    lengths = np.asarray([record.length for record in records], dtype=LENGTH_DTYPE)
    write_binary(token_path, tokens)
    write_binary(mask_path, masks)
    write_binary(length_path, lengths)
    return {
        "index": shard_index,
        "records": len(records),
        "tokens_path": relative_to_output(token_path),
        "loss_mask_path": relative_to_output(mask_path),
        "lengths_path": relative_to_output(length_path),
        "tokens_sha256": sha256_file(token_path),
        "loss_mask_sha256": sha256_file(mask_path),
        "lengths_sha256": sha256_file(length_path),
    }


def build_runtime_config(input_manifest_sha256: str, tokenizer_sha256: str) -> dict[str, Any]:
    inputs = {}
    for split in SPLITS:
        path = INPUT_ROOT / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"缺少 SFT 输入：{path}")
        stat = path.stat()
        inputs[split] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {
        "format_version": FORMAT_VERSION,
        "input_manifest_sha256": input_manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "model_context_length": MODEL_CONTEXT_LENGTH,
        "record_tokens": RECORD_TOKENS,
        "records_per_shard": RECORDS_PER_SHARD,
        "inputs": inputs,
    }


def load_or_create_checkpoint(runtime_config: dict[str, Any]) -> dict[str, Any]:
    fingerprint = sha256_json(runtime_config)
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if checkpoint.get("config_fingerprint") != fingerprint:
            raise RuntimeError("SFT tokenize 配置或输入已变化，请保留旧输出并使用新的输出版本目录")
        return checkpoint
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "config_fingerprint": fingerprint,
        "runtime_config": runtime_config,
        "splits": {
            split: {
                "next_line": 0,
                "next_shard": 0,
                "completed": False,
                "shards": [],
                "stats": initial_stats(),
            }
            for split in SPLITS
        },
    }
    atomic_write_json(CHECKPOINT_PATH, checkpoint)
    return checkpoint


def process_split(
    split: str, checkpoint: dict[str, Any], raw_tokenizer: Tokenizer, special_ids: dict[str, int]
) -> None:
    state = checkpoint["splits"][split]
    if state["completed"]:
        print(f"{split} 已完成，跳过。")
        return
    path = INPUT_ROOT / f"{split}.jsonl"
    buffer = []
    processed_line = int(state["next_line"])
    print()
    print("=" * 72)
    print(f"开始 tokenize：{split}")
    print(f"恢复行：{processed_line:,}")
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line_number <= processed_line:
                continue
            record = json.loads(line)
            buffer.append(encode_record(record, raw_tokenizer, special_ids))
            if len(buffer) < RECORDS_PER_SHARD:
                continue
            shard = write_shard(split, int(state["next_shard"]), buffer)
            update_stats(state["stats"], buffer)
            state["shards"].append(shard)
            state["next_shard"] += 1
            state["next_line"] = line_number
            checkpoint["updated_at"] = now_iso()
            atomic_write_json(CHECKPOINT_PATH, checkpoint)
            ensure_free_disk_space()
            print(
                f"{split}：{state['stats']['records']:,} 条，{state['stats']['stored_tokens']:,} 有效 tokens，分片 {state['next_shard']:,}"
            )
            buffer = []
        if buffer:
            shard = write_shard(split, int(state["next_shard"]), buffer)
            update_stats(state["stats"], buffer)
            state["shards"].append(shard)
            state["next_shard"] += 1
            state["next_line"] = line_number
    state["completed"] = True
    checkpoint["updated_at"] = now_iso()
    atomic_write_json(CHECKPOINT_PATH, checkpoint)
    print(
        f"{split} 完成：{state['stats']['records']:,} 条，{state['stats']['stored_tokens']:,} 有效 tokens"
    )


def build_manifest(
    checkpoint: dict[str, Any], tokenizer: Tokenizer, special_ids: dict[str, int]
) -> dict[str, Any]:
    splits = {}
    total_records = 0
    total_tokens = 0
    total_supervised_tokens = 0
    for split in SPLITS:
        state = checkpoint["splits"][split]
        splits[split] = {
            "completed": state["completed"],
            "stats": state["stats"],
            "shards": state["shards"],
        }
        total_records += state["stats"]["records"]
        total_tokens += state["stats"]["stored_tokens"]
        total_supervised_tokens += state["stats"]["supervised_tokens"]
    return {
        "format_version": FORMAT_VERSION,
        "created_at": now_iso(),
        "input_manifest": str(INPUT_MANIFEST_PATH),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST_PATH),
        "tokenizer": {
            "path": str(TOKENIZER_PATH),
            "sha256": sha256_file(TOKENIZER_PATH),
            "vocab_size": tokenizer.get_vocab_size(),
            "special_token_ids": special_ids,
        },
        "storage": {
            "record_tokens": RECORD_TOKENS,
            "model_context_length": MODEL_CONTEXT_LENGTH,
            "tokens_dtype": TOKEN_DTYPE.str,
            "loss_mask_dtype": MASK_DTYPE.str,
            "lengths_dtype": LENGTH_DTYPE.str,
            "records_per_shard": RECORDS_PER_SHARD,
        },
        "chat_template": "<|bos|><|im_start|>{role}\\n{content}<|im_end|>\\n...<|eos|>",
        "loss_policy": "assistant content, assistant <|im_end|>, and final <|eos|> only",
        "truncation_policy": {
            "drop_oldest_complete_turns_first": True,
            "last_user_max_tokens": MAX_LAST_USER_TOKENS,
            "last_assistant_min_tokens": MIN_LAST_ASSISTANT_TOKENS,
            "long_user_middle_truncation": True,
            "long_assistant_keep_prefix": True,
        },
        "totals": {
            "records": total_records,
            "effective_tokens": total_tokens,
            "supervised_tokens": total_supervised_tokens,
            "fixed_token_slots": total_records * RECORD_TOKENS,
        },
        "splits": splits,
        "config_fingerprint": checkpoint["config_fingerprint"],
    }


def verify_shard(shard: dict[str, Any], vocab_size: int, pad_id: int) -> None:
    records = int(shard["records"])
    token_path = OUTPUT_ROOT / shard["tokens_path"]
    mask_path = OUTPUT_ROOT / shard["loss_mask_path"]
    length_path = OUTPUT_ROOT / shard["lengths_path"]
    expected_sizes = {
        token_path: records * RECORD_TOKENS * TOKEN_DTYPE.itemsize,
        mask_path: records * RECORD_TOKENS * MASK_DTYPE.itemsize,
        length_path: records * LENGTH_DTYPE.itemsize,
    }
    for path, expected_size in expected_sizes.items():
        if not path.exists() or path.stat().st_size != expected_size:
            raise RuntimeError(f"分片大小错误：{path}")
    if (
        sha256_file(token_path) != shard["tokens_sha256"]
        or sha256_file(mask_path) != shard["loss_mask_sha256"]
        or sha256_file(length_path) != shard["lengths_sha256"]
    ):
        raise RuntimeError(f"分片 SHA256 错误：{token_path.name}")
    tokens = np.memmap(token_path, dtype=TOKEN_DTYPE, mode="r", shape=(records, RECORD_TOKENS))
    masks = np.memmap(mask_path, dtype=MASK_DTYPE, mode="r", shape=(records, RECORD_TOKENS))
    lengths = np.memmap(length_path, dtype=LENGTH_DTYPE, mode="r", shape=(records,))
    if tokens.size and int(tokens.max()) >= vocab_size:
        raise RuntimeError(f"Token ID 越界：{token_path}")
    if np.any((masks != 0) & (masks != 1)):
        raise RuntimeError(f"Loss mask 不是二值：{mask_path}")
    if np.any(lengths < 3) or np.any(lengths > RECORD_TOKENS):
        raise RuntimeError(f"Length 越界：{length_path}")
    if np.any(masks.sum(axis=1) < 2):
        raise RuntimeError(f"存在无监督目标记录：{mask_path}")
    for index in range(records):
        length = int(lengths[index])
        if np.any(masks[index, length:] != 0) or np.any(tokens[index, length:] != pad_id):
            raise RuntimeError(f"Padding 验收失败：{token_path}:{index}")


def verify_dataset() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"缺少 manifest：{MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if sha256_file(INPUT_MANIFEST_PATH) != manifest["input_manifest_sha256"]:
        raise RuntimeError("输入 manifest SHA256 已变化")
    if sha256_file(TOKENIZER_PATH) != manifest["tokenizer"]["sha256"]:
        raise RuntimeError("Tokenizer SHA256 已变化")
    records = 0
    for split in SPLITS:
        entry = manifest["splits"][split]
        if not entry["completed"]:
            raise RuntimeError(f"{split} 尚未完成")
        split_records = 0
        for shard in entry["shards"]:
            print(f"验证：{split}/{Path(shard['tokens_path']).name}")
            verify_shard(
                shard,
                int(manifest["tokenizer"]["vocab_size"]),
                int(manifest["tokenizer"]["special_token_ids"]["<|pad|>"]),
            )
            split_records += int(shard["records"])
        if split_records != int(entry["stats"]["records"]):
            raise RuntimeError(f"{split} 记录汇总错误")
        records += split_records
    if records != int(manifest["totals"]["records"]):
        raise RuntimeError("全局记录汇总错误")
    return manifest["totals"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 XiangLM SFT 转换为定长 token、loss mask 和 length 分片"
    )
    parser.add_argument("--verify-only", action="store_true", help="只验证已生成的 SFT token 分片")
    parser.add_argument("--no-verify", action="store_true", help="完成 tokenize 后暂时跳过全量验证")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only and args.no_verify:
        raise ValueError("--verify-only 和 --no-verify 不能同时使用")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("XiangLM SFT Tokenize")
    print("=" * 72)
    print(f"输入目录：{INPUT_ROOT}")
    print(f"Tokenizer：{TOKENIZER_PATH}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"模型上下文：{MODEL_CONTEXT_LENGTH:,}；存储记录：{RECORD_TOKENS:,} tokens")
    if args.verify_only:
        totals = verify_dataset()
        print(f"验收通过：{totals}")
        return
    control_tokenizer, raw_tokenizer, special_ids = load_tokenizers()
    runtime_config = build_runtime_config(
        sha256_file(INPUT_MANIFEST_PATH), sha256_file(TOKENIZER_PATH)
    )
    checkpoint = load_or_create_checkpoint(runtime_config)
    if checkpoint["status"] == "complete":
        print("Checkpoint 显示 tokenize 已完成。")
        if not args.no_verify:
            totals = verify_dataset()
            print(f"验收通过：{totals}")
        return
    try:
        for split in SPLITS:
            process_split(split, checkpoint, raw_tokenizer, special_ids)
    except KeyboardInterrupt:
        checkpoint["updated_at"] = now_iso()
        atomic_write_json(CHECKPOINT_PATH, checkpoint)
        print("检测到用户中断；已完成分片均已保存，重新运行即可续传。")
        raise SystemExit(130)
    manifest = build_manifest(checkpoint, control_tokenizer, special_ids)
    atomic_write_json(MANIFEST_PATH, manifest)
    checkpoint["status"] = "complete"
    checkpoint["completed_at"] = now_iso()
    checkpoint["updated_at"] = now_iso()
    atomic_write_json(CHECKPOINT_PATH, checkpoint)
    print()
    print("=" * 72)
    print("SFT tokenize 完成")
    print(f"记录数：{manifest['totals']['records']:,}")
    print(f"有效 tokens：{manifest['totals']['effective_tokens']:,}")
    print(f"监督 tokens：{manifest['totals']['supervised_tokens']:,}")
    print(f"Manifest：{MANIFEST_PATH}")
    if not args.no_verify:
        totals = verify_dataset()
        print(f"完整验收通过：{totals}")


if __name__ == "__main__":
    main()
