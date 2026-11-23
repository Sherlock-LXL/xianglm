from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from hashlib import blake2b, sha256
from pathlib import Path
from typing import Any, Iterator
import argparse
import ast
import heapq
import json
import os
import re
import unicodedata
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "post_training" / "raw" / "sft"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "post_training" / "processed" / "sft"
STAGING_ROOT = OUTPUT_ROOT / "staging"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
REPORT_PATH = OUTPUT_ROOT / "report.json"

RANDOM_SEED = 20_260_831
PARQUET_BATCH_ROWS = 2048
PROGRESS_EVERY_ROWS = 100_000
SPLIT_BUCKETS = 10_000
VALIDATION_BUCKETS = 100
TEST_BUCKETS = 100
MIN_USER_CHARS = 2
MIN_ASSISTANT_CHARS = 2
MAX_USER_CHARS = 3_000
MAX_ASSISTANT_CHARS = 6_000
MAX_TOTAL_CHARS = 10_000
MAX_MESSAGES = 12
SOURCE_TARGETS = {
    "magpie": 80_000,
    "fineweb_edu_qa": 60_000,
    "qwen3_distill": 35_000,
    "deepseek_r1_distill": 25_000,
}
SMOLTALK_TARGETS = {
    "advice-seeking": 2_500,
    "brainstorming": 2_500,
    "coding": 3_500,
    "creative-writing": 3_000,
    "data-analysis": 2_500,
    "document-qa": 2_500,
    "editing": 2_500,
    "everyday": 2_000,
    "format-constrain": 2_000,
    "information-seeking": 2_500,
    "math": 3_000,
    "math23k_zh_fixed": 1_500,
    "planning": 2_000,
    "reasoning": 3_000,
    "rewrite": 1_500,
    "role-playing": 2_500,
    "safe": 1_500,
    "summary": 1_500,
    "translate": 1_000,
}
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
BLANK_LINES_PATTERN = re.compile(r"\n{3,}")
WHITESPACE_PATTERN = re.compile(r"\s+")
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
TEACHER_IDENTITY_PATTERNS = (
    "我是chatgpt",
    "我是 qwen",
    "我是qwen",
    "作为deepseek",
    "我是deepseek",
    "由openai开发",
)


@dataclass
class SamplingResult:
    source: str
    detail: str
    target: int
    scanned: int
    eligible: int
    selected: int
    rejected: dict[str, int]
    recovered_json_lines: int
    staging_path: str


class PrioritySampler:
    def __init__(self, target: int) -> None:
        self.target = target
        self.heap: list[tuple[int, str, dict[str, Any]]] = []
        self.seen: set[str] = set()

    def offer(self, record: dict[str, Any]) -> bool:
        record_id = record["id"]
        if record_id in self.seen:
            return False
        self.seen.add(record_id)
        priority = stable_int(f"{RANDOM_SEED}:{record_id}")
        item = (-priority, record_id, record)
        if len(self.heap) < self.target:
            heapq.heappush(self.heap, item)
            return True
        if priority < -self.heap[0][0]:
            heapq.heapreplace(self.heap, item)
            return True
        return False

    def records(self) -> list[dict[str, Any]]:
        return sorted((item[2] for item in self.heap), key=lambda record: record["id"])


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def stable_digest(text: str, digest_size: int = 16) -> str:
    return blake2b(text.encode("utf-8"), digest_size=digest_size).hexdigest()


def stable_int(text: str) -> int:
    return int.from_bytes(blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def sha256_file(path: Path, chunk_size: int = 16 * 1024**2) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, path)


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = (
        unicodedata.normalize("NFC", value)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\t", " ")
    )
    text = CONTROL_PATTERN.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    return BLANK_LINES_PATTERN.sub("\n\n", text)


def dedup_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub("", unicodedata.normalize("NFKC", text).casefold())


def canonical_messages(messages: list[dict[str, str]]) -> str:
    return json.dumps(
        [[message["role"], dedup_text(message["content"])] for message in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def repeated_line_ratio(text: str) -> float:
    lines = [dedup_text(line) for line in text.splitlines() if len(dedup_text(line)) >= 4]
    if len(lines) < 4:
        return 0.0
    return max(Counter(lines).values()) / len(lines)


def normalize_messages(raw_messages: Any) -> tuple[list[dict[str, str]] | None, str | None]:
    if not isinstance(raw_messages, list) or not raw_messages:
        return None, "messages_not_list"
    messages = []
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "user": "user",
        "assistant": "assistant",
        "system": "system",
    }
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            return None, "message_not_dict"
        raw_role = str(raw_message.get("role") or raw_message.get("from") or "").lower()
        role = role_map.get(raw_role)
        content = normalize_text(
            raw_message.get("content") if "content" in raw_message else raw_message.get("value")
        )
        if role is None or not content:
            return None, "invalid_role_or_content"
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = normalize_text(messages[-1]["content"] + "\n" + content)
        else:
            messages.append({"role": role, "content": content})
    if messages and messages[0]["role"] == "system":
        if len(messages) < 3:
            return None, "system_without_dialogue"
        dialogue = messages[1:]
    else:
        dialogue = messages
    if not dialogue or dialogue[0]["role"] != "user" or dialogue[-1]["role"] != "assistant":
        return None, "dialogue_boundary"
    if any(message["role"] not in {"user", "assistant"} for message in dialogue):
        return None, "system_in_middle"
    if any(
        message["role"] == dialogue[index - 1]["role"]
        for index, message in enumerate(dialogue[1:], 1)
    ):
        return None, "role_order"
    if len(messages) > MAX_MESSAGES:
        return None, "too_many_messages"
    return messages, None


def prepare_record(
    source: str, detail: str, raw_messages: Any, metadata: dict[str, Any] | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    messages, error = normalize_messages(raw_messages)
    if messages is None:
        return None, error
    user_text = "\n".join(message["content"] for message in messages if message["role"] == "user")
    assistant_text = "\n".join(
        message["content"] for message in messages if message["role"] == "assistant"
    )
    if len(user_text) < MIN_USER_CHARS:
        return None, "user_too_short"
    if len(assistant_text) < MIN_ASSISTANT_CHARS:
        return None, "assistant_too_short"
    if len(user_text) > MAX_USER_CHARS:
        return None, "user_too_long"
    if len(assistant_text) > MAX_ASSISTANT_CHARS:
        return None, "assistant_too_long"
    if len(user_text) + len(assistant_text) > MAX_TOTAL_CHARS:
        return None, "conversation_too_long"
    if len(CJK_PATTERN.findall(user_text + assistant_text)) < 4:
        return None, "not_enough_chinese"
    if repeated_line_ratio(assistant_text) >= 0.6:
        return None, "assistant_repetition"
    identity_text = dedup_text(assistant_text)
    if any(pattern in identity_text for pattern in TEACHER_IDENTITY_PATTERNS):
        return None, "teacher_identity"
    canonical = canonical_messages(messages)
    record_id = f"{source}:{stable_digest(canonical)}"
    cleaned_metadata = {
        str(key): value
        for key, value in (metadata or {}).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    return {
        "id": record_id,
        "source": source,
        "source_detail": detail,
        "messages": messages,
        "metadata": cleaned_metadata,
    }, None


def parse_conversations(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None


def iter_magpie() -> Iterator[tuple[str, Any, dict[str, Any]]]:
    root = RAW_ROOT / "magpie-qwen2-pro-200k-chinese"
    columns = [
        "uuid",
        "instruction",
        "response",
        "task_category",
        "difficulty",
        "input_quality",
        "llama_guard_2",
        "instruct_reward",
        "language",
    ]
    for path in sorted(root.rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=PARQUET_BATCH_ROWS, columns=columns
        ):
            for row in batch.to_pylist():
                try:
                    reward = float(row.get("instruct_reward"))
                except (TypeError, ValueError):
                    reward = -999.0
                if (
                    str(row.get("language", "")).upper() != "ZH"
                    or str(row.get("llama_guard_2", "")).lower() != "safe"
                    or str(row.get("input_quality", "")).lower() == "poor"
                    or reward < 0
                ):
                    yield "quality_filter", None, {}
                    continue
                messages = [
                    {"role": "user", "content": row.get("instruction")},
                    {"role": "assistant", "content": row.get("response")},
                ]
                metadata = {
                    "uuid": row.get("uuid"),
                    "task_category": row.get("task_category"),
                    "difficulty": row.get("difficulty"),
                    "instruct_reward": reward,
                }
                yield str(row.get("task_category") or "unknown"), messages, metadata


def iter_fineweb_edu_qa() -> Iterator[tuple[str, Any, dict[str, Any]]]:
    root = RAW_ROOT / "fineweb-edu-chinese-v2_2-sft-qa" / "sft" / "cleaned"
    for path in sorted(root.glob("*.jsonl"), key=lambda value: int(value.stem)):
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    yield "json_error", None, {"file": path.name, "line": line_number}
                    continue
                instruction = normalize_text(row.get("instruction"))
                extra_input = normalize_text(row.get("input"))
                user_content = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": row.get("output")},
                ]
                yield "pure_qa", messages, {"file": path.name, "line": line_number}


def iter_smoltalk_file(path: Path) -> Iterator[tuple[str, Any, dict[str, Any]]]:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [
        column
        for column in (
            "conversations",
            "score",
            "difficulty",
            "classify",
            "system_prompt_key",
            "magpie_model",
            "n_turn",
        )
        if column in available
    ]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=columns):
        for row in batch.to_pylist():
            score_value = row.get("score")
            if score_value not in (None, "", "none"):
                try:
                    if float(score_value) < 4:
                        yield "quality_filter", None, {}
                        continue
                except (TypeError, ValueError):
                    yield "score_error", None, {}
                    continue
            messages = parse_conversations(row.get("conversations"))
            metadata = {
                "score": score_value,
                "difficulty": row.get("difficulty"),
                "classify": row.get("classify"),
                "model": row.get("magpie_model"),
                "turns": row.get("n_turn"),
            }
            yield path.stem, messages, metadata


def iter_qwen3_distill() -> Iterator[tuple[str, Any, dict[str, Any]]]:
    path = (
        RAW_ROOT
        / "chinese-qwen3-235b-distill-100k"
        / "Chinese-Qwen3-235b-a22b-Thinking-2507-Distill-data-100K_final.jsonl"
    )
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield "json_error", None, {"line": line_number}
                continue
            messages = [
                {"role": "user", "content": row.get("Input")},
                {"role": "assistant", "content": row.get("Answer_content")},
            ]
            metadata = {
                "from": row.get("From"),
                "generator": row.get("Generator"),
                "answer_chars": row.get("anwser_len"),
                "reasoning_removed": True,
            }
            yield str(row.get("From") or "unknown"), messages, metadata


def iter_deepseek_r1_distill() -> Iterator[tuple[str, Any, dict[str, Any]]]:
    path = RAW_ROOT / "chinese-deepseek-r1-distill-110k" / "distill_r1_110k.jsonl"
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield "json_error", None, {"line": line_number}
                continue
            try:
                score = float(row.get("score"))
            except (TypeError, ValueError):
                score = -1.0
            if score < 8:
                yield "quality_filter", None, {}
                continue
            messages = [
                {"role": "user", "content": row.get("input")},
                {"role": "assistant", "content": row.get("content")},
            ]
            metadata = {
                "repo_name": row.get("repo_name"),
                "score": score,
                "content_tokens_len": row.get("content_tokens_len"),
                "reasoning_removed": True,
            }
            yield str(row.get("repo_name") or "unknown"), messages, metadata


def sample_records(
    source: str,
    detail: str,
    iterator: Iterator[tuple[str, Any, dict[str, Any]]],
    target: int,
    staging_path: Path,
    recovered_json_lines: int = 0,
) -> SamplingResult:
    sampler = PrioritySampler(target)
    rejected: Counter[str] = Counter()
    scanned = 0
    eligible = 0
    for raw_detail, raw_messages, metadata in iterator:
        scanned += 1
        if raw_messages is None:
            rejected[raw_detail] += 1
        else:
            record, error = prepare_record(source, raw_detail, raw_messages, metadata)
            if record is None:
                rejected[str(error)] += 1
            else:
                eligible += 1
                if not sampler.offer(record):
                    rejected["duplicate_or_outside_sample"] += 1
        if scanned % PROGRESS_EVERY_ROWS == 0:
            print(
                f"{source}/{detail}：扫描 {scanned:,}，合格 {eligible:,}，当前保留 {len(sampler.heap):,}"
            )
    records = sampler.records()
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = staging_path.with_suffix(staging_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary_path, staging_path)
    print(
        f"{source}/{detail} 完成：扫描 {scanned:,}，合格 {eligible:,}，选择 {len(records):,}/{target:,}"
    )
    return SamplingResult(
        source,
        detail,
        target,
        scanned,
        eligible,
        len(records),
        dict(rejected),
        recovered_json_lines,
        str(staging_path),
    )


def build_staging() -> list[SamplingResult]:
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    results.append(
        sample_records(
            "magpie", "all", iter_magpie(), SOURCE_TARGETS["magpie"], STAGING_ROOT / "magpie.jsonl"
        )
    )
    results.append(
        sample_records(
            "fineweb_edu_qa",
            "all",
            iter_fineweb_edu_qa(),
            SOURCE_TARGETS["fineweb_edu_qa"],
            STAGING_ROOT / "fineweb_edu_qa.jsonl",
        )
    )
    smoltalk_root = RAW_ROOT / "smoltalk-chinese"
    for detail, target in SMOLTALK_TARGETS.items():
        path = smoltalk_root / f"{detail}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"缺少 SmolTalk 分类文件：{path}")
        results.append(
            sample_records(
                "smoltalk",
                detail,
                iter_smoltalk_file(path),
                target,
                STAGING_ROOT / f"smoltalk-{detail}.jsonl",
            )
        )
    results.append(
        sample_records(
            "qwen3_distill",
            "all",
            iter_qwen3_distill(),
            SOURCE_TARGETS["qwen3_distill"],
            STAGING_ROOT / "qwen3_distill.jsonl",
        )
    )
    results.append(
        sample_records(
            "deepseek_r1_distill",
            "all",
            iter_deepseek_r1_distill(),
            SOURCE_TARGETS["deepseek_r1_distill"],
            STAGING_ROOT / "deepseek_r1_distill.jsonl",
        )
    )
    return results


def split_for_record(record_id: str) -> str:
    bucket = stable_int(f"split:{record_id}") % SPLIT_BUCKETS
    if bucket < VALIDATION_BUCKETS:
        return "validation"
    if bucket < VALIDATION_BUCKETS + TEST_BUCKETS:
        return "test"
    return "train"


def merge_staging(results: list[SamplingResult]) -> dict[str, Any]:
    split_paths = {
        split: OUTPUT_ROOT / f"{split}.jsonl" for split in ("train", "validation", "test")
    }
    temporary_paths = {
        split: path.with_suffix(path.suffix + ".tmp") for split, path in split_paths.items()
    }
    files = {
        split: path.open("w", encoding="utf-8", newline="\n")
        for split, path in temporary_paths.items()
    }
    seen = set()
    split_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    duplicate_count = 0
    try:
        for result in results:
            path = Path(result.staging_path)
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    record = json.loads(line)
                    if record["id"] in seen:
                        duplicate_count += 1
                        continue
                    seen.add(record["id"])
                    split = split_for_record(record["id"])
                    files[split].write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    split_counts[split] += 1
                    source_counts.setdefault(record["source"], Counter())[split] += 1
    finally:
        for file in files.values():
            file.close()
    for split, path in split_paths.items():
        os.replace(temporary_paths[split], path)
    return {
        "splits": dict(split_counts),
        "sources": {source: dict(counts) for source, counts in source_counts.items()},
        "cross_source_duplicates": duplicate_count,
        "total_records": sum(split_counts.values()),
        "paths": {split: str(path) for split, path in split_paths.items()},
    }


def validate_record(record: Any) -> str | None:
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("id"), str)
        or not isinstance(record.get("source"), str)
    ):
        return "record_schema"
    messages, error = normalize_messages(record.get("messages"))
    if messages is None or messages != record["messages"]:
        return error or "messages_not_normalized"
    return None


def verify_dataset() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"缺少 manifest：{MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    seen = set()
    for split in ("train", "validation", "test"):
        path = OUTPUT_ROOT / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"缺少数据文件：{path}")
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"JSONL 解析失败：{path}:{line_number}: {error}") from error
                validation_error = validate_record(record)
                if validation_error:
                    raise RuntimeError(f"记录验收失败：{path}:{line_number}: {validation_error}")
                if record["id"] in seen:
                    raise RuntimeError(f"发现重复 ID：{record['id']}")
                if split_for_record(record["id"]) != split:
                    raise RuntimeError(f"划分不一致：{record['id']}")
                seen.add(record["id"])
                counts[split] += 1
                source_counts[record["source"]] += 1
        expected = int(manifest["output"]["splits"].get(split, 0))
        if counts[split] != expected:
            raise RuntimeError(f"{split} 数量错误：{counts[split]} != {expected}")
    if len(seen) != int(manifest["output"]["total_records"]):
        raise RuntimeError("总记录数与 manifest 不一致")
    return {"splits": dict(counts), "sources": dict(source_counts), "total_records": len(seen)}


def cleanup_staging(results: list[SamplingResult]) -> None:
    for result in results:
        path = Path(result.staging_path)
        if path.exists():
            path.unlink()
    if STAGING_ROOT.exists() and not any(STAGING_ROOT.iterdir()):
        STAGING_ROOT.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗、统一、去重并分层抽取 XiangLM SFT 数据")
    parser.add_argument("--verify-only", action="store_true", help="只验证已生成的数据集")
    parser.add_argument("--keep-staging", action="store_true", help="成功后保留中间候选文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("XiangLM SFT 数据构建")
    print("=" * 72)
    print(f"原始目录：{RAW_ROOT}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"计划候选数：{sum(SOURCE_TARGETS.values()) + sum(SMOLTALK_TARGETS.values()):,}")
    if args.verify_only:
        verified = verify_dataset()
        print(f"验收通过：{verified['total_records']:,} 条；{verified['splits']}")
        return
    results = build_staging()
    output = merge_staging(results)
    source_files = [
        path
        for path in RAW_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".parquet", ".jsonl"}
    ]
    manifest = {
        "format_version": 1,
        "created_at": now_iso(),
        "random_seed": RANDOM_SEED,
        "schema": {
            "id": "string",
            "source": "string",
            "source_detail": "string",
            "messages": "list[role,content]",
            "metadata": "object",
        },
        "filters": {
            "min_user_chars": MIN_USER_CHARS,
            "min_assistant_chars": MIN_ASSISTANT_CHARS,
            "max_user_chars": MAX_USER_CHARS,
            "max_assistant_chars": MAX_ASSISTANT_CHARS,
            "max_total_chars": MAX_TOTAL_CHARS,
            "max_messages": MAX_MESSAGES,
            "magpie_reward_min": 0,
            "smoltalk_score_min": 4,
            "deepseek_r1_score_min": 8,
            "long_cot_removed": True,
        },
        "targets": {"sources": SOURCE_TARGETS, "smoltalk_categories": SMOLTALK_TARGETS},
        "sampling": [result.__dict__ for result in results],
        "split_policy": {
            "buckets": SPLIT_BUCKETS,
            "validation_buckets": VALIDATION_BUCKETS,
            "test_buckets": TEST_BUCKETS,
            "method": "blake2b-64 of record id",
        },
        "output": output,
        "inputs": [
            {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_files
        ],
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    verified = verify_dataset()
    report = {
        "created_at": now_iso(),
        "verified": verified,
        "sampling": [result.__dict__ for result in results],
        "output": output,
    }
    atomic_write_json(REPORT_PATH, report)
    if not args.keep_staging:
        cleanup_staging(results)
    print()
    print("=" * 72)
    print("SFT 构建并验收完成")
    print(f"总记录：{verified['total_records']:,}")
    print(f"划分：{verified['splits']}")
    print(f"来源：{verified['sources']}")
    print(f"Manifest：{MANIFEST_PATH}")


if __name__ == "__main__":
    main()
