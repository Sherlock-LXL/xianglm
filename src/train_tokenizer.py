from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import blake2b
from pathlib import Path
from random import Random
from typing import Iterator
import json
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

from src.inspect_data import (
    SOURCES,
    SourceConfig,
    find_source_files,
    iter_source_records,
    normalize_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = PROJECT_ROOT / "data" / "tokenizer" / "xianglm-32k"
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
REPORT_PATH = TOKENIZER_DIR / "training_report.json"

VOCAB_SIZE = 32_768
TOTAL_SAMPLE_BYTES = 128 * 1024**2
SOURCE_WEIGHTS = {
    "chinese_web": 0.60,
    "chinese_education": 0.25,
    "english_education": 0.10,
    "code": 0.05,
}

RANDOM_SEED = 20_260_822
MAX_DOCUMENT_CHARS = 50_000
BATCH_DOCUMENTS = 256
ROW_GROUPS_PER_VISIT = 8
CODE_SAMPLE_BUCKETS = 16
EVAL_DOCUMENTS_PER_SOURCE = 100

PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
EOT_TOKEN = "<|endoftext|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    EOT_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
]


@dataclass
class SamplingStats:
    source: str
    target_bytes: int
    accepted_bytes: int = 0
    accepted_documents: int = 0

    @property
    def completion_ratio(self) -> float:
        if self.target_bytes == 0:
            return 1.0
        return self.accepted_bytes / self.target_bytes


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def build_source_quotas() -> dict[str, int]:
    missing = {source.name for source in SOURCES if source.name not in SOURCE_WEIGHTS}
    if missing:
        raise RuntimeError(f"SOURCE_WEIGHTS 缺少来源：{sorted(missing)}")
    weights_sum = sum(SOURCE_WEIGHTS[source.name] for source in SOURCES)
    if abs(weights_sum - 1.0) > 1e-9:
        raise RuntimeError(f"数据比例之和必须为 1，目前是 {weights_sum}")
    quotas: dict[str, int] = {}
    remaining = TOTAL_SAMPLE_BYTES
    for source in SOURCES[:-1]:
        quota = int(TOTAL_SAMPLE_BYTES * SOURCE_WEIGHTS[source.name])
        quotas[source.name] = quota
        remaining -= quota
    quotas[SOURCES[-1].name] = remaining
    return quotas


def crop_document(text: str, rng: Random) -> str:
    if len(text) <= MAX_DOCUMENT_CHARS:
        return text
    max_start = len(text) - MAX_DOCUMENT_CHARS
    start = rng.randint(0, max_start)
    end = start + MAX_DOCUMENT_CHARS
    return text[start:end]


def iter_random_parquet_texts(source: SourceConfig, rng: Random) -> Iterator[str]:
    files = find_source_files(source)
    if not files:
        raise RuntimeError(f"{source.name} 没有找到 parquet 文件")
    states: list[tuple[Path, list[int]]] = []
    print(f"扫描 {source.name} 的 parquet 元数据……")
    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        row_groups = list(range(parquet_file.metadata.num_row_groups))
        rng.shuffle(row_groups)
        states.append((file_path, row_groups))
    while states:
        rng.shuffle(states)
        next_states: list[tuple[Path, list[int]]] = []
        for file_path, remaining_groups in states:
            take_count = min(ROW_GROUPS_PER_VISIT, len(remaining_groups))
            selected_groups = [remaining_groups.pop() for _ in range(take_count)]
            parquet_file = pq.ParquetFile(file_path)
            for row_group_index in selected_groups:
                table = parquet_file.read_row_group(row_group_index, columns=["text"])
                texts = table.column("text").to_pylist()
                rng.shuffle(texts)
                for text in texts:
                    if isinstance(text, str):
                        yield text
            if remaining_groups:
                next_states.append((file_path, remaining_groups))
        states = next_states


def stable_bucket(value: str, bucket_count: int) -> int:
    digest = blake2b(value.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % bucket_count


def iter_sampled_code_texts(source: SourceConfig) -> Iterator[str]:
    for record in iter_source_records(source):
        doc_id = str(record["doc_id"])
        if stable_bucket(doc_id, CODE_SAMPLE_BUCKETS) != 0:
            continue
        text = record["text"]
        if isinstance(text, str):
            yield text


def iter_source_texts(source: SourceConfig, rng: Random) -> Iterator[str]:
    if source.kind == "parquet":
        yield from iter_random_parquet_texts(source, rng)
        return
    if source.kind == "json_gzip":
        yield from iter_sampled_code_texts(source)
        return
    raise ValueError(f"未知数据类型：{source.kind}")


class TrainingCorpus:
    def __init__(self, quotas: dict[str, int]) -> None:
        self.quotas = quotas
        self.stats = {
            source.name: SamplingStats(source=source.name, target_bytes=quotas[source.name])
            for source in SOURCES
        }

    def __iter__(self) -> Iterator[str]:
        for source_index, source in enumerate(SOURCES):
            source_seed = RANDOM_SEED + source_index * 100_003
            rng = Random(source_seed)
            stats = self.stats[source.name]
            print()
            print("=" * 72)
            print(f"开始抽样：{source.name}")
            print(f"目标大小：{format_bytes(stats.target_bytes)}")
            for raw_text in iter_source_texts(source, rng):
                text = normalize_text(raw_text)
                if not text.strip():
                    continue
                text = crop_document(text, rng)
                encoded_size = len(text.encode("utf-8", errors="strict"))
                if encoded_size == 0:
                    continue
                stats.accepted_bytes += encoded_size
                stats.accepted_documents += 1
                yield text
                if stats.accepted_documents % 10_000 == 0:
                    print(
                        f"{source.name}: {stats.accepted_documents:,} 篇，{format_bytes(stats.accepted_bytes)}"
                    )
                if stats.accepted_bytes >= stats.target_bytes:
                    break
            if stats.accepted_bytes < stats.target_bytes:
                raise RuntimeError(
                    f"{source.name} 抽样不足：需要 {format_bytes(stats.target_bytes)}，实际只有 {format_bytes(stats.accepted_bytes)}"
                )
            print(
                f"{source.name} 抽样完成：{stats.accepted_documents:,} 篇，{format_bytes(stats.accepted_bytes)}"
            )

    def batch_iterator(self) -> Iterator[list[str]]:
        batch: list[str] = []
        for text in self:
            batch.append(text)
            if len(batch) >= BATCH_DOCUMENTS:
                yield batch
                batch = []
        if batch:
            yield batch


def build_tokenizer() -> tuple[Tokenizer, BpeTrainer]:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        max_token_length=32,
    )
    return tokenizer, trainer


def evaluate_tokenizer(tokenizer: Tokenizer) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    unk_id = tokenizer.token_to_id(UNK_TOKEN)
    for source in SOURCES:
        document_count = 0
        character_count = 0
        byte_count = 0
        token_count = 0
        unknown_count = 0
        roundtrip_failures = 0
        for record in iter_source_records(source):
            text = normalize_text(record["text"])
            if not text:
                continue
            encoding = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
            document_count += 1
            character_count += len(text)
            byte_count += len(text.encode("utf-8"))
            token_count += len(encoding.ids)
            if unk_id is not None:
                unknown_count += encoding.ids.count(unk_id)
            if decoded != text:
                roundtrip_failures += 1
            if document_count >= EVAL_DOCUMENTS_PER_SOURCE:
                break
        if document_count == 0:
            raise RuntimeError(f"{source.name} 没有可用于验收的文本")
        result = {
            "documents": document_count,
            "characters": character_count,
            "utf8_bytes": byte_count,
            "tokens": token_count,
            "characters_per_token": character_count / token_count if token_count else 0.0,
            "bytes_per_token": byte_count / token_count if token_count else 0.0,
            "unknown_tokens": unknown_count,
            "roundtrip_failures": roundtrip_failures,
        }
        results[source.name] = result
    return results


def print_evaluation(results: dict[str, dict[str, float | int]]) -> None:
    print()
    print("=" * 72)
    print("Tokenizer 验收")
    for source_name, result in results.items():
        print()
        print(f"来源：{source_name}")
        print(f"文档数：{result['documents']:,}")
        print(f"字符数：{result['characters']:,}")
        print(f"Token 数：{result['tokens']:,}")
        print(f"字符 / token：{result['characters_per_token']:.3f}")
        print(f"UTF-8 字节 / token：{result['bytes_per_token']:.3f}")
        print(f"UNK 数：{result['unknown_tokens']}")
        print(f"回译失败数：{result['roundtrip_failures']}")


def main() -> None:
    print("=" * 72)
    print("XiangLM 32K Byte-level BPE Tokenizer")
    print("=" * 72)
    print(f"项目目录：{PROJECT_ROOT}")
    print(f"输出目录：{TOKENIZER_DIR}")
    print(f"词表大小：{VOCAB_SIZE:,}")
    print(f"抽样大小：{format_bytes(TOTAL_SAMPLE_BYTES)}")
    existing_outputs = [
        path
        for path in (
            TOKENIZER_JSON,
            TOKENIZER_DIR / "vocab.json",
            TOKENIZER_DIR / "merges.txt",
            REPORT_PATH,
        )
        if path.exists()
    ]
    if existing_outputs:
        formatted = "\n".join(f"- {path}" for path in existing_outputs)
        raise RuntimeError(
            f"检测到已有 tokenizer 输出。为防止意外覆盖，请先确认并手动移动或删除：\n{formatted}"
        )
    quotas = build_source_quotas()
    print()
    print("训练语料分配：")
    for source in SOURCES:
        quota = quotas[source.name]
        weight = SOURCE_WEIGHTS[source.name]
        print(f"- {source.name}: {weight:.0%}, {format_bytes(quota)}")
    corpus = TrainingCorpus(quotas)
    tokenizer, trainer = build_tokenizer()
    print()
    print("=" * 72)
    print("开始训练 tokenizer")
    print("此阶段可能长时间停留在 BPE pair 统计，属于正常现象。")
    tokenizer.train_from_iterator(corpus.batch_iterator(), trainer=trainer)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_JSON), pretty=True)
    tokenizer.model.save(str(TOKENIZER_DIR))
    special_token_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    print()
    print("=" * 72)
    print("特殊 token ID")
    for token, token_id in special_token_ids.items():
        print(f"{token}: {token_id}")
    evaluation = evaluate_tokenizer(tokenizer)
    print_evaluation(evaluation)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "tokenizer_type": "Byte-level BPE",
        "vocab_size_requested": VOCAB_SIZE,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "sample_target_bytes": TOTAL_SAMPLE_BYTES,
        "source_weights": SOURCE_WEIGHTS,
        "sampling_stats": {name: asdict(stats) for name, stats in corpus.stats.items()},
        "special_token_ids": special_token_ids,
        "configuration": {
            "random_seed": RANDOM_SEED,
            "max_document_chars": MAX_DOCUMENT_CHARS,
            "batch_documents": BATCH_DOCUMENTS,
            "min_frequency": 2,
            "max_token_length": 32,
            "add_prefix_space": False,
            "unicode_normalizer": None,
        },
        "evaluation": evaluation,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    total_unknown = sum(int(result["unknown_tokens"]) for result in evaluation.values())
    total_roundtrip_failures = sum(
        int(result["roundtrip_failures"]) for result in evaluation.values()
    )
    print()
    print("=" * 72)
    print("训练完成")
    print(f"Tokenizer：{TOKENIZER_JSON}")
    print(f"训练报告：{REPORT_PATH}")
    print(f"实际词表大小：{tokenizer.get_vocab_size():,}")
    if total_unknown != 0:
        raise RuntimeError(f"验收失败：发现 {total_unknown} 个 UNK")
    if total_roundtrip_failures != 0:
        raise RuntimeError(f"验收失败：发现 {total_roundtrip_failures} 个回译错误")
    print("验收通过：UNK = 0，回译失败 = 0")


if __name__ == "__main__":
    main()
