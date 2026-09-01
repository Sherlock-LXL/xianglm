from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from src.inspect_data import PROJECT_ROOT, SOURCES, SourceConfig, find_source_files, make_fallback_id, normalize_text

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

TOKENIZER_PATH = PROJECT_ROOT / "data" / "tokenizer" / "xianglm-32k" / "tokenizer.json"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "tokenized" / "xianglm-32k"
CHECKPOINT_PATH = OUTPUT_ROOT / "checkpoint.json"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"

FORMAT_VERSION = 1
TOKEN_DTYPE = np.dtype("<u2")
TOKEN_BYTES = TOKEN_DTYPE.itemsize
SHARD_SIZE_TOKENS = 100_000_000
VALIDATION_BUCKETS = 10_000
VALIDATION_THRESHOLD = 50
MAX_BATCH_DOCUMENTS = 512
MAX_BATCH_UTF8_BYTES = 4 * 1024**2
LONG_DOCUMENT_UTF8_BYTES = 4 * 1024**2
LONG_DOCUMENT_CHUNK_CHARS = 250_000
CHECKPOINT_EVERY_UTF8_BYTES = 64 * 1024**2
CHECKPOINT_EVERY_SECONDS = 120.0
MIN_FREE_DISK_BYTES = 20 * 1024**3
PARQUET_BATCH_ROWS = 1024
VERIFY_CHUNK_BYTES = 64 * 1024**2

RECOMMENDED_TRAINING_MIXTURE = {"chinese_web": 0.60, "chinese_education": 0.25, "english_education": 0.10, "code": 0.05}
SPECIAL_TOKEN_STRINGS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]

@dataclass(frozen=True)
class InputRecord:
    doc_id: str
    text: str
    next_cursor: dict[str, int]

@dataclass(frozen=True)
class PreparedRecord:
    doc_id: str
    text: str
    utf8_bytes: int
    split: str
    next_cursor: dict[str, int]

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()

def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"

def sha256_file(path: Path, chunk_size: int = 16 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)

def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()

def relative_to_output(path: Path) -> str:
    return path.resolve().relative_to(OUTPUT_ROOT.resolve()).as_posix()

def ensure_free_disk_space() -> None:
    free_bytes = shutil.disk_usage(OUTPUT_ROOT).free
    if free_bytes < MIN_FREE_DISK_BYTES:
        raise RuntimeError(f"磁盘剩余空间低于安全线：{format_bytes(free_bytes)}。\n为防止磁盘写满，编码已安全停止。")

def load_tokenizers() -> tuple[Tokenizer, Tokenizer, dict[str, int]]:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"没有找到 tokenizer：{TOKENIZER_PATH}")
    control_tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    special_token_ids: dict[str, int] = {}
    for token in SPECIAL_TOKEN_STRINGS:
        token_id = control_tokenizer.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f"Tokenizer 缺少特殊 token：{token}")
        special_token_ids[token] = token_id
    if control_tokenizer.get_vocab_size() > 65_536:
        raise RuntimeError("词表大于 65,536，不能使用 uint16")
    tokenizer_payload = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    tokenizer_payload["added_tokens"] = []
    raw_tokenizer = Tokenizer.from_buffer(json.dumps(tokenizer_payload, ensure_ascii=False).encode("utf-8"))
    raw_tokenizer.no_padding()
    raw_tokenizer.no_truncation()
    probe = "普通文本<|endoftext|>literal<|im_start|>测试"
    probe_encoding = raw_tokenizer.encode(probe, add_special_tokens=False)
    reserved_ids = set(special_token_ids.values())
    accidental_reserved_ids = reserved_ids.intersection(probe_encoding.ids)
    if accidental_reserved_ids:
        raise RuntimeError(f"普通正文 tokenizer 仍然产生了保留 token ID：{sorted(accidental_reserved_ids)}")
    probe_decoded = raw_tokenizer.decode(probe_encoding.ids, skip_special_tokens=False)
    if probe_decoded != probe:
        raise RuntimeError("普通正文 tokenizer 回译测试失败")
    return control_tokenizer, raw_tokenizer, special_token_ids

def build_runtime_config(tokenizer_sha256: str) -> dict[str, Any]:
    source_files: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCES:
        files = find_source_files(source)
        if not files:
            raise RuntimeError(f"{source.name} 没有找到输入文件")
        entries: list[dict[str, Any]] = []
        for file_path in files:
            stat = file_path.stat()
            entries.append({"path": relative_to_project(file_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        source_files[source.name] = entries
    return {
        "format_version": FORMAT_VERSION,
        "tokenizer_path": relative_to_project(TOKENIZER_PATH),
        "tokenizer_sha256": tokenizer_sha256,
        "token_dtype": TOKEN_DTYPE.str,
        "shard_size_tokens": SHARD_SIZE_TOKENS,
        "validation_buckets": VALIDATION_BUCKETS,
        "validation_threshold": VALIDATION_THRESHOLD,
        "max_batch_documents": MAX_BATCH_DOCUMENTS,
        "max_batch_utf8_bytes": MAX_BATCH_UTF8_BYTES,
        "long_document_utf8_bytes": LONG_DOCUMENT_UTF8_BYTES,
        "long_document_chunk_chars": LONG_DOCUMENT_CHUNK_CHARS,
        "source_files": source_files,
    }

def initial_cursor(source: SourceConfig) -> dict[str, int]:
    if source.kind == "parquet":
        return {"row_group": 0, "row_offset": 0}
    if source.kind == "json_gzip":
        return {"line_number": 0}
    raise ValueError(f"未知输入格式：{source.kind}")

def initial_writer_state() -> dict[str, Any]:
    return {"shard_index": 0, "tokens_in_partial": 0, "documents_in_partial": 0, "total_tokens": 0, "total_documents": 0, "completed_shards": []}

def create_checkpoint(runtime_config: dict[str, Any], config_fingerprint: str) -> dict[str, Any]:
    source_states: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        source_states[source.name] = {
            "completed": False,
            "input_file_index": 0,
            "cursor": initial_cursor(source),
            "processed_documents": 0,
            "processed_utf8_bytes": 0,
            "writers": {"train": initial_writer_state(), "validation": initial_writer_state()},
        }
    return {
        "format_version": FORMAT_VERSION,
        "status": "running",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "config_fingerprint": config_fingerprint,
        "runtime_config": runtime_config,
        "sources": source_states,
    }

def load_or_create_checkpoint(runtime_config: dict[str, Any], config_fingerprint: str) -> dict[str, Any]:
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        previous_fingerprint = checkpoint.get("config_fingerprint")
        if previous_fingerprint != config_fingerprint:
            raise RuntimeError(f"当前输入文件、tokenizer 或编码配置与 checkpoint 不一致。\n为防止混合两种数据格式，程序拒绝续跑。\n已有指纹：{previous_fingerprint}\n当前指纹：{config_fingerprint}")
        print("检测到 checkpoint，将从上次提交位置继续。")
        return checkpoint
    if OUTPUT_ROOT.exists():
        existing_files = [path for path in OUTPUT_ROOT.rglob("*") if path.is_file()]
        if existing_files:
            preview = "\n".join(f"- {path}" for path in existing_files[:10])
            raise RuntimeError(f"输出目录中已有文件，但没有 checkpoint。\n程序不会猜测这些文件是否完整，也不会覆盖：\n{preview}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = create_checkpoint(runtime_config, config_fingerprint)
    atomic_write_json(CHECKPOINT_PATH, checkpoint)
    return checkpoint

def iter_parquet_file_records(source: SourceConfig, file_path: Path, cursor: dict[str, int]) -> Iterator[InputRecord]:
    parquet_file = pq.ParquetFile(file_path)
    available_columns = set(parquet_file.schema_arrow.names)
    if "text" not in available_columns:
        raise RuntimeError(f"{file_path} 不包含 text 列")
    columns = ["text"]
    if source.id_column is not None and source.id_column in available_columns:
        columns.append(source.id_column)
    row_group_count = parquet_file.metadata.num_row_groups
    start_row_group = cursor.get("row_group", 0)
    start_row_offset = cursor.get("row_offset", 0)
    if start_row_group > row_group_count:
        raise RuntimeError(f"{file_path} 的 row group cursor 越界")
    row_group_bases: list[int] = []
    accumulated_rows = 0
    for index in range(row_group_count):
        row_group_bases.append(accumulated_rows)
        accumulated_rows += parquet_file.metadata.row_group(index).num_rows
    for row_group_index in range(start_row_group, row_group_count):
        row_group_rows = parquet_file.metadata.row_group(row_group_index).num_rows
        required_offset = start_row_offset if row_group_index == start_row_group else 0
        if required_offset > row_group_rows:
            raise RuntimeError(f"{file_path} 的 row offset 越界")
        position_in_group = 0
        batches = parquet_file.iter_batches(batch_size=PARQUET_BATCH_ROWS, row_groups=[row_group_index], columns=columns)
        for batch in batches:
            batch_start = position_in_group
            batch_end = batch_start + batch.num_rows
            position_in_group = batch_end
            if batch_end <= required_offset:
                continue
            values = batch.to_pydict()
            texts = values["text"]
            ids = None
            if source.id_column is not None:
                ids = values.get(source.id_column)
            local_start = max(0, required_offset - batch_start)
            for local_index in range(local_start, batch.num_rows):
                row_offset = batch_start + local_index
                next_row_group = row_group_index
                next_row_offset = row_offset + 1
                if next_row_offset >= row_group_rows:
                    next_row_group += 1
                    next_row_offset = 0
                raw_text = texts[local_index]
                if not isinstance(raw_text, str):
                    continue
                text = normalize_text(raw_text)
                if not text.strip():
                    continue
                raw_id = ids[local_index] if ids is not None else None
                absolute_row_number = row_group_bases[row_group_index] + row_offset
                doc_id = str(raw_id) if raw_id is not None else make_fallback_id(source, file_path, absolute_row_number)
                yield InputRecord(doc_id=doc_id, text=text, next_cursor={"row_group": next_row_group, "row_offset": next_row_offset})

def iter_gzip_file_records(source: SourceConfig, file_path: Path, cursor: dict[str, int]) -> Iterator[InputRecord]:
    start_line_number = cursor.get("line_number", 0)
    with gzip.open(file_path, mode="rt", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file):
            if line_number < start_line_number:
                continue
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{file_path} 第 {line_number + 1} 行 JSON 损坏") from error
            raw_text = row.get("text")
            if not isinstance(raw_text, str):
                continue
            text = normalize_text(raw_text)
            if not text.strip():
                continue
            raw_id = row.get("id")
            doc_id = str(raw_id) if raw_id is not None else make_fallback_id(source, file_path, line_number)
            yield InputRecord(doc_id=doc_id, text=text, next_cursor={"line_number": line_number + 1})

def iter_input_file_records(source: SourceConfig, file_path: Path, cursor: dict[str, int]) -> Iterator[InputRecord]:
    if source.kind == "parquet":
        yield from iter_parquet_file_records(source, file_path, cursor)
        return
    if source.kind == "json_gzip":
        yield from iter_gzip_file_records(source, file_path, cursor)
        return
    raise ValueError(f"未知输入格式：{source.kind}")

def prepare_record(record: InputRecord) -> PreparedRecord:
    encoded_text = record.text.encode("utf-8", errors="strict")
    digest = hashlib.blake2b(encoded_text, digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="little", signed=False) % VALIDATION_BUCKETS
    split = "validation" if bucket < VALIDATION_THRESHOLD else "train"
    return PreparedRecord(doc_id=record.doc_id, text=record.text, utf8_bytes=len(encoded_text), split=split, next_cursor=record.next_cursor)

def iter_long_document_chunks(text: str) -> Iterator[str]:
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + LONG_DOCUMENT_CHUNK_CHARS, text_length)
        if end < text_length:
            search_start = max(start + 1, end - LONG_DOCUMENT_CHUNK_CHARS // 5)
            newline_position = text.rfind("\n", search_start, end)
            if newline_position > start:
                end = newline_position + 1
        if end <= start:
            end = min(start + LONG_DOCUMENT_CHUNK_CHARS, text_length)
        yield text[start:end]
        start = end

class ShardWriter:
    def __init__(self, source_name: str, split: str, state: dict[str, Any], vocab_size: int) -> None:
        self.source_name = source_name
        self.split = split
        self.state = state
        self.vocab_size = vocab_size
        self.directory = OUTPUT_ROOT / source_name / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._recover_filesystem_to_checkpoint()

    def _final_path(self, shard_index: int) -> Path:
        return self.directory / f"shard_{shard_index:05d}.bin"

    def _partial_path(self, shard_index: int) -> Path:
        return self.directory / f"shard_{shard_index:05d}.bin.partial"

    def _parse_shard_index(self, path: Path) -> int | None:
        match = re.fullmatch(r"shard_(\d+)\.bin(?:\.partial)?", path.name)
        if match is None:
            return None
        return int(match.group(1))

    def _recover_filesystem_to_checkpoint(self) -> None:
        current_index = int(self.state["shard_index"])
        expected_partial_tokens = int(self.state["tokens_in_partial"])
        expected_partial_bytes = expected_partial_tokens * TOKEN_BYTES
        completed_entries = self.state["completed_shards"]
        completed_indices = {int(entry["index"]) for entry in completed_entries}
        for entry in completed_entries:
            path = OUTPUT_ROOT / entry["path"]
            if not path.exists():
                raise RuntimeError(f"Checkpoint 记录的完整分片不存在：{path}")
            if path.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"完整分片大小与 checkpoint 不符：{path}")
        final_current = self._final_path(current_index)
        partial_current = self._partial_path(current_index)
        if final_current.exists():
            if partial_current.exists():
                raise RuntimeError(f"同时发现 final 和 partial：\n{final_current}\n{partial_current}")
            print(f"恢复未提交分片：{final_current.name}")
            os.replace(final_current, partial_current)
        for path in self.directory.glob("shard_*"):
            index = self._parse_shard_index(path)
            if index is None:
                continue
            if index < current_index:
                if path.suffix == ".bin" and index in completed_indices:
                    continue
                if path.name.endswith(".bin.partial"):
                    raise RuntimeError(f"发现旧分片的 partial 文件：{path}")
                if index not in completed_indices:
                    raise RuntimeError(f"发现 checkpoint 未记录的旧分片：{path}")
            if index > current_index:
                print(f"删除 checkpoint 之后的未提交输出：{path}")
                path.unlink()
        if expected_partial_bytes == 0:
            if partial_current.exists():
                with partial_current.open("r+b") as file:
                    file.truncate(0)
                    file.flush()
                    os.fsync(file.fileno())
                partial_current.unlink()
            return
        if not partial_current.exists():
            raise RuntimeError(f"Checkpoint 记录了 partial token，但对应文件不存在：{partial_current}")
        actual_bytes = partial_current.stat().st_size
        if actual_bytes < expected_partial_bytes:
            raise RuntimeError(f"Partial 文件短于 checkpoint：{partial_current}\n文件：{actual_bytes} 字节\ncheckpoint：{expected_partial_bytes} 字节")
        if actual_bytes > expected_partial_bytes:
            print(f"回滚 checkpoint 之后的未提交字节：{partial_current.name}，{actual_bytes - expected_partial_bytes:,} 字节")
            with partial_current.open("r+b") as file:
                file.truncate(expected_partial_bytes)
                file.flush()
                os.fsync(file.fileno())

    def _ensure_open(self) -> None:
        if self._handle is not None:
            return
        partial_path = self._partial_path(int(self.state["shard_index"]))
        self._handle = partial_path.open("ab")

    def flush(self) -> None:
        if self._handle is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is None:
            return
        self.flush()
        self._handle.close()
        self._handle = None

    def append_tokens(self, token_ids: Sequence[int], *, document_end: bool = False) -> None:
        if not token_ids:
            if document_end:
                raise ValueError("document_end 必须随至少一个 token 写入")
            return
        minimum_id = min(token_ids)
        maximum_id = max(token_ids)
        if minimum_id < 0:
            raise RuntimeError(f"发现负数 token ID：{minimum_id}")
        if maximum_id >= self.vocab_size:
            raise RuntimeError(f"Token ID 超过词表范围：{maximum_id} >= {self.vocab_size}")
        offset = 0
        total_length = len(token_ids)
        while offset < total_length:
            remaining_capacity = SHARD_SIZE_TOKENS - int(self.state["tokens_in_partial"])
            take = min(remaining_capacity, total_length - offset)
            end = offset + take
            is_document_end_piece = document_end and end == total_length
            piece = np.asarray(token_ids[offset:end], dtype=TOKEN_DTYPE)
            self._ensure_open()
            piece.tofile(self._handle)
            self.state["tokens_in_partial"] += take
            self.state["total_tokens"] += take
            if is_document_end_piece:
                self.state["documents_in_partial"] += 1
                self.state["total_documents"] += 1
            offset = end
            if self.state["tokens_in_partial"] == SHARD_SIZE_TOKENS:
                self._finalize_current_shard()

    def _finalize_current_shard(self) -> None:
        token_count = int(self.state["tokens_in_partial"])
        if token_count == 0:
            return
        self.close()
        shard_index = int(self.state["shard_index"])
        partial_path = self._partial_path(shard_index)
        final_path = self._final_path(shard_index)
        if not partial_path.exists():
            raise RuntimeError(f"准备完成分片时没有找到 partial：{partial_path}")
        expected_bytes = token_count * TOKEN_BYTES
        actual_bytes = partial_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(f"Partial 分片大小错误：{partial_path}\n期望：{expected_bytes}\n实际：{actual_bytes}")
        if final_path.exists():
            raise RuntimeError(f"目标分片已经存在：{final_path}")
        os.replace(partial_path, final_path)
        checksum = sha256_file(final_path)
        document_count = int(self.state["documents_in_partial"])
        entry = {"index": shard_index, "path": relative_to_output(final_path), "tokens": token_count, "bytes": expected_bytes, "documents": document_count, "sha256": checksum}
        self.state["completed_shards"].append(entry)
        self.state["shard_index"] += 1
        self.state["tokens_in_partial"] = 0
        self.state["documents_in_partial"] = 0
        print(f"完成分片：{self.source_name}/{self.split}/{final_path.name}，{token_count:,} tokens，{format_bytes(expected_bytes)}")

    def finalize_tail(self) -> None:
        self._finalize_current_shard()

def validate_content_token_ids(token_ids: Sequence[int], doc_id: str, reserved_ids: set[int]) -> None:
    accidental_ids = reserved_ids.intersection(token_ids)
    if accidental_ids:
        raise RuntimeError(f"普通正文产生了保留 token ID。\ndoc_id：{doc_id}\n保留 ID：{sorted(accidental_ids)}")

def encode_regular_batch(records: list[PreparedRecord], raw_tokenizer: Tokenizer, writers: dict[str, ShardWriter], eot_id: int, reserved_ids: set[int], source_state: dict[str, Any]) -> int:
    if not records:
        return 0
    encodings = raw_tokenizer.encode_batch([record.text for record in records], add_special_tokens=False)
    if len(encodings) != len(records):
        raise RuntimeError("encode_batch 返回数量异常")
    processed_bytes = 0
    for record, encoding in zip(records, encodings, strict=True):
        validate_content_token_ids(encoding.ids, record.doc_id, reserved_ids)
        writer = writers[record.split]
        writer.append_tokens(encoding.ids)
        writer.append_tokens([eot_id], document_end=True)
        source_state["processed_documents"] += 1
        source_state["processed_utf8_bytes"] += record.utf8_bytes
        source_state["cursor"] = record.next_cursor
        processed_bytes += record.utf8_bytes
    return processed_bytes

def encode_long_document(record: PreparedRecord, raw_tokenizer: Tokenizer, writers: dict[str, ShardWriter], eot_id: int, reserved_ids: set[int], source_state: dict[str, Any]) -> int:
    writer = writers[record.split]
    for chunk in iter_long_document_chunks(record.text):
        encoding = raw_tokenizer.encode(chunk, add_special_tokens=False)
        validate_content_token_ids(encoding.ids, record.doc_id, reserved_ids)
        writer.append_tokens(encoding.ids)
    writer.append_tokens([eot_id], document_end=True)
    source_state["processed_documents"] += 1
    source_state["processed_utf8_bytes"] += record.utf8_bytes
    source_state["cursor"] = record.next_cursor
    return record.utf8_bytes

class CheckpointCommitter:
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        self.checkpoint = checkpoint
        self.bytes_since_commit = 0
        self.last_commit_time = time.monotonic()

    def add_processed_bytes(self, value: int) -> None:
        self.bytes_since_commit += value

    def should_commit(self) -> bool:
        if self.bytes_since_commit >= CHECKPOINT_EVERY_UTF8_BYTES:
            return True
        elapsed = time.monotonic() - self.last_commit_time
        return elapsed >= CHECKPOINT_EVERY_SECONDS

    def commit(self, writers: dict[str, ShardWriter], *, force: bool = False, progress_label: str | None = None) -> None:
        if not force and not self.should_commit():
            return
        for writer in writers.values():
            writer.flush()
        self.checkpoint["updated_at"] = now_iso()
        atomic_write_json(CHECKPOINT_PATH, self.checkpoint)
        ensure_free_disk_space()
        if progress_label is not None:
            print(progress_label)
        self.bytes_since_commit = 0
        self.last_commit_time = time.monotonic()

def process_source(source: SourceConfig, checkpoint: dict[str, Any], raw_tokenizer: Tokenizer, vocab_size: int, special_token_ids: dict[str, int]) -> None:
    source_state = checkpoint["sources"][source.name]
    if source_state["completed"]:
        print(f"{source.name} 已完成，跳过。")
        return
    files = find_source_files(source)
    configured_files = checkpoint["runtime_config"]["source_files"][source.name]
    current_relative_files = [relative_to_project(path) for path in files]
    configured_relative_files = [entry["path"] for entry in configured_files]
    if current_relative_files != configured_relative_files:
        raise RuntimeError(f"{source.name} 输入文件顺序发生变化")
    writers = {split: ShardWriter(source_name=source.name, split=split, state=source_state["writers"][split], vocab_size=vocab_size) for split in ("train", "validation")}
    eot_id = special_token_ids["<|endoftext|>"]
    reserved_ids = set(special_token_ids.values())
    committer = CheckpointCommitter(checkpoint)
    start_file_index = int(source_state["input_file_index"])
    if start_file_index > len(files):
        raise RuntimeError(f"{source.name} 的文件 cursor 越界")
    print()
    print("=" * 72)
    print(f"开始来源：{source.name}")
    print(f"输入文件数：{len(files):,}")
    print(f"恢复文件位置：{start_file_index + 1}/{len(files)}")
    for file_index in range(start_file_index, len(files)):
        file_path = files[file_index]
        if file_index == start_file_index:
            cursor = source_state["cursor"]
        else:
            cursor = initial_cursor(source)
        source_state["input_file_index"] = file_index
        source_state["cursor"] = cursor
        print()
        print(f"[{file_index + 1}/{len(files)}] {file_path}")
        print(f"文件大小：{format_bytes(file_path.stat().st_size)}")
        print(f"恢复 cursor：{cursor}")
        pending_records: list[PreparedRecord] = []
        pending_utf8_bytes = 0

        def flush_pending() -> None:
            nonlocal pending_records
            nonlocal pending_utf8_bytes
            if not pending_records:
                return
            processed_bytes = (
                encode_regular_batch(records=pending_records, raw_tokenizer=raw_tokenizer, writers=writers, eot_id=eot_id, reserved_ids=reserved_ids, source_state=source_state)
            )
            committer.add_processed_bytes(processed_bytes)
            pending_records = []
            pending_utf8_bytes = 0
            if committer.should_commit():
                train_tokens = source_state["writers"]["train"]["total_tokens"]
                validation_tokens = source_state["writers"]["validation"]["total_tokens"]
                progress = (
                    f"{source.name}：{source_state['processed_documents']:,} 篇，"
                    f"{format_bytes(source_state['processed_utf8_bytes'])} 原文，"
                    f"{train_tokens + validation_tokens:,} tokens"
                )
                committer.commit(writers, progress_label=progress)
        records = iter_input_file_records(source, file_path, cursor)
        for input_record in records:
            record = prepare_record(input_record)
            is_long_document = record.utf8_bytes >= LONG_DOCUMENT_UTF8_BYTES
            if is_long_document:
                flush_pending()
                processed_bytes = (encode_long_document(record=record, raw_tokenizer=raw_tokenizer, writers=writers, eot_id=eot_id, reserved_ids=reserved_ids, source_state=source_state))
                committer.add_processed_bytes(processed_bytes)
                if committer.should_commit():
                    train_tokens = source_state["writers"]["train"]["total_tokens"]
                    validation_tokens = source_state["writers"]["validation"]["total_tokens"]
                    progress = (
                        f"{source.name}：{source_state['processed_documents']:,} 篇，"
                        f"{format_bytes(source_state['processed_utf8_bytes'])} 原文，"
                        f"{train_tokens + validation_tokens:,} tokens"
                    )
                    committer.commit(writers, progress_label=progress)
                continue
            would_exceed_documents = len(pending_records) >= MAX_BATCH_DOCUMENTS
            would_exceed_bytes = pending_records and pending_utf8_bytes + record.utf8_bytes > MAX_BATCH_UTF8_BYTES
            if would_exceed_documents or would_exceed_bytes:
                flush_pending()
            pending_records.append(record)
            pending_utf8_bytes += record.utf8_bytes
            if len(pending_records) >= MAX_BATCH_DOCUMENTS or pending_utf8_bytes >= MAX_BATCH_UTF8_BYTES:
                flush_pending()
        flush_pending()
        source_state["input_file_index"] = file_index + 1
        source_state["cursor"] = initial_cursor(source)
        progress = f"文件完成：{file_path.name}；来源累计文档：{source_state['processed_documents']:,}"
        committer.commit(writers, force=True, progress_label=progress)
    for writer in writers.values():
        writer.finalize_tail()
    source_state["completed"] = True
    source_state["input_file_index"] = len(files)
    source_state["cursor"] = initial_cursor(source)
    committer.commit(writers, force=True, progress_label=f"{source.name} 全部完成")
    for writer in writers.values():
        writer.close()

def build_manifest(checkpoint: dict[str, Any], tokenizer_sha256: str, special_token_ids: dict[str, int], vocab_size: int) -> dict[str, Any]:
    manifest_sources: dict[str, dict[str, Any]] = {}
    total_tokens = 0
    total_documents = 0
    for source in SOURCES:
        state = checkpoint["sources"][source.name]
        source_entry: dict[str, Any] = {
            "completed": state["completed"],
            "processed_documents": state["processed_documents"],
            "processed_utf8_bytes": state["processed_utf8_bytes"],
            "splits": {},
        }
        for split in ("train", "validation"):
            writer_state = state["writers"][split]
            if writer_state["tokens_in_partial"] != 0:
                raise RuntimeError(f"{source.name}/{split} 仍有未完成 partial")
            split_entry = {"tokens": writer_state["total_tokens"], "documents": writer_state["total_documents"], "shards": writer_state["completed_shards"]}
            source_entry["splits"][split] = split_entry
            total_tokens += writer_state["total_tokens"]
            total_documents += writer_state["total_documents"]
        manifest_sources[source.name] = source_entry
    return {
        "format_version": FORMAT_VERSION,
        "created_at": now_iso(),
        "tokenizer": {"path": relative_to_project(TOKENIZER_PATH), "sha256": tokenizer_sha256, "vocab_size": vocab_size, "special_token_ids": special_token_ids},
        "storage": {
            "dtype": "uint16",
            "numpy_dtype": TOKEN_DTYPE.str,
            "endianness": "little",
            "bytes_per_token": TOKEN_BYTES,
            "shard_size_tokens": SHARD_SIZE_TOKENS,
            "document_separator": "<|endoftext|>",
        },
        "split_policy": {
            "method": "blake2b-64 of normalized UTF-8 text",
            "buckets": VALIDATION_BUCKETS,
            "validation_threshold": VALIDATION_THRESHOLD,
            "validation_ratio": VALIDATION_THRESHOLD / VALIDATION_BUCKETS,
        },
        "normalization": {"crlf_to_lf": True, "cr_to_lf": True, "remove_null_bytes": True, "unicode_normalization": None},
        "raw_special_token_policy": "added_tokens removed while encoding ordinary source text",
        "recommended_training_mixture": RECOMMENDED_TRAINING_MIXTURE,
        "totals": {"tokens": total_tokens, "documents": total_documents, "binary_bytes": total_tokens * TOKEN_BYTES},
        "sources": manifest_sources,
        "input_files": checkpoint["runtime_config"]["source_files"],
        "config_fingerprint": checkpoint["config_fingerprint"],
    }

def scan_and_verify_shard(path: Path, expected_sha256: str, expected_tokens: int, expected_documents: int, vocab_size: int, eot_id: int) -> None:
    expected_bytes = expected_tokens * TOKEN_BYTES
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(f"分片大小错误：{path}\n期望：{expected_bytes}\n实际：{actual_bytes}")
    if actual_bytes % TOKEN_BYTES != 0:
        raise RuntimeError(f"分片不是完整 uint16：{path}")
    digest = hashlib.sha256()
    maximum_token_id = -1
    eot_count = 0
    scanned_tokens = 0
    with path.open("rb") as file:
        while True:
            chunk = file.read(VERIFY_CHUNK_BYTES)
            if not chunk:
                break
            if len(chunk) % TOKEN_BYTES != 0:
                raise RuntimeError(f"读取到不完整 uint16：{path}")
            digest.update(chunk)
            tokens = np.frombuffer(chunk, dtype=TOKEN_DTYPE)
            if tokens.size:
                maximum_token_id = max(maximum_token_id, int(tokens.max()))
                eot_count += int(np.count_nonzero(tokens == eot_id))
                scanned_tokens += int(tokens.size)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"SHA256 不一致：{path}")
    if scanned_tokens != expected_tokens:
        raise RuntimeError(f"Token 数量不一致：{path}")
    if maximum_token_id >= vocab_size:
        raise RuntimeError(f"发现越界 token ID：{path}\n{maximum_token_id} >= {vocab_size}")
    if eot_count != expected_documents:
        raise RuntimeError(f"EOT 数量与文档数不一致：{path}\nEOT：{eot_count}\n文档：{expected_documents}")

def verify_dataset() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"没有找到 manifest：{MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current_tokenizer_sha256 = sha256_file(TOKENIZER_PATH)
    expected_tokenizer_sha256 = manifest["tokenizer"]["sha256"]
    if current_tokenizer_sha256 != expected_tokenizer_sha256:
        raise RuntimeError("Tokenizer SHA256 与 manifest 不一致")
    vocab_size = int(manifest["tokenizer"]["vocab_size"])
    eot_id = int(manifest["tokenizer"]["special_token_ids"]["<|endoftext|>"])
    partial_files = list(OUTPUT_ROOT.rglob("*.partial"))
    if partial_files:
        preview = "\n".join(f"- {path}" for path in partial_files[:10])
        raise RuntimeError(f"完整数据集中仍有 partial 文件：\n{preview}")
    verified_tokens = 0
    verified_documents = 0
    verified_shards = 0
    print()
    print("=" * 72)
    print("开始完整验证")
    for source_name, source_entry in manifest["sources"].items():
        if not source_entry["completed"]:
            raise RuntimeError(f"{source_name} 尚未完成")
        for split, split_entry in source_entry["splits"].items():
            split_tokens = 0
            split_documents = 0
            for shard in split_entry["shards"]:
                path = OUTPUT_ROOT / shard["path"]
                if not path.exists():
                    raise FileNotFoundError(f"分片不存在：{path}")
                print(f"验证：{source_name}/{split}/{path.name}")
                scan_and_verify_shard(
                    path=path,
                    expected_sha256=shard["sha256"],
                    expected_tokens=int(shard["tokens"]),
                    expected_documents=int(shard["documents"]),
                    vocab_size=vocab_size,
                    eot_id=eot_id,
                )
                split_tokens += int(shard["tokens"])
                split_documents += int(shard["documents"])
                verified_shards += 1
            if split_tokens != split_entry["tokens"]:
                raise RuntimeError(f"{source_name}/{split} token 汇总错误")
            if split_documents != split_entry["documents"]:
                raise RuntimeError(f"{source_name}/{split} 文档汇总错误")
            verified_tokens += split_tokens
            verified_documents += split_documents
    if verified_tokens != manifest["totals"]["tokens"]:
        raise RuntimeError("全局 token 总数错误")
    if verified_documents != manifest["totals"]["documents"]:
        raise RuntimeError("全局文档总数错误")
    print()
    print("=" * 72)
    print("完整验证通过")
    print(f"分片数：{verified_shards:,}")
    print(f"文档数：{verified_documents:,}")
    print(f"Token 数：{verified_tokens:,}")
    print(f"二进制数据大小：{format_bytes(verified_tokens * TOKEN_BYTES)}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 XiangLM 全量预训练数据编码为 uint16 token shards")
    parser.add_argument("--verify-only", action="store_true", help="只验证已经生成的数据")
    parser.add_argument("--no-verify", action="store_true", help="编码完成后暂时跳过全量读取验证")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    print("=" * 72)
    print("XiangLM 全量 Tokenize")
    print("=" * 72)
    print(f"项目目录：{PROJECT_ROOT}")
    print(f"Tokenizer：{TOKENIZER_PATH}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"分片大小：{SHARD_SIZE_TOKENS:,} tokens，{format_bytes(SHARD_SIZE_TOKENS * TOKEN_BYTES)}")
    print("验证集比例：{VALIDATION_THRESHOLD / VALIDATION_BUCKETS:.2%}")
    if args.verify_only:
        verify_dataset()
        return
    (control_tokenizer, raw_tokenizer, special_token_ids) = load_tokenizers()
    vocab_size = control_tokenizer.get_vocab_size()
    tokenizer_sha256 = sha256_file(TOKENIZER_PATH)
    runtime_config = build_runtime_config(tokenizer_sha256)
    config_fingerprint = sha256_json(runtime_config)
    checkpoint = load_or_create_checkpoint(runtime_config, config_fingerprint)
    print(f"Tokenizer SHA256：{tokenizer_sha256}")
    print(f"配置指纹：{config_fingerprint}")
    print(f"当前磁盘剩余：{format_bytes(shutil.disk_usage(OUTPUT_ROOT).free)}")
    if checkpoint.get("status") == "complete":
        print("Checkpoint 显示全量编码已经完成。")
        if not args.no_verify:
            verify_dataset()
        return
    for source in SOURCES:
        process_source(source=source, checkpoint=checkpoint, raw_tokenizer=raw_tokenizer, vocab_size=vocab_size, special_token_ids=special_token_ids)
    manifest = build_manifest(checkpoint=checkpoint, tokenizer_sha256=tokenizer_sha256, special_token_ids=special_token_ids, vocab_size=vocab_size)
    atomic_write_json(MANIFEST_PATH, manifest)
    checkpoint["status"] = "complete"
    checkpoint["completed_at"] = now_iso()
    checkpoint["updated_at"] = now_iso()
    checkpoint["manifest_path"] = relative_to_project(MANIFEST_PATH)
    atomic_write_json(CHECKPOINT_PATH, checkpoint)
    print()
    print("=" * 72)
    print("全量编码完成")
    print(f"Manifest：{MANIFEST_PATH}")
    print(f"总文档数：{manifest['totals']['documents']:,}")
    print(f"总 Token 数：{manifest['totals']['tokens']:,}")
    print(f"Token 数据大小：{format_bytes(manifest['totals']['binary_bytes'])}")
    if not args.no_verify:
        verify_dataset()
    else:
        print("已按照 --no-verify 跳过完整验证。")
        print("稍后可以运行：py -u -m src.tokenize_data --verify-only")

if __name__ == "__main__":
    main()