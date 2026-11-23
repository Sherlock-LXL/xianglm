from __future__ import annotations
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterator
import gzip
import json
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = PROJECT_ROOT / "data" / "downloads"


@dataclass(frozen=True)
class SourceConfig:
    name: str
    directory: Path
    kind: str
    pattern: str
    id_column: str | None


SOURCES = (
    SourceConfig(
        name="chinese_web",
        directory=DOWNLOAD_ROOT / "fineweb-2-cmn-hani",
        kind="parquet",
        pattern="*.parquet",
        id_column="id",
    ),
    SourceConfig(
        name="chinese_education",
        directory=DOWNLOAD_ROOT / "chinese-fineweb-edu-v2",
        kind="parquet",
        pattern="*.parquet",
        id_column=None,
    ),
    SourceConfig(
        name="english_education",
        directory=DOWNLOAD_ROOT / "fineweb-edu-english",
        kind="parquet",
        pattern="*.parquet",
        id_column="id",
    ),
    SourceConfig(
        name="code",
        directory=DOWNLOAD_ROOT / "dolma-v1_7-starcoder",
        kind="json_gzip",
        pattern="*.json.gz",
        id_column="id",
    ),
)


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def make_fallback_id(source: SourceConfig, file_path: Path, row_number: int) -> str:
    relative_path = file_path.relative_to(PROJECT_ROOT)
    return f"{source.name}:{relative_path.as_posix()}:{row_number}"


def iter_parquet_records(source: SourceConfig, file_path: Path) -> Iterator[dict]:
    parquet_file = pq.ParquetFile(file_path)
    available_columns = set(parquet_file.schema_arrow.names)
    if "text" not in available_columns:
        raise RuntimeError(f"{file_path} 不包含 text 列")
    columns = ["text"]
    if source.id_column is not None and source.id_column in available_columns:
        columns.append(source.id_column)
    row_number = 0
    for batch in parquet_file.iter_batches(batch_size=1024, columns=columns):
        values = batch.to_pydict()
        texts = values["text"]
        ids = None
        if source.id_column is not None:
            ids = values.get(source.id_column)
        for index, raw_text in enumerate(texts):
            current_row = row_number
            row_number += 1
            if not isinstance(raw_text, str):
                continue
            text = normalize_text(raw_text)
            if not text.strip():
                continue
            raw_id = ids[index] if ids is not None else None
            doc_id = (
                str(raw_id)
                if raw_id is not None
                else make_fallback_id(source, file_path, current_row)
            )
            yield {"doc_id": doc_id, "source": source.name, "text": text}


def iter_json_gzip_records(source: SourceConfig, file_path: Path) -> Iterator[dict]:
    with gzip.open(file_path, mode="rt", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{file_path} 第{line_number + 1}行 JSON 损坏") from error
            raw_text = row.get("text")
            if not isinstance(raw_text, str):
                continue
            text = normalize_text(raw_text)
            if not text.strip():
                continue
            raw_id = row.get("id")
            doc_id = (
                str(raw_id)
                if raw_id is not None
                else make_fallback_id(source, file_path, line_number)
            )
            yield {"doc_id": doc_id, "source": source.name, "text": text}


def find_source_files(source: SourceConfig) -> list[Path]:
    return sorted(source.directory.rglob(source.pattern))


def iter_source_records(source: SourceConfig) -> Iterator[dict]:
    files = find_source_files(source)
    for file_path in files:
        if source.kind == "parquet":
            yield from iter_parquet_records(source, file_path)
        elif source.kind == "json_gzip":
            yield from iter_json_gzip_records(source, file_path)
        else:
            raise ValueError(f"未知格式：{source.kind}")


def inspect_source(source: SourceConfig) -> None:
    files = find_source_files(source)
    print("=" * 72)
    print(f"来源：{source.name}")
    print(f"目录：{source.directory}")
    print(f"文件数：{len(files)}")
    if not files:
        raise RuntimeError(f"{source.name}没有找到文件")
    first_file = files[0]
    print(f"样例文件：{first_file}")
    if source.kind == "parquet":
        parquet_file = pq.ParquetFile(first_file)
        print("字段：" + ", ".join(parquet_file.schema_arrow.names))
        print(f"该文件文档数：{parquet_file.metadata.num_rows:,}")
    records = iter_source_records(source)
    for number, record in enumerate(islice(records, 2), start=1):
        preview = record["text"][:300].replace("\n", "\\n")
        print()
        print(f"样例{number}")
        print(f"doc_id：{record['doc_id']}")
        print(f"字符数：{len(record['text']):,}")
        print(f"正文：{preview}")
    print()


def main() -> None:
    for source in SOURCES:
        inspect_source(source)


if __name__ == "__main__":
    main()
