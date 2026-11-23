from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import gzip
import json
import os
import random
import shutil
import time
import pyarrow.parquet as pq
from modelscope_hub import HubApi
from concurrent.futures import ThreadPoolExecutor, as_completed

GIB = 1024**3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DOWNLOAD_ROOT = DATA_ROOT / "downloads"
MANIFEST_ROOT = DATA_ROOT / "manifests"
LOG_ROOT = DATA_ROOT / "logs"

PLAN_PATH = MANIFEST_ROOT / "pretraining_download_plan.json"
FINAL_MANIFEST_PATH = MANIFEST_ROOT / "pretraining_download_manifest.json"

MAX_WORKERS = 3
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 30
FREE_SPACE_RESERVE_GIB = 10
RANDOM_SEED = 20260822


@dataclass(frozen=True)
class SourceSpec:
    name: str
    category: str
    repo_id: str
    local_name: str
    target_gib: float
    license_name: str
    source_url: str
    prefixes: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()

    @property
    def local_dir(self) -> Path:
        return DOWNLOAD_ROOT / self.local_name

    @property
    def target_bytes(self) -> int:
        return int(self.target_gib * GIB)


SOURCES = [
    SourceSpec(
        name="FineWeb 2 Mandarin Chinese",
        category="chinese_web",
        repo_id="HuggingFaceFW/fineweb-2",
        local_name="fineweb-2-cmn-hani",
        target_gib=36,
        license_name="ODC-By 1.0",
        source_url="https://modelscope.cn/datasets/HuggingFaceFW/fineweb-2",
        prefixes=("data/cmn_Hani/train/",),
        suffixes=(".parquet",),
    ),
    SourceSpec(
        name="Chinese FineWeb Edu V2",
        category="chinese_education",
        repo_id="opencsg/chinese-fineweb-edu-v2",
        local_name="chinese-fineweb-edu-v2",
        target_gib=16,
        license_name="OpenCSG Community License + Apache 2.0",
        source_url="https://modelscope.cn/datasets/opencsg/chinese-fineweb-edu-v2",
        prefixes=("data/",),
        suffixes=(".parquet",),
    ),
    SourceSpec(
        name="FineWeb Edu English",
        category="english_web_education",
        repo_id="HuggingFaceFW/fineweb-edu",
        local_name="fineweb-edu-english",
        target_gib=5,
        license_name="ODC-By 1.0",
        source_url="https://modelscope.cn/datasets/HuggingFaceFW/fineweb-edu",
        prefixes=("sample/10BT/", "data/sample/10BT/"),
        suffixes=(".parquet",),
    ),
    SourceSpec(
        name="Dolma v1.7 StarCoder",
        category="code",
        repo_id="modelscope/dolma",
        local_name="dolma-v1_7-starcoder",
        target_gib=3,
        license_name="ODC-By 1.0 and source licenses",
        source_url="https://modelscope.cn/datasets/modelscope/dolma",
        prefixes=("data/v1_7/",),
        contains=("starcoder",),
        suffixes=(".json.gz", ".jsonl.gz"),
    ),
]


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise RuntimeError("无法格式化文件大小")


def ensure_directories() -> None:
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        source.local_dir.mkdir(parents=True, exist_ok=True)


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def matches_source(path: str, source: SourceSpec) -> bool:
    normalized = normalize_path(path)
    lowercase = normalized.lower()
    if source.prefixes:
        if not any(normalized.startswith(prefix) for prefix in source.prefixes):
            return False
    if source.contains:
        if not all(required.lower() in lowercase for required in source.contains):
            return False
    if source.suffixes:
        if not any(lowercase.endswith(suffix.lower()) for suffix in source.suffixes):
            return False
    if "/test/" in lowercase:
        return False
    return True


def list_candidate_files(api: HubApi, source: SourceSpec) -> list[dict]:
    print()
    print("=" * 72)
    print(f"读取仓库文件列表：{source.name}")
    print(f"Repo：{source.repo_id}")
    roots = tuple(dict.fromkeys(prefix.rstrip("/") for prefix in source.prefixes))
    if not roots:
        roots = ("/",)
    entries_by_path: dict[str, dict] = {}
    for root in roots:
        print(f"分页读取目录：{root}")
        raw_entries = api.legacy.list_dataset_files_paginated(
            repo_id=source.repo_id, revision="master", page_size=25, root_path=root
        )
        print(f"目录返回条目数：{len(raw_entries):,}")
        for entry in raw_entries:
            entry_type = entry.get("Type") or entry.get("type") or "blob"
            if entry_type == "tree":
                continue
            path = entry.get("Path") or entry.get("path") or entry.get("Name") or entry.get("name")
            raw_size = entry.get("Size") or entry.get("size")
            if not path or raw_size is None:
                continue
            try:
                size_bytes = int(raw_size)
            except (TypeError, ValueError):
                continue
            if size_bytes <= 0:
                continue
            normalized_path = normalize_path(str(path))
            entries_by_path[normalized_path] = {
                "path": normalized_path,
                "size_bytes": size_bytes,
                "sha256": entry.get("Sha256") or entry.get("sha256"),
            }
    candidates = [item for item in entries_by_path.values() if matches_source(item["path"], source)]
    candidates.sort(key=lambda item: item["path"])
    if not candidates:
        sample_paths = list(entries_by_path.keys())[:30]
        raise RuntimeError(
            f"{source.name}没有找到匹配文件。\n仓库前30个路径：\n" + "\n".join(sample_paths)
        )
    total_available = sum(item["size_bytes"] for item in candidates)
    print(f"候选文件数：{len(candidates):,}")
    print(f"候选文件总大小：{format_bytes(total_available)}")
    if total_available < source.target_bytes:
        raise RuntimeError(
            f"{source.name}候选文件容量不足：需要{format_bytes(source.target_bytes)}，只有{format_bytes(total_available)}。"
        )
    return candidates


def is_complete_local_file(source: SourceSpec, item: dict) -> bool:
    local_path = source.local_dir / Path(item["path"])
    return local_path.is_file() and local_path.stat().st_size == item["size_bytes"]


def select_files(source: SourceSpec, candidates: list[dict], seed: int) -> list[dict]:
    completed = [item for item in candidates if is_complete_local_file(source, item)]
    remaining = [item for item in candidates if not is_complete_local_file(source, item)]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    completed.sort(key=lambda item: item["path"])
    ordered = completed + remaining
    selected = []
    selected_size = 0
    for item in ordered:
        selected.append(item)
        selected_size += item["size_bytes"]
        if selected_size >= source.target_bytes:
            break
    print(f"目标大小：{source.target_gib:.1f} GiB")
    print(f"选择文件数：{len(selected):,}")
    print(f"预计下载集大小：{format_bytes(selected_size)}")
    print(f"已存在完整文件：{sum(is_complete_local_file(source, item) for item in selected)}")
    return selected


def build_download_plan(api: HubApi) -> list[dict]:
    plan = []
    for index, source in enumerate(SOURCES):
        candidates = list_candidate_files(api, source)
        selected = select_files(source, candidates, seed=RANDOM_SEED + index)
        plan.append({"source": source, "files": selected})
    return plan


def serialize_plan(plan: list[dict]) -> dict:
    sources = []
    for item in plan:
        source: SourceSpec = item["source"]
        selected_files: list[dict] = item["files"]
        selected_size = sum(file["size_bytes"] for file in selected_files)
        existing_size = sum(
            file["size_bytes"] for file in selected_files if is_complete_local_file(source, file)
        )
        sources.append(
            {
                "name": source.name,
                "category": source.category,
                "repo_id": source.repo_id,
                "source_url": source.source_url,
                "license": source.license_name,
                "local_dir": str(source.local_dir),
                "target_gib": source.target_gib,
                "selected_file_count": len(selected_files),
                "selected_size_bytes": selected_size,
                "selected_size_human": format_bytes(selected_size),
                "already_complete_bytes": existing_size,
                "already_complete_human": format_bytes(existing_size),
                "files": selected_files,
            }
        )
    total_size = sum(source["selected_size_bytes"] for source in sources)
    existing_size = sum(source["already_complete_bytes"] for source in sources)
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "download_root": str(DOWNLOAD_ROOT),
        "requested_total_gib": sum(source.target_gib for source in SOURCES),
        "selected_total_bytes": total_size,
        "selected_total_human": format_bytes(total_size),
        "already_complete_bytes": existing_size,
        "already_complete_human": format_bytes(existing_size),
        "remaining_bytes": total_size - existing_size,
        "remaining_human": format_bytes(total_size - existing_size),
        "sources": sources,
    }


def save_json(path: Path, content: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def check_disk_space(plan_dict: dict) -> None:
    free_bytes = shutil.disk_usage(DATA_ROOT).free
    remaining_bytes = plan_dict["remaining_bytes"]
    required_bytes = remaining_bytes + FREE_SPACE_RESERVE_GIB * GIB
    print()
    print("=" * 72)
    print("磁盘检查")
    print(f"剩余空间：{format_bytes(free_bytes)}")
    print(f"尚需下载：{format_bytes(remaining_bytes)}")
    print(f"安全预留：{FREE_SPACE_RESERVE_GIB} GiB")
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"磁盘空间不足。至少还需要：{format_bytes(required_bytes)}，当前只有：{format_bytes(free_bytes)}。"
        )


def download_selected_file(source: SourceSpec, item: dict) -> str:
    if is_complete_local_file(source, item):
        return item["path"]
    worker_api = HubApi()
    worker_api.download_file(
        repo_id=source.repo_id,
        repo_type="dataset",
        file_path=item["path"],
        local_dir=str(source.local_dir),
        expected_sha256=item.get("sha256") or None,
    )
    if not is_complete_local_file(source, item):
        local_path = source.local_dir / Path(item["path"])
        actual_size = local_path.stat().st_size if local_path.exists() else 0
        raise RuntimeError(
            f"文件下载后大小不正确：{item['path']}，预期{item['size_bytes']}字节，实际{actual_size}字节。"
        )
    return item["path"]


def download_one_source(api: HubApi, source: SourceSpec, files: list[dict]) -> dict:
    del api
    complete_before = sum(is_complete_local_file(source, item) for item in files)
    print()
    print("=" * 72)
    print(f"开始下载：{source.name}")
    print(f"Repo：{source.repo_id}")
    print(f"目录：{source.local_dir}")
    print(f"文件数：{len(files)}")
    print(f"已有完整文件：{complete_before}")
    print("计划总大小：" + format_bytes(sum(item["size_bytes"] for item in files)))
    source.local_dir.mkdir(parents=True, exist_ok=True)
    pending = [item for item in files if not is_complete_local_file(source, item)]
    start_time = time.perf_counter()
    last_errors: list[tuple[dict, Exception]] = []
    for attempt in range(1, MAX_RETRIES + 1):
        if not pending:
            break
        print()
        print(f"下载尝试 {attempt}/{MAX_RETRIES}")
        print(f"本轮待下载文件：{len(pending)}")
        failed: list[tuple[dict, Exception]] = []
        worker_count = min(MAX_WORKERS, len(pending))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_item = {
                executor.submit(download_selected_file, source, item): item for item in pending
            }
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    finished_path = future.result()
                    print(f"完成：{finished_path}")
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    failed.append((item, error))
                    print(f"失败：{item['path']}：{type(error).__name__}: {error}")
        pending = [item for item, _ in failed]
        last_errors = failed
        if pending and attempt < MAX_RETRIES:
            wait_seconds = RETRY_WAIT_SECONDS * attempt
            print(f"{len(pending)}个文件将在{wait_seconds}秒后重试；已完成文件会被复用。")
            time.sleep(wait_seconds)
    elapsed = time.perf_counter() - start_time
    if pending:
        error_message = "; ".join(
            (f"{item['path']}: {type(error).__name__}: {error}") for item, error in last_errors
        )
        return {
            "status": "download_failed",
            "error_type": "PartialDownloadError",
            "error_message": error_message,
            "elapsed_seconds": elapsed,
        }
    invalid_files = []
    for item in files:
        local_path = source.local_dir / Path(item["path"])
        if not local_path.is_file():
            invalid_files.append({"path": item["path"], "reason": "missing"})
            continue
        actual_size = local_path.stat().st_size
        if actual_size != item["size_bytes"]:
            invalid_files.append(
                {
                    "path": item["path"],
                    "reason": "size_mismatch",
                    "expected_size": item["size_bytes"],
                    "actual_size": actual_size,
                }
            )
    if invalid_files:
        return {
            "status": "validation_failed",
            "invalid_files": invalid_files,
            "elapsed_seconds": elapsed,
        }
    print(f"{source.name}完成，耗时{elapsed / 3600:.2f}小时。")
    return {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }


def inspect_parquet_sample(file_path: Path) -> dict:
    parquet_file = pq.ParquetFile(file_path)
    first_batch = next(parquet_file.iter_batches(batch_size=1))
    first_row = first_batch.to_pylist()[0]
    preview = {}
    for key, value in first_row.items():
        text = str(value)
        if len(text) > 300:
            text = text[:300] + "..."
        preview[key] = text
    return {
        "path": str(file_path),
        "num_rows": parquet_file.metadata.num_rows,
        "num_row_groups": parquet_file.metadata.num_row_groups,
        "schema": str(parquet_file.schema_arrow),
        "first_row_preview": preview,
    }


def inspect_gzip_sample(file_path: Path) -> dict:
    with gzip.open(file_path, mode="rt", encoding="utf-8", errors="replace") as file:
        first_line = file.readline().strip()
    if len(first_line) > 1000:
        first_line = first_line[:1000] + "..."
    parsed: Any
    try:
        parsed = json.loads(first_line)
    except json.JSONDecodeError:
        parsed = first_line
    return {"path": str(file_path), "first_record_preview": parsed}


def inspect_one_file(source: SourceSpec, files: list[dict]) -> dict | None:
    if not files:
        return None
    first_file = source.local_dir / Path(files[0]["path"])
    if not first_file.exists():
        return None
    lowercase = first_file.name.lower()
    if lowercase.endswith(".parquet"):
        return inspect_parquet_sample(first_file)
    if lowercase.endswith(".json.gz") or lowercase.endswith(".jsonl.gz"):
        return inspect_gzip_sample(first_file)
    return {"path": str(first_file), "note": "未配置样本读取器"}


def build_final_manifest(plan_dict: dict, results: list[dict]) -> dict:
    final_sources = []
    for planned_source, result in zip(plan_dict["sources"], results, strict=True):
        final_source = dict(planned_source)
        final_source["result"] = result
        final_sources.append(final_source)
    successful_bytes = sum(
        source["selected_size_bytes"]
        for source in final_sources
        if source["result"]["status"] == "complete"
    )
    all_complete = all(source["result"]["status"] == "complete" for source in final_sources)
    return {
        "finished_at": datetime.now().astimezone().isoformat(),
        "all_complete": all_complete,
        "successful_bytes": successful_bytes,
        "successful_human": format_bytes(successful_bytes),
        "sources": final_sources,
    }


def main() -> None:
    ensure_directories()
    print("=" * 72)
    print("Scratch Chinese LM 数据下载")
    print("=" * 72)
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"数据目录：{DATA_ROOT}")
    print(f"目标下载量：{sum(source.target_gib for source in SOURCES)} GiB")
    print("实际可能因完整分片而略高于目标。")
    api = HubApi()
    plan = build_download_plan(api)
    plan_dict = serialize_plan(plan)
    save_json(PLAN_PATH, plan_dict)
    print()
    print("=" * 72)
    print("下载计划已生成")
    print(f"计划文件：{PLAN_PATH}")
    print(f"选择总大小：{plan_dict['selected_total_human']}")
    print(f"已有完整数据：{plan_dict['already_complete_human']}")
    print(f"尚需下载：{plan_dict['remaining_human']}")
    check_disk_space(plan_dict)
    results = []
    for item in plan:
        source: SourceSpec = item["source"]
        files: list[dict] = item["files"]
        try:
            result = download_one_source(api, source, files)
            if result["status"] == "complete":
                result["sample_inspection"] = inspect_one_file(source, files)
        except KeyboardInterrupt:
            print()
            print("用户中断。重新运行即可断点续传。")
            raise SystemExit(130)
        except Exception as error:
            result = {
                "status": "unexpected_error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        results.append(result)
        partial_manifest = build_final_manifest(
            plan_dict,
            results + [{"status": "not_started"} for _ in range(len(plan) - len(results))],
        )
        save_json(FINAL_MANIFEST_PATH, partial_manifest)
    final_manifest = build_final_manifest(plan_dict, results)
    save_json(FINAL_MANIFEST_PATH, final_manifest)
    print()
    print("=" * 72)
    print("下载任务结束")
    print(f"成功数据大小：{final_manifest['successful_human']}")
    print(f"Manifest：{FINAL_MANIFEST_PATH}")
    for source, result in zip(SOURCES, results, strict=True):
        print(f"{source.category}: {result['status']}")
    if not final_manifest["all_complete"]:
        print()
        print("部分来源未完成。重新运行相同命令即可继续。")
        raise SystemExit(1)
    print()
    print("所有来源下载并验证完成。")


if __name__ == "__main__":
    main()
