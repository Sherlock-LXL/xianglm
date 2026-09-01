from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
import json
import os
import shutil
import time
import pyarrow.parquet as pq
from modelscope_hub import HubApi

GIB = 1024**3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
POST_TRAINING_ROOT = DATA_ROOT / "post_training"
RAW_ROOT = POST_TRAINING_ROOT / "raw" / "sft"
PROCESSED_ROOT = POST_TRAINING_ROOT / "processed" / "sft"
TOKENIZED_ROOT = POST_TRAINING_ROOT / "tokenized" / "sft"
MANIFEST_ROOT = DATA_ROOT / "manifests" / "post_training"

PLAN_PATH = MANIFEST_ROOT / "sft_download_plan.json"
FINAL_MANIFEST_PATH = MANIFEST_ROOT / "sft_download_manifest.json"

DEFAULT_WORKERS = 3
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 20
FREE_SPACE_RESERVE_GIB = 10
REVISION = "master"

METADATA_FILENAMES = frozenset({"README.md", "dataset_infos.json"})

@dataclass(frozen=True)
class SourceSpec:
    key: str
    name: str
    repo_id: str
    local_name: str
    root_paths: tuple[str, ...]
    exact_files: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()
    excluded_prefixes: tuple[str, ...] = ()
    intended_role: str = ""
    planned_weight: float = 0.0
    planned_examples: int = 0
    license_name: str = ""
    source_url: str = ""
    notes: str = ""

    @property
    def local_dir(self) -> Path:
        return RAW_ROOT / self.local_name

SOURCES = (
    SourceSpec(
        key="magpie",
        name="Magpie Qwen2 Pro 200K Chinese",
        repo_id="AI-ModelScope/Magpie-Qwen2-Pro-200K-Chinese",
        local_name="magpie-qwen2-pro-200k-chinese",
        root_paths=("/",),
        prefixes=("data/",),
        suffixes=(".parquet",),
        intended_role="通用中文单轮指令遵循",
        planned_weight=0.35,
        planned_examples=42_000,
        license_name="以数据集卡和上游 Magpie/Qwen 条款为准",
        source_url="https://modelscope.cn/datasets/AI-ModelScope/Magpie-Qwen2-Pro-200K-Chinese",
        notes="下载完整 200K 原始集；混合阶段再进行质量、长度和安全筛选。",
    ),
    SourceSpec(
        key="fineweb_edu_qa",
        name="Chinese FineWeb Edu V2.2 Pure QA",
        repo_id="opencsg/Fineweb-Edu-Chinese-V2.2",
        local_name="fineweb-edu-chinese-v2_2-sft-qa",
        root_paths=("sft",),
        prefixes=("sft/cleaned/",),
        suffixes=(".jsonl",),
        intended_role="事实锚定的教育、科普和解释型问答",
        planned_weight=0.20,
        planned_examples=24_000,
        license_name="OpenCSG Community License + Apache-2.0",
        source_url="https://modelscope.cn/datasets/opencsg/Fineweb-Edu-Chinese-V2.2",
        notes="只下载 sft/cleaned/*.jsonl 纯 QA（约 3.4 GB）；明确排除约 14.6 GB 的上下文版和 TB 级预训练目录。",
    ),
    SourceSpec(
        key="smoltalk",
        name="Chinese SmolTalk",
        repo_id="opencsg/smoltalk-chinese",
        local_name="smoltalk-chinese",
        root_paths=("/",),
        suffixes=(".parquet",),
        excluded_prefixes=("old/",),
        intended_role="多任务、多轮聊天、写作、总结和角色扮演",
        planned_weight=0.15,
        planned_examples=18_000,
        license_name="Apache-2.0；同时遵守 OpenCSG 数据集卡条款",
        source_url="https://modelscope.cn/datasets/opencsg/smoltalk-chinese",
        notes="排除 old/ 中的旧分片；混合阶段按任务类型分层抽样。",
    ),
    SourceSpec(
        key="qwen3_distill",
        name="Chinese Qwen3 235B Thinking Distill 100K",
        repo_id="Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k",
        local_name="chinese-qwen3-235b-distill-100k",
        root_paths=("/",),
        exact_files=("Chinese-Qwen3-235b-a22b-Thinking-2507-Distill-data-100K_final.jsonl",),
        intended_role="复杂任务的最终答案、数学、科学、编程和结构化写作",
        planned_weight=0.15,
        planned_examples=18_000,
        license_name="Apache-2.0；同时遵守上游模型和种子数据条款",
        source_url="https://modelscope.cn/datasets/Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k",
        notes="混合阶段默认删除长 CoT，只保留问题和 final answer，并限制答案长度。",
    ),
    SourceSpec(
        key="deepseek_r1_distill",
        name="Chinese DeepSeek R1 Distill 110K",
        repo_id="liucong/Chinese-DeepSeek-R1-Distill-data-110k",
        local_name="chinese-deepseek-r1-distill-110k",
        root_paths=("/",),
        exact_files=("distill_r1_110k.jsonl",),
        intended_role="少量精选的推理、数学、STEM 和通用复杂问答",
        planned_weight=0.10,
        planned_examples=12_000,
        license_name="Apache-2.0；同时遵守上游种子数据条款",
        source_url="https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k",
        notes="下载保留 reasoning_content、content、score 和 repo_name 的原始版；混合阶段按 score 筛选并默认只训练最终答案。",
    ),
)

SOURCE_BY_KEY = {source.key: source for source in SOURCES}

LOCAL_SOURCE_PATHS = {}

RECOMMENDED_MIXTURE = {
    "target_examples": 120_000,
    "downloaded_sources": {source.key: {"weight": source.planned_weight, "examples": source.planned_examples} for source in SOURCES},
    "note": "比例仅用于下一阶段清洗混合，不代表本脚本会从原始仓库裁样。各项相加为 120000 条左右，适合先为 200M 模型制作 SFT 数据。",
}

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()

def format_bytes(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise RuntimeError("无法格式化文件大小")

def normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")

def ensure_directories() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    TOKENIZED_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        source.local_dir.mkdir(parents=True, exist_ok=True)

def atomic_write_json(path: Path, content: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)

def entry_path(entry: dict[str, Any]) -> str | None:
    raw_path = (entry.get("Path") or entry.get("path") or entry.get("Name") or entry.get("name"))
    if not raw_path:
        return None
    return normalize_path(str(raw_path))

def entry_size(entry: dict[str, Any]) -> int | None:
    raw_size = entry.get("Size")
    if raw_size is None:
        raw_size = entry.get("size")
    try:
        size = int(raw_size)
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None

def entry_sha256(entry: dict[str, Any]) -> str | None:
    value = entry.get("Sha256") or entry.get("sha256")
    return str(value) if value else None

def entry_is_file(entry: dict[str, Any]) -> bool:
    entry_type = str(entry.get("Type") or entry.get("type") or "blob")
    return entry_type != "tree"

def matches_data_file(path: str, source: SourceSpec) -> bool:
    if source.exact_files and path not in source.exact_files:
        return False
    if source.prefixes and not any(path.startswith(prefix) for prefix in source.prefixes):
        return False
    if source.excluded_prefixes and any(path.startswith(prefix) for prefix in source.excluded_prefixes):
        return False
    lowercase = path.lower()
    if source.suffixes and not any(lowercase.endswith(suffix.lower()) for suffix in source.suffixes):
        return False
    return True

def matches_metadata_file(path: str) -> bool:
    return "/" not in path and path in METADATA_FILENAMES

def list_source_files(api: HubApi, source: SourceSpec) -> list[dict[str, Any]]:
    print()
    print("=" * 72)
    print(f"读取仓库：{source.name}")
    print(f"Repo：{source.repo_id}")
    entries_by_path: dict[str, dict[str, Any]] = {}
    for root_path in source.root_paths:
        print(f"读取目录：{root_path}")
        entries = api.legacy.list_dataset_files_paginated(repo_id=source.repo_id, revision=REVISION, page_size=1000, root_path=root_path)
        print(f"目录返回：{len(entries):,} 项")
        for entry in entries:
            if not entry_is_file(entry):
                continue
            path = entry_path(entry)
            size_bytes = entry_size(entry)
            if path is None or size_bytes is None:
                continue
            entries_by_path[path] = {"path": path, "size_bytes": size_bytes, "sha256": entry_sha256(entry), "revision": entry.get("Revision") or entry.get("revision")}
    selected = [item for path, item in entries_by_path.items() if matches_data_file(path, source) or matches_metadata_file(path)]
    selected.sort(key=lambda item: item["path"])
    data_files = [item for item in selected if matches_data_file(item["path"], source)]
    if not data_files:
        available_preview = "\n".join(sorted(entries_by_path)[:30])
        raise RuntimeError(f"{source.name} 没有匹配到数据文件。\n仓库路径预览：\n{available_preview}")
    data_size = sum(item["size_bytes"] for item in data_files)
    metadata_count = len(selected) - len(data_files)
    print(f"数据文件：{len(data_files):,}")
    print(f"元数据文件：{metadata_count:,}")
    print(f"数据大小：{format_bytes(data_size)}")
    return selected

def local_file_path(source: SourceSpec, item: dict[str, Any]) -> Path:
    return source.local_dir / Path(item["path"])

def is_complete_local_file(source: SourceSpec, item: dict[str, Any]) -> bool:
    path = local_file_path(source, item)
    return path.is_file() and path.stat().st_size == item["size_bytes"]

def serialize_source_plan(source: SourceSpec, files: list[dict[str, Any]]) -> dict[str, Any]:
    total_size = sum(item["size_bytes"] for item in files)
    complete_size = sum(item["size_bytes"] for item in files if is_complete_local_file(source, item))
    return {
        "key": source.key,
        "name": source.name,
        "repo_id": source.repo_id,
        "revision": REVISION,
        "source_url": source.source_url,
        "license": source.license_name,
        "intended_role": source.intended_role,
        "notes": source.notes,
        "local_dir": str(source.local_dir),
        "file_count": len(files),
        "selected_size_bytes": total_size,
        "selected_size_human": format_bytes(total_size),
        "already_complete_bytes": complete_size,
        "already_complete_human": format_bytes(complete_size),
        "remaining_bytes": total_size - complete_size,
        "remaining_human": format_bytes(total_size - complete_size),
        "files": files,
    }

def inspect_local_sources() -> dict[str, dict[str, Any]]:
    result = {}
    for name, path in LOCAL_SOURCE_PATHS.items():
        exists = path.is_file()
        result[name] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else 0,
            "size_human": format_bytes(path.stat().st_size) if exists else "0.00 B",
            "downloaded_by_this_script": False,
        }
    return result

def build_plan(api: HubApi, selected_sources: tuple[SourceSpec, ...]) -> tuple[list[tuple[SourceSpec, list[dict[str, Any]]]], dict[str, Any]]:
    runtime_plan = []
    serialized_sources = []
    for source in selected_sources:
        files = list_source_files(api, source)
        runtime_plan.append((source, files))
        serialized_sources.append(serialize_source_plan(source, files))
    total_size = sum(item["selected_size_bytes"] for item in serialized_sources)
    complete_size = sum(item["already_complete_bytes"] for item in serialized_sources)
    serialized_plan = {
        "format_version": 1,
        "created_at": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "raw_root": str(RAW_ROOT),
        "processed_root": str(PROCESSED_ROOT),
        "tokenized_root": str(TOKENIZED_ROOT),
        "selected_source_keys": [source.key for source in selected_sources],
        "selected_size_bytes": total_size,
        "selected_size_human": format_bytes(total_size),
        "already_complete_bytes": complete_size,
        "already_complete_human": format_bytes(complete_size),
        "remaining_bytes": total_size - complete_size,
        "remaining_human": format_bytes(total_size - complete_size),
        "sources": serialized_sources,
        "local_sources": inspect_local_sources(),
        "recommended_mixture": RECOMMENDED_MIXTURE,
        "preference_data_policy": "UltraFeedback-Chinese 不属于第一阶段 SFT 下载范围；完成 SFT 评测后再为 DPO 单独准备。",
    }
    return runtime_plan, serialized_plan

def check_disk_space(plan: dict[str, Any]) -> None:
    remaining_bytes = int(plan["remaining_bytes"])
    free_bytes = shutil.disk_usage(POST_TRAINING_ROOT).free
    required_bytes = remaining_bytes + FREE_SPACE_RESERVE_GIB * GIB
    print()
    print("=" * 72)
    print("磁盘检查")
    print(f"剩余空间：{format_bytes(free_bytes)}")
    print(f"尚需下载：{format_bytes(remaining_bytes)}")
    print(f"安全预留：{FREE_SPACE_RESERVE_GIB} GiB")
    if free_bytes < required_bytes:
        raise RuntimeError(f"磁盘空间不足。至少需要 {format_bytes(required_bytes)}，当前只有 {format_bytes(free_bytes)}。")

def download_file(source: SourceSpec, item: dict[str, Any]) -> str:
    if is_complete_local_file(source, item):
        return item["path"]
    worker_api = HubApi()
    worker_api.download_file(repo_id=source.repo_id, repo_type="dataset", file_path=item["path"], revision=REVISION, local_dir=str(source.local_dir), expected_sha256=item.get("sha256"))
    path = local_file_path(source, item)
    actual_size = path.stat().st_size if path.exists() else 0
    if actual_size != item["size_bytes"]:
        raise RuntimeError(f"{item['path']} 下载后大小错误：预期 {item['size_bytes']}，实际 {actual_size}")
    return item["path"]

def download_source(source: SourceSpec, files: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    print()
    print("=" * 72)
    print(f"开始下载：{source.name}")
    print(f"本地目录：{source.local_dir}")
    print(f"计划文件：{len(files):,}")
    print("计划大小：" + format_bytes(sum(item["size_bytes"] for item in files)))
    pending = [item for item in files if not is_complete_local_file(source, item)]
    print(f"已有完整文件：{len(files) - len(pending):,}")
    started = time.perf_counter()
    last_errors: list[tuple[dict[str, Any], Exception]] = []
    for attempt in range(1, MAX_RETRIES + 1):
        if not pending:
            break
        print()
        print(f"下载尝试 {attempt}/{MAX_RETRIES}；待下载 {len(pending):,} 个文件")
        failures: list[tuple[dict[str, Any], Exception]] = []
        worker_count = min(workers, len(pending))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_item = {executor.submit(download_file, source, item): item for item in pending}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    finished_path = future.result()
                    print(f"完成：{finished_path}")
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    failures.append((item, error))
                    print(f"失败：{item['path']}：{type(error).__name__}: {error}")
        pending = [item for item, _ in failures]
        last_errors = failures
        if pending and attempt < MAX_RETRIES:
            wait_seconds = RETRY_WAIT_SECONDS * attempt
            print(f"{len(pending)} 个文件将在 {wait_seconds} 秒后重试。")
            time.sleep(wait_seconds)
    elapsed = time.perf_counter() - started
    if pending:
        error_message = "; ".join(f"{item['path']}: {type(error).__name__}: {error}" for item, error in last_errors)
        return {"status": "download_failed", "elapsed_seconds": elapsed, "pending_files": [item["path"] for item in pending], "error_message": error_message}
    invalid_files = []
    for item in files:
        path = local_file_path(source, item)
        actual_size = path.stat().st_size if path.exists() else 0
        if actual_size != item["size_bytes"]:
            invalid_files.append({"path": item["path"], "expected_size": item["size_bytes"], "actual_size": actual_size})
    if invalid_files:
        return {"status": "validation_failed", "elapsed_seconds": elapsed, "invalid_files": invalid_files}
    return {"status": "complete", "elapsed_seconds": elapsed, "file_count": len(files), "total_size_bytes": sum(item["size_bytes"] for item in files)}

def preview_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:300] + ("..." if len(value) > 300 else "")
    if isinstance(value, list):
        return [preview_value(item) for item in value[:2]]
    if isinstance(value, dict):
        return {str(key): preview_value(item) for key, item in list(value.items())[:10]}
    return value

def inspect_parquet(path: Path) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    first_batch = next(parquet_file.iter_batches(batch_size=1))
    first_row = first_batch.to_pylist()[0]
    return {"path": str(path), "format": "parquet", "rows": parquet_file.metadata.num_rows, "columns": parquet_file.schema_arrow.names, "first_row_preview": preview_value(first_row)}

def inspect_jsonl(path: Path) -> dict[str, Any]:
    first_record: Any = None
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            first_record = json.loads(line)
            break
        else:
            raise RuntimeError(f"JSONL 文件没有有效记录：{path}")
    return {"path": str(path), "format": "jsonl", "first_record_line": line_number, "first_record_preview": preview_value(first_record)}

def inspect_source(source: SourceSpec, files: list[dict[str, Any]]) -> dict[str, Any]:
    data_items = [item for item in files if matches_data_file(item["path"], source)]
    first_path = local_file_path(source, data_items[0])
    lowercase = first_path.name.lower()
    if lowercase.endswith(".parquet"):
        return inspect_parquet(first_path)
    if lowercase.endswith(".jsonl") or lowercase.endswith(".json"):
        return inspect_jsonl(first_path)
    return {"path": str(first_path), "format": "unknown"}

def validate_existing_source(source: SourceSpec, files: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_files = []
    for item in files:
        path = local_file_path(source, item)
        actual_size = path.stat().st_size if path.exists() else 0
        if actual_size != item["size_bytes"]:
            invalid_files.append({"path": item["path"], "expected_size": item["size_bytes"], "actual_size": actual_size})
    if invalid_files:
        return {"status": "validation_failed", "invalid_files": invalid_files}
    return {"status": "complete", "file_count": len(files), "total_size_bytes": sum(item["size_bytes"] for item in files), "sample_inspection": inspect_source(source, files)}

def build_final_manifest(plan: dict[str, Any], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_entries = []
    for planned_source in plan["sources"]:
        entry = dict(planned_source)
        entry["result"] = results.get(planned_source["key"], {"status": "not_started"})
        source_entries.append(entry)
    return {
        "format_version": 1,
        "updated_at": now_iso(),
        "all_complete": all(entry["result"]["status"] == "complete" for entry in source_entries),
        "sources": source_entries,
        "local_sources": inspect_local_sources(),
        "recommended_mixture": RECOMMENDED_MIXTURE,
    }

def resolve_selected_sources(keys: list[str] | None) -> tuple[SourceSpec, ...]:
    if not keys:
        return SOURCES
    ordered_keys = list(dict.fromkeys(keys))
    return tuple(SOURCE_BY_KEY[key] for key in ordered_keys)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 XiangLM 第一阶段 SFT 原始数据，并生成可追溯 manifest")
    parser.add_argument("--source", action="append", choices=tuple(SOURCE_BY_KEY), help="只处理指定来源；可以重复传入。默认处理全部来源。")
    parser.add_argument("--plan-only", action="store_true", help="只生成计划、统计大小和检查磁盘，不下载文件。")
    parser.add_argument("--verify-only", action="store_true", help="只按远端文件大小验证已下载文件，并读取每个来源的一个样本。")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"同一来源并发下载数，默认 {DEFAULT_WORKERS}。")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers 必须在 1 到 8 之间")
    if args.plan_only and args.verify_only:
        raise ValueError("--plan-only 和 --verify-only 不能同时使用")
    selected_sources = resolve_selected_sources(args.source)
    ensure_directories()
    print("=" * 72)
    print("XiangLM 第一阶段 SFT 原始数据准备")
    print("=" * 72)
    print(f"项目目录：{PROJECT_ROOT}")
    print(f"原始数据目录：{RAW_ROOT}")
    print("来源：" + ", ".join(source.key for source in selected_sources))
    print("本阶段只下载原始文件，不清洗、不混合、不 tokenize、不开始训练。")
    api = HubApi()
    runtime_plan, plan = build_plan(api, selected_sources)
    atomic_write_json(PLAN_PATH, plan)
    print()
    print("=" * 72)
    print("下载计划已生成")
    print(f"计划文件：{PLAN_PATH}")
    print(f"选择总大小：{plan['selected_size_human']}")
    print(f"已有完整数据：{plan['already_complete_human']}")
    print(f"尚需下载：{plan['remaining_human']}")
    check_disk_space(plan)
    if args.plan_only:
        print("计划模式完成，没有下载正文数据。")
        return
    results: dict[str, dict[str, Any]] = {}
    for source, files in runtime_plan:
        try:
            if args.verify_only:
                result = validate_existing_source(source, files)
            else:
                result = download_source(source, files, args.workers)
                if result["status"] == "complete":
                    result["sample_inspection"] = inspect_source(source, files)
        except KeyboardInterrupt:
            print()
            print("检测到用户中断。已完成文件会被保留，重新运行即可断点续传。")
            partial_manifest = build_final_manifest(plan, results)
            atomic_write_json(FINAL_MANIFEST_PATH, partial_manifest)
            raise SystemExit(130)
        except Exception as error:
            result = {"status": "unexpected_error", "error_type": type(error).__name__, "error_message": str(error)}
        results[source.key] = result
        partial_manifest = build_final_manifest(plan, results)
        atomic_write_json(FINAL_MANIFEST_PATH, partial_manifest)
        print(f"{source.key}: {result['status']}")
    final_manifest = build_final_manifest(plan, results)
    atomic_write_json(FINAL_MANIFEST_PATH, final_manifest)
    print()
    print("=" * 72)
    print("SFT 原始数据准备结束")
    print(f"Manifest：{FINAL_MANIFEST_PATH}")
    for entry in final_manifest["sources"]:
        print(f"{entry['key']}: {entry['result']['status']}")
    if not final_manifest["all_complete"]:
        raise RuntimeError("仍有来源未完成。修复网络问题后重新运行即可续传。")
    print("全部来源下载并完成基础验收。下一步是统一格式、清洗和分层抽样。")

if __name__ == "__main__":
    main()