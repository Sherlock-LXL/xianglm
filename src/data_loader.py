from __future__ import annotations
import bisect
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import Tensor

from src.config import DataConfig


@dataclass
class TokenBatch:
    input_ids: Tensor
    labels: Tensor
    source_counts: dict[str, int]


@dataclass(frozen=True)
class ShardInfo:
    source: str
    split: str
    path: Path
    token_count: int

    def valid_window_count(self, context_length: int) -> int:
        return max(0, self.token_count - context_length)


class MemmapCache:
    def __init__(self, dtype: np.dtype, max_open_files: int = 8) -> None:
        if max_open_files <= 0:
            raise ValueError("max_open_files必须为正数")
        self.dtype = dtype
        self.max_open_files = max_open_files
        self._cache: OrderedDict[Path, np.memmap] = OrderedDict()

    def get(self, shard: ShardInfo) -> np.memmap:
        cached = self._cache.pop(shard.path, None)
        if cached is not None:
            self._cache[shard.path] = cached
            return cached
        mapped = np.memmap(shard.path, mode="r", dtype=self.dtype, shape=(shard.token_count,))
        self._cache[shard.path] = mapped
        while len(self._cache) > self.max_open_files:
            _, old_map = self._cache.popitem(last=False)
            del old_map
        return mapped

    def close(self) -> None:
        self._cache.clear()


class SourceShardPool:
    def __init__(
        self, source: str, split: str, shards: list[ShardInfo], context_length: int
    ) -> None:
        self.source = source
        self.split = split
        self.context_length = context_length
        self.shards: list[ShardInfo] = []
        self.cumulative_windows: list[int] = []
        cumulative = 0
        for shard in shards:
            windows = shard.valid_window_count(context_length)
            if windows <= 0:
                continue
            cumulative += windows
            self.shards.append(shard)
            self.cumulative_windows.append(cumulative)
        self.total_windows = cumulative
        if self.total_windows <= 0:
            raise RuntimeError(f"{source}/{split}没有足够长的分片，context_length={context_length}")

    def sample_location(self, rng: random.Random) -> tuple[ShardInfo, int]:
        global_window = rng.randrange(self.total_windows)
        shard_index = bisect.bisect_right(self.cumulative_windows, global_window)
        previous_cumulative = 0 if shard_index == 0 else self.cumulative_windows[shard_index - 1]
        local_start = global_window - previous_cumulative
        return self.shards[shard_index], local_start


class MixedPretrainingLoader:
    def __init__(
        self,
        data_config: DataConfig,
        context_length: int,
        split: str,
        *,
        seed: int,
        max_open_files: int = 8,
    ) -> None:
        data_config.validate()
        if context_length <= 0:
            raise ValueError("context_length必须为正数")
        self.data_config = data_config
        self.context_length = context_length
        self.split = split
        self.seed = seed
        self.rng = random.Random(seed)
        manifest = json.loads(data_config.manifest_path.read_text(encoding="utf-8"))
        self.manifest = manifest
        self.root = data_config.manifest_path.parent
        dtype_string = manifest["storage"]["numpy_dtype"]
        self.dtype = np.dtype(dtype_string)
        if self.dtype != np.dtype("<u2"):
            raise RuntimeError(f"当前DataLoader只支持小端uint16，manifest为{self.dtype.str}")
        self.vocab_size = int(manifest["tokenizer"]["vocab_size"])
        self.cache = MemmapCache(dtype=self.dtype, max_open_files=max_open_files)
        weight_dict = data_config.source_weight_dict
        self.source_names = list(weight_dict)
        self.source_weights = [weight_dict[name] for name in self.source_names]
        self.pools: dict[str, SourceShardPool] = {}
        for source_name in self.source_names:
            try:
                split_entry = manifest["sources"][source_name]["splits"][split]
            except KeyError as error:
                raise RuntimeError(f"Manifest缺少数据：{source_name}/{split}") from error
            shards: list[ShardInfo] = []
            for shard_entry in split_entry["shards"]:
                shard_path = self.root / shard_entry["path"]
                if not shard_path.exists():
                    raise FileNotFoundError(f"Token分片不存在：{shard_path}")
                expected_bytes = int(shard_entry["tokens"]) * self.dtype.itemsize
                if shard_path.stat().st_size != expected_bytes:
                    raise RuntimeError(f"Token分片大小错误：{shard_path}")
                shards.append(
                    ShardInfo(
                        source=source_name,
                        split=split,
                        path=shard_path,
                        token_count=int(shard_entry["tokens"]),
                    )
                )
            self.pools[source_name] = SourceShardPool(
                source=source_name, split=split, shards=shards, context_length=context_length
            )
        self.source_sample_counts = {name: 0 for name in self.source_names}

    def _sample_source(self) -> str:
        return self.rng.choices(self.source_names, weights=self.source_weights, k=1)[0]

    def _sample_sequence(self, source_name: str) -> np.ndarray:
        try:
            pool = self.pools[source_name]
        except KeyError as error:
            raise ValueError(f"未知来源：{source_name}") from error
        shard, start = pool.sample_location(self.rng)
        mapped = self.cache.get(shard)
        end = start + self.context_length + 1
        return np.array(mapped[start:end], dtype=np.int64, copy=True)

    def next_batch(
        self, batch_size: int, *, device: torch.device | None = None, source_name: str | None = None
    ) -> TokenBatch:
        if batch_size <= 0:
            raise ValueError("batch_size必须为正数")
        sequences = np.empty((batch_size, self.context_length + 1), dtype=np.int64)
        source_counts = {name: 0 for name in self.source_names}
        for batch_index in range(batch_size):
            selected_source = source_name if source_name is not None else self._sample_source()
            sequences[batch_index] = self._sample_sequence(selected_source)
            source_counts[selected_source] += 1
            self.source_sample_counts[selected_source] += 1
        tokens = torch.from_numpy(sequences)
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]
        if self.data_config.pin_memory and torch.cuda.is_available():
            input_ids = input_ids.pin_memory()
            labels = labels.pin_memory()
        if device is not None:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        return TokenBatch(
            input_ids=input_ids,
            labels=labels,
            source_counts={name: count for name, count in source_counts.items() if count > 0},
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "split": self.split,
            "context_length": self.context_length,
            "rng_state": self.rng.getstate(),
            "source_sample_counts": dict(self.source_sample_counts),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["split"] != self.split:
            raise RuntimeError("DataLoader split与checkpoint不一致")
        if int(state["context_length"]) != self.context_length:
            raise RuntimeError("DataLoader context_length与checkpoint不一致")
        self.rng.setstate(state["rng_state"])
        restored_counts = state.get("source_sample_counts", {})
        for source_name in self.source_names:
            self.source_sample_counts[source_name] = int(restored_counts.get(source_name, 0))

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> MixedPretrainingLoader:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
