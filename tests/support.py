from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from src.config import DataConfig


def create_tokenized_fixture(
    root: Path,
    source_weights: tuple[tuple[str, float], ...],
    *,
    vocab_size: int = 128,
    token_count: int = 4_096,
) -> DataConfig:
    sources = {}
    for source_index, (source, _) in enumerate(source_weights):
        splits = {}
        for split_index, split in enumerate(("train", "validation")):
            path = root / f"{source}-{split}.tokens.bin"
            tokens = (
                (np.arange(token_count, dtype=np.uint32) + source_index * 17 + split_index * 31)
                % vocab_size
            ).astype("<u2")
            tokens.tofile(path)
            splits[split] = {"shards": [{"path": path.name, "tokens": token_count}]}
        sources[source] = {"splits": splits}
    manifest_path = root / "manifest.json"
    manifest = {
        "storage": {"numpy_dtype": "<u2"},
        "tokenizer": {"vocab_size": vocab_size},
        "sources": sources,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return DataConfig(
        manifest_path=manifest_path,
        source_weights=source_weights,
        train_split="train",
        validation_split="validation",
        pin_memory=False,
        seed=42,
    )
