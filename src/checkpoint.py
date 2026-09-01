from __future__ import annotations
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from src.config import ExperimentConfig, PRETRAIN_BEST_CHECKPOINT_PATH, PRETRAIN_LATEST_CHECKPOINT_PATH
from src.data_loader import MixedPretrainingLoader

LATEST_CHECKPOINT_NAME = PRETRAIN_LATEST_CHECKPOINT_PATH.name
BEST_CHECKPOINT_NAME = PRETRAIN_BEST_CHECKPOINT_PATH.name

@dataclass(frozen=True)
class TrainingState:
    step: int
    tokens_processed: int
    best_validation_loss: float

def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(), "torch_cuda": None}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_rng_state = state["torch_cpu"]
    if not isinstance(cpu_rng_state, torch.Tensor):
        raise TypeError("torch_cpu RNG state必须是Tensor")
    cpu_rng_state = cpu_rng_state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    torch.set_rng_state(cpu_rng_state)
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        normalized_cuda_state = []
        for generator_state in cuda_state:
            if not isinstance(generator_state, torch.Tensor):
                raise TypeError("CUDA RNG state必须是Tensor")
            normalized_cuda_state.append(generator_state.detach().to(device="cpu", dtype=torch.uint8).contiguous())
        torch.cuda.set_rng_state_all(normalized_cuda_state)

def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)

def copy_checkpoint_atomic(source: Path, destination: Path) -> None:
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    shutil.copy2(source, temporary_path)
    os.replace(temporary_path, destination)

def save_training_checkpoint(
    *,
    directory: Path,
    experiment: ExperimentConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    train_loader: MixedPretrainingLoader,
    step: int,
    tokens_processed: int,
    best_validation_loss: float,
    is_best: bool,
) -> Path:
    latest_path = directory / LATEST_CHECKPOINT_NAME
    payload = {
        "format_version": 1,
        "experiment_name": experiment.name,
        "experiment_config": asdict(experiment),
        "step": step,
        "tokens_processed": tokens_processed,
        "best_validation_loss": best_validation_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "train_loader": train_loader.state_dict(),
        "rng_state": capture_rng_state(),
    }
    atomic_torch_save(payload, latest_path)
    if is_best:
        best_path = directory / BEST_CHECKPOINT_NAME
        copy_checkpoint_atomic(latest_path, best_path)
    return latest_path

def load_training_checkpoint(
    *,
    path: Path,
    experiment: ExperimentConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    train_loader: MixedPretrainingLoader,
    map_location: torch.device | str,
) -> TrainingState:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    checkpoint_experiment = checkpoint["experiment_name"]
    if checkpoint_experiment != experiment.name:
        raise RuntimeError(f"Checkpoint配置不匹配：{checkpoint_experiment} != {experiment.name}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler_state = checkpoint.get("scaler")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    train_loader.load_state_dict(checkpoint["train_loader"])
    restore_rng_state(checkpoint["rng_state"])
    return TrainingState(step=int(checkpoint["step"]), tokens_processed=int(checkpoint["tokens_processed"]), best_validation_loss=float(checkpoint["best_validation_loss"]))

def load_model_checkpoint(*, path: Path, model: nn.Module, expected_experiment_name: str, map_location: torch.device | str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    actual_name = checkpoint["experiment_name"]
    if actual_name != expected_experiment_name:
        raise RuntimeError(f"Checkpoint配置不匹配：{actual_name} != {expected_experiment_name}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint

def latest_checkpoint_path(directory: Path) -> Path | None:
    path = directory / LATEST_CHECKPOINT_NAME
    return path if path.exists() else None