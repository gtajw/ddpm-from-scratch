"""YAML configuration and reproducibility helpers."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
import yaml

def load_config(path: str | Path) -> dict:
    """Load a YAML mapping."""
    with Path(path).open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    if not isinstance(config, dict): raise ValueError("Config must be a YAML mapping")
    return config

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
