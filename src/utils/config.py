"""Configuration loading and validation utility."""
import os
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure output directories exist
    paths = config.get("paths", {})
    for dir_key, dir_path in paths.items():
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    return config


def get_device(device_setting: str = "auto") -> str:
    """Resolve compute device based on configuration and hardware availability."""
    import torch
    if device_setting == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif device_setting == "cpu":
        return "cpu"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"
