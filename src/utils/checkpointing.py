"""Model checkpointing and manifest management."""
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def save_model_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save model, tokenizer, and metadata to specified directory."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Save huggingface model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save custom metadata
    if metadata:
        meta_file = path / "checkpoint_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return str(path)


def load_model_checkpoint(checkpoint_dir: str, model_class, device: str = "cpu"):
    """Load model and tokenizer from checkpoint directory."""
    path = Path(checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = model_class.from_pretrained(checkpoint_dir).to(device)
    model.eval()
    return model, tokenizer


def update_manifest(manifest_path: str, experiment_entry: Dict[str, Any]) -> None:
    """Add or update an experiment in the results manifest JSON."""
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = {"experiments": []}
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"experiments": []}

    # Replace existing entry if experiment_id matches, else append
    exp_id = experiment_entry.get("experiment_id")
    experiments = [e for e in data.get("experiments", []) if e.get("experiment_id") != exp_id]
    experiments.append(experiment_entry)
    data["experiments"] = experiments

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
