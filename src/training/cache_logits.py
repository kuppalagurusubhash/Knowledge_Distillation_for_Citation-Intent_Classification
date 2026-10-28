"""Teacher logit and representation caching system for efficient knowledge distillation."""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from src.data.preprocessing import CitationDataset
from src.models.teacher import TeacherModel


def compute_cache_fingerprint(
    teacher_checkpoint: str,
    dataset_length: int,
    max_length: int,
    split_name: str = "train",
) -> str:
    """Compute a deterministic hash fingerprint to validate cache consistency."""
    hash_str = f"{teacher_checkpoint}_{dataset_length}_{max_length}_{split_name}"
    return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:16]


def cache_teacher_outputs(
    teacher: TeacherModel,
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    cache_path: str,
    batch_size: int = 16,
    max_length: int = 128,
    text_col: str = "string",
    id_col: str = "id",
    label_col: str = "label",
    device: str = "cpu",
    save_hidden_states: bool = False,
) -> Dict[str, Any]:
    """Generate and cache pre-softmax teacher logits and sample identifiers."""
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    texts = dataset[text_col]
    labels = dataset[label_col] if label_col in dataset.column_names else None
    ids = dataset[id_col] if id_col in dataset.column_names else [str(i) for i in range(len(texts))]

    ds = CitationDataset(texts, labels, ids, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    teacher.eval()
    teacher.to(device)

    all_logits = []
    all_sample_ids = []
    all_hidden_states = []

    print(f"Caching teacher outputs for {len(dataset)} examples...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Teacher Cache"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_hidden_states=save_hidden_states,
            )

            all_logits.append(outputs["logits"].cpu())
            all_sample_ids.extend(batch["id"])

            if save_hidden_states and "hidden_states" in outputs:
                # Save last hidden state
                all_hidden_states.append(outputs["hidden_states"][-1].cpu())

    logits_tensor = torch.cat(all_logits, dim=0)

    fingerprint = compute_cache_fingerprint(
        teacher.model_name, len(dataset), max_length, split_name="train"
    )

    cache_data = {
        "logits": logits_tensor,
        "sample_ids": all_sample_ids,
        "num_samples": len(all_sample_ids),
        "fingerprint": fingerprint,
        "teacher_model": teacher.model_name,
        "max_length": max_length,
    }

    if save_hidden_states and all_hidden_states:
        cache_data["last_hidden_states"] = torch.cat(all_hidden_states, dim=0)

    torch.save(cache_data, cache_file)

    meta_file = cache_file.with_suffix(".json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_file": str(cache_file),
                "num_samples": len(all_sample_ids),
                "fingerprint": fingerprint,
                "teacher_model": teacher.model_name,
                "max_length": max_length,
            },
            f,
            indent=2,
        )

    print(f"Cached teacher logits saved to {cache_file} (Fingerprint: {fingerprint})")
    return cache_data


def load_cached_teacher_outputs(
    cache_path: str,
    expected_fingerprint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load cached teacher outputs with integrity/fingerprint verification."""
    p = Path(cache_path)
    if not p.exists():
        return None

    try:
        cache = torch.load(p, map_location="cpu", weights_only=False)
        if expected_fingerprint and cache.get("fingerprint") != expected_fingerprint:
            print(f"Cache fingerprint mismatch: {cache.get('fingerprint')} != {expected_fingerprint}. Invalidating cache.")
            return None
        return cache
    except Exception as e:
        print(f"Error loading cache: {e}. Invalidating cache.")
        return None
