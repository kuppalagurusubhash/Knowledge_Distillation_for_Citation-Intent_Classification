"""Dataset loading, validation, and split management for SciCite citation intent classification."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# Canonical SciCite label mapping
LABEL_NAMES = ["method", "background", "result"]
LABEL_MAP = {name: idx for idx, name in enumerate(LABEL_NAMES)}
ID2LABEL = {idx: name for idx, name in enumerate(LABEL_NAMES)}


def get_dataset_stats(dataset_dict: DatasetDict, text_col: str = "string", label_col: str = "label") -> Dict[str, Any]:
    """Compute comprehensive dataset statistics across splits without test leakage."""
    stats: Dict[str, Any] = {
        "splits": {},
        "label_mapping": ID2LABEL,
        "minority_class": {"id": 2, "name": "result"},
        "data_leakage_check": {},
    }

    train_texts = set()

    for split_name, split_data in dataset_dict.items():
        df = split_data.to_pandas()
        total = len(df)
        label_counts = df[label_col].value_counts().to_dict()
        label_dist = {
            ID2LABEL.get(int(k), str(k)): {
                "count": int(v),
                "percentage": round(float(v) / total * 100, 2) if total > 0 else 0,
            }
            for k, v in label_counts.items()
        }

        # Word length statistics
        word_lengths = df[text_col].astype(str).apply(lambda x: len(x.split()))
        char_lengths = df[text_col].astype(str).apply(len)

        stats["splits"][split_name] = {
            "total_samples": total,
            "label_distribution": label_dist,
            "word_length": {
                "mean": round(float(word_lengths.mean()), 2) if total > 0 else 0,
                "median": round(float(word_lengths.median()), 2) if total > 0 else 0,
                "std": round(float(word_lengths.std()), 2) if total > 0 else 0,
                "p95": round(float(np.percentile(word_lengths, 95)), 2) if total > 0 else 0,
                "min": int(word_lengths.min()) if total > 0 else 0,
                "max": int(word_lengths.max()) if total > 0 else 0,
            },
            "char_length": {
                "mean": round(float(char_lengths.mean()), 2) if total > 0 else 0,
                "median": round(float(char_lengths.median()), 2) if total > 0 else 0,
            },
        }

        if split_name == "train":
            train_texts = set(df[text_col].str.strip().tolist())
        else:
            split_texts = set(df[text_col].str.strip().tolist())
            overlap = train_texts.intersection(split_texts)
            stats["data_leakage_check"][f"train_{split_name}_overlap_count"] = len(overlap)

    # Compute inverse-frequency class weights for training
    train_df = dataset_dict["train"].to_pandas()
    counts = [train_df[label_col].value_counts().get(i, 1) for i in range(len(LABEL_NAMES))]
    total_train = len(train_df)
    n_classes = len(LABEL_NAMES)
    class_weights = [round(total_train / (n_classes * c), 4) for c in counts]
    stats["class_weights"] = class_weights

    return stats


def load_scicite_data(
    config: Dict[str, Any],
    save_stats: bool = True
) -> Tuple[DatasetDict, Dict[str, Any]]:
    """Load SciCite dataset, validate schema, optionally apply stratified subsampling for quick runs."""
    ds_cfg = config.get("dataset", {})
    text_col = ds_cfg.get("text_col", "string")
    label_col = ds_cfg.get("label_col", "label")
    seed = config.get("seed", 42)

    logger.info("Loading SciCite dataset from Hugging Face Datasets...")
    raw_dataset = load_dataset("scicite", trust_remote_code=True)

    processed_splits = {}
    for split_name in ["train", "validation", "test"]:
        split_data = raw_dataset[split_name]
        max_samples_key = f"max_{split_name[:3]}_samples"
        max_samples = ds_cfg.get(max_samples_key) or ds_cfg.get(f"max_{split_name}_samples")

        if max_samples and max_samples < len(split_data):
            logger.info(
                f"Subsampling {split_name} split from {len(split_data)} to {max_samples} "
                f"using StratifiedShuffleSplit (seed={seed})"
            )
            df = split_data.to_pandas()
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
            indices, _ = next(splitter.split(df, df[label_col]))
            sub_df = df.iloc[indices].reset_index(drop=True)
            processed_splits[split_name] = Dataset.from_pandas(sub_df)
        else:
            processed_splits[split_name] = split_data

    dataset_dict = DatasetDict(processed_splits)
    stats = get_dataset_stats(dataset_dict, text_col=text_col, label_col=label_col)

    if save_stats:
        results_dir = Path(config.get("paths", {}).get("results_dir", "results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        stats_path = results_dir / "dataset_summary.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Dataset statistics saved to {stats_path}")

    return dataset_dict, stats
