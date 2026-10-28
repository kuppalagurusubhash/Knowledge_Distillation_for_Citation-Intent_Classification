"""Model loading, parameter counting, and model inspection utilities."""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

ID2LABEL = {0: "method", 1: "background", 2: "result"}
LABEL2ID = {"method": 0, "background": 1, "result": 2}


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Calculate total and trainable parameter counts and size in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate parameter size in MB (assuming 32-bit float = 4 bytes per param)
    param_size_mb = (total_params * 4) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "param_size_mb": round(param_size_mb, 2),
    }


def get_checkpoint_size_mb(checkpoint_dir: str) -> float:
    """Calculate the total size of files inside a checkpoint directory in megabytes."""
    path = Path(checkpoint_dir)
    if not path.exists():
        return 0.0

    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)

    return round(total_bytes / (1024 * 1024), 2)


def load_model_and_tokenizer(
    model_name_or_path: str,
    num_labels: int = 3,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    device: str = "cpu",
    output_hidden_states: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load Hugging Face sequence classification model and tokenizer with citation labels."""
    if id2label is None:
        id2label = ID2LABEL
    if label2id is None:
        label2id = LABEL2ID

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        output_hidden_states=output_hidden_states,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
    ).to(device)

    return model, tokenizer
