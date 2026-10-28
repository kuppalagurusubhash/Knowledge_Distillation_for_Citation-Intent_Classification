"""Robustness evaluation harness applying controlled text perturbations and measuring degradation."""
import copy
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.data.preprocessing import CitationDataset
from src.evaluation.classification_metrics import compute_metrics
from src.training.distillation import perturb_text


def evaluate_robustness(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    clean_texts: List[str],
    labels: List[int],
    device: str = "cpu",
    batch_size: int = 16,
    max_length: int = 128,
    perturbation_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate model robustness across clean and controlled perturbed citation test sets.
    Computes absolute and relative Macro-F1 degradation and minority class resilience.
    """
    if perturbation_types is None:
        perturbation_types = ["whitespace_normalize", "minor_punctuation", "typo_insertion"]

    model.eval()
    model.to(device)

    def _eval_on_texts(texts: List[str]) -> Dict[str, Any]:
        ds = CitationDataset(texts, labels, None, tokenizer, max_length=max_length)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                b_labels = batch["labels"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids is not None and "token_type_ids" in tokenizer.model_input_names:
                    kwargs["token_type_ids"] = token_type_ids

                outputs = model(**kwargs)
                preds = torch.argmax(outputs.logits if hasattr(outputs, "logits") else outputs["logits"], dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(b_labels.cpu().numpy().tolist())

        return compute_metrics(all_labels, all_preds)

    # 1. Clean Test Performance
    clean_metrics = _eval_on_texts(clean_texts)
    clean_macro_f1 = clean_metrics["macro_f1"]
    clean_minority_f1 = clean_metrics["minority_class"]["f1_score"]

    perturbation_results = {}

    # 2. Perturbation Evaluations
    for p_type in perturbation_types:
        perturbed_texts = [perturb_text(t, perturbation_type=p_type) for t in clean_texts]
        pert_metrics = _eval_on_texts(perturbed_texts)

        pert_macro_f1 = pert_metrics["macro_f1"]
        pert_minority_f1 = pert_metrics["minority_class"]["f1_score"]

        abs_deg = round(clean_macro_f1 - pert_macro_f1, 4)
        rel_deg_pct = round((abs_deg / (clean_macro_f1 + 1e-8)) * 100, 2)
        robustness_retention = round(100.0 - rel_deg_pct, 2)

        min_abs_deg = round(clean_minority_f1 - pert_minority_f1, 4)
        min_rel_deg_pct = round((min_abs_deg / (clean_minority_f1 + 1e-8)) * 100, 2)

        perturbation_results[p_type] = {
            "perturbed_accuracy": pert_metrics["accuracy"],
            "perturbed_macro_f1": pert_macro_f1,
            "absolute_f1_degradation": abs_deg,
            "relative_f1_degradation_pct": rel_deg_pct,
            "robustness_retention_pct": robustness_retention,
            "minority_clean_f1": clean_minority_f1,
            "minority_perturbed_f1": pert_minority_f1,
            "minority_f1_degradation": min_abs_deg,
            "minority_rel_deg_pct": min_rel_deg_pct,
        }

    # Aggregate average degradation across all perturbations
    avg_rel_deg = round(
        float(np.mean([res["relative_f1_degradation_pct"] for res in perturbation_results.values()])), 2
    )

    return {
        "clean_accuracy": clean_metrics["accuracy"],
        "clean_macro_f1": clean_macro_f1,
        "clean_minority_f1": clean_minority_f1,
        "average_relative_degradation_pct": avg_rel_deg,
        "overall_robustness_score": round(100.0 - avg_rel_deg, 2),
        "perturbations": perturbation_results,
    }
