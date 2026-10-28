"""Classification evaluation metrics: Accuracy, Macro-F1, Weighted-F1, Per-class, and Confusion Matrix."""
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABEL_NAMES = ["method", "background", "result"]


def compute_metrics(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    class_names: List[str] = LABEL_NAMES,
    minority_class_id: int = 2,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics adhering to scientific rigor."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    minority_name = class_names[minority_class_id]
    minority_metrics = per_class[minority_name]

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "minority_class": {
            "name": minority_name,
            "id": minority_class_id,
            "recall": minority_metrics["recall"],
            "f1_score": minority_metrics["f1_score"],
            "support": minority_metrics["support"],
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "total_evaluated": int(len(y_true)),
    }
