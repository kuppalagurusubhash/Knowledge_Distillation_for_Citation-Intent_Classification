"""Unit tests for classification metrics, per-class evaluation, and Pareto frontiers."""
import numpy as np
from src.evaluation.classification_metrics import compute_metrics
from src.evaluation.pareto import is_pareto_efficient, compute_pareto_frontiers


def test_compute_metrics():
    y_true = [0, 1, 2, 0, 1, 2, 1, 1]
    y_pred = [0, 1, 2, 0, 1, 1, 1, 0]

    metrics = compute_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "minority_class" in metrics
    assert metrics["minority_class"]["id"] == 2
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 3


def test_pareto_efficiency_basic():
    # Objectives: [Cost (min), Score (max)]
    costs = [
        [10, 0.85],  # Model A: smaller, good score
        [20, 0.88],  # Model B: larger, better score
        [30, 0.84],  # Model C: dominated by A and B
    ]
    # Maximize: [False (min cost), True (max score)]
    efficient = is_pareto_efficient(costs, maximize=[False, True])
    assert efficient[0] is True   # Model A is efficient
    assert efficient[1] is True   # Model B is efficient
    assert efficient[2] is False  # Model C is strictly dominated


def test_compute_pareto_frontiers():
    models = [
        {"model_alias": "SciBERT", "total_params": 110_000_000, "test_macro_f1": 0.86, "latency_single_ms": 45.0, "throughput_single_eps": 22.0},
        {"model_alias": "DistilBERT", "total_params": 66_000_000, "test_macro_f1": 0.84, "latency_single_ms": 25.0, "throughput_single_eps": 40.0},
        {"model_alias": "BERT-Mini", "total_params": 11_000_000, "test_macro_f1": 0.81, "latency_single_ms": 8.0, "throughput_single_eps": 125.0},
        {"model_alias": "InefficientModel", "total_params": 100_000_000, "test_macro_f1": 0.70, "latency_single_ms": 50.0, "throughput_single_eps": 20.0},
    ]
    res = compute_pareto_frontiers(models)
    assert "BERT-Mini" in res["pareto_size_vs_f1_names"]
    assert "InefficientModel" not in res["pareto_size_vs_f1_names"]
