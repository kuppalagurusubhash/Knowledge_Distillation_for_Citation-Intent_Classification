"""Pareto frontier calculation and multi-objective trade-off analysis."""
from typing import Any, Dict, List


def is_pareto_efficient(costs: List[List[float]], maximize: List[bool]) -> List[bool]:
    """
    Find Pareto-efficient points given a list of multi-objective metrics.
    maximize: List of booleans indicating whether each dimension should be maximized.
    """
    n_points = len(costs)
    if n_points == 0:
        return []

    # Standardize so that higher is better for all objectives
    norm_scores = []
    for row in costs:
        norm_row = [val if max_flag else -val for val, max_flag in zip(row, maximize)]
        norm_scores.append(norm_row)

    is_efficient = [True] * n_points
    for i in range(n_points):
        if not is_efficient[i]:
            continue
        for j in range(n_points):
            if i == j:
                continue
            # Point j dominates point i if j is >= i on all objectives and > on at least one
            all_gte = all(norm_scores[j][k] >= norm_scores[i][k] for k in range(len(maximize)))
            any_gt = any(norm_scores[j][k] > norm_scores[i][k] for k in range(len(maximize)))
            if all_gte and any_gt:
                is_efficient[i] = False
                break

    return is_efficient


def compute_pareto_frontiers(models_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute Pareto frontiers across multiple critical trade-offs:
    1. Size vs Macro-F1 (size: min, macro-f1: max)
    2. Latency vs Macro-F1 (latency: min, macro-f1: max)
    3. Size vs Throughput (size: min, throughput: max)
    """
    if not models_data:
        return {"models": [], "pareto_size_vs_f1": [], "pareto_latency_vs_f1": []}

    # Trade-off 1: Size (min) vs Macro-F1 (max)
    costs_size_f1 = [
        [m.get("total_params", 0), m.get("test_macro_f1", 0.0)]
        for m in models_data
    ]
    eff_size_f1 = is_pareto_efficient(costs_size_f1, maximize=[False, True])

    # Trade-off 2: Latency (min) vs Macro-F1 (max)
    costs_lat_f1 = [
        [m.get("latency_single_ms", 100.0), m.get("test_macro_f1", 0.0)]
        for m in models_data
    ]
    eff_lat_f1 = is_pareto_efficient(costs_lat_f1, maximize=[False, True])

    # Trade-off 3: Size (min) vs Throughput (max)
    costs_size_tp = [
        [m.get("total_params", 0), m.get("throughput_single_eps", 0.0)]
        for m in models_data
    ]
    eff_size_tp = is_pareto_efficient(costs_size_tp, maximize=[False, True])

    annotated_models = []
    for idx, m in enumerate(models_data):
        entry = dict(m)
        entry["is_pareto_size_vs_f1"] = eff_size_f1[idx]
        entry["is_pareto_latency_vs_f1"] = eff_lat_f1[idx]
        entry["is_pareto_size_vs_throughput"] = eff_size_tp[idx]
        entry["is_globally_pareto"] = eff_size_f1[idx] or eff_lat_f1[idx]
        annotated_models.append(entry)

    return {
        "models": annotated_models,
        "pareto_size_vs_f1_names": [m["model_alias"] for idx, m in enumerate(annotated_models) if eff_size_f1[idx]],
        "pareto_latency_vs_f1_names": [m["model_alias"] for idx, m in enumerate(annotated_models) if eff_lat_f1[idx]],
    }
