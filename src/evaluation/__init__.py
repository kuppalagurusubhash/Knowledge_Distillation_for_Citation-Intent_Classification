# Evaluation package
from .classification_metrics import compute_metrics, LABEL_NAMES
from .benchmark import benchmark_model
from .robustness import evaluate_robustness
from .pareto import compute_pareto_frontiers, is_pareto_efficient

__all__ = [
    "compute_metrics",
    "LABEL_NAMES",
    "benchmark_model",
    "evaluate_robustness",
    "compute_pareto_frontiers",
    "is_pareto_efficient",
]
