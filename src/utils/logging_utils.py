"""Logging and experiment tracking utilities."""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


def get_logger(name: str = "KD-Citation", log_file: Optional[str] = None) -> logging.Logger:
    """Create a structured console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentTracker:
    """Tracks training logs and metrics, saving to JSON and CSV."""

    def __init__(self, experiment_id: str, output_dir: str = "results"):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self.final_metrics: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "created_at": datetime.now().isoformat(),
        }

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Log per-epoch metrics."""
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **metrics}
        self.history.append(entry)

    def log_final(self, metrics: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record final evaluation metrics and metadata."""
        self.final_metrics = metrics
        if metadata:
            self.metadata.update(metadata)
        self.save()

    def save(self) -> None:
        """Save history and final metrics to files."""
        # Save JSON record
        record_path = self.output_dir / f"{self.experiment_id}_metrics.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": self.metadata,
                    "history": self.history,
                    "final_metrics": self.final_metrics,
                },
                f,
                indent=2,
            )

        # Save history CSV if epochs logged
        if self.history:
            df_history = pd.DataFrame(self.history)
            df_history.to_csv(self.output_dir / f"{self.experiment_id}_history.csv", index=False)
