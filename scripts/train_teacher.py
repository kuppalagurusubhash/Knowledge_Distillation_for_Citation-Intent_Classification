"""Script to train SciBERT teacher model and cache training logits."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_teacher import train_teacher


def main():
    parser = argparse.ArgumentParser(description="Train SciBERT teacher model.")
    parser.add_argument("--config", type=str, default="configs/quick_run.yaml", help="Path to config YAML")
    parser.add_argument("--experiment_id", type=str, default="teacher_scibert", help="Experiment ID")
    args = parser.parse_args()

    print(f"Starting teacher training with config {args.config}...")
    train_teacher(config_path=args.config, experiment_id=args.experiment_id)
    print("Teacher training complete.")


if __name__ == "__main__":
    main()
