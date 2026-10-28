"""Script to orchestrate student model training across methods and architectures."""
import argparse
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.train_student import train_student


def main():
    parser = argparse.ArgumentParser(description="Train student models across distillation methods.")
    parser.add_argument("--config", type=str, default="configs/quick_run.yaml")
    parser.add_argument("--students", nargs="+", default=["bert_mini"], help="List of student keys (distilbert, bert_mini, bert_tiny)")
    parser.add_argument("--methods", nargs="+", default=["supervised", "logit_distillation"], help="List of methods (supervised, logit_distillation, feature_distillation, augmented_distillation)")
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    args = parser.parse_args()

    print(f"Starting training run for students: {args.students} | Methods: {args.methods}")
    for student_key in args.students:
        for method in args.methods:
            exp_id = f"student_{student_key}_{method}"
            print(f"\n==========================================")
            print(f"Executing Experiment: {exp_id}")
            print(f"==========================================")
            try:
                train_student(
                    student_key=student_key,
                    method=method,
                    config_path=args.config,
                    experiment_id=exp_id,
                    teacher_checkpoint=args.teacher_checkpoint,
                )
                print(f"[SUCCESS] Finished experiment: {exp_id}")
            except Exception as e:
                print(f"[ERROR] Failed experiment {exp_id}: {e}")


if __name__ == "__main__":
    main()
