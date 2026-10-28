"""Run controlled robustness evaluations on all trained models and record degradation."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data.dataset_loader import load_scicite_data
from src.evaluation.robustness import evaluate_robustness
from src.utils.config import get_device, load_config
from src.utils.logging_utils import get_logger


def main():
    parser = argparse.ArgumentParser(description="Run robustness evaluations on trained models.")
    parser.add_argument("--config", type=str, default="configs/quick_run.yaml")
    args = parser.parse_args()

    logger = get_logger("Robustness-Runner")
    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))

    manifest_path = Path(config["paths"]["results_dir"]) / "manifest.json"
    if not manifest_path.exists():
        logger.warning(f"Manifest not found at {manifest_path}. Please train models first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    experiments = manifest.get("experiments", [])
    if not experiments:
        logger.warning("No completed experiments found in manifest.")
        return

    # Load test dataset
    dataset_dict, _ = load_scicite_data(config, save_stats=False)
    test_texts = dataset_dict["test"][config["dataset"]["text_col"]]
    test_labels = dataset_dict["test"][config["dataset"]["label_col"]]

    logger.info(f"Evaluating robustness across {len(experiments)} models on {len(test_texts)} test citations...")
    robustness_summary = {}
    table_rows = []

    for exp in experiments:
        ckpt_dir = exp.get("checkpoint_dir")
        if not ckpt_dir or not Path(ckpt_dir).exists():
            continue

        exp_id = exp.get("experiment_id")
        alias = exp.get("model_alias", exp_id)
        logger.info(f"Evaluating robustness for: {alias} ({exp_id})...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
            model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)

            rob_res = evaluate_robustness(
                model=model,
                tokenizer=tokenizer,
                clean_texts=test_texts,
                labels=test_labels,
                device=device,
                batch_size=config["training"]["teacher"].get("batch_size", 16),
                max_length=config["dataset"].get("max_length", 128),
                perturbation_types=config["evaluation"].get(
                    "perturbations", ["whitespace_normalize", "minor_punctuation", "typo_insertion"]
                ),
            )

            robustness_summary[exp_id] = {
                "model_alias": alias,
                "method": exp.get("method"),
                **rob_res,
            }

            row = {
                "experiment_id": exp_id,
                "model_alias": alias,
                "method": exp.get("method"),
                "clean_macro_f1": rob_res["clean_macro_f1"],
                "clean_minority_f1": rob_res["clean_minority_f1"],
                "overall_robustness_score": rob_res["overall_robustness_score"],
                "avg_relative_degradation_pct": rob_res["average_relative_degradation_pct"],
            }
            # Add per-perturbation degradation
            for p_name, p_data in rob_res["perturbations"].items():
                row[f"{p_name}_f1"] = p_data["perturbed_macro_f1"]
                row[f"{p_name}_deg_pct"] = p_data["relative_f1_degradation_pct"]

            table_rows.append(row)
            logger.info(
                f"[DONE] {alias} - Clean F1: {rob_res['clean_macro_f1']} | "
                f"Avg Degradation: {rob_res['average_relative_degradation_pct']}% | "
                f"Robustness Score: {rob_res['overall_robustness_score']}%"
            )

        except Exception as e:
            logger.error(f"Error in robustness evaluation of {exp_id}: {e}")

    results_dir = Path(config["paths"]["results_dir"])
    with open(results_dir / "robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(robustness_summary, f, indent=2)

    if table_rows:
        df_rob = pd.DataFrame(table_rows)
        df_rob.to_csv(results_dir / "robustness_table.csv", index=False)
        logger.info(f"Robustness table saved to {results_dir / 'robustness_table.csv'}")


if __name__ == "__main__":
    main()
