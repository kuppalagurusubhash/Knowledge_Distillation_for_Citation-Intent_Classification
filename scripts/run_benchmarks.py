"""Run inference latency, throughput benchmarks, and compute Pareto frontiers for all completed models."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.evaluation.benchmark import benchmark_model
from src.evaluation.pareto import compute_pareto_frontiers
from src.utils.config import get_device, load_config
from src.utils.logging_utils import get_logger


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks and Pareto analysis.")
    parser.add_argument("--config", type=str, default="configs/quick_run.yaml")
    args = parser.parse_args()

    logger = get_logger("Benchmark-Runner")
    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))

    manifest_path = Path(config["paths"]["results_dir"]) / "manifest.json"
    if not manifest_path.exists():
        logger.warning(f"Manifest not found at {manifest_path}. Please run training first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    experiments = manifest.get("experiments", [])
    if not experiments:
        logger.warning("No completed experiments found in manifest.")
        return

    logger.info(f"Found {len(experiments)} models to benchmark on {device}...")
    benchmarks_list = []

    for exp in experiments:
        ckpt_dir = exp.get("checkpoint_dir")
        if not ckpt_dir or not Path(ckpt_dir).exists():
            logger.warning(f"Checkpoint for {exp.get('experiment_id')} not found at {ckpt_dir}. Skipping.")
            continue

        exp_id = exp.get("experiment_id")
        alias = exp.get("model_alias", exp_id)
        logger.info(f"Benchmarking model: {alias} ({exp_id})...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
            model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
            model.eval()

            b_res = benchmark_model(
                model=model,
                tokenizer=tokenizer,
                device=device,
                seq_length=config["dataset"].get("max_length", 128),
                batch_sizes=config["evaluation"].get("batch_sizes", [1, 8]),
                num_runs=config["evaluation"].get("benchmark_runs", 15),
                warmup_runs=config["evaluation"].get("warmup_runs", 3),
                checkpoint_dir=ckpt_dir,
            )

            # Combine with test metrics
            combined = {
                "experiment_id": exp_id,
                "model_alias": alias,
                "model_type": exp.get("model_type"),
                "method": exp.get("method"),
                "test_accuracy": exp.get("test_accuracy"),
                "test_macro_f1": exp.get("test_macro_f1"),
                "minority_f1": exp.get("minority_f1"),
                "total_params": b_res["total_parameters"],
                "param_size_mb": b_res["param_size_mb"],
                "checkpoint_size_mb": b_res["checkpoint_size_mb"],
                "latency_single_ms": b_res["latency_single_ms"],
                "latency_p95_ms": b_res["latency_p95_ms"],
                "throughput_single_eps": b_res["throughput_single_eps"],
                "batch_benchmarks": b_res["batch_benchmarks"],
                "device": device,
            }
            benchmarks_list.append(combined)
            logger.info(
                f"[DONE] {alias} - Median Latency: {b_res['latency_single_ms']} ms | "
                f"Throughput: {b_res['throughput_single_eps']} ex/s | "
                f"Params: {b_res['total_parameters']:,}"
            )
        except Exception as e:
            logger.error(f"Error benchmarking {exp_id}: {e}")

    # Compute Pareto Frontiers
    pareto_results = compute_pareto_frontiers(benchmarks_list)

    results_dir = Path(config["paths"]["results_dir"])
    with open(results_dir / "benchmarks_summary.json", "w", encoding="utf-8") as f:
        json.dump(pareto_results, f, indent=2)

    # Save CSV
    if pareto_results["models"]:
        df_bench = pd.DataFrame(pareto_results["models"])
        # Drop nested batch dict for clean CSV
        csv_df = df_bench.drop(columns=["batch_benchmarks"], errors="ignore")
        csv_df.to_csv(results_dir / "benchmark_table.csv", index=False)
        logger.info(f"Saved benchmark summary table to {results_dir / 'benchmark_table.csv'}")

    logger.info("Pareto-optimal models (Size vs Macro-F1): " + ", ".join(pareto_results.get("pareto_size_vs_f1_names", [])))
    logger.info("Pareto-optimal models (Latency vs Macro-F1): " + ", ".join(pareto_results.get("pareto_latency_vs_f1_names", [])))


if __name__ == "__main__":
    main()
