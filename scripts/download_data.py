"""Script to download, validate, and summarize the SciCite dataset."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.data.dataset_loader import load_scicite_data


def main():
    parser = argparse.ArgumentParser(description="Download and inspect SciCite dataset.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    logger = get_logger("Data-Download")
    config = load_config(args.config)
    logger.info("Initializing dataset download and validation...")
    dataset_dict, stats = load_scicite_data(config, save_stats=True)

    logger.info("Dataset inspection completed successfully!")
    for split, info in stats["splits"].items():
        logger.info(
            f"Split '{split}': {info['total_samples']} samples. "
            f"Label counts: { {k: v['count'] for k, v in info['label_distribution'].items()} }"
        )
    logger.info(f"Class weights (inverse freq): {stats['class_weights']}")


if __name__ == "__main__":
    main()
