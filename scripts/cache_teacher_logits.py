"""Script to standalone cache teacher logits from a trained teacher checkpoint."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset_loader import load_scicite_data
from src.models.teacher import TeacherModel
from src.training.cache_logits import cache_teacher_outputs
from src.utils.config import get_device, load_config
from src.utils.logging_utils import get_logger


def main():
    parser = argparse.ArgumentParser(description="Cache SciBERT teacher logits.")
    parser.add_argument("--config", type=str, default="configs/quick_run.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="artifacts/cache/teacher_train_logits.pt")
    args = parser.parse_args()

    logger = get_logger("Cache-Logits")
    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))

    checkpoint_dir = args.checkpoint
    if not checkpoint_dir:
        checkpoint_dir = str(Path(config["paths"]["checkpoints_dir"]) / "teacher_scibert")

    logger.info(f"Loading teacher model from {checkpoint_dir}...")
    teacher = TeacherModel(model_name_or_path=checkpoint_dir, num_labels=3, device=device)

    dataset_dict, _ = load_scicite_data(config, save_stats=False)
    batch_size = config["training"]["teacher"].get("batch_size", 16)
    max_length = config["dataset"].get("max_length", 128)

    cache_teacher_outputs(
        teacher=teacher,
        dataset=dataset_dict["train"],
        tokenizer=teacher.tokenizer,
        cache_path=args.output,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        save_hidden_states=True,
    )
    logger.info(f"Teacher logits cached successfully to {args.output}")


if __name__ == "__main__":
    main()
