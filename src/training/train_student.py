"""Student model training pipeline across methods (Supervised, Logit KD, Feature KD, Augmented KD)."""
import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data.dataset_loader import load_scicite_data
from src.data.preprocessing import CitationDataset, create_dataloaders
from src.evaluation.classification_metrics import compute_metrics
from src.models.model_loader import count_parameters, get_checkpoint_size_mb
from src.models.student import StudentModel
from src.models.teacher import TeacherModel
from src.training.cache_logits import load_cached_teacher_outputs
from src.training.distillation import DistillationTrainer, perturb_text
from src.training.train_teacher import evaluate_model
from src.utils.checkpointing import save_model_checkpoint, update_manifest
from src.utils.config import get_device, load_config
from src.utils.logging_utils import ExperimentTracker, get_logger
from src.utils.seed import set_seed


def train_student(
    student_key: str = "bert_mini",
    method: str = "logit_distillation",
    config_path: str = "configs/default.yaml",
    experiment_id: Optional[str] = None,
    teacher_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a student model with the specified distillation method and evaluate on test set."""
    config = load_config(config_path)
    set_seed(config.get("seed", 42), config.get("deterministic", True))
    device = get_device(config.get("device", "auto"))

    if experiment_id is None:
        experiment_id = f"student_{student_key}_{method}"

    logs_dir = Path(config["paths"]["logs_dir"])
    logger = get_logger("Train-Student", log_file=str(logs_dir / f"{experiment_id}.log"))
    logger.info(
        f"Initializing Student Training. Student: {student_key} | Method: {method} | "
        f"Device: {device} | Experiment: {experiment_id}"
    )

    # 1. Dataset Loading
    dataset_dict, data_stats = load_scicite_data(config, save_stats=False)

    # Apply data augmentation to train split if method == augmented_distillation
    if method == "augmented_distillation":
        logger.info("Applying controlled label-preserving text augmentation to training set...")
        train_df = dataset_dict["train"].to_pandas()
        aug_texts = [perturb_text(t, perturbation_type="minor_punctuation") for t in train_df["string"]]
        train_df["string"] = aug_texts
        from datasets import Dataset
        dataset_dict["train"] = Dataset.from_pandas(train_df)

    # 2. Student Configuration
    s_cfg = config["training"]["students"].get(student_key)
    if not s_cfg:
        raise ValueError(f"Student key '{student_key}' not found in configuration training.students.")

    model_name = s_cfg["model_name"]
    alias = s_cfg.get("alias", student_key.upper())
    batch_size = s_cfg.get("batch_size", 16)
    epochs = s_cfg.get("epochs", 4)
    lr = float(s_cfg.get("learning_rate", 5e-5))
    weight_decay = float(s_cfg.get("weight_decay", 0.01))
    max_length = config["dataset"].get("max_length", 128)

    # 3. Initialize Student Model
    student = StudentModel(
        model_name_or_path=model_name,
        num_labels=3,
        teacher_hidden_size=768,
        device=device,
    )
    param_stats = student.get_parameter_stats()
    logger.info(
        f"Initialized Student {alias} ({model_name}). "
        f"Parameters: {param_stats['total_parameters']:,} ({param_stats['param_size_mb']} MB)"
    )

    # 4. Load or Prepare Teacher Logits/Model
    teacher = None
    cached_logits = None
    cached_ids = None

    if method != "supervised":
        cache_path = Path(config["paths"]["cache_dir"]) / "teacher_train_logits.pt"
        cached_data = load_cached_teacher_outputs(str(cache_path))
        if cached_data is not None:
            logger.info(f"Loaded cached teacher logits from {cache_path} ({cached_data['num_samples']} samples)")
            cached_logits = cached_data["logits"]
            cached_ids = cached_data["sample_ids"]
        else:
            logger.info("No cached teacher logits found or cache invalid. Loading live teacher model...")
            t_model_name = teacher_checkpoint or config["training"]["teacher"]["model_name"]
            teacher = TeacherModel(model_name_or_path=t_model_name, num_labels=3, device=device)
            teacher.eval()

    # 5. Distillation Trainer
    dist_cfg = config.get("distillation", {})
    trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        cached_teacher_logits=cached_logits,
        cached_sample_ids=cached_ids,
        method=method,
        temperature=dist_cfg.get("temperature", 3.0),
        alpha=dist_cfg.get("alpha", 0.6),
        feature_weight=dist_cfg.get("feature_distill", {}).get("weight", 0.5),
        device=device,
    )

    # 6. DataLoaders
    loaders = create_dataloaders(
        dataset_dict=dataset_dict,
        tokenizer=student.tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        text_col=config["dataset"]["text_col"],
        label_col=config["dataset"]["label_col"],
    )

    # 7. Optimizer & Scheduler
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if trainer.feat_loss_fn and not isinstance(trainer.feat_loss_fn.projection, nn.Identity):
        trainable_params += list(trainer.feat_loss_fn.parameters())

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = len(loaders["train"]) * epochs
    warmup_steps = int(total_steps * s_cfg.get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    tracker = ExperimentTracker(experiment_id=experiment_id, output_dir=config["paths"]["results_dir"])

    best_macro_f1 = -1.0
    best_weights = None
    best_epoch = -1
    start_time = time.time()

    logger.info(f"Training {alias} using {method} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        student.train()
        epoch_loss = 0.0
        train_samples = 0

        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch}/{epochs} [{alias}]")
        for batch in pbar:
            optimizer.zero_grad()
            loss, loss_breakdown = trainer.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            batch_len = len(batch["input_ids"])
            epoch_loss += loss.item() * batch_len
            train_samples += batch_len
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / train_samples if train_samples > 0 else 0.0

        # Validate
        val_metrics = evaluate_model(student, loaders["validation"], device=device)
        logger.info(
            f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Minority F1: {val_metrics['minority_class']['f1_score']:.4f}"
        )

        tracker.log_epoch(
            epoch=epoch,
            metrics={
                "train_loss": round(avg_loss, 4),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_minority_f1": val_metrics["minority_class"]["f1_score"],
            },
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_weights = copy.deepcopy(student.model.state_dict())
            best_epoch = epoch
            logger.info(f"[*] New best validation Macro-F1: {best_macro_f1:.4f} at epoch {epoch}")

    training_time = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {training_time}s. Best epoch: {best_epoch} (Val Macro-F1: {best_macro_f1:.4f})")

    # Load best weights
    if best_weights is not None:
        student.model.load_state_dict(best_weights)

    # 8. Test Set Evaluation
    logger.info("Evaluating best student checkpoint on untouched test set...")
    test_metrics = evaluate_model(student, loaders["test"], device=device)
    logger.info(
        f"Test Results ({alias} | {method}) - Acc: {test_metrics['accuracy']:.4f} | "
        f"Macro-F1: {test_metrics['macro_f1']:.4f} | "
        f"Weighted-F1: {test_metrics['weighted_f1']:.4f} | "
        f"Minority F1: {test_metrics['minority_class']['f1_score']:.4f}"
    )

    # 9. Save Checkpoint
    checkpoint_dir = Path(config["paths"]["checkpoints_dir"]) / experiment_id
    save_model_checkpoint(
        model=student.model,
        tokenizer=student.tokenizer,
        save_dir=str(checkpoint_dir),
        metadata={
            "experiment_id": experiment_id,
            "student_key": student_key,
            "method": method,
            "model_name": model_name,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "test_macro_f1": test_metrics["macro_f1"],
            "training_time_seconds": training_time,
            "device": device,
        },
    )
    checkpoint_size_mb = get_checkpoint_size_mb(str(checkpoint_dir))

    # 10. Record Final Results & Manifest
    final_record = {
        "model_name": model_name,
        "model_alias": alias,
        "model_type": "student",
        "student_key": student_key,
        "method": method,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_size_mb": checkpoint_size_mb,
        "parameters": param_stats,
        "best_epoch": best_epoch,
        "training_time_seconds": training_time,
        "device": device,
        "test_metrics": test_metrics,
        "status": "completed",
    }
    tracker.log_final(metrics=final_record)

    manifest_path = Path(config["paths"]["results_dir"]) / "manifest.json"
    update_manifest(
        manifest_path=str(manifest_path),
        experiment_entry={
            "experiment_id": experiment_id,
            "model_alias": alias,
            "model_type": "student",
            "student_key": student_key,
            "method": method,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "minority_f1": test_metrics["minority_class"]["f1_score"],
            "total_params": param_stats["total_parameters"],
            "checkpoint_size_mb": checkpoint_size_mb,
            "checkpoint_dir": str(checkpoint_dir),
            "status": "completed",
        },
    )

    return final_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student model using knowledge distillation.")
    parser.add_argument("--student", type=str, default="bert_mini", choices=["distilbert", "bert_mini", "bert_tiny"])
    parser.add_argument("--method", type=str, default="logit_distillation", choices=["supervised", "logit_distillation", "feature_distillation", "augmented_distillation"])
    parser.add_argument("--config", type=str, default="configs/quick_run.yaml")
    parser.add_argument("--experiment_id", type=str, default=None)
    args = parser.parse_args()
    train_student(
        student_key=args.student,
        method=args.method,
        config_path=args.config,
        experiment_id=args.experiment_id,
    )
