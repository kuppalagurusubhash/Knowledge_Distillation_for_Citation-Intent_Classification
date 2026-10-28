"""SciBERT Teacher training, validation, early stopping, and test evaluation pipeline."""
import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data.dataset_loader import load_scicite_data
from src.data.preprocessing import CitationDataset, create_dataloaders
from src.evaluation.classification_metrics import compute_metrics
from src.models.model_loader import count_parameters, get_checkpoint_size_mb
from src.models.teacher import TeacherModel
from src.training.cache_logits import cache_teacher_outputs
from src.utils.checkpointing import save_model_checkpoint, update_manifest
from src.utils.config import get_device, load_config
from src.utils.logging_utils import ExperimentTracker, get_logger
from src.utils.seed import set_seed


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    class_names: Optional[list] = None,
) -> Dict[str, Any]:
    """Evaluate model on a DataLoader and compute loss and classification metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids

            outputs = model(**kwargs)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    metrics = compute_metrics(
        all_labels, all_preds, class_names=class_names or ["method", "background", "result"]
    )
    metrics["loss"] = round(avg_loss, 4)
    return metrics


def train_teacher(
    config_path: str = "configs/default.yaml",
    experiment_id: str = "teacher_scibert",
) -> Dict[str, Any]:
    """Train SciBERT teacher model with early stopping and logit caching."""
    config = load_config(config_path)
    set_seed(config.get("seed", 42), config.get("deterministic", True))
    device = get_device(config.get("device", "auto"))

    logs_dir = Path(config["paths"]["logs_dir"])
    logger = get_logger("Train-Teacher", log_file=str(logs_dir / f"{experiment_id}.log"))
    logger.info(f"Starting teacher training. Device: {device}. Experiment: {experiment_id}")

    # 1. Load Data
    dataset_dict, data_stats = load_scicite_data(config, save_stats=True)
    t_cfg = config["training"]["teacher"]
    model_name = t_cfg["model_name"]
    batch_size = t_cfg.get("batch_size", 16)
    max_length = config["dataset"].get("max_length", 128)
    epochs = t_cfg.get("epochs", 3)
    lr = float(t_cfg.get("learning_rate", 2e-5))
    weight_decay = float(t_cfg.get("weight_decay", 0.01))
    patience = t_cfg.get("early_stopping_patience", 2)

    # 2. Initialize Model and Tokenizer
    teacher = TeacherModel(model_name_or_path=model_name, num_labels=3, device=device)
    param_stats = teacher.get_parameter_stats()
    logger.info(f"Loaded Teacher ({model_name}). Parameters: {param_stats['total_parameters']:,} ({param_stats['param_size_mb']} MB)")

    # 3. Create DataLoaders
    loaders = create_dataloaders(
        dataset_dict=dataset_dict,
        tokenizer=teacher.tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        text_col=config["dataset"]["text_col"],
        label_col=config["dataset"]["label_col"],
    )

    # 4. Optimizer & Scheduler
    optimizer = AdamW(teacher.model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(loaders["train"]) * epochs
    warmup_steps = int(total_steps * t_cfg.get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    tracker = ExperimentTracker(experiment_id=experiment_id, output_dir=config["paths"]["results_dir"])

    best_macro_f1 = -1.0
    best_weights = None
    best_epoch = -1
    no_improve_epochs = 0

    # 5. Training Loop
    logger.info(f"Beginning training for {epochs} epochs (Total training steps: {total_steps})...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        teacher.train()
        train_loss = 0.0
        train_samples = 0

        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs["logits"]
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.model.parameters(), max_norm=t_cfg.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)
            train_samples += len(labels)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0

        # Validate
        val_metrics = evaluate_model(teacher, loaders["validation"], device=device)
        logger.info(
            f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Minority F1 ({val_metrics['minority_class']['name']}): {val_metrics['minority_class']['f1_score']:.4f}"
        )

        tracker.log_epoch(
            epoch=epoch,
            metrics={
                "train_loss": round(avg_train_loss, 4),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_minority_f1": val_metrics["minority_class"]["f1_score"],
            },
        )

        # Early stopping check on validation Macro-F1
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_weights = copy.deepcopy(teacher.model.state_dict())
            best_epoch = epoch
            no_improve_epochs = 0
            logger.info(f"[*] New best validation Macro-F1: {best_macro_f1:.4f} at epoch {epoch}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without validation Macro-F1 improvement.")
                break

    training_time = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {training_time}s. Best epoch: {best_epoch} (Val Macro-F1: {best_macro_f1:.4f})")

    # Load best weights before test evaluation
    if best_weights is not None:
        teacher.model.load_state_dict(best_weights)

    # 6. Evaluation on UNTOUCHED Test Set
    logger.info("Evaluating best teacher checkpoint on the untouched test set...")
    test_metrics = evaluate_model(teacher, loaders["test"], device=device)
    logger.info(
        f"Test Results - Acc: {test_metrics['accuracy']:.4f} | "
        f"Macro-F1: {test_metrics['macro_f1']:.4f} | "
        f"Weighted-F1: {test_metrics['weighted_f1']:.4f} | "
        f"Minority F1: {test_metrics['minority_class']['f1_score']:.4f}"
    )

    # 7. Save Checkpoint
    checkpoint_dir = Path(config["paths"]["checkpoints_dir"]) / experiment_id
    save_model_checkpoint(
        model=teacher.model,
        tokenizer=teacher.tokenizer,
        save_dir=str(checkpoint_dir),
        metadata={
            "experiment_id": experiment_id,
            "model_name": model_name,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "test_macro_f1": test_metrics["macro_f1"],
            "training_time_seconds": training_time,
            "device": device,
        },
    )
    checkpoint_size_mb = get_checkpoint_size_mb(str(checkpoint_dir))

    # 8. Record Final Experiment Summary
    final_record = {
        "model_name": model_name,
        "model_alias": t_cfg.get("alias", "SciBERT-Teacher"),
        "model_type": "teacher",
        "method": "reference_teacher",
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

    # Update global manifest
    manifest_path = Path(config["paths"]["results_dir"]) / "manifest.json"
    update_manifest(
        manifest_path=str(manifest_path),
        experiment_entry={
            "experiment_id": experiment_id,
            "model_alias": t_cfg.get("alias", "SciBERT-Teacher"),
            "model_type": "teacher",
            "method": "reference_teacher",
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "minority_f1": test_metrics["minority_class"]["f1_score"],
            "total_params": param_stats["total_parameters"],
            "checkpoint_size_mb": checkpoint_size_mb,
            "checkpoint_dir": str(checkpoint_dir),
            "status": "completed",
        },
    )

    # 9. Cache Teacher Logits on the Training Set
    cache_path = Path(config["paths"]["cache_dir"]) / "teacher_train_logits.pt"
    logger.info(f"Caching teacher logits on training set to {cache_path}...")
    cache_teacher_outputs(
        teacher=teacher,
        dataset=dataset_dict["train"],
        tokenizer=teacher.tokenizer,
        cache_path=str(cache_path),
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        save_hidden_states=True,
    )

    return final_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SciBERT teacher model on SciCite.")
    parser.add_argument("--config", type=str, default="configs/quick_run.yaml", help="Path to config YAML")
    parser.add_argument("--experiment_id", type=str, default="teacher_scibert", help="Unique experiment ID")
    args = parser.parse_args()
    train_teacher(config_path=args.config, experiment_id=args.experiment_id)
