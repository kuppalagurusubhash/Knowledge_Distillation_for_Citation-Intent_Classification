"""Knowledge Distillation framework supporting Supervised, Logit, Feature, and Augmented Distillation."""
import random
import re
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from src.training.losses import DistillationLoss, FeatureDistillationLoss


def perturb_text(text: str, perturbation_type: str = "minor_punctuation") -> str:
    """
    Apply controlled, label-preserving text perturbation for scientific citation sentences.
    Preserves citation semantics and factual references.
    """
    if not text or len(text.strip()) == 0:
        return text

    if perturbation_type == "whitespace_normalize":
        # Normalize redundant spaces and tab characters
        return " ".join(text.split())

    elif perturbation_type == "minor_punctuation":
        # Interchange bracket styles for citations, e.g. [12] -> (12) or vice versa
        if "[" in text and "]" in text:
            return re.sub(r"\[(\d+(?:,\s*\d+)*)\]", r"(\1)", text)
        elif "(" in text and ")" in text:
            return re.sub(r"\((\d+(?:,\s*\d+)*)\)", r"[\1]", text)
        else:
            # Minor harmless trailing punctuation normalization
            return text.rstrip(".") + "."

    elif perturbation_type == "typo_insertion":
        # Minor harmless typo in a non-keyword token (swap two adjacent characters in a long word > 5 chars)
        words = text.split()
        candidate_indices = [i for i, w in enumerate(words) if len(w) > 5 and w.isalpha()]
        if candidate_indices:
            idx = random.choice(candidate_indices)
            w = list(words[idx])
            # Swap middle characters
            swap_pos = len(w) // 2
            w[swap_pos], w[swap_pos + 1] = w[swap_pos + 1], w[swap_pos]
            words[idx] = "".join(w)
            return " ".join(words)
        return text

    return text


class DistillationTrainer:
    """Unified trainer for Knowledge Distillation experiments across student architectures."""

    def __init__(
        self,
        student_model,
        teacher_model=None,
        cached_teacher_logits: Optional[torch.Tensor] = None,
        cached_sample_ids: Optional[List[str]] = None,
        method: str = "logit_distillation",  # 'supervised', 'logit_distillation', 'feature_distillation', 'augmented_distillation'
        temperature: float = 3.0,
        alpha: float = 0.6,
        feature_weight: float = 0.5,
        device: str = "cpu",
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.method = method
        self.device = device
        self.temperature = temperature
        self.alpha = alpha if method != "supervised" else 0.0
        self.feature_weight = feature_weight

        # Caching lookup table by ID
        self.cached_logits = cached_teacher_logits
        self.cached_id_map = (
            {cid: idx for idx, cid in enumerate(cached_sample_ids)}
            if cached_sample_ids is not None
            else None
        )

        # Loss functions
        self.kd_loss_fn = DistillationLoss(temperature=temperature, alpha=self.alpha)

        if method == "feature_distillation":
            teacher_dim = getattr(
                teacher_model, "hidden_size", 768
            ) if teacher_model else 768
            self.feat_loss_fn = FeatureDistillationLoss(
                student_dim=student_model.hidden_size,
                teacher_dim=teacher_dim,
                loss_type="mse",
                normalize=True,
            ).to(device)
        else:
            self.feat_loss_fn = None

    def compute_loss(
        self,
        batch: Dict[str, Any],
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss based on selected distillation method."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(self.device)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None and "token_type_ids" in self.student.tokenizer.model_input_names:
            token_type_ids = token_type_ids.to(self.device)
        else:
            token_type_ids = None

        # Student Forward pass
        need_hidden = self.method == "feature_distillation"
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_hidden_states=need_hidden,
        )
        student_logits = student_outputs["logits"]

        teacher_logits = None
        teacher_hidden = None

        # Retrieve Teacher Logits
        if self.method in ["logit_distillation", "feature_distillation", "augmented_distillation"]:
            # Check cached logits first
            if self.cached_logits is not None and self.cached_id_map is not None and "id" in batch:
                batch_ids = batch["id"]
                indices = [self.cached_id_map[cid] for cid in batch_ids if cid in self.cached_id_map]
                if len(indices) == len(batch_ids):
                    teacher_logits = self.cached_logits[indices].to(self.device)

            # Fallback to live teacher if cached logits not available
            if teacher_logits is None and self.teacher is not None:
                with torch.no_grad():
                    t_out = self.teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        return_hidden_states=need_hidden,
                    )
                    teacher_logits = t_out["logits"]
                    if need_hidden and "hidden_states" in t_out:
                        teacher_hidden = t_out["hidden_states"][-1]

        # Calculate Base Distillation Loss (CE + Logit KD)
        total_loss = self.kd_loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
        )
        loss_breakdown = {"base_loss": total_loss.item()}

        # Feature distillation addition
        if self.method == "feature_distillation" and self.feat_loss_fn is not None:
            if teacher_hidden is not None and "projected_hidden" in student_outputs:
                feat_loss = self.feat_loss_fn(
                    student_hidden=student_outputs["projected_hidden"],
                    teacher_hidden=teacher_hidden,
                    attention_mask=attention_mask,
                )
                total_loss = total_loss + self.feature_weight * feat_loss
                loss_breakdown["feature_loss"] = feat_loss.item()

        loss_breakdown["total_loss"] = total_loss.item()
        return total_loss, loss_breakdown
