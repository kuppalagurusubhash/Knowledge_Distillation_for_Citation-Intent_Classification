"""Loss functions for Knowledge Distillation: Logit KL Divergence, Supervised Cross-Entropy, and Feature Projection Loss."""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Combined Cross-Entropy and Temperature-Scaled KL-Divergence Loss."""

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.6,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute total distillation loss.
        If teacher_logits is None or alpha == 0, falls back to standard supervised cross-entropy.
        """
        loss = torch.tensor(0.0, device=student_logits.device)

        if labels is not None:
            ce_loss = self.ce_loss_fn(student_logits, labels)
        else:
            ce_loss = torch.tensor(0.0, device=student_logits.device)

        if teacher_logits is not None and self.alpha > 0.0:
            # Scale logits by temperature T
            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

            # KL Divergence scaled by T^2 as prescribed by Hinton et al.
            kd_loss = self.kl_loss_fn(student_soft, teacher_soft) * (self.temperature ** 2)

            loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss
        else:
            loss = ce_loss

        return loss


class FeatureDistillationLoss(nn.Module):
    """
    Hidden-state / feature-based distillation loss.
    Projects student hidden dimension to teacher hidden dimension and computes
    mean squared error or cosine distance over non-padding tokens.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        loss_type: str = "mse",
        normalize: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize

        # Learnable projection head if dimensions differ
        if student_dim != teacher_dim:
            self.projection = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        student_hidden: [batch_size, seq_len, student_dim]
        teacher_hidden: [batch_size, seq_len, teacher_dim]
        attention_mask: [batch_size, seq_len] with 1 for valid tokens, 0 for pad
        """
        # Project student hidden states to teacher space
        proj_student = self.projection(student_hidden)

        if self.normalize:
            proj_student = F.normalize(proj_student, p=2, dim=-1)
            teacher_hidden = F.normalize(teacher_hidden, p=2, dim=-1)

        if self.loss_type == "cosine":
            # 1 - cosine similarity
            token_loss = 1.0 - F.cosine_similarity(proj_student, teacher_hidden, dim=-1)  # [B, L]
        else:
            # MSE per token
            token_loss = torch.sum((proj_student - teacher_hidden) ** 2, dim=-1)  # [B, L]

        if attention_mask is not None:
            # Mask out padding tokens
            mask = attention_mask.float()
            token_loss = token_loss * mask
            total_tokens = mask.sum()
            loss = token_loss.sum() / (total_tokens + 1e-8)
        else:
            loss = token_loss.mean()

        return loss
