"""Unit tests for distillation and feature matching loss functions."""
import torch
from src.training.losses import DistillationLoss, FeatureDistillationLoss


def test_distillation_loss_ce_only():
    loss_fn = DistillationLoss(temperature=3.0, alpha=0.0)
    student_logits = torch.tensor([[2.0, -1.0, 0.5], [-0.5, 3.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])

    loss = loss_fn(student_logits, teacher_logits=None, labels=labels)
    assert loss.item() > 0
    loss.backward()
    assert student_logits.grad is not None


def test_distillation_loss_combined():
    loss_fn = DistillationLoss(temperature=2.0, alpha=0.5)
    student_logits = torch.tensor([[1.5, 0.2, -0.8], [0.1, 2.5, 0.4]], requires_grad=True)
    teacher_logits = torch.tensor([[2.0, 0.1, -0.5], [0.0, 3.0, 0.2]])
    labels = torch.tensor([0, 1])

    loss = loss_fn(student_logits, teacher_logits, labels)
    assert loss.item() > 0
    loss.backward()
    assert student_logits.grad is not None


def test_feature_distillation_loss_projection():
    # Student dim 128, Teacher dim 768
    loss_fn = FeatureDistillationLoss(student_dim=128, teacher_dim=768, loss_type="mse")
    student_hidden = torch.randn(2, 8, 128, requires_grad=True)
    teacher_hidden = torch.randn(2, 8, 768)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]])

    loss = loss_fn(student_hidden, teacher_hidden, attention_mask=mask)
    assert loss.item() > 0
    loss.backward()
    assert student_hidden.grad is not None
