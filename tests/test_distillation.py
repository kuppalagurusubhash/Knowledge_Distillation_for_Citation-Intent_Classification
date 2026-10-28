"""Unit test verifying the end-to-end distillation trainer across methods."""
import torch
from transformers import AutoTokenizer
from src.models.student import StudentModel
from src.training.distillation import DistillationTrainer


def test_distillation_trainer_step():
    model_name = "prajjwal1/bert-tiny"
    student = StudentModel(model_name_or_path=model_name, num_labels=3, teacher_hidden_size=768, device="cpu")

    trainer = DistillationTrainer(
        student_model=student,
        teacher_model=None,
        method="logit_distillation",
        temperature=2.0,
        alpha=0.5,
        device="cpu",
    )

    tokenizer = student.tokenizer
    encoded = tokenizer(["Sample citation sentence.", "Another reference."], padding=True, return_tensors="pt")
    batch = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor([1, 0]),
        "id": ["s1", "s2"],
    }

    loss, breakdown = trainer.compute_loss(batch)
    assert loss.item() > 0
    assert "total_loss" in breakdown
    loss.backward()

    # Verify gradients computed
    has_grad = any(p.grad is not None for p in student.parameters() if p.requires_grad)
    assert has_grad
