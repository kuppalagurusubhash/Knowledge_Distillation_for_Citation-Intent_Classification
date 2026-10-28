"""Student model family and wrappers supporting multiple compact architectures."""
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from .model_loader import load_model_and_tokenizer, count_parameters


class StudentModel(nn.Module):
    """
    Student model wrapper for compact transformer architectures
    (e.g., DistilBERT, BERT-mini, BERT-tiny).
    """

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased",
        num_labels: int = 3,
        teacher_hidden_size: Optional[int] = 768,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name_or_path
        self.device = device
        self.num_labels = num_labels

        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name_or_path,
            num_labels=num_labels,
            device=device,
            output_hidden_states=True,
        )

        self.hidden_size = getattr(
            self.model.config, "hidden_size", getattr(self.model.config, "dim", 768)
        )

        # Learnable projection head for feature distillation
        if teacher_hidden_size is not None and self.hidden_size != teacher_hidden_size:
            self.feature_projector = nn.Linear(self.hidden_size, teacher_hidden_size, bias=False).to(device)
        else:
            self.feature_projector = nn.Identity().to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through Student model."""
        kwargs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "return_dict": True,
            "output_hidden_states": True,
        }
        if token_type_ids is not None and "token_type_ids" in self.tokenizer.model_input_names:
            kwargs["token_type_ids"] = token_type_ids.to(self.device)
        if labels is not None:
            kwargs["labels"] = labels.to(self.device)

        outputs = self.model(**kwargs)
        res = {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
        }

        if return_hidden_states and outputs.hidden_states is not None:
            res["hidden_states"] = outputs.hidden_states
            # Project the final hidden state to teacher dimension
            last_hidden = outputs.hidden_states[-1]
            res["projected_hidden"] = self.feature_projector(last_hidden)

        return res

    def get_parameter_stats(self) -> Dict[str, Any]:
        """Compute parameter counts and size in MB."""
        stats = count_parameters(self.model)
        if not isinstance(self.feature_projector, nn.Identity):
            proj_params = sum(p.numel() for p in self.feature_projector.parameters())
            stats["projector_parameters"] = proj_params
        return stats
