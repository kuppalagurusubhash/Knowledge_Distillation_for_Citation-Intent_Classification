"""SciBERT Teacher model wrapper."""
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from .model_loader import load_model_and_tokenizer, count_parameters


class TeacherModel(nn.Module):
    """Teacher model wrapper for SciBERT citation intent classification."""

    def __init__(
        self,
        model_name_or_path: str = "allenai/scibert_scivocab_uncased",
        num_labels: int = 3,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name_or_path
        self.device = device
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name_or_path,
            num_labels=num_labels,
            device=device,
            output_hidden_states=True,
        )
        self.num_labels = num_labels
        self.hidden_size = getattr(
            self.model.config, "hidden_size", getattr(self.model.config, "dim", 768)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through SciBERT teacher."""
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

        return res

    def get_parameter_stats(self) -> Dict[str, Any]:
        """Get parameter counts and footprint."""
        return count_parameters(self.model)
