"""Preprocessing, tokenization, and PyTorch Dataset creation for citation intent models."""
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


class CitationDataset(Dataset):
    """PyTorch Dataset for Citation-Intent classification."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        ids: Optional[List[str]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ):
        self.texts = [str(t) if t is not None else "" for t in texts]
        self.labels = labels
        self.ids = ids if ids is not None else [str(i) for i in range(len(self.texts))]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "sample_idx": idx,
            "id": self.ids[idx],
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def tokenize_batch(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Tokenize a list of texts and return PyTorch tensor batch on target device."""
    clean_texts = [str(t) if t is not None else "" for t in texts]
    batch = tokenizer(
        clean_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in batch.items()}


def create_dataloaders(
    dataset_dict: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    max_length: int = 128,
    text_col: str = "string",
    label_col: str = "label",
    id_col: str = "id",
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test splits."""
    loaders = {}
    for split in ["train", "validation", "test"]:
        if split not in dataset_dict:
            continue
        data = dataset_dict[split]
        texts = data[text_col]
        labels = data[label_col] if label_col in data.column_names else None
        ids = data[id_col] if id_col in data.column_names else None

        ds = CitationDataset(
            texts=texts,
            labels=labels,
            ids=ids,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        is_train = split == "train"
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders
