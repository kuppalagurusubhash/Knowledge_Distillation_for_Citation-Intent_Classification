"""Unit tests for dataset loading, schema verification, and preprocessing."""
import pytest
from transformers import AutoTokenizer
from src.utils.config import load_config
from src.data.dataset_loader import load_scicite_data, get_dataset_stats, LABEL_NAMES
from src.data.preprocessing import CitationDataset, create_dataloaders


def test_config_loading():
    config = load_config("configs/quick_run.yaml")
    assert "dataset" in config
    assert "training" in config
    assert config["dataset"]["class_names"] == ["method", "background", "result"]


def test_scicite_loading_and_subsampling():
    config = load_config("configs/quick_run.yaml")
    # Set small subset for testing
    config["dataset"]["max_train_samples"] = 50
    config["dataset"]["max_val_samples"] = 20
    config["dataset"]["max_test_samples"] = 20

    dataset_dict, stats = load_scicite_data(config, save_stats=False)

    assert "train" in dataset_dict
    assert "validation" in dataset_dict
    assert "test" in dataset_dict
    assert len(dataset_dict["train"]) == 50
    assert len(dataset_dict["validation"]) == 20
    assert len(dataset_dict["test"]) == 20

    assert "splits" in stats
    assert stats["minority_class"]["name"] == "result"
    assert len(stats["class_weights"]) == 3


def test_citation_dataset_tokenization():
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    texts = [
        "We compare our approach against the baseline method.",
        "Previous studies established the foundation for this theory.",
    ]
    labels = [2, 1]
    ids = ["id_1", "id_2"]

    ds = CitationDataset(texts, labels, ids, tokenizer, max_length=32)
    assert len(ds) == 2

    item0 = ds[0]
    assert "input_ids" in item0
    assert "attention_mask" in item0
    assert item0["input_ids"].shape[0] == 32
    assert item0["labels"].item() == 2
    assert item0["id"] == "id_1"
