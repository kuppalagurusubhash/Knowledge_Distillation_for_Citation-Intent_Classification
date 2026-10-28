"""Unit tests for inference latency, throughput benchmarking, and robustness evaluation."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.evaluation.benchmark import benchmark_model
from src.evaluation.robustness import evaluate_robustness


def test_benchmark_model_execution():
    # Use lightweight bert-tiny for testing
    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    results = benchmark_model(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        seq_length=32,
        batch_sizes=[1, 2],
        num_runs=3,
        warmup_runs=1,
    )

    assert "latency_single_ms" in results
    assert "throughput_single_eps" in results
    assert "total_parameters" in results
    assert results["latency_single_ms"] > 0
    assert "batch_1" in results["batch_benchmarks"]
    assert "batch_2" in results["batch_benchmarks"]


def test_evaluate_robustness():
    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    texts = [
        "We contrast our findings [12] with previous work.",
        "The standard framework was implemented following Smith et al. [15].",
    ]
    labels = [2, 0]

    robustness = evaluate_robustness(
        model=model,
        tokenizer=tokenizer,
        clean_texts=texts,
        labels=labels,
        device="cpu",
        batch_size=2,
        max_length=32,
        perturbation_types=["whitespace_normalize", "minor_punctuation"],
    )

    assert "clean_macro_f1" in robustness
    assert "perturbations" in robustness
    assert "whitespace_normalize" in robustness["perturbations"]
    assert "minor_punctuation" in robustness["perturbations"]
