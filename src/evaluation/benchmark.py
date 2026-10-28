"""Inference latency, throughput, and hardware benchmarking harness."""
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from src.models.model_loader import count_parameters, get_checkpoint_size_mb


def benchmark_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str = "cpu",
    seq_length: int = 128,
    batch_sizes: Optional[List[int]] = None,
    num_runs: int = 25,
    warmup_runs: int = 5,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure single-example latency, batched latency, throughput, and parameter metrics."""
    if batch_sizes is None:
        batch_sizes = [1, 8]

    model.eval()
    model.to(device)

    is_cuda = (device == "cuda") and torch.cuda.is_available()

    # Create dummy synthetic input to measure pure model inference time (excluding tokenization)
    dummy_text = "This is a representative scientific citation sentence describing previous methodology and findings."
    encoded = tokenizer(
        [dummy_text] * max(batch_sizes),
        max_length=seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    all_input_ids = encoded["input_ids"].to(device)
    all_attention_mask = encoded["attention_mask"].to(device)

    token_type_ids = None
    if "token_type_ids" in encoded and "token_type_ids" in tokenizer.model_input_names:
        token_type_ids = encoded["token_type_ids"].to(device)

    results_by_batch: Dict[str, Any] = {}

    for bs in batch_sizes:
        cur_input_ids = all_input_ids[:bs]
        cur_attention_mask = all_attention_mask[:bs]
        cur_token_types = token_type_ids[:bs] if token_type_ids is not None else None

        kwargs = {
            "input_ids": cur_input_ids,
            "attention_mask": cur_attention_mask,
        }
        if cur_token_types is not None:
            kwargs["token_type_ids"] = cur_token_types

        # Warmup iterations
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(**kwargs)
                if is_cuda:
                    torch.cuda.synchronize()

        # Benchmark runs
        latencies: List[float] = []
        with torch.no_grad():
            for _ in range(num_runs):
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                _ = model(**kwargs)

                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # Convert to milliseconds

        lat_array = np.array(latencies)
        mean_lat = float(np.mean(lat_array))
        median_lat = float(np.median(lat_array))
        p95_lat = float(np.percentile(lat_array, 95))
        throughput = float((bs / (median_lat / 1000.0))) if median_lat > 0 else 0.0

        results_by_batch[f"batch_{bs}"] = {
            "batch_size": bs,
            "latency_mean_ms": round(mean_lat, 2),
            "latency_median_ms": round(median_lat, 2),
            "latency_p95_ms": round(p95_lat, 2),
            "throughput_examples_per_sec": round(throughput, 1),
            "num_runs": num_runs,
        }

    # Parameter and disk size
    param_info = count_parameters(model)
    disk_size_mb = get_checkpoint_size_mb(checkpoint_dir) if checkpoint_dir else param_info["param_size_mb"]

    single_batch = results_by_batch.get("batch_1", {})

    return {
        "device": device,
        "is_cuda": is_cuda,
        "sequence_length": seq_length,
        "total_parameters": param_info["total_parameters"],
        "param_size_mb": param_info["param_size_mb"],
        "checkpoint_size_mb": disk_size_mb,
        "latency_single_ms": single_batch.get("latency_median_ms", 0.0),
        "latency_p95_ms": single_batch.get("latency_p95_ms", 0.0),
        "throughput_single_eps": single_batch.get("throughput_examples_per_sec", 0.0),
        "batch_benchmarks": results_by_batch,
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "processor": platform.processor(),
            "os": platform.platform(),
        },
    }
