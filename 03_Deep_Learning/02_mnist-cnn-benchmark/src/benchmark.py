"""Measure checkpoint inference latency on the selected CPU or GPU device."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

from config import parser_with_config
from evaluate import load_checkpoint
from train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = parser_with_config("Benchmark MNIST checkpoint inference latency.", {"model": "both", "models_dir": "models", "results_dir": "results", "device": "auto"})
    parser.add_argument("--model", choices=["fcn", "cnn", "both"], default=argparse.SUPPRESS)
    parser.add_argument("--batch-sizes", default="1,128")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--models-dir", default=argparse.SUPPRESS)
    parser.add_argument("--results-dir", default=argparse.SUPPRESS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=argparse.SUPPRESS)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(model: torch.nn.Module, batch_size: int, device: torch.device, warmup: int, iterations: int) -> dict[str, float | int]:
    inputs = torch.randn(batch_size, 1, 28, 28, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)
        synchronize(device)
        measurements = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(inputs)
            synchronize(device)
            measurements.append((time.perf_counter() - started) * 1000)
    mean_ms = statistics.fmean(measurements)
    return {
        "batch_size": batch_size,
        "mean_latency_ms": mean_ms,
        "p50_latency_ms": statistics.median(measurements),
        "throughput_images_per_second": batch_size / (mean_ms / 1000),
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive.")
    device = resolve_device(args.device)
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if any(value < 1 for value in batch_sizes):
        raise ValueError("batch sizes must be positive.")
    names = ["fcn", "cnn"] if args.model == "both" else [args.model]
    rows = []
    for name in names:
        model = load_checkpoint(name, Path(args.models_dir), device)
        for batch_size in batch_sizes:
            rows.append({"model": name, "device": str(device), **measure(model, batch_size, device, args.warmup, args.iterations)})
    destination = Path(args.results_dir) / "inference-benchmark.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved inference benchmark: {destination}")


if __name__ == "__main__":
    main()
