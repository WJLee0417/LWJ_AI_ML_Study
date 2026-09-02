"""Train FCN and CNN models without using the held-out MNIST test set."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
import torch
from torch import nn, optim

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import parser_with_config
from data import build_dataloaders
from models import build_model, count_parameters
from utils import save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = parser_with_config("Train reproducible MNIST FCN/CNN benchmarks.", {"model": "both", "epochs": 5, "batch_size": 128, "learning_rate": 1e-3, "seed": 42, "num_workers": 0, "data_dir": "data/raw", "models_dir": "models", "results_dir": "results", "device": "auto"})
    parser.add_argument("--model", choices=["fcn", "cnn", "both"], default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--data-dir", default=argparse.SUPPRESS)
    parser.add_argument("--models-dir", default=argparse.SUPPRESS)
    parser.add_argument("--results-dir", default=argparse.SUPPRESS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one train or validation epoch and return mean loss and accuracy."""
    is_training = optimizer is not None
    model.train(is_training)
    loss_total, correct, examples = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if is_training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
        loss_total += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        examples += labels.size(0)
    return loss_total / examples, correct / examples


def save_checkpoint(
    model: nn.Module,
    model_name: str,
    epoch: int,
    validation_accuracy: float,
    path: Path,
) -> None:
    torch.save(
        {
            "architecture": model_name,
            "epoch": epoch,
            "validation_accuracy": validation_accuracy,
            "state_dict": model.state_dict(),
        },
        path,
    )


def train_model(
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_path = Path(args.models_dir) / f"best_{model_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"[{model_name.upper()}] epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f}, val loss {validation_loss:.4f}, "
            f"val accuracy {validation_accuracy:.4f}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(model, model_name, epoch, validation_accuracy, checkpoint_path)

    duration = time.perf_counter() - start_time
    return (
        {
            "model": model_name,
            "parameters": count_parameters(model),
            "best_validation_accuracy": best_accuracy,
            "training_time_seconds": duration,
            "checkpoint": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        },
        history,
    )


def write_validation_comparison(rows: list[dict[str, object]], results_dir: Path) -> None:
    destination = results_dir / "validation-model-comparison.csv"
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_curves(histories: dict[str, list[dict[str, float]]], path: Path) -> None:
    """Save train/validation loss curves for each trained model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, history in histories.items():
        epochs = [entry["epoch"] for entry in history]
        axis.plot(epochs, [entry["train_loss"] for entry in history], label=f"{name.upper()} train")
        axis.plot(
            epochs,
            [entry["validation_loss"] for entry in history],
            linestyle="--",
            label=f"{name.upper()} validation",
        )
    axis.set(title="MNIST training and validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive integers.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    train_loader, validation_loader, _ = build_dataloaders(
        args.data_dir, args.batch_size, args.seed, num_workers=args.num_workers
    )
    names = ["fcn", "cnn"] if args.model == "both" else [args.model]
    rows, histories = [], {}
    for name in names:
        result, history = train_model(name, train_loader, validation_loader, args, device)
        rows.append(result)
        histories[name] = history

    save_json(results_dir / "training-history.json", histories)
    save_json(results_dir / "validation-metrics.json", rows)
    save_json(
        results_dir / "runtime-info.json",
        {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "seed": args.seed,
            "config": vars(args),
        },
    )
    write_validation_comparison(rows, results_dir)
    plot_loss_curves(histories, Path("assets") / "loss-curve.png")
    print("Training complete. Run `python src/evaluate.py` to evaluate the untouched test set.")


if __name__ == "__main__":
    main()
