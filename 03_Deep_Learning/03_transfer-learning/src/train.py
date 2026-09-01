"""Train reproducible scratch and ResNet18 waste-image classifiers."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import balanced_class_weights, build_dataloaders
from models import build_model, count_parameters
from utils import save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a waste-classification transfer-learning experiment.")
    parser.add_argument("--model", choices=["scratch", "resnet-feature", "resnet-finetune"], required=True)
    parser.add_argument("--run-name", help="Unique experiment name. Defaults to model plus augmentation mode.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--augmentation", choices=["on", "off"], default="on")
    parser.add_argument("--class-weighting", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    loss_sum, examples = 0.0, 0
    labels_all, predictions_all = [], []
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
        loss_sum += loss.item() * labels.size(0)
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
        examples += labels.size(0)
    return {
        "loss": loss_sum / examples,
        "accuracy": sum(actual == predicted for actual, predicted in zip(labels_all, predictions_all)) / examples,
        "macro_f1": f1_score(labels_all, predictions_all, average="macro", zero_division=0),
    }


def update_csv(path: Path, row: dict[str, object]) -> None:
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            rows = [existing for existing in csv.DictReader(file) if existing["experiment"] != str(row["experiment"])]
    rows.append({key: str(value) for key, value in row.items()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history: list[dict[str, float]], run_name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    for metric, axis, label in (("loss", axes[0], "Cross-entropy loss"), ("macro_f1", axes[1], "Macro F1")):
        axis.plot(epochs, [row[f"train_{metric}"] for row in history], label="train")
        axis.plot(epochs, [row[f"validation_{metric}"] for row in history], label="validation")
        axis.set(xlabel="Epoch", ylabel=label, title=f"{run_name}: {label}")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch-size must be positive.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    augmented = args.augmentation == "on"
    run_name = args.run_name or f"{args.model}-{'aug' if augmented else 'noaug'}"
    results_dir, models_dir = Path(args.results_dir), Path(args.models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, _, class_names, targets = build_dataloaders(
        args.processed_dir, args.batch_size, augmented, args.num_workers
    )
    model = build_model(args.model, len(class_names), pretrained=not args.no_pretrained).to(device)
    weights = balanced_class_weights(targets, len(class_names)).to(device) if args.class_weighting == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    checkpoint_path = models_dir / f"best_{run_name}.pt"
    best_f1, stale_epochs, history = -1.0, 0, []
    started = time.perf_counter()
    print(f"Using {device}; experiment: {run_name}; classes: {', '.join(class_names)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_metrics["macro_f1"])
        history.append({"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in validation_metrics.items()}})
        print(f"[{run_name}] epoch {epoch}/{args.epochs} | val loss {validation_metrics['loss']:.4f} | val macro F1 {validation_metrics['macro_f1']:.4f}")
        if validation_metrics["macro_f1"] > best_f1:
            best_f1, stale_epochs = validation_metrics["macro_f1"], 0
            torch.save({"architecture": args.model, "class_names": class_names, "augmentation": args.augmentation, "class_weighting": args.class_weighting, "epoch": epoch, "validation_macro_f1": best_f1, "state_dict": model.state_dict()}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    elapsed = time.perf_counter() - started
    summary = {"experiment": run_name, "model": args.model, "augmentation": args.augmentation, "class_weighting": args.class_weighting, "best_validation_macro_f1": best_f1, "training_time_seconds": elapsed, "trainable_parameters": count_parameters(model), "total_parameters": count_parameters(model, trainable_only=False), "checkpoint_size_bytes": checkpoint_path.stat().st_size}
    save_json(results_dir / f"{run_name}-history.json", history)
    update_csv(results_dir / "validation-model-comparison.csv", summary)
    plot_history(history, run_name, Path(args.assets_dir) / f"{run_name}-learning-curve.png")
    print(f"Saved best validation checkpoint: {checkpoint_path}")
    print("Run evaluate.py once after selecting this experiment; the test set was not used during training.")


if __name__ == "__main__":
    main()
