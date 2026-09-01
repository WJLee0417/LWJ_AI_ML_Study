"""Evaluate saved MNIST checkpoints once on the untouched test set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from data import build_dataloaders
from models import build_model, count_parameters
from train import resolve_device
from utils import save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MNIST FCN/CNN checkpoints.")
    parser.add_argument("--model", choices=["fcn", "cnn", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_checkpoint(model_name: str, models_dir: Path, device: torch.device) -> torch.nn.Module:
    checkpoint_path = models_dir / f"best_{model_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual, predicted, images_out = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            actual.extend(labels.numpy())
            predicted.extend(logits.argmax(dim=1).cpu().numpy())
            images_out.extend(images.numpy())
    return np.asarray(actual), np.asarray(predicted), np.asarray(images_out)


def create_confusion_matrix_plot(results: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    figure, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    for axis, (name, result) in zip(axes[0], results.items()):
        matrix = confusion_matrix(result["actual"], result["predicted"], labels=range(10))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(title=f"{name.upper()} confusion matrix", xlabel="Predicted", ylabel="Actual")
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))
        for row in range(10):
            for column in range(10):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def create_misclassified_plot(results: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    figure, axes = plt.subplots(len(results), 8, figsize=(14, 2.5 * len(results)), squeeze=False)
    for row, (name, result) in enumerate(results.items()):
        errors = np.flatnonzero(result["actual"] != result["predicted"])[:8]
        for column, axis in enumerate(axes[row]):
            axis.axis("off")
            if column < len(errors):
                index = errors[column]
                axis.imshow(result["images"][index].squeeze(), cmap="gray")
                axis.set_title(
                    f"{name.upper()}\ntrue {result['actual'][index]}, pred {result['predicted'][index]}", fontsize=8
                )
    figure.suptitle("First eight misclassified test samples per model", y=1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_comparison(rows: list[dict[str, object]], results_dir: Path) -> None:
    destination = results_dir / "model-comparison.csv"
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    assets_dir, results_dir = Path(args.assets_dir), Path(args.results_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    _, _, test_loader = build_dataloaders(
        args.data_dir, args.batch_size, args.seed, num_workers=args.num_workers
    )
    names = ["fcn", "cnn"] if args.model == "both" else [args.model]
    rows, plot_data, detailed_metrics = [], {}, {}
    for name in names:
        model = load_checkpoint(name, Path(args.models_dir), device)
        actual, predicted, images = predict(model, test_loader, device)
        report = classification_report(actual, predicted, output_dict=True, zero_division=0)
        checkpoint_path = Path(args.models_dir) / f"best_{name}.pt"
        rows.append(
            {
                "model": name,
                "test_accuracy": report["accuracy"],
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "parameters": count_parameters(model),
                "model_size_bytes": checkpoint_path.stat().st_size,
            }
        )
        detailed_metrics[name] = report
        plot_data[name] = {"actual": actual, "predicted": predicted, "images": images}
        print(f"[{name.upper()}] test accuracy: {report['accuracy']:.4f}")

    write_comparison(rows, results_dir)
    save_json(results_dir / "metrics.json", {"models": detailed_metrics, "comparison": rows})
    create_confusion_matrix_plot(plot_data, assets_dir / "confusion-matrix.png")
    create_misclassified_plot(plot_data, assets_dir / "misclassified-samples.png")
    print(f"Saved metrics to {results_dir} and diagnostic images to {assets_dir}.")


if __name__ == "__main__":
    main()
