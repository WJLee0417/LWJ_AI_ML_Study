"""Final test evaluation and Grad-CAM diagnostics for a selected experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import classification_report, confusion_matrix

from data import IMAGENET_MEAN, IMAGENET_STD, build_dataloaders
from models import build_model, count_parameters, gradcam_target_layer
from train import resolve_device
from utils import save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one selected waste-classification experiment.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def upsert_csv(path: Path, row: dict[str, object]) -> None:
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            rows = [existing for existing in csv.DictReader(file) if existing["experiment"] != str(row["experiment"])]
    rows.append({key: str(value) for key, value in row.items()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)


def predict(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, int, int]]]:
    """Collect metrics and only the first few errors needed for visual diagnostics."""
    actual, predicted, errors = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_images, labels in loader:
            logits = model(batch_images.to(device))
            batch_predictions = logits.argmax(dim=1).cpu()
            actual.extend(labels.tolist())
            predicted.extend(batch_predictions.tolist())
            for index, (label, prediction) in enumerate(zip(labels.tolist(), batch_predictions.tolist())):
                if label != prediction and len(errors) < 6:
                    errors.append((batch_images[index].numpy(), label, prediction))
    return np.asarray(actual), np.asarray(predicted), errors


class GradCAM:
    """Minimal Grad-CAM implementation that exposes the final convolutional activations."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model, self.activations, self.gradients = model, None, None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def heatmap(self, image: torch.Tensor, target: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        logits[0, target].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = functional.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = heatmap.squeeze().cpu().numpy()
        return heatmap / (heatmap.max() + 1e-8)

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def denormalize(image: np.ndarray) -> np.ndarray:
    mean = np.asarray(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD).reshape(3, 1, 1)
    return np.clip(np.transpose(image * std + mean, (1, 2, 0)), 0, 1)


def save_confusion_matrix(actual: np.ndarray, predicted: np.ndarray, labels: list[str], path: Path) -> None:
    matrix = confusion_matrix(actual, predicted, labels=range(len(labels)))
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(title="Test confusion matrix", xlabel="Predicted", ylabel="Actual")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_gradcam_samples(
    model: torch.nn.Module,
    architecture: str,
    errors: list[tuple[np.ndarray, int, int]],
    labels: list[str],
    device: torch.device,
    path: Path,
) -> None:
    if not errors:
        return
    cam = GradCAM(model, gradcam_target_layer(model, architecture))
    figure, axes = plt.subplots(len(errors), 2, figsize=(7, 3 * len(errors)), squeeze=False)
    try:
        for row, (source_image, actual_class, predicted_class) in enumerate(errors):
            image = torch.tensor(source_image[None, ...], device=device)
            heatmap = cam.heatmap(image, predicted_class)
            visible = denormalize(source_image)
            axes[row, 0].imshow(visible)
            axes[row, 0].set(title=f"True: {labels[actual_class]}\nPredicted: {labels[predicted_class]}")
            axes[row, 1].imshow(visible)
            axes[row, 1].imshow(heatmap, cmap="jet", alpha=0.45)
            axes[row, 1].set(title="Grad-CAM for predicted class")
            for axis in axes[row]:
                axis.axis("off")
        figure.tight_layout()
        figure.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        cam.close()
        plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.models_dir) / f"best_{args.run_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train this experiment first.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["architecture"], len(checkpoint["class_names"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    _, _, test_loader, classes, _ = build_dataloaders(args.processed_dir, args.batch_size, augment=False, num_workers=args.num_workers)
    if classes != checkpoint["class_names"]:
        raise ValueError("Class order differs from the checkpoint. Recreate the data split consistently.")
    actual, predicted, errors = predict(model, test_loader, device)
    report = classification_report(actual, predicted, target_names=classes, output_dict=True, zero_division=0)
    assets_dir, results_dir = Path(args.assets_dir), Path(args.results_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": args.run_name, "test_accuracy": report["accuracy"], "macro_precision": report["macro avg"]["precision"], "macro_recall": report["macro avg"]["recall"], "macro_f1": report["macro avg"]["f1-score"], "trainable_parameters": count_parameters(model), "model_size_bytes": checkpoint_path.stat().st_size}
    save_json(results_dir / f"{args.run_name}-test-metrics.json", {"summary": summary, "classes": classes, "metrics": report})
    upsert_csv(results_dir / "model-comparison.csv", summary)
    save_confusion_matrix(actual, predicted, classes, assets_dir / f"{args.run_name}-confusion-matrix.png")
    save_gradcam_samples(model, checkpoint["architecture"], errors, classes, device, assets_dir / f"{args.run_name}-gradcam-errors.png")
    print(f"[{args.run_name}] test accuracy={summary['test_accuracy']:.4f}, macro F1={summary['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
