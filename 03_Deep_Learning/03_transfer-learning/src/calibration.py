"""Temperature-scale validation logits and export a deployment manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import log_loss
from torch import nn, optim

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import build_dataloaders
from models import build_model
from train import resolve_device
from utils import save_json, set_seed


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temperature.exp().clamp_min(1e-3)


def collect_logits(model: nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            logits.append(model(images.to(device)).cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    scaler = TemperatureScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(scaler.log_temperature.exp().item())


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    boundaries = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if in_bin.any():
            error += abs(correct[in_bin].mean() - confidence[in_bin].mean()) * in_bin.mean()
    return float(error)


def choose_review_threshold(labels: np.ndarray, probabilities: np.ndarray, minimum_precision: float) -> dict[str, float | int]:
    confidence, predicted = probabilities.max(axis=1), probabilities.argmax(axis=1)
    best = None
    for threshold in np.unique(confidence):
        automatic = confidence >= threshold
        precision = float((labels[automatic] == predicted[automatic]).mean())
        count = int(automatic.sum())
        if precision >= minimum_precision and (best is None or count > best["automatic_count"]):
            best = {"review_threshold": float(threshold), "automatic_count": count, "automatic_precision": precision}
    return best or {"review_threshold": 1.0, "automatic_count": 0, "automatic_precision": 0.0}


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_reliability(before: np.ndarray, after: np.ndarray, labels: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    for name, probabilities in (("Before scaling", before), ("After scaling", after)):
        confidence, correct = probabilities.max(axis=1), probabilities.argmax(axis=1) == labels
        x_values, y_values = [], []
        for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
            if mask.any():
                x_values.append(confidence[mask].mean())
                y_values.append(correct[mask].mean())
        axis.plot(x_values, y_values, marker="o", label=name)
    axis.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axis.set(xlabel="Mean confidence", ylabel="Observed accuracy", title="Validation reliability diagram")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a selected transfer-learning checkpoint on validation data.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--minimum-auto-precision", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.minimum_auto_precision <= 1:
        raise ValueError("minimum-auto-precision must be in (0, 1].")
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.models_dir) / f"best_{args.run_name}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _, validation_loader, _, classes, _ = build_dataloaders(args.processed_dir, args.batch_size, augment=False, num_workers=args.num_workers)
    if checkpoint["class_names"] != classes:
        raise ValueError("Dataset class order differs from the checkpoint.")
    model = build_model(checkpoint["architecture"], len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    logits, labels = collect_logits(model, validation_loader, device)
    labels_array = labels.numpy()
    before = torch.softmax(logits, dim=1).numpy()
    temperature = fit_temperature(logits, labels)
    after = torch.softmax(logits / temperature, dim=1).numpy()
    policy = choose_review_threshold(labels_array, after, args.minimum_auto_precision)
    artifact_dir = Path(args.artifacts_dir) / args.run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_version": args.model_version,
        "architecture": checkpoint["architecture"],
        "class_names": classes,
        "image_size": checkpoint.get("image_size", 224),
        "normalization": checkpoint.get("normalization", "imagenet"),
        "review_threshold": policy["review_threshold"],
        "validation_macro_f1": checkpoint["validation_macro_f1"],
        "calibration_temperature": temperature,
        "validation_nll_before": log_loss(labels_array, before, labels=range(len(classes))),
        "validation_nll_after": log_loss(labels_array, after, labels=range(len(classes))),
        "validation_ece_before": expected_calibration_error(before, labels_array),
        "validation_ece_after": expected_calibration_error(after, labels_array),
        "minimum_auto_precision": args.minimum_auto_precision,
        "automatic_validation_count": policy["automatic_count"],
        "automatic_validation_precision": policy["automatic_precision"],
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_hash(checkpoint_path),
    }
    save_json(artifact_dir / "manifest.json", manifest)
    plot_reliability(before, after, labels_array, Path(args.assets_dir) / f"{args.run_name}-reliability.png")
    print(f"Saved calibrated deployment manifest: {artifact_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
