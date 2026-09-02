"""Validation-only temperature scaling and automation-policy export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss
from torch import nn, optim

try:  # Supports both `python src/...` and `from src...` test execution.
    from .data import load_csv
    from .inference import load_baseline_artifact, predict_baseline, resolve_device, transformer_predictions
    from .policy import choose_review_threshold, operational_metrics
    from .utils import save_json
except ImportError:  # pragma: no cover - exercised by command-line scripts.
    from data import load_csv
    from inference import load_baseline_artifact, predict_baseline, resolve_device, transformer_predictions
    from policy import choose_review_threshold, operational_metrics
    from utils import save_json


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temperature.exp().clamp_min(1e-3)


def probability_logits(probabilities: np.ndarray) -> np.ndarray:
    """Use log probabilities as logits; softmax(log(p)) reconstructs p."""

    return np.log(np.clip(probabilities, 1e-12, 1.0))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    logits = probability_logits(probabilities) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, actual: np.ndarray) -> float:
    logits = torch.tensor(probability_logits(probabilities), dtype=torch.float32)
    labels = torch.tensor(actual, dtype=torch.long)
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


def expected_calibration_error(probabilities: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == actual
    boundaries = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            error += abs(correct[mask].mean() - confidence[mask].mean()) * mask.mean()
    return float(error)


def load_validation_probabilities(
    run_name: str, validation, models_dir: Path, batch_size: int, device: str
) -> tuple[str, list[str], np.ndarray, np.ndarray]:
    baseline_path = models_dir / f"{run_name}.joblib"
    if baseline_path.exists():
        artifact = load_baseline_artifact(baseline_path)
        encoder = artifact["label_encoder"]
        _, probabilities = predict_baseline(artifact, validation["text"])
        return "tfidf-logreg", encoder.classes_.tolist(), encoder.transform(validation["label"]), probabilities
    model_dir = models_dir / run_name
    if not model_dir.exists():
        raise FileNotFoundError(f"No baseline or Transformer artifact found for {run_name}.")
    classes, _, probabilities, _ = transformer_predictions(model_dir, validation["text"].tolist(), batch_size, resolve_device(device))
    if not set(validation["label"]).issubset(classes):
        raise ValueError("Validation labels are not compatible with the saved Transformer mapping.")
    return "transformer", classes, np.asarray([classes.index(label) for label in validation["label"]]), probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate one selected inquiry model on validation data only.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--validation-csv", default="data/processed/validation.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-auto-precision", type=float, default=0.9)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.minimum_auto_precision <= 1:
        raise ValueError("minimum-auto-precision must be in (0, 1].")
    validation = load_csv(args.validation_csv)
    model, classes, actual, before = load_validation_probabilities(
        args.run_name, validation, Path(args.models_dir), args.batch_size, args.device
    )
    temperature = fit_temperature(before, actual)
    after = apply_temperature(before, temperature)
    predicted, confidence = after.argmax(axis=1), after.max(axis=1)
    policy = choose_review_threshold(actual, predicted, confidence, args.minimum_auto_precision)
    policy.update({
        "minimum_auto_precision": args.minimum_auto_precision,
        "derived_from": "validation-temperature-scaling",
        **operational_metrics(actual, predicted, confidence, float(policy["threshold"])),
    })
    manifest = {
        "artifact_type": "temperature-calibration-v1",
        "run_name": args.run_name,
        "model": model,
        "class_names": classes,
        "calibration_temperature": temperature,
        "validation_nll_before": log_loss(actual, before, labels=range(len(classes))),
        "validation_nll_after": log_loss(actual, after, labels=range(len(classes))),
        "validation_ece_before": expected_calibration_error(before, actual),
        "validation_ece_after": expected_calibration_error(after, actual),
        "policy": policy,
    }
    artifact_path = Path(args.artifacts_dir) / args.run_name / "manifest.json"
    save_json(artifact_path, manifest)
    save_json(Path(args.results_dir) / f"{args.run_name}-calibration.json", manifest)
    print(f"Saved validation calibration manifest: {artifact_path}")


if __name__ == "__main__":
    main()
