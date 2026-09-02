"""Inference-only model loading and prediction utilities.

This module deliberately does not import either training entry point. Baseline
evaluation therefore does not initialize Transformer training code.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class TokenizedTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int) -> None:
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            **{key: torch.tensor(value[index]) for key, value in self.encodings.items()},
            "labels": torch.tensor(self.labels[index]),
        }


def load_baseline_artifact(path: Path) -> dict[str, object]:
    return joblib.load(path)


def predict_baseline(artifact: dict[str, object], texts) -> tuple[np.ndarray, np.ndarray]:
    probabilities = artifact["pipeline"].predict_proba(texts)
    return probabilities.argmax(axis=1), probabilities


def transformer_predictions(
    model_dir: Path, texts: list[str], batch_size: int, device: torch.device
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, object]]:
    """Return label IDs and probability rows from a saved Transformer artifact."""

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    classes = metadata["class_names"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    dataset = TokenizedTextDataset(texts, np.zeros(len(texts), dtype=np.int64), tokenizer, metadata["max_length"])
    loader = DataLoader(dataset, batch_size=batch_size)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            rows.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(rows, axis=0)
    return classes, probabilities.argmax(axis=1), probabilities, metadata
