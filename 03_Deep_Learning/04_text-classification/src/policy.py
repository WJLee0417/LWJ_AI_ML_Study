"""Validation-derived confidence policy for automation versus human review."""

from __future__ import annotations

import numpy as np


def choose_review_threshold(
    actual: np.ndarray, predicted: np.ndarray, confidence: np.ndarray, minimum_precision: float
) -> dict[str, float | int]:
    """Maximize automated volume while meeting the requested validation precision."""
    if not 0 < minimum_precision <= 1:
        raise ValueError("minimum_precision must be in (0, 1].")
    candidates = np.unique(confidence)
    best = None
    for threshold in candidates:
        automated = confidence >= threshold
        count = int(automated.sum())
        if not count:
            continue
        precision = float((actual[automated] == predicted[automated]).mean())
        if precision >= minimum_precision and (best is None or count > best["automated_count"]):
            best = {"threshold": float(threshold), "automated_count": count, "automated_precision": precision}
    if best is None:
        return {"threshold": 1.01, "automated_count": 0, "automated_precision": 0.0}
    return best


def apply_review_policy(confidence: np.ndarray, threshold: float) -> np.ndarray:
    return confidence < threshold


def operational_metrics(actual: np.ndarray, predicted: np.ndarray, confidence: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Report the automation and human-review metrics used for operations."""

    needs_review = apply_review_policy(confidence, threshold)
    automated = ~needs_review
    automated_count = int(automated.sum())
    automated_precision = float((actual[automated] == predicted[automated]).mean()) if automated_count else 0.0
    return {
        "automated_count": automated_count,
        "review_count": int(needs_review.sum()),
        "automated_coverage": float(automated.mean()),
        "automated_precision": automated_precision,
        "review_rate": float(needs_review.mean()),
    }
