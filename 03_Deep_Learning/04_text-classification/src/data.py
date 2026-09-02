"""CSV contract, PII masking, and deterministic split utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:  # Supports both `python src/...` and `from src...` test execution.
    from .pii import mask_pii
except ImportError:  # pragma: no cover - exercised by command-line scripts.
    from pii import mask_pii

REQUIRED_COLUMNS = {"text", "label"}
TIMESTAMP_COLUMN = "timestamp"


def load_csv(path: str | Path, include_timestamp: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    columns = ["text", "label"]
    if include_timestamp:
        if TIMESTAMP_COLUMN not in frame.columns:
            raise ValueError(f"{path} needs a '{TIMESTAMP_COLUMN}' column for a temporal split.")
        columns.append(TIMESTAMP_COLUMN)
    frame = frame.loc[:, columns].dropna().copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["label"] != "")]
    if include_timestamp:
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce", utc=True)
        if frame[TIMESTAMP_COLUMN].isna().any():
            raise ValueError(f"{path} has invalid '{TIMESTAMP_COLUMN}' values; use ISO-8601 dates.")
    masked = frame["text"].map(mask_pii)
    frame["text"] = [item[0] for item in masked]
    if frame["label"].nunique() < 2:
        raise ValueError("At least two labels are required.")
    return frame.reset_index(drop=True)


def pii_masking_summary(original: pd.Series) -> dict[str, int]:
    """Count replacements without persisting the original PII-bearing text."""

    counts = {"email": 0, "phone": 0, "order_id": 0}
    for text in original.astype(str):
        _, replacements = mask_pii(text)
        for name, count in replacements.items():
            counts[name] += count
    return counts


def stratified_split(
    frame: pd.DataFrame, validation_fraction: float, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be positive and sum to less than 1.")
    counts = frame["label"].value_counts()
    if counts.min() < 4:
        raise ValueError("Each label needs at least four examples for a stratified three-way split.")
    train, held_out = train_test_split(
        frame, test_size=validation_fraction + test_fraction, random_state=seed, stratify=frame["label"]
    )
    validation_fraction_of_held_out = validation_fraction / (validation_fraction + test_fraction)
    validation, test = train_test_split(
        held_out,
        test_size=1 - validation_fraction_of_held_out,
        random_state=seed,
        stratify=held_out["label"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def temporal_split(
    frame: pd.DataFrame, validation_fraction: float, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train/validation/test splits without future leakage."""

    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"Temporal split needs a '{TIMESTAMP_COLUMN}' column.")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be positive and sum to less than 1.")
    ordered = frame.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
    validation_size = max(1, round(len(ordered) * validation_fraction))
    test_size = max(1, round(len(ordered) * test_fraction))
    train_size = len(ordered) - validation_size - test_size
    if train_size < 2:
        raise ValueError("Not enough rows for a chronological three-way split.")
    train = ordered.iloc[:train_size].copy()
    validation = ordered.iloc[train_size : train_size + validation_size].copy()
    test = ordered.iloc[train_size + validation_size :].copy()
    unseen = (set(validation["label"]) | set(test["label"])) - set(train["label"])
    if unseen:
        raise ValueError(
            "Temporal split leaves labels unseen in training: " + ", ".join(sorted(unseen)) + ". "
            "Collect earlier examples or route those labels to human review."
        )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
