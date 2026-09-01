"""CSV contract and deterministic train/validation/test split utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"text", "label"}


def load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    frame = frame.loc[:, ["text", "label"]].dropna().copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["label"] != "")]
    if frame["label"].nunique() < 2:
        raise ValueError("At least two labels are required.")
    return frame.reset_index(drop=True)


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
