"""Reproducibility, JSON, and experiment-table helpers."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path: str | Path, content: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_csv(path: str | Path, row: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if destination.exists():
        with destination.open(encoding="utf-8", newline="") as file:
            rows = [existing for existing in csv.DictReader(file) if existing["experiment"] != str(row["experiment"])]
    rows.append({key: str(value) for key, value in row.items()})
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
