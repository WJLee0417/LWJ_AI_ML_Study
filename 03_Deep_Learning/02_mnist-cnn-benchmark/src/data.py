"""MNIST data loading and deterministic train/validation splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def mnist_transform() -> transforms.Compose:
    """Return the normalization shared by train, validation, and test data."""
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )


def stratified_indices(
    labels: Iterable[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split MNIST training labels into reproducible, class-balanced indices."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    indices = np.arange(len(labels))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=np.asarray(labels),
    )
    return train_indices.tolist(), validation_indices.tolist()


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    seed: int,
    validation_fraction: float = 0.2,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Download MNIST and return train, validation, and untouched test loaders."""
    data_path = Path(data_dir)
    dataset = datasets.MNIST(
        root=str(data_path), train=True, download=True, transform=mnist_transform()
    )
    test_dataset = datasets.MNIST(
        root=str(data_path), train=False, download=True, transform=mnist_transform()
    )
    train_indices, validation_indices = stratified_indices(
        dataset.targets.tolist(), validation_fraction, seed
    )

    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        Subset(dataset, train_indices), shuffle=True, generator=generator, **loader_options
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), shuffle=False, **loader_options
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_loader, validation_loader, test_loader
