"""ImageFolder loading, augmentation, and class-imbalance helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(augment: bool) -> transforms.Compose:
    operations: list[object] = []
    if augment:
        operations.extend(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ]
        )
    else:
        operations.extend([transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)])
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(operations)


def build_dataloaders(
    processed_dir: str | Path,
    batch_size: int,
    augment: bool,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], list[int]]:
    """Load a fixed train/validation/test directory split without reshuffling membership."""
    root = Path(processed_dir)
    train_dataset = datasets.ImageFolder(root / "train", transform=build_transform(augment))
    evaluation_transform = build_transform(False)
    validation_dataset = datasets.ImageFolder(root / "validation", transform=evaluation_transform)
    test_dataset = datasets.ImageFolder(root / "test", transform=evaluation_transform)
    if train_dataset.classes != validation_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError("train, validation, and test class directories must match.")
    options = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_dataset, shuffle=True, **options),
        DataLoader(validation_dataset, shuffle=False, **options),
        DataLoader(test_dataset, shuffle=False, **options),
        train_dataset.classes,
        train_dataset.targets,
    )


def balanced_class_weights(targets: list[int], class_count: int) -> torch.Tensor:
    """Give minority classes higher cross-entropy loss weight."""
    counts = Counter(targets)
    if len(counts) != class_count:
        raise ValueError("Every class needs at least one training image.")
    total = len(targets)
    return torch.tensor([total / (class_count * counts[index]) for index in range(class_count)], dtype=torch.float32)
