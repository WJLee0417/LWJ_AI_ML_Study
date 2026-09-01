"""Create a deterministic train/validation/test split from class-folder images."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw waste images by class without test leakage.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def class_file_map(raw_dir: Path) -> dict[str, list[Path]]:
    classes = {}
    for directory in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        files = sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if files:
            classes[directory.name] = files
    if len(classes) < 2:
        raise ValueError("Place images in at least two class directories under data/raw.")
    return classes


def split_paths(
    paths: list[Path], validation_fraction: float, test_fraction: float, seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be positive and sum to less than 1.")
    if len(paths) < 10:
        raise ValueError("Each class needs at least 10 images for a 3-way split.")
    train_paths, held_out = train_test_split(paths, test_size=validation_fraction + test_fraction, random_state=seed)
    validation_share = validation_fraction / (validation_fraction + test_fraction)
    validation_paths, test_paths = train_test_split(held_out, test_size=1 - validation_share, random_state=seed)
    return list(train_paths), list(validation_paths), list(test_paths)


def main() -> None:
    args = parse_args()
    raw_dir, output_dir = Path(args.raw_dir), Path(args.output_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty. Choose a new output directory to preserve its split.")
    class_paths = class_file_map(raw_dir)
    rows = []
    for class_name, paths in class_paths.items():
        train_paths, validation_paths, test_paths = split_paths(
            paths, args.validation_fraction, args.test_fraction, args.seed
        )
        for split_name, selected_paths in {
            "train": train_paths,
            "validation": validation_paths,
            "test": test_paths,
        }.items():
            destination = output_dir / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(selected_paths):
                target = destination / f"{index:05d}_{source.name}"
                shutil.copy2(source, target)
            rows.append({"class": class_name, "split": split_name, "images": len(selected_paths)})
    with (output_dir / "split-summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "split", "images"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created deterministic split at {output_dir}")


if __name__ == "__main__":
    main()
