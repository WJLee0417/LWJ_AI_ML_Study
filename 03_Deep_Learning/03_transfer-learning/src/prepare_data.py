"""Create a deterministic train/validation/test split from class-folder images."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw waste images by class without test leakage.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        help="Optional CSV with relative_path,label,group_id. Use it to keep related images in one split.",
    )
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


def load_group_manifest(manifest_path: Path, raw_dir: Path) -> dict[str, list[tuple[Path, str]]]:
    """Load class-labelled image groups and reject cross-class group IDs."""
    required = {"relative_path", "label", "group_id"}
    grouped: dict[str, list[tuple[Path, str]]] = {}
    group_labels: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8", newline="") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            if not required.issubset(row):
                raise ValueError("Manifest needs relative_path,label,group_id columns.")
            relative_path, label, group_id = row["relative_path"].strip(), row["label"].strip(), row["group_id"].strip()
            if not relative_path or not label or not group_id:
                raise ValueError(f"Manifest row {row_number} has an empty required value.")
            source = raw_dir / relative_path
            if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
                raise FileNotFoundError(f"Manifest row {row_number} points to an invalid image: {source}")
            if group_id in group_labels and group_labels[group_id] != label:
                raise ValueError(f"group_id '{group_id}' appears under multiple labels.")
            group_labels[group_id] = label
            grouped.setdefault(label, []).append((source, group_id))
    if len(grouped) < 2:
        raise ValueError("Manifest must contain at least two labels.")
    return grouped


def split_grouped_paths(
    records: list[tuple[Path, str]], validation_fraction: float, test_fraction: float, seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split whole image groups, preventing near-duplicate leakage across splits."""
    groups: dict[str, list[Path]] = {}
    for path, group_id in records:
        groups.setdefault(group_id, []).append(path)
    group_ids = sorted(groups)
    if len(group_ids) < 7:
        raise ValueError("Each class needs at least seven unique groups for the default 70/15/15 grouped split.")
    train_groups, held_out_groups = train_test_split(
        group_ids, test_size=validation_fraction + test_fraction, random_state=seed
    )
    validation_share = validation_fraction / (validation_fraction + test_fraction)
    validation_groups, test_groups = train_test_split(
        held_out_groups, test_size=1 - validation_share, random_state=seed
    )
    def paths_for(selected_groups: list[str]) -> list[Path]:
        return [path for group_id in selected_groups for path in groups[group_id]]
    return paths_for(train_groups), paths_for(validation_groups), paths_for(test_groups)


def main() -> None:
    args = parse_args()
    raw_dir, output_dir = Path(args.raw_dir), Path(args.output_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    if output_dir.exists() and any(path for path in output_dir.iterdir() if path.name != ".gitkeep"):
        raise FileExistsError(f"{output_dir} is not empty. Choose a new output directory to preserve its split.")
    class_paths = class_file_map(raw_dir) if not args.manifest else None
    grouped_paths = load_group_manifest(Path(args.manifest), raw_dir) if args.manifest else None
    rows = []
    source_classes = grouped_paths if grouped_paths is not None else class_paths
    for class_name, source_records in source_classes.items():
        if grouped_paths is not None:
            train_paths, validation_paths, test_paths = split_grouped_paths(
                source_records, args.validation_fraction, args.test_fraction, args.seed
            )
            groups_by_path = dict(source_records)
        else:
            train_paths, validation_paths, test_paths = split_paths(
                source_records, args.validation_fraction, args.test_fraction, args.seed
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
            unique_groups = len({groups_by_path[path] for path in selected_paths}) if grouped_paths is not None else None
            rows.append({"class": class_name, "split": split_name, "images": len(selected_paths), "unique_groups": unique_groups})
    with (output_dir / "split-summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "split", "images", "unique_groups"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created deterministic split at {output_dir}")


if __name__ == "__main__":
    main()
