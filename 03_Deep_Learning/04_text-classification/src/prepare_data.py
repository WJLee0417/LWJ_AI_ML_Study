"""Split a raw inquiry CSV once, preserving an untouched final test CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data import load_csv, pii_masking_summary, stratified_split, temporal_split
from utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible inquiry-classification split.")
    parser.add_argument("--input-csv", default="data/raw/inquiries.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-strategy", choices=["stratified", "temporal"], default="stratified")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    existing_outputs = [path for path in output_dir.iterdir() if path.name != ".gitkeep"] if output_dir.exists() else []
    if existing_outputs:
        raise FileExistsError(f"{output_dir} is not empty. Choose a new output directory to preserve the split.")
    raw = pd.read_csv(args.input_csv)
    if "text" not in raw.columns:
        raise ValueError(f"{args.input_csv} is missing required column: text")
    frame = load_csv(args.input_csv, include_timestamp=args.split_strategy == "temporal")
    if args.split_strategy == "temporal":
        train, validation, test = temporal_split(frame, args.validation_fraction, args.test_fraction)
    else:
        train, validation, test = stratified_split(frame, args.validation_fraction, args.test_fraction, args.seed)
    output_dir.mkdir(parents=True)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        split.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")
    summary = {
        "seed": args.seed,
        "split_strategy": args.split_strategy,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "label_counts": {name: split["label"].value_counts().sort_index().to_dict() for name, split in (("train", train), ("validation", validation), ("test", test))},
        "pii_replacements": pii_masking_summary(raw["text"]),
    }
    save_json(output_dir / "split-summary.json", summary)
    print(f"Created fixed split at {output_dir}")


if __name__ == "__main__":
    main()
