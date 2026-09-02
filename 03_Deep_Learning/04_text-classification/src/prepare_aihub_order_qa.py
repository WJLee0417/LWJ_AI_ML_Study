"""Build leakage-safe inquiry CSVs from AI Hub order question-answer files."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

try:  # Supports both `python src/...` and `from src...` test execution.
    from .pii import mask_pii
    from .utils import save_json
except ImportError:  # pragma: no cover - exercised by command-line scripts.
    from pii import mask_pii
    from utils import save_json

REQUIRED_COLUMNS = {"IDX", "발화자", "발화문", "QA여부", "인텐트", "날짜", "상담번호"}


def load_mapping(path: str | Path) -> list[dict[str, object]]:
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = content.get("labels", [])
    if len(labels) < 2:
        raise ValueError("The intent mapping must define at least two target labels.")
    return labels


def map_intent(intent: str, mapping: list[dict[str, object]]) -> str | None:
    for rule in mapping:
        if any(intent.startswith(prefix) for prefix in rule["intent_prefixes"]):
            return str(rule["label"])
    return None


class LabelReservoir:
    """Deterministically keep a balanced bounded sample without loading every row."""

    def __init__(self, max_per_label: int, seed: int) -> None:
        self.max_per_label = max_per_label
        self.random = random.Random(seed)
        self.rows: dict[str, list[dict[str, object]]] = defaultdict(list)
        self.seen: Counter[str] = Counter()

    def add(self, row: dict[str, object]) -> None:
        label = str(row["label"])
        self.seen[label] += 1
        bucket = self.rows[label]
        if len(bucket) < self.max_per_label:
            bucket.append(row)
            return
        replacement = self.random.randrange(self.seen[label])
        if replacement < self.max_per_label:
            bucket[replacement] = row

    def frame(self) -> pd.DataFrame:
        rows = [row for label in sorted(self.rows) for row in self.rows[label]]
        if not rows:
            raise ValueError("No customer-question rows matched the configured intent mapping.")
        return pd.DataFrame(rows)


def collect_customer_questions(
    directory: str | Path, mapping: list[dict[str, object]], max_per_label: int, seed: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    reservoir = LabelReservoir(max_per_label, seed)
    stats: dict[str, object] = {
        "files": 0,
        "rows": 0,
        "customer_questions": 0,
        "dated_customer_questions": 0,
        "matched_by_label": Counter(),
        "excluded_intents": Counter(),
    }
    for file_path in sorted(Path(directory).glob("*.csv")):
        stats["files"] = int(stats["files"]) + 1
        for chunk in pd.read_csv(file_path, dtype=str, keep_default_na=False, chunksize=100_000):
            missing = REQUIRED_COLUMNS - set(chunk.columns)
            if missing:
                raise ValueError(f"{file_path} is missing columns: {', '.join(sorted(missing))}")
            stats["rows"] = int(stats["rows"]) + len(chunk)
            questions = chunk[(chunk["발화자"] == "c") & (chunk["QA여부"] == "q")]
            stats["customer_questions"] = int(stats["customer_questions"]) + len(questions)
            stats["dated_customer_questions"] = int(stats["dated_customer_questions"]) + int(questions["날짜"].str.strip().ne("").sum())
            for row in questions.itertuples(index=False):
                record = row._asdict()
                intent = record["인텐트"].strip()
                label = map_intent(intent, mapping)
                if label is None:
                    stats["excluded_intents"][intent] += 1
                    continue
                text, pii_counts = mask_pii(record["발화문"].strip())
                if not text:
                    continue
                source_group = record["상담번호"].strip() or record["IDX"].strip()
                reservoir.add(
                    {
                        "text": text,
                        "label": label,
                        "group_id": f"{file_path.stem}:{source_group}",
                        "source_intent": intent,
                        "pii_counts": pii_counts,
                    }
                )
                stats["matched_by_label"][label] += 1
    frame = reservoir.frame()
    stats["matched_by_label"] = dict(stats["matched_by_label"])
    stats["excluded_intents"] = dict(stats["excluded_intents"].most_common(30))
    stats["sampled_by_label"] = frame["label"].value_counts().sort_index().to_dict()
    stats["date_coverage"] = float(stats["dated_customer_questions"]) / int(stats["customer_questions"]) if stats["customer_questions"] else 0.0
    return frame, stats


def group_stratified_split(frame: pd.DataFrame, validation_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation-fraction must be in (0, 0.5).")
    n_splits = max(2, round(1 / validation_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    target_rows = len(frame) * validation_fraction
    candidates = list(splitter.split(frame, frame["label"], groups=frame["group_id"]))
    train_indices, validation_indices = min(candidates, key=lambda pair: abs(len(pair[1]) - target_rows))
    train, validation = frame.iloc[train_indices].copy(), frame.iloc[validation_indices].copy()
    if set(train["group_id"]) & set(validation["group_id"]):
        raise RuntimeError("Conversation groups leaked between train and validation.")
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def output_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ["text", "label", "group_id"]].sort_values(["label", "group_id"], kind="stable").reset_index(drop=True)


def split_summary(name: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "groups": int(frame["group_id"].nunique()),
        "label_counts": frame["label"].value_counts().sort_index().to_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AI Hub order Q/A data into inquiry-classification splits.")
    parser.add_argument("--training-dir", default="data/raw/Training")
    parser.add_argument("--source-validation-dir", default="data/raw/Validation")
    parser.add_argument("--intent-map", default="configs/aihub-intent-map.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--max-train-per-label", type=int, default=12000)
    parser.add_argument("--max-test-per-label", type=int, default=3000)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    existing = [path for path in output_dir.iterdir() if path.name != ".gitkeep"] if output_dir.exists() else []
    if existing:
        raise FileExistsError(f"{output_dir} is not empty. Choose a new output directory to preserve the split.")
    mapping = load_mapping(args.intent_map)
    source_train, train_stats = collect_customer_questions(args.training_dir, mapping, args.max_train_per_label, args.seed)
    source_test, test_stats = collect_customer_questions(args.source_validation_dir, mapping, args.max_test_per_label, args.seed + 1)
    train, validation = group_stratified_split(source_train, args.validation_fraction, args.seed)
    test = source_test.reset_index(drop=True)
    if set(train["group_id"]) & set(test["group_id"]):
        raise RuntimeError("Source training and validation conversation groups overlap.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        output_frame(frame).to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")
    pii_counts = Counter()
    for frame in (train, validation, test):
        for counts in frame["pii_counts"]:
            pii_counts.update(counts)
    summary = {
        "source": "AI Hub 소상공인 고객 주문 질의-응답 텍스트",
        "split_strategy": "official-source-validation-holdout + stratified-group-kfold-on-source-training",
        "seed": args.seed,
        "intent_mapping": args.intent_map,
        "source_train": train_stats,
        "source_validation": test_stats,
        "splits": {name: split_summary(name, frame) for name, frame in (("train", train), ("validation", validation), ("test", test))},
        "pii_replacements_in_export": dict(pii_counts),
        "timestamp_note": "The source date field is incomplete, so this export intentionally does not claim a date-based test split.",
    }
    save_json(output_dir / "split-summary.json", summary)
    print(f"Created AI Hub inquiry splits at {output_dir}")


if __name__ == "__main__":
    main()
