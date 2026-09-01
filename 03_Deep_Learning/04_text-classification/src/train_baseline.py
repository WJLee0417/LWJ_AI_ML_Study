"""Train TF-IDF + Logistic Regression as the non-neural baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from data import load_csv
from policy import choose_review_threshold
from utils import save_json, set_seed, upsert_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TF-IDF Logistic Regression baseline.")
    parser.add_argument("--run-name", default="tfidf-logreg")
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--validation-csv", default="data/processed/validation.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--minimum-auto-precision", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train, validation = load_csv(args.train_csv), load_csv(args.validation_csv)
    encoder = LabelEncoder().fit(train["label"])
    train_targets, validation_targets = encoder.transform(train["label"]), encoder.transform(validation["label"])
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
        ]
    )
    started = time.perf_counter()
    pipeline.fit(train["text"], train_targets)
    elapsed = time.perf_counter() - started
    probabilities = pipeline.predict_proba(validation["text"])
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    report = classification_report(validation_targets, predicted, target_names=encoder.classes_, output_dict=True, zero_division=0)
    policy = choose_review_threshold(validation_targets, predicted, confidence, args.minimum_auto_precision)
    policy.update({"minimum_auto_precision": args.minimum_auto_precision, "derived_from": "validation"})
    models_dir, results_dir = Path(args.models_dir), Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / f"{args.run_name}.joblib"
    if artifact_path.exists():
        raise FileExistsError(f"{artifact_path} already exists. Use a new --run-name to preserve this experiment.")
    joblib.dump({"pipeline": pipeline, "label_encoder": encoder, "policy": policy}, artifact_path)
    summary = {"experiment": args.run_name, "model": "tfidf-logreg", "validation_accuracy": report["accuracy"], "validation_macro_f1": report["macro avg"]["f1-score"], "training_time_seconds": elapsed, "model_size_bytes": artifact_path.stat().st_size}
    save_json(results_dir / f"{args.run_name}-validation-metrics.json", {"summary": summary, "metrics": report, "review_policy": policy})
    upsert_csv(results_dir / "validation-model-comparison.csv", summary)
    print(f"[{args.run_name}] validation macro F1={summary['validation_macro_f1']:.4f}; review threshold={policy['threshold']:.4f}")


if __name__ == "__main__":
    main()
