"""Final test evaluation for a saved baseline or Transformer experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_csv
from policy import apply_review_policy
from train_transformer import InquiryDataset, resolve_device
from torch.utils.data import DataLoader
from utils import save_json, upsert_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a selected inquiry-classification experiment on the final test set.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def transformer_predictions(
    model_dir: Path, frame: pd.DataFrame, batch_size: int, device: torch.device
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, object]]:
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    classes = metadata["class_names"]
    label_to_id = {label: index for index, label in enumerate(classes)}
    if not set(frame["label"]).issubset(label_to_id):
        raise ValueError("Test labels are not compatible with the saved Transformer label mapping.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    dataset = InquiryDataset(frame["text"].tolist(), np.asarray([label_to_id[label] for label in frame["label"]]), tokenizer, metadata["max_length"])
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    predictions, confidence = [], []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            probabilities = torch.softmax(model(**{key: value.to(device) for key, value in batch.items()}).logits, dim=1).cpu().numpy()
            predictions.extend(probabilities.argmax(axis=1).tolist())
            confidence.extend(probabilities.max(axis=1).tolist())
    return classes, np.asarray(predictions), np.asarray(confidence), metadata["policy"]


def main() -> None:
    args = parse_args()
    test = load_csv(args.test_csv)
    models_dir, results_dir = Path(args.models_dir), Path(args.results_dir)
    baseline_path = models_dir / f"{args.run_name}.joblib"
    if baseline_path.exists():
        artifact = joblib.load(baseline_path)
        encoder, pipeline, policy = artifact["label_encoder"], artifact["pipeline"], artifact["policy"]
        classes = encoder.classes_.tolist()
        actual = encoder.transform(test["label"])
        probabilities = pipeline.predict_proba(test["text"])
        predicted, confidence = probabilities.argmax(axis=1), probabilities.max(axis=1)
        model_name, model_size = "tfidf-logreg", baseline_path.stat().st_size
    else:
        model_dir = models_dir / args.run_name
        if not model_dir.exists():
            raise FileNotFoundError(f"No baseline or Transformer artifact found for {args.run_name}.")
        classes, predicted, confidence, policy = transformer_predictions(model_dir, test, args.batch_size, resolve_device(args.device))
        actual = np.asarray([classes.index(label) for label in test["label"]])
        model_name = "transformer"
        model_size = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
    report = classification_report(actual, predicted, target_names=classes, output_dict=True, zero_division=0)
    needs_review = apply_review_policy(confidence, float(policy["threshold"]))
    automated = ~needs_review
    automated_precision = float((actual[automated] == predicted[automated]).mean()) if automated.any() else 0.0
    summary = {"experiment": args.run_name, "model": model_name, "test_accuracy": report["accuracy"], "test_macro_f1": report["macro avg"]["f1-score"], "review_threshold": policy["threshold"], "automated_coverage": float(automated.mean()), "automated_precision": automated_precision, "model_size_bytes": model_size}
    save_json(results_dir / f"{args.run_name}-test-metrics.json", {"summary": summary, "metrics": report, "validation_review_policy": policy})
    upsert_csv(results_dir / "model-comparison.csv", summary)
    errors = test.loc[actual != predicted, ["text", "label"]].copy()
    errors["text_excerpt"] = errors.pop("text").str.slice(0, 120)
    errors["predicted_label"] = [classes[index] for index in predicted[actual != predicted]]
    errors["confidence"] = confidence[actual != predicted]
    errors["needs_review"] = needs_review[actual != predicted]
    generated_dir = results_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(generated_dir / f"{args.run_name}-misclassifications.csv", index=False, encoding="utf-8")
    print(f"[{args.run_name}] test macro F1={summary['test_macro_f1']:.4f}; automatic coverage={summary['automated_coverage']:.2%}")


if __name__ == "__main__":
    main()
