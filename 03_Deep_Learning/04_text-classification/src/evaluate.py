"""Final test evaluation for a saved baseline or Transformer experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from calibration import apply_temperature
from data import load_csv
from inference import load_baseline_artifact, predict_baseline, resolve_device, transformer_predictions
from policy import apply_review_policy, operational_metrics
from utils import save_json, upsert_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a selected inquiry-classification experiment on the final test set.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--calibration-path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_calibration(path: Path, classes: list[str]) -> dict[str, object] | None:
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("class_names") != classes:
        raise ValueError("Calibration class order differs from the saved model.")
    return manifest


def main() -> None:
    args = parse_args()
    test = load_csv(args.test_csv)
    models_dir, results_dir = Path(args.models_dir), Path(args.results_dir)
    baseline_path = models_dir / f"{args.run_name}.joblib"
    if baseline_path.exists():
        artifact = load_baseline_artifact(baseline_path)
        encoder = artifact["label_encoder"]
        classes = encoder.classes_.tolist()
        actual = encoder.transform(test["label"])
        _, probabilities = predict_baseline(artifact, test["text"])
        legacy_policy = artifact["policy"]
        model_name, model_size = "tfidf-logreg", baseline_path.stat().st_size
    else:
        model_dir = models_dir / args.run_name
        if not model_dir.exists():
            raise FileNotFoundError(f"No baseline or Transformer artifact found for {args.run_name}.")
        classes, _, probabilities, metadata = transformer_predictions(
            model_dir, test["text"].tolist(), args.batch_size, resolve_device(args.device)
        )
        if not set(test["label"]).issubset(classes):
            raise ValueError("Test labels are not compatible with the saved Transformer label mapping.")
        actual = np.asarray([classes.index(label) for label in test["label"]])
        legacy_policy = metadata["policy"]
        model_name = "transformer"
        model_size = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())

    calibration_path = Path(args.calibration_path) if args.calibration_path else Path("artifacts") / args.run_name / "manifest.json"
    calibration = load_calibration(calibration_path, classes)
    if calibration:
        probabilities = apply_temperature(probabilities, float(calibration["calibration_temperature"]))
        policy = calibration["policy"]
        calibration_source = str(calibration_path)
    else:
        policy = legacy_policy
        calibration_source = None

    predicted, confidence = probabilities.argmax(axis=1), probabilities.max(axis=1)
    report = classification_report(actual, predicted, target_names=classes, output_dict=True, zero_division=0)
    operations = operational_metrics(actual, predicted, confidence, float(policy["threshold"]))
    needs_review = apply_review_policy(confidence, float(policy["threshold"]))
    summary = {
        "experiment": args.run_name,
        "model": model_name,
        "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "review_threshold": policy["threshold"],
        "model_size_bytes": model_size,
        "calibration_source": calibration_source,
        **operations,
    }
    save_json(
        results_dir / f"{args.run_name}-test-metrics.json",
        {"summary": summary, "metrics": report, "review_policy": policy, "calibration": calibration},
    )
    upsert_csv(results_dir / "model-comparison.csv", summary)
    errors = test.loc[actual != predicted, ["text", "label"]].copy()
    errors["text_excerpt"] = errors.pop("text").str.slice(0, 120)
    errors["predicted_label"] = [classes[index] for index in predicted[actual != predicted]]
    errors["confidence"] = confidence[actual != predicted]
    errors["needs_review"] = needs_review[actual != predicted]
    generated_dir = results_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(generated_dir / f"{args.run_name}-misclassifications.csv", index=False, encoding="utf-8")
    print(
        f"[{args.run_name}] test macro F1={summary['test_macro_f1']:.4f}; "
        f"automatic coverage={summary['automated_coverage']:.2%}; review rate={summary['review_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
