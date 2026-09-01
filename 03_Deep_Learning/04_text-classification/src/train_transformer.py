"""Fine-tune a Korean pretrained Transformer for inquiry classification."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import load_csv
from policy import choose_review_threshold
from utils import save_json, set_seed, upsert_csv


class InquiryDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int) -> None:
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {**{key: torch.tensor(value[index]) for key, value in self.encodings.items()}, "labels": torch.tensor(self.labels[index])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Korean Transformer for inquiry classification.")
    parser.add_argument("--run-name", default="klue-bert-finetune")
    parser.add_argument("--pretrained-model", default="klue/bert-base")
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--validation-csv", default="data/processed/validation.csv")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--minimum-auto-precision", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def evaluate(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    actual, predicted, confidence = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            actual.extend(labels.tolist())
            predicted.extend(probabilities.argmax(axis=1).tolist())
            confidence.extend(probabilities.max(axis=1).tolist())
    return np.asarray(actual), np.asarray(predicted), np.asarray(confidence)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch-size must be positive.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = Path(args.models_dir) / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists. Use a new --run-name to preserve this experiment.")
    train, validation = load_csv(args.train_csv), load_csv(args.validation_csv)
    encoder = LabelEncoder().fit(train["label"])
    train_targets, validation_targets = encoder.transform(train["label"]), encoder.transform(validation["label"])
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    train_loader = DataLoader(InquiryDataset(train["text"].tolist(), train_targets, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(InquiryDataset(validation["text"].tolist(), validation_targets, tokenizer, args.max_length), batch_size=args.batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=len(encoder.classes_), id2label={index: label for index, label in enumerate(encoder.classes_)}, label2id={label: index for index, label in enumerate(encoder.classes_)}).to(device)
    counts = Counter(train_targets.tolist())
    weights = torch.tensor([len(train_targets) / (len(encoder.classes_) * counts[index]) for index in range(len(encoder.classes_))], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * args.epochs)
    best_f1, best_accuracy, stale, history = -1.0, 0.0, 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            optimizer.zero_grad()
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        actual, predicted, confidence = evaluate(model, validation_loader, device)
        report = classification_report(actual, predicted, target_names=encoder.classes_, output_dict=True, zero_division=0)
        history.append({"epoch": epoch, "validation_accuracy": report["accuracy"], "validation_macro_f1": report["macro avg"]["f1-score"]})
        print(f"[{args.run_name}] epoch {epoch}/{args.epochs} | validation macro F1={report['macro avg']['f1-score']:.4f}")
        if report["macro avg"]["f1-score"] > best_f1:
            best_f1, best_accuracy, stale = report["macro avg"]["f1-score"], report["accuracy"], 0
            run_dir.mkdir(parents=True)
            model.save_pretrained(run_dir)
            tokenizer.save_pretrained(run_dir)
            policy = choose_review_threshold(actual, predicted, confidence, args.minimum_auto_precision)
            policy.update({"minimum_auto_precision": args.minimum_auto_precision, "derived_from": "validation"})
            save_json(run_dir / "metadata.json", {"model_type": "transformer", "class_names": encoder.classes_.tolist(), "max_length": args.max_length, "policy": policy})
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break
    elapsed = time.perf_counter() - started
    summary = {"experiment": args.run_name, "model": "transformer", "pretrained_model": args.pretrained_model, "validation_accuracy": best_accuracy, "validation_macro_f1": best_f1, "training_time_seconds": elapsed, "model_size_bytes": sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())}
    save_json(Path(args.results_dir) / f"{args.run_name}-validation-history.json", history)
    upsert_csv(Path(args.results_dir) / "validation-model-comparison.csv", summary)
    print(f"Saved best validation checkpoint to {run_dir}")


if __name__ == "__main__":
    main()
