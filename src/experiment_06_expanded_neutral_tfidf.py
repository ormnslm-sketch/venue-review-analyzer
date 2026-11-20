"""Run the unchanged TF-IDF baseline on Neutral Expansion v1 train data."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from train_sentiment_baseline import (
    ASPECTS,
    SENTIMENTS,
    build_inputs,
    build_model,
    load_rows,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "expanded_neutral_v1" / "train.csv"
DEV_PATH = ROOT / "data" / "expanded_neutral_v1" / "dev.csv"
CLEANED_DEV_PATH = ROOT / "data" / "cleaned" / "dev.csv"
REPORT_PATH = ROOT / "data" / "analysis" / "experiment_06_expanded_neutral_tfidf.txt"
PREDICTIONS_PATH = ROOT / "data" / "analysis" / "experiment_06_dev_predictions.csv"

EXPECTED_TRAIN = Counter({"positive": 2173, "neutral": 300, "negative": 1050})
EXPECTED_DEV = Counter({"positive": 586, "neutral": 57, "negative": 258})

OLD = {
    "accuracy": 0.7603,
    "macro_f1": 0.5891,
    "positive_precision": 0.8714,
    "positive_recall": 0.8208,
    "positive_f1": 0.8453,
    "neutral_precision": 0.3667,
    "neutral_recall": 0.1930,
    "neutral_f1": 0.2529,
    "negative_precision": 0.6050,
    "negative_recall": 0.7481,
    "negative_f1": 0.6690,
    "tr_macro_f1": 0.5936,
    "en_macro_f1": 0.5998,
    "es_macro_f1": 0.5734,
    "taste_macro_f1": 0.5902,
    "atmosphere_macro_f1": 0.3210,
    "service_macro_f1": 0.5251,
    "value_macro_f1": 0.6988,
}

LANGUAGES = (("tr", "TR"), ("en", "EN"), ("es", "ES"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_metrics(y_true, y_pred):
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=SENTIMENTS,
        zero_division=0,
    )
    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1_values[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(SENTIMENTS)
    }


def neutral_metrics(y_true, y_pred):
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["neutral"],
        zero_division=0,
    )
    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1_values[0]),
        "support": int(support[0]),
    }


def grouped_metrics(rows, predictions, column, groups):
    results = {}
    for value, display_name in groups:
        indices = [index for index, row in enumerate(rows) if row[column] == value]
        y_true = [rows[index]["sentiment"] for index in indices]
        y_pred = [predictions[index] for index in indices]
        results[display_name] = {
            "n": len(indices),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "neutral": neutral_metrics(y_true, y_pred),
        }
    return results


def delta(new_value: float, old_value: float) -> float:
    return new_value - old_value


def save_predictions(dev_rows, predictions) -> None:
    fields = [
        "sentence",
        "language",
        "aspect",
        "true_sentiment",
        "predicted_sentiment",
    ]
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row, prediction in zip(dev_rows, predictions):
            writer.writerow(
                {
                    "sentence": row["sentence"],
                    "language": row["language"],
                    "aspect": row["aspect"],
                    "true_sentiment": row["sentiment"],
                    "predicted_sentiment": prediction,
                }
            )


def main() -> None:
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")
    train_counts = Counter(row["sentiment"] for row in train_rows)
    dev_counts = Counter(row["sentiment"] for row in dev_rows)

    if train_counts != EXPECTED_TRAIN or len(train_rows) != 3523:
        raise ValueError(
            f"Train distribution mismatch: rows={len(train_rows)}, counts={dict(train_counts)}"
        )
    if dev_counts != EXPECTED_DEV or len(dev_rows) != 901:
        raise ValueError(
            f"Dev distribution mismatch: rows={len(dev_rows)}, counts={dict(dev_counts)}"
        )
    if file_hash(DEV_PATH) != file_hash(CLEANED_DEV_PATH):
        raise ValueError("Expanded dev.csv differs from cleaned dev.csv")

    model = build_model()
    model.fit(build_inputs(train_rows), [row["sentiment"] for row in train_rows])
    predictions = model.predict(build_inputs(dev_rows)).tolist()

    y_true = [row["sentiment"] for row in dev_rows]
    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(y_true, predictions, average="macro", zero_division=0)
    sentiments = class_metrics(y_true, predictions)
    matrix = confusion_matrix(y_true, predictions, labels=SENTIMENTS)
    predicted_counts = Counter(predictions)
    language_metrics = grouped_metrics(dev_rows, predictions, "language", LANGUAGES)
    aspect_metrics = grouped_metrics(
        dev_rows,
        predictions,
        "aspect",
        [(aspect, aspect) for aspect in ASPECTS],
    )

    save_predictions(dev_rows, predictions)

    report = [
        "EXPERIMENT 06 — EXPANDED NEUTRAL V1 TF-IDF BASELINE",
        "",
        "CONTROLLED SETTINGS",
        "Train: data/expanded_neutral_v1/train.csv",
        "Dev: data/expanded_neutral_v1/dev.csv",
        "Test: not loaded or evaluated",
        "Input: aspect=<aspect> [SEP] <sentence>",
        "TF-IDF: ngram_range=(1, 2), min_df=2; all other settings unchanged/default",
        "LogisticRegression: class_weight=balanced, max_iter=1000, random_state=42; "
        "solver and other settings unchanged/default",
        "Hyperparameter tuning: none",
        "",
        "DATA DISTRIBUTION",
        f"Train: positive={train_counts['positive']}, neutral={train_counts['neutral']}, "
        f"negative={train_counts['negative']}, total={len(train_rows)}",
        f"Dev: positive={dev_counts['positive']}, neutral={dev_counts['neutral']}, "
        f"negative={dev_counts['negative']}, total={len(dev_rows)}",
        "Dev matches cleaned dev byte-for-byte: True",
        "",
        "OVERALL",
        f"Accuracy: {accuracy:.4f}",
        f"Macro F1: {macro_f1:.4f}",
        "",
        "PER-CLASS METRICS",
    ]

    for sentiment in SENTIMENTS:
        metrics = sentiments[sentiment]
        report.extend(
            [
                f"{sentiment}:",
                f"  precision: {metrics['precision']:.4f}",
                f"  recall: {metrics['recall']:.4f}",
                f"  F1: {metrics['f1']:.4f}",
                f"  support: {metrics['support']}",
            ]
        )

    report.extend(
        [
            "",
            "CONFUSION MATRIX",
            "Rows=actual, columns=predicted",
            "                 positive  neutral  negative",
        ]
    )
    for sentiment, row in zip(SENTIMENTS, matrix):
        report.append(f"{sentiment:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}")

    report.extend(
        [
            "",
            "PREDICTION DISTRIBUTION",
            *[f"{label}: {predicted_counts[label]}" for label in SENTIMENTS],
            "",
            "LANGUAGE METRICS",
        ]
    )
    for language in ("TR", "EN", "ES"):
        metrics = language_metrics[language]
        neutral = metrics["neutral"]
        report.extend(
            [
                f"{language}: n={metrics['n']} | accuracy={metrics['accuracy']:.4f} | "
                f"macro F1={metrics['macro_f1']:.4f}",
                f"  neutral: precision={neutral['precision']:.4f} | "
                f"recall={neutral['recall']:.4f} | F1={neutral['f1']:.4f} | "
                f"support={neutral['support']}",
            ]
        )

    report.extend(["", "ASPECT METRICS"])
    for aspect in ASPECTS:
        metrics = aspect_metrics[aspect]
        report.append(
            f"{aspect}: n={metrics['n']} | accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f} | "
            f"neutral F1={metrics['neutral']['f1']:.4f}"
        )

    report.extend(
        [
            "",
            "DELTA VS CLEANED BASELINE",
            f"Accuracy: {delta(accuracy, OLD['accuracy']):+.4f}",
            f"Macro F1: {delta(macro_f1, OLD['macro_f1']):+.4f}",
            f"positive F1: {delta(sentiments['positive']['f1'], OLD['positive_f1']):+.4f}",
            f"neutral F1: {delta(sentiments['neutral']['f1'], OLD['neutral_f1']):+.4f}",
            f"negative F1: {delta(sentiments['negative']['f1'], OLD['negative_f1']):+.4f}",
            f"neutral precision: {delta(sentiments['neutral']['precision'], OLD['neutral_precision']):+.4f}",
            f"neutral recall: {delta(sentiments['neutral']['recall'], OLD['neutral_recall']):+.4f}",
            "",
            "Language macro F1 delta:",
            *[
                f"{language}: {delta(language_metrics[language]['macro_f1'], OLD[f'{language.lower()}_macro_f1']):+.4f}"
                for language in ("TR", "EN", "ES")
            ],
            "",
            "Aspect macro F1 delta:",
            *[
                f"{aspect}: {delta(aspect_metrics[aspect]['macro_f1'], OLD[f'{aspect}_macro_f1']):+.4f}"
                for aspect in ASPECTS
            ],
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
