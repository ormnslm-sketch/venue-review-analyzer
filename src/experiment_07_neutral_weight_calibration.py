"""Calibrate only the neutral class weight on Expanded Neutral v1."""

from __future__ import annotations

import csv
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
CSV_PATH = ROOT / "data" / "analysis" / "experiment_07_neutral_weight_sweep.csv"
REPORT_PATH = ROOT / "data" / "analysis" / "experiment_07_neutral_weight_calibration.txt"

EXPECTED_TRAIN = Counter({"positive": 2173, "neutral": 300, "negative": 1050})
EXPECTED_DEV = Counter({"positive": 586, "neutral": 57, "negative": 258})

POSITIVE_WEIGHT = 0.54042031
NEGATIVE_WEIGHT = 1.11841270
NEUTRAL_WEIGHTS = (
    2.0,
    3.0,
    3.91444444,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    11.88530466,
    13.0,
    15.0,
)

LANGUAGES = (("tr", "TR"), ("en", "EN"), ("es", "ES"))

HISTORICAL_CLEANED = {
    "macro_f1": 0.5891,
    "neutral_precision": 0.3667,
    "neutral_recall": 0.1930,
    "neutral_f1": 0.2529,
}

CSV_FIELDS = [
    "neutral_weight",
    "accuracy",
    "macro_f1",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "neutral_precision",
    "neutral_recall",
    "neutral_f1",
    "negative_precision",
    "negative_recall",
    "negative_f1",
    "pred_positive",
    "pred_neutral",
    "pred_negative",
]


def format_weight(weight: float) -> str:
    return f"{weight:.8f}".rstrip("0").rstrip(".")


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
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "neutral": neutral_metrics(y_true, y_pred),
        }
    return results


def run_weight(weight, train_rows, dev_rows, train_inputs, dev_inputs):
    class_weight = {
        "positive": POSITIVE_WEIGHT,
        "neutral": weight,
        "negative": NEGATIVE_WEIGHT,
    }
    model = build_model()
    model.set_params(classifier__class_weight=class_weight)
    model.fit(train_inputs, [row["sentiment"] for row in train_rows])
    predictions = model.predict(dev_inputs).tolist()
    y_true = [row["sentiment"] for row in dev_rows]

    return {
        "neutral_weight": weight,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "classes": class_metrics(y_true, predictions),
        "predicted": Counter(predictions),
        "matrix": confusion_matrix(y_true, predictions, labels=SENTIMENTS),
        "languages": grouped_metrics(dev_rows, predictions, "language", LANGUAGES),
        "aspects": grouped_metrics(
            dev_rows,
            predictions,
            "aspect",
            [(aspect, aspect) for aspect in ASPECTS],
        ),
    }


def selection_key(result):
    neutral = result["classes"]["neutral"]
    return (
        result["macro_f1"],
        neutral["f1"],
        neutral["precision"],
        -result["neutral_weight"],
    )


def save_csv(results) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item["neutral_weight"]):
            positive = result["classes"]["positive"]
            neutral = result["classes"]["neutral"]
            negative = result["classes"]["negative"]
            writer.writerow(
                {
                    "neutral_weight": format_weight(result["neutral_weight"]),
                    "accuracy": f"{result['accuracy']:.8f}",
                    "macro_f1": f"{result['macro_f1']:.8f}",
                    "positive_precision": f"{positive['precision']:.8f}",
                    "positive_recall": f"{positive['recall']:.8f}",
                    "positive_f1": f"{positive['f1']:.8f}",
                    "neutral_precision": f"{neutral['precision']:.8f}",
                    "neutral_recall": f"{neutral['recall']:.8f}",
                    "neutral_f1": f"{neutral['f1']:.8f}",
                    "negative_precision": f"{negative['precision']:.8f}",
                    "negative_recall": f"{negative['recall']:.8f}",
                    "negative_f1": f"{negative['f1']:.8f}",
                    "pred_positive": result["predicted"]["positive"],
                    "pred_neutral": result["predicted"]["neutral"],
                    "pred_negative": result["predicted"]["negative"],
                }
            )


def append_weight_details(report, result):
    weight = format_weight(result["neutral_weight"])
    report.extend(
        [
            "",
            f"NEUTRAL WEIGHT = {weight}",
            f"Accuracy: {result['accuracy']:.4f}",
            f"Macro F1: {result['macro_f1']:.4f}",
        ]
    )
    for sentiment in SENTIMENTS:
        metrics = result["classes"][sentiment]
        report.append(
            f"{sentiment}: precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | "
            f"support={metrics['support']}"
        )
    report.extend(
        [
            "Predicted: "
            f"positive={result['predicted']['positive']}, "
            f"neutral={result['predicted']['neutral']}, "
            f"negative={result['predicted']['negative']}",
            "Confusion matrix (rows=actual, columns=predicted):",
            "                 positive  neutral  negative",
        ]
    )
    for sentiment, row in zip(SENTIMENTS, result["matrix"]):
        report.append(f"{sentiment:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}")


def append_top_three_breakdown(report, top_three):
    report.extend(["", "TOP 3 MACRO F1 CONFIGURATIONS"])
    for rank, result in enumerate(top_three, start=1):
        report.extend(
            [
                "",
                f"Rank {rank}: neutral_weight={format_weight(result['neutral_weight'])}",
                f"Accuracy={result['accuracy']:.4f} | Macro F1={result['macro_f1']:.4f}",
                "Language Macro F1 / Neutral F1:",
            ]
        )
        for language in ("TR", "EN", "ES"):
            metrics = result["languages"][language]
            report.append(
                f"{language}: macro F1={metrics['macro_f1']:.4f} | "
                f"neutral F1={metrics['neutral']['f1']:.4f}"
            )
        report.append("Aspect Macro F1 / Neutral F1:")
        for aspect in ASPECTS:
            metrics = result["aspects"][aspect]
            report.append(
                f"{aspect}: macro F1={metrics['macro_f1']:.4f} | "
                f"neutral F1={metrics['neutral']['f1']:.4f}"
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

    train_inputs = build_inputs(train_rows)
    dev_inputs = build_inputs(dev_rows)
    results = []
    for weight in NEUTRAL_WEIGHTS:
        print(f"Running neutral_weight={format_weight(weight)}...")
        results.append(
            run_weight(weight, train_rows, dev_rows, train_inputs, dev_inputs)
        )

    best = max(results, key=selection_key)
    top_three = sorted(results, key=selection_key, reverse=True)[:3]
    save_csv(results)

    reference_pipeline = build_model()
    tfidf = reference_pipeline.named_steps["tfidf"]
    classifier = reference_pipeline.named_steps["classifier"]
    report = [
        "EXPERIMENT 07 — EXPANDED NEUTRAL CLASS-WEIGHT CALIBRATION",
        "",
        "CONTROLLED SETTINGS",
        "Train: data/expanded_neutral_v1/train.csv",
        "Dev: data/expanded_neutral_v1/dev.csv",
        "Test: not loaded or evaluated",
        "Input: aspect=<aspect> [SEP] <sentence>",
        f"TF-IDF: ngram_range={tfidf.ngram_range}, min_df={tfidf.min_df}, "
        f"max_df={tfidf.max_df}, max_features={tfidf.max_features}",
        f"LogisticRegression: solver={classifier.solver}, max_iter={classifier.max_iter}, "
        f"random_state={classifier.random_state}, warm_start={classifier.warm_start}",
        f"Fixed positive weight: {POSITIVE_WEIGHT:.8f}",
        f"Fixed negative weight: {NEGATIVE_WEIGHT:.8f}",
        "Only neutral class weight changes.",
        "Primary selection metric: Dev Macro F1",
        "Tie-breaks: neutral F1, neutral precision, then lower neutral weight",
        "",
        "DATA VALIDATION",
        f"Train: positive={train_counts['positive']}, neutral={train_counts['neutral']}, "
        f"negative={train_counts['negative']}, total={len(train_rows)}",
        f"Dev: positive={dev_counts['positive']}, neutral={dev_counts['neutral']}, "
        f"negative={dev_counts['negative']}, total={len(dev_rows)}",
    ]

    for result in sorted(results, key=lambda item: item["neutral_weight"]):
        append_weight_details(report, result)

    append_top_three_breakdown(report, top_three)

    neutral = best["classes"]["neutral"]
    report.extend(
        [
            "",
            "SORTED SUMMARY",
            "Weight | Accuracy | Macro F1 | Neutral P | Neutral R | Neutral F1 | Pred Neutral",
        ]
    )
    for result in sorted(results, key=lambda item: item["neutral_weight"]):
        current_neutral = result["classes"]["neutral"]
        report.append(
            f"{format_weight(result['neutral_weight'])} | "
            f"{result['accuracy']:.4f} | {result['macro_f1']:.4f} | "
            f"{current_neutral['precision']:.4f} | "
            f"{current_neutral['recall']:.4f} | "
            f"{current_neutral['f1']:.4f} | "
            f"{result['predicted']['neutral']}"
        )

    report.extend(
        [
            "",
            f"BEST WEIGHT: {format_weight(best['neutral_weight'])}",
            f"Best Accuracy: {best['accuracy']:.4f}",
            f"Best Macro F1: {best['macro_f1']:.4f}",
            f"Best Neutral P: {neutral['precision']:.4f}",
            f"Best Neutral R: {neutral['recall']:.4f}",
            f"Best Neutral F1: {neutral['f1']:.4f}",
            f"Pred Neutral: {best['predicted']['neutral']}",
            "",
            "DELTA VS CLEANED BASELINE",
            f"Macro F1: {best['macro_f1'] - HISTORICAL_CLEANED['macro_f1']:+.4f}",
            f"Neutral P: {neutral['precision'] - HISTORICAL_CLEANED['neutral_precision']:+.4f}",
            f"Neutral R: {neutral['recall'] - HISTORICAL_CLEANED['neutral_recall']:+.4f}",
            f"Neutral F1: {neutral['f1'] - HISTORICAL_CLEANED['neutral_f1']:+.4f}",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
