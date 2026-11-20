"""Experiment 11: calibrate the training influence of added neutral rows.

The test split is never loaded. TF-IDF is fitted once on the full expanded
training corpus; only LogisticRegression row-level sample weights change.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from train_sentiment_baseline import ASPECTS, SENTIMENTS, build_inputs, load_rows


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "expanded_neutral_v1" / "train.csv"
DEV_PATH = ROOT / "data" / "expanded_neutral_v1" / "dev.csv"
MANIFEST_PATH = ROOT / "data" / "expanded_neutral_v1" / "added_neutral_manifest.csv"
CSV_PATH = ROOT / "data" / "analysis" / "experiment_11_added_neutral_soft_dose.csv"
REPORT_PATH = ROOT / "data" / "analysis" / "experiment_11_added_neutral_soft_dose.txt"

FIXED_OLD_WEIGHTS = {
    "positive": 0.5086669734621875,
    "neutral": 11.885304659498209,
    "negative": 1.0526984126984127,
}
ALPHAS = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00)
LANGUAGES = ("tr", "en", "es")

HISTORICAL = {
    "accuracy": 0.7603,
    "macro_f1": 0.5891,
    "neutral_precision": 0.3667,
    "neutral_recall": 0.1930,
    "neutral_f1": 0.2529,
}
FULL_EXPANDED_REFERENCE = {
    "macro_f1": 0.5805,
    "neutral_precision": 0.2066,
    "neutral_recall": 0.4386,
    "neutral_f1": 0.2809,
    "pred_neutral": 121,
    "false_neutral": 96,
}


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def row_key(row):
    return (
        row["sentence"],
        row["language"],
        row["aspect"],
        row["sentiment"],
    )


def validate_and_build_added_mask(train_rows, manifest):
    if len(train_rows) != 3523 or len(manifest) != 207:
        raise ValueError(
            f"Expected train=3523 and manifest=207; got {len(train_rows)} and {len(manifest)}"
        )
    expected_train = Counter({"positive": 2173, "neutral": 300, "negative": 1050})
    actual_train = Counter(row["sentiment"] for row in train_rows)
    if actual_train != expected_train:
        raise ValueError(f"Unexpected train distribution: {actual_train}")

    manifest_keys = [row_key(row) for row in manifest]
    if len(set(manifest_keys)) != 207:
        raise ValueError("Manifest contains duplicate sentence/language/aspect/sentiment keys")
    invalid_manifest = [
        row["candidate_id"]
        for row in manifest
        if row["sentiment"] != "neutral"
        or row["human_verified"].lower() != "true"
        or (
            row["language"] != "en"
            and row["translation_human_verified"].lower() != "true"
        )
    ]
    if invalid_manifest:
        raise ValueError(f"Invalid manifest rows: {invalid_manifest[:10]}")

    manifest_key_set = set(manifest_keys)
    added_mask = np.asarray(
        [row_key(row) in manifest_key_set for row in train_rows], dtype=bool
    )
    if int(added_mask.sum()) != 207:
        raise ValueError(
            f"Manifest-to-train matching failed: matched {int(added_mask.sum())} rows"
        )
    base_rows = len(train_rows) - int(added_mask.sum())
    if base_rows != 3316:
        raise ValueError(f"Expected 3316 base rows, found {base_rows}")
    matched_keys = {row_key(row) for row, is_added in zip(train_rows, added_mask) if is_added}
    if matched_keys != manifest_key_set:
        raise ValueError("Matched added-row keys differ from manifest keys")
    return added_mask


def class_metrics(y_true, y_pred):
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true, y_pred, labels=SENTIMENTS, zero_division=0
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
        y_true, y_pred, labels=["neutral"], zero_division=0
    )
    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1_values[0]),
        "support": int(support[0]),
    }


def grouped_metrics(dev_rows, predictions, column, values):
    output = {}
    for value in values:
        indices = [index for index, row in enumerate(dev_rows) if row[column] == value]
        y_true = [dev_rows[index]["sentiment"] for index in indices]
        y_pred = [predictions[index] for index in indices]
        output[value] = {
            "n": len(indices),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "neutral": neutral_metrics(y_true, y_pred),
        }
    return output


def evaluate(dev_rows, predictions):
    y_true = [row["sentiment"] for row in dev_rows]
    classes = class_metrics(y_true, predictions)
    false_indices = [
        index
        for index, prediction in enumerate(predictions)
        if prediction == "neutral" and y_true[index] != "neutral"
    ]
    positive_to_neutral = sum(y_true[index] == "positive" for index in false_indices)
    negative_to_neutral = sum(y_true[index] == "negative" for index in false_indices)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "classes": classes,
        "predicted": Counter(predictions),
        "positive_to_neutral": positive_to_neutral,
        "negative_to_neutral": negative_to_neutral,
        "false_neutral_total": len(false_indices),
        "matrix": confusion_matrix(y_true, predictions, labels=SENTIMENTS),
        "languages": grouped_metrics(dev_rows, predictions, "language", LANGUAGES),
        "aspects": grouped_metrics(dev_rows, predictions, "aspect", ASPECTS),
        "false_neutral_by_aspect": Counter(
            dev_rows[index]["aspect"] for index in false_indices
        ),
    }


def run_alpha(alpha, train_matrix, train_labels, dev_matrix, dev_rows, added_mask):
    sample_weights = np.where(added_mask, alpha, 1.0)
    classifier = LogisticRegression(
        class_weight=FIXED_OLD_WEIGHTS,
        max_iter=1000,
        random_state=42,
    )
    classifier.fit(train_matrix, train_labels, sample_weight=sample_weights)
    predictions = classifier.predict(dev_matrix).tolist()
    return {
        "alpha": alpha,
        "effective_added_neutral_multiplier": FIXED_OLD_WEIGHTS["neutral"] * alpha,
        "metrics": evaluate(dev_rows, predictions),
    }


def selection_key(result):
    metrics = result["metrics"]
    neutral = metrics["classes"]["neutral"]
    return (
        metrics["macro_f1"],
        neutral["f1"],
        neutral["precision"],
        -metrics["false_neutral_total"],
        -result["alpha"],
    )


def csv_row(result):
    metrics = result["metrics"]
    row = {
        "alpha": result["alpha"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }
    for sentiment in SENTIMENTS:
        for metric in ("precision", "recall", "f1"):
            row[f"{sentiment}_{metric}"] = metrics["classes"][sentiment][metric]
    row.update(
        {
            "pred_positive": metrics["predicted"]["positive"],
            "pred_neutral": metrics["predicted"]["neutral"],
            "pred_negative": metrics["predicted"]["negative"],
            "positive_to_neutral": metrics["positive_to_neutral"],
            "negative_to_neutral": metrics["negative_to_neutral"],
            "false_neutral_total": metrics["false_neutral_total"],
        }
    )
    return row


def save_csv(results):
    rows = [csv_row(result) for result in sorted(results, key=lambda item: item["alpha"])]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.8f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def format_matrix(matrix):
    lines = ["                 positive  neutral  negative"]
    for label, row in zip(SENTIMENTS, matrix):
        lines.append(f"{label:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}")
    return lines


def append_result_details(report, result):
    metrics = result["metrics"]
    report.extend(
        [
            "",
            f"ALPHA = {result['alpha']:.2f}",
            "Effective added-neutral multiplier: "
            f"{result['effective_added_neutral_multiplier']:.6f}",
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Macro F1: {metrics['macro_f1']:.4f}",
        ]
    )
    for sentiment in SENTIMENTS:
        item = metrics["classes"][sentiment]
        report.append(
            f"{sentiment}: P={item['precision']:.4f} | R={item['recall']:.4f} | "
            f"F1={item['f1']:.4f} | support={item['support']}"
        )
    report.extend(
        [
            "Predicted: "
            f"positive={metrics['predicted']['positive']}, "
            f"neutral={metrics['predicted']['neutral']}, "
            f"negative={metrics['predicted']['negative']}",
            "False-neutral: "
            f"positive->neutral={metrics['positive_to_neutral']}, "
            f"negative->neutral={metrics['negative_to_neutral']}, "
            f"total={metrics['false_neutral_total']}",
            "Confusion matrix (rows=actual, columns=predicted):",
            *format_matrix(metrics["matrix"]),
        ]
    )


def main():
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")
    manifest = read_csv(MANIFEST_PATH)
    added_mask = validate_and_build_added_mask(train_rows, manifest)

    train_inputs = build_inputs(train_rows)
    dev_inputs = build_inputs(dev_rows)
    train_labels = [row["sentiment"] for row in train_rows]

    # This is deliberately fitted once and reused for all alpha conditions.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    train_matrix = vectorizer.fit_transform(train_inputs)
    dev_matrix = vectorizer.transform(dev_inputs)

    results = [
        run_alpha(
            alpha,
            train_matrix,
            train_labels,
            dev_matrix,
            dev_rows,
            added_mask,
        )
        for alpha in ALPHAS
    ]
    best = max(results, key=selection_key)
    best_nonzero = max(
        (result for result in results if result["alpha"] > 0), key=selection_key
    )
    top_three = sorted(results, key=selection_key, reverse=True)[:3]
    save_csv(results)

    best_metrics = best["metrics"]
    best_neutral = best_metrics["classes"]["neutral"]
    best_nonzero_metrics = best_nonzero["metrics"]
    best_nonzero_neutral = best_nonzero_metrics["classes"]["neutral"]
    best_nonzero_success = (
        best_nonzero_metrics["macro_f1"] >= HISTORICAL["macro_f1"]
        and best_nonzero_neutral["f1"] > HISTORICAL["neutral_f1"]
    )
    ideal_success = (
        best_metrics["macro_f1"] >= HISTORICAL["macro_f1"]
        and best_neutral["f1"] > HISTORICAL["neutral_f1"]
    )
    overprediction_reduced = (
        best_metrics["predicted"]["neutral"] < FULL_EXPANDED_REFERENCE["pred_neutral"]
        and best_metrics["false_neutral_total"] < FULL_EXPANDED_REFERENCE["false_neutral"]
    )
    partial_gain = (
        best_metrics["macro_f1"] >= HISTORICAL["macro_f1"]
        or best_neutral["f1"] > HISTORICAL["neutral_f1"]
    )
    if ideal_success:
        verdict = "YES"
    elif overprediction_reduced and partial_gain:
        verdict = "PARTIALLY"
    else:
        verdict = "NO"

    report = [
        "EXPERIMENT 11 — ADDED NEUTRAL EFFECTIVE-DOSE CALIBRATION",
        "",
        "Test split was not loaded or evaluated.",
        "Base train rows: 3316",
        "Manifest-matched added neutral rows: 207",
        "Total expanded train rows: 3523",
        f"Fixed old class weights: {FIXED_OLD_WEIGHTS}",
        "Representation: aspect=<aspect> [SEP] <sentence>",
        "TF-IDF: ngram_range=(1, 2), min_df=2; fitted once on all 3523 train rows.",
        "LogisticRegression: max_iter=1000, random_state=42; new fit per alpha.",
        "Alpha=0 keeps added rows in TF-IDF vocabulary/IDF but gives them zero "
        "LogisticRegression sample weight; it is not the historical cleaned model.",
        "",
        "SUMMARY",
        "Alpha | Accuracy | Macro F1 | Neutral P | Neutral R | Neutral F1 | "
        "Pred Neutral | False Neutral",
    ]
    for result in sorted(results, key=lambda item: item["alpha"]):
        metrics = result["metrics"]
        neutral = metrics["classes"]["neutral"]
        report.append(
            f"{result['alpha']:.2f} | {metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | {neutral['precision']:.4f} | "
            f"{neutral['recall']:.4f} | {neutral['f1']:.4f} | "
            f"{metrics['predicted']['neutral']} | {metrics['false_neutral_total']}"
        )

    report.extend(
        [
            "",
            "HISTORICAL REFERENCES",
            "Cleaned baseline: Accuracy=0.7603, Macro F1=0.5891, "
            "Neutral P/R/F1=0.3667/0.1930/0.2529.",
            "Full expanded calibrated: Macro F1=0.5805, "
            "Neutral P/R/F1=0.2066/0.4386/0.2809, Pred neutral=121, "
            "False neutral=96.",
            "That full-expanded reference used different positive/negative weights. "
            "Therefore alpha=1.00 in this experiment is not expected to reproduce it; "
            "all Experiment 11 conditions use fixed old cleaned weights.",
        ]
    )
    for result in results:
        append_result_details(report, result)

    report.extend(["", "TOP 3 CONFIGURATIONS BY DEV MACRO F1"])
    for rank, result in enumerate(top_three, start=1):
        metrics = result["metrics"]
        report.extend(
            [
                "",
                f"Rank {rank}: alpha={result['alpha']:.2f}",
                f"Accuracy={metrics['accuracy']:.4f} | Macro F1={metrics['macro_f1']:.4f}",
                "Language Macro F1 / Neutral F1:",
            ]
        )
        for language in LANGUAGES:
            group = metrics["languages"][language]
            report.append(
                f"{language.upper()}: Macro F1={group['macro_f1']:.4f} | "
                f"Neutral F1={group['neutral']['f1']:.4f}"
            )
        report.append("Aspect Macro F1 / Neutral F1 / False-neutral:")
        for aspect in ASPECTS:
            group = metrics["aspects"][aspect]
            report.append(
                f"{aspect}: Macro F1={group['macro_f1']:.4f} | "
                f"Neutral F1={group['neutral']['f1']:.4f} | "
                f"False-neutral={metrics['false_neutral_by_aspect'][aspect]}"
            )

    report.extend(
        [
            "",
            "BEST CONFIGURATION",
            f"Best alpha: {best['alpha']:.2f}",
            f"Accuracy: {best_metrics['accuracy']:.4f}",
            f"Macro F1: {best_metrics['macro_f1']:.4f}",
            f"Neutral precision: {best_neutral['precision']:.4f}",
            f"Neutral recall: {best_neutral['recall']:.4f}",
            f"Neutral F1: {best_neutral['f1']:.4f}",
            f"Pred neutral: {best_metrics['predicted']['neutral']}",
            f"False neutral: {best_metrics['false_neutral_total']}",
            "",
            "BEST NONZERO-DOSE CONFIGURATION",
            f"Alpha: {best_nonzero['alpha']:.2f}",
            f"Macro F1: {best_nonzero_metrics['macro_f1']:.4f}",
            f"Neutral precision: {best_nonzero_neutral['precision']:.4f}",
            f"Neutral recall: {best_nonzero_neutral['recall']:.4f}",
            f"Neutral F1: {best_nonzero_neutral['f1']:.4f}",
            f"Pred neutral: {best_nonzero_metrics['predicted']['neutral']}",
            f"False neutral: {best_nonzero_metrics['false_neutral_total']}",
            f"Ideal success thresholds met by best nonzero dose: {best_nonzero_success}",
            "",
            "DELTA VS CLEANED BASELINE",
            f"Macro F1: {best_metrics['macro_f1'] - HISTORICAL['macro_f1']:+.4f}",
            f"Neutral precision: {best_neutral['precision'] - HISTORICAL['neutral_precision']:+.4f}",
            f"Neutral recall: {best_neutral['recall'] - HISTORICAL['neutral_recall']:+.4f}",
            f"Neutral F1: {best_neutral['f1'] - HISTORICAL['neutral_f1']:+.4f}",
            "",
            f"DOES SOFT DOSE CONTROL SOLVE THE OVER-PREDICTION PROBLEM? {verdict}",
            f"Ideal success criteria met: {ideal_success}",
            f"Over-prediction reduced vs full expanded reference: {overprediction_reduced}",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("Alpha | Accuracy | Macro F1 | Neutral P | Neutral R | Neutral F1 | "
          "Pred Neutral | False Neutral")
    for result in sorted(results, key=lambda item: item["alpha"]):
        metrics = result["metrics"]
        neutral = metrics["classes"]["neutral"]
        print(
            f"{result['alpha']:.2f} | {metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | {neutral['precision']:.4f} | "
            f"{neutral['recall']:.4f} | {neutral['f1']:.4f} | "
            f"{metrics['predicted']['neutral']} | {metrics['false_neutral_total']}"
        )
    print("\nBEST ALPHA:", f"{best['alpha']:.2f}")
    print(f"Macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"Neutral P: {best_neutral['precision']:.4f}")
    print(f"Neutral R: {best_neutral['recall']:.4f}")
    print(f"Neutral F1: {best_neutral['f1']:.4f}")
    print(f"Pred Neutral: {best_metrics['predicted']['neutral']}")
    print(f"False Neutral: {best_metrics['false_neutral_total']}")
    print("\nBEST NONZERO ALPHA:", f"{best_nonzero['alpha']:.2f}")
    print(f"Macro F1: {best_nonzero_metrics['macro_f1']:.4f}")
    print(f"Neutral F1: {best_nonzero_neutral['f1']:.4f}")
    print(f"False Neutral: {best_nonzero_metrics['false_neutral_total']}")
    print(f"Ideal thresholds met: {best_nonzero_success}")
    print("\nDelta vs cleaned baseline:")
    print(f"Macro F1: {best_metrics['macro_f1'] - HISTORICAL['macro_f1']:+.4f}")
    print(
        f"Neutral P: {best_neutral['precision'] - HISTORICAL['neutral_precision']:+.4f}"
    )
    print(f"Neutral R: {best_neutral['recall'] - HISTORICAL['neutral_recall']:+.4f}")
    print(f"Neutral F1: {best_neutral['f1'] - HISTORICAL['neutral_f1']:+.4f}")
    print("\nDOES SOFT DOSE CONTROL SOLVE THE OVER-PREDICTION PROBLEM?", verdict)
    print(f"Report: {REPORT_PATH}")
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
