"""Run the Experiment 06B neutral-data/class-weight ablation."""

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
OLD_TRAIN_PATH = ROOT / "data" / "cleaned" / "train.csv"
NEW_TRAIN_PATH = ROOT / "data" / "expanded_neutral_v1" / "train.csv"
DEV_PATH = ROOT / "data" / "cleaned" / "dev.csv"
EXPANDED_DEV_PATH = ROOT / "data" / "expanded_neutral_v1" / "dev.csv"
REPORT_PATH = ROOT / "data" / "analysis" / "experiment_06b_weight_ablation.txt"
PREDICTIONS_TEMPLATE = ROOT / "data" / "analysis" / "experiment_06b_predictions_{condition}.csv"

EXPECTED_OLD_TRAIN = Counter({"positive": 2173, "neutral": 93, "negative": 1050})
EXPECTED_NEW_TRAIN = Counter({"positive": 2173, "neutral": 300, "negative": 1050})
EXPECTED_DEV = Counter({"positive": 586, "neutral": 57, "negative": 258})

FIXED_OLD_WEIGHTS = {
    "positive": 0.5086669734621875,
    "neutral": 11.885304659498209,
    "negative": 1.0526984126984127,
}

CONDITIONS = {
    "A": {
        "dataset": "OLD",
        "train_path": OLD_TRAIN_PATH,
        "weighting": "None",
        "class_weight": None,
    },
    "B": {
        "dataset": "EXPANDED",
        "train_path": NEW_TRAIN_PATH,
        "weighting": "None",
        "class_weight": None,
    },
    "C": {
        "dataset": "OLD",
        "train_path": OLD_TRAIN_PATH,
        "weighting": "Fixed old weights",
        "class_weight": FIXED_OLD_WEIGHTS,
    },
    "D": {
        "dataset": "EXPANDED",
        "train_path": NEW_TRAIN_PATH,
        "weighting": "Fixed old weights",
        "class_weight": FIXED_OLD_WEIGHTS,
    },
}

LANGUAGES = (("tr", "TR"), ("en", "EN"), ("es", "ES"))

HISTORICAL_CLEANED = {
    "accuracy": 0.7603,
    "macro_f1": 0.5891,
    "neutral_precision": 0.3667,
    "neutral_recall": 0.1930,
    "neutral_f1": 0.2529,
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_distribution(rows, expected: Counter, name: str) -> None:
    actual = Counter(row["sentiment"] for row in rows)
    if actual != expected:
        raise ValueError(f"{name} distribution mismatch: {dict(actual)}")


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


def save_predictions(condition: str, dev_rows, predictions) -> Path:
    path = Path(str(PREDICTIONS_TEMPLATE).format(condition=condition))
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sentence",
        "language",
        "aspect",
        "true_sentiment",
        "predicted_sentiment",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
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
    return path


def run_condition(condition: str, config, train_rows, dev_rows):
    model = build_model()
    model.set_params(classifier__class_weight=config["class_weight"])
    model.fit(
        build_inputs(train_rows),
        [row["sentiment"] for row in train_rows],
    )
    predictions = model.predict(build_inputs(dev_rows)).tolist()
    y_true = [row["sentiment"] for row in dev_rows]

    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "classes": class_metrics(y_true, predictions),
        "matrix": confusion_matrix(y_true, predictions, labels=SENTIMENTS),
        "predicted": Counter(predictions),
        "languages": grouped_metrics(dev_rows, predictions, "language", LANGUAGES),
        "aspects": grouped_metrics(
            dev_rows,
            predictions,
            "aspect",
            [(aspect, aspect) for aspect in ASPECTS],
        ),
    }
    result["predictions_path"] = save_predictions(condition, dev_rows, predictions)
    return result


def metric_delta(new, old):
    return {
        "accuracy": new["accuracy"] - old["accuracy"],
        "macro_f1": new["macro_f1"] - old["macro_f1"],
        "neutral_precision": (
            new["classes"]["neutral"]["precision"]
            - old["classes"]["neutral"]["precision"]
        ),
        "neutral_recall": (
            new["classes"]["neutral"]["recall"]
            - old["classes"]["neutral"]["recall"]
        ),
        "neutral_f1": (
            new["classes"]["neutral"]["f1"]
            - old["classes"]["neutral"]["f1"]
        ),
    }


def append_condition_report(report, condition, config, result):
    report.extend(
        [
            "",
            f"CONDITION {condition}",
            f"Dataset: {config['dataset']}",
            f"Weighting: {config['weighting']}",
            f"Accuracy: {result['accuracy']:.4f}",
            f"Macro F1: {result['macro_f1']:.4f}",
            "",
            "Per-class metrics:",
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
            "",
            "Confusion matrix (rows=actual, columns=predicted):",
            "                 positive  neutral  negative",
        ]
    )
    for sentiment, row in zip(SENTIMENTS, result["matrix"]):
        report.append(f"{sentiment:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}")

    report.extend(
        [
            "",
            "Predicted counts:",
            *[f"{label}: {result['predicted'][label]}" for label in SENTIMENTS],
            "",
            "Language metrics:",
        ]
    )
    for language in ("TR", "EN", "ES"):
        metrics = result["languages"][language]
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

    report.extend(["", "Aspect metrics:"])
    for aspect in ASPECTS:
        metrics = result["aspects"][aspect]
        report.append(
            f"{aspect}: n={metrics['n']} | accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f} | "
            f"neutral F1={metrics['neutral']['f1']:.4f}"
        )


def append_delta(report, title, delta):
    report.extend(
        [
            title,
            f"  Delta Accuracy: {delta['accuracy']:+.4f}",
            f"  Delta Macro F1: {delta['macro_f1']:+.4f}",
            f"  Delta Neutral precision: {delta['neutral_precision']:+.4f}",
            f"  Delta Neutral recall: {delta['neutral_recall']:+.4f}",
            f"  Delta Neutral F1: {delta['neutral_f1']:+.4f}",
        ]
    )


def main() -> None:
    old_train_rows = load_rows(OLD_TRAIN_PATH, expected_split="train")
    new_train_rows = load_rows(NEW_TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")

    validate_distribution(old_train_rows, EXPECTED_OLD_TRAIN, "Old train")
    validate_distribution(new_train_rows, EXPECTED_NEW_TRAIN, "Expanded train")
    validate_distribution(dev_rows, EXPECTED_DEV, "Dev")
    if file_hash(DEV_PATH) != file_hash(EXPANDED_DEV_PATH):
        raise ValueError("Expanded dev.csv differs from cleaned dev.csv")

    train_rows_by_dataset = {
        "OLD": old_train_rows,
        "EXPANDED": new_train_rows,
    }
    results = {}
    for condition, config in CONDITIONS.items():
        print(f"Running Condition {condition}...")
        results[condition] = run_condition(
            condition,
            config,
            train_rows_by_dataset[config["dataset"]],
            dev_rows,
        )

    tolerance = 0.0001
    condition_c_reproduces = (
        abs(results["C"]["accuracy"] - HISTORICAL_CLEANED["accuracy"]) <= tolerance
        and abs(results["C"]["macro_f1"] - HISTORICAL_CLEANED["macro_f1"]) <= tolerance
        and abs(
            results["C"]["classes"]["neutral"]["f1"]
            - HISTORICAL_CLEANED["neutral_f1"]
        )
        <= tolerance
    )

    pipeline = build_model()
    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    report = [
        "EXPERIMENT 06B — NEUTRAL DATA x CLASS WEIGHT ABLATION",
        "",
        "CONTROLLED PIPELINE",
        "Dev: data/cleaned/dev.csv for every condition",
        "Test: not loaded or evaluated",
        "Input: aspect=<aspect> [SEP] <sentence>",
        f"TF-IDF: ngram_range={tfidf.ngram_range}, min_df={tfidf.min_df}, "
        f"max_df={tfidf.max_df}, max_features={tfidf.max_features}",
        f"LogisticRegression: solver={classifier.solver}, max_iter={classifier.max_iter}, "
        f"random_state={classifier.random_state}",
        f"Fixed old weights: {FIXED_OLD_WEIGHTS}",
        "Hyperparameter tuning: none",
        "",
        "DATA VALIDATION",
        f"Old train: {dict(Counter(row['sentiment'] for row in old_train_rows))} | "
        f"total={len(old_train_rows)}",
        f"Expanded train: {dict(Counter(row['sentiment'] for row in new_train_rows))} | "
        f"total={len(new_train_rows)}",
        f"Dev: {dict(Counter(row['sentiment'] for row in dev_rows))} | "
        f"total={len(dev_rows)}",
        "Expanded dev matches cleaned dev byte-for-byte: True",
    ]

    for condition, config in CONDITIONS.items():
        append_condition_report(report, condition, config, results[condition])

    comparisons = {
        "B - A (data effect without weighting)": metric_delta(results["B"], results["A"]),
        "D - C (data effect with fixed weights)": metric_delta(results["D"], results["C"]),
        "C - A (weighting effect on old data)": metric_delta(results["C"], results["A"]),
        "D - B (weighting effect on expanded data)": metric_delta(results["D"], results["B"]),
    }

    report.extend(["", "CRITICAL COMPARISONS"])
    for title, comparison in comparisons.items():
        append_delta(report, title, comparison)

    report.extend(
        [
            "",
            "COMPACT COMPARISON",
            "Condition | Dataset | Weighting | Accuracy | Macro F1 | Neutral P | Neutral R | Neutral F1 | Pred Neutral",
        ]
    )
    for condition, config in CONDITIONS.items():
        result = results[condition]
        neutral = result["classes"]["neutral"]
        report.append(
            f"{condition} | {config['dataset']} | {config['weighting']} | "
            f"{result['accuracy']:.4f} | {result['macro_f1']:.4f} | "
            f"{neutral['precision']:.4f} | {neutral['recall']:.4f} | "
            f"{neutral['f1']:.4f} | {result['predicted']['neutral']}"
        )

    report.extend(
        [
            "",
            f"C reproduces old baseline: {'YES' if condition_c_reproduces else 'NO'}",
            f"Reproducibility tolerance against rounded historical metrics: {tolerance:.4f}",
        ]
    )
    if not condition_c_reproduces:
        report.append(
            "Interpretation withheld: Condition C materially differs from the historical baseline."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
