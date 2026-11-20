"""Aynı sentiment baseline modelini cleaned train/dev üzerinde değerlendir."""

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from train_sentiment_baseline import (
    ASPECTS,
    LANGUAGE_NAMES,
    SENTIMENTS,
    build_inputs,
    build_model,
    load_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "cleaned" / "train.csv"
DEV_PATH = PROJECT_ROOT / "data" / "cleaned" / "dev.csv"
REPORT_PATH = (
    PROJECT_ROOT / "data" / "analysis" / "experiment_02_clean_baseline.txt"
)

ORIGINAL_METRICS = {
    "accuracy": 0.7614,
    "macro_f1": 0.5725,
    "positive_f1": 0.8460,
    "neutral_f1": 0.2174,
    "negative_f1": 0.6541,
    "Turkish_macro_f1": 0.5896,
    "English_macro_f1": 0.5681,
    "Spanish_macro_f1": 0.5606,
    "taste_macro_f1": 0.5807,
    "atmosphere_macro_f1": 0.3189,
    "service_macro_f1": 0.5273,
    "value_macro_f1": 0.6500,
}


def group_metrics(rows, predictions, column, groups):
    """Language veya aspect grupları için accuracy ve macro F1 hesapla."""
    results = {}
    for group_value, display_name in groups:
        indices = [
            index
            for index, row in enumerate(rows)
            if row[column] == group_value
        ]
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
        }
    return results


def difference_line(label, original, cleaned):
    """Original-cleaned karşılaştırmasını tek satırda biçimlendir."""
    difference = cleaned - original
    return (
        f"{label}: original={original:.4f} | cleaned={cleaned:.4f} | "
        f"difference={difference:+.4f}"
    )


def build_report(train_rows, dev_rows, predictions):
    """Deney metriklerini ve original karşılaştırmasını metne dönüştür."""
    y_true = [row["sentiment"] for row in dev_rows]
    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(y_true, predictions, average="macro", zero_division=0)

    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=SENTIMENTS,
        zero_division=0,
    )
    sentiment_metrics = {
        sentiment: {
            "precision": precision[index],
            "recall": recall[index],
            "f1": class_f1[index],
            "support": int(support[index]),
        }
        for index, sentiment in enumerate(SENTIMENTS)
    }

    matrix = confusion_matrix(y_true, predictions, labels=SENTIMENTS)
    language_metrics = group_metrics(
        dev_rows,
        predictions,
        "language",
        list(LANGUAGE_NAMES.items()),
    )
    aspect_metrics = group_metrics(
        dev_rows,
        predictions,
        "aspect",
        [(aspect, aspect) for aspect in ASPECTS],
    )

    report = [
        "EXPERIMENT 02 — CLEANED DATASET SENTIMENT BASELINE",
        "",
        "Controlled experiment:",
        "- Training data: data/cleaned/train.csv",
        "- Evaluation data: data/cleaned/dev.csv",
        "- Test data: not loaded or evaluated",
        "- Input: aspect=<aspect> [SEP] <sentence>",
        "- TF-IDF: ngram_range=(1, 2), min_df=2",
        "- Logistic Regression: class_weight=balanced, max_iter=1000, random_state=42",
        f"- Train rows: {len(train_rows)}",
        f"- Dev rows: {len(dev_rows)}",
        "",
        "OVERALL",
        f"Accuracy: {accuracy:.4f}",
        f"Macro F1: {macro_f1:.4f}",
        "",
        "SENTIMENT METRICS",
    ]

    for sentiment in SENTIMENTS:
        metrics = sentiment_metrics[sentiment]
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
        report.append(
            f"{sentiment:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}"
        )

    report.extend(["", "LANGUAGE METRICS"])
    for language_name, metrics in language_metrics.items():
        report.append(
            f"{language_name}: n={metrics['n']} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f}"
        )

    report.extend(["", "ASPECT METRICS"])
    for aspect, metrics in aspect_metrics.items():
        report.append(
            f"{aspect}: n={metrics['n']} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f}"
        )

    report.extend(
        [
            "",
            "ORIGINAL VS CLEANED",
            difference_line("Accuracy", ORIGINAL_METRICS["accuracy"], accuracy),
            difference_line("Macro F1", ORIGINAL_METRICS["macro_f1"], macro_f1),
        ]
    )

    for sentiment in SENTIMENTS:
        report.append(
            difference_line(
                f"{sentiment} F1",
                ORIGINAL_METRICS[f"{sentiment}_f1"],
                sentiment_metrics[sentiment]["f1"],
            )
        )

    for language_name, metrics in language_metrics.items():
        report.append(
            difference_line(
                f"{language_name} macro F1",
                ORIGINAL_METRICS[f"{language_name}_macro_f1"],
                metrics["macro_f1"],
            )
        )

    for aspect, metrics in aspect_metrics.items():
        report.append(
            difference_line(
                f"{aspect} macro F1",
                ORIGINAL_METRICS[f"{aspect}_macro_f1"],
                metrics["macro_f1"],
            )
        )

    return "\n".join(report) + "\n"


def main():
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")

    model = build_model()
    model.fit(
        build_inputs(train_rows),
        [row["sentiment"] for row in train_rows],
    )
    predictions = model.predict(build_inputs(dev_rows))

    report = build_report(train_rows, dev_rows, predictions)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
