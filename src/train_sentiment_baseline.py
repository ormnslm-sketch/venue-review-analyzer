"""TF-IDF ve Logistic Regression ile sentiment baseline modeli eğit."""

import csv
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
DEV_PATH = PROJECT_ROOT / "data" / "processed" / "dev.csv"

REQUIRED_COLUMNS = {"sentence", "language", "split", "aspect", "sentiment"}
SENTIMENTS = ("positive", "neutral", "negative")
LANGUAGE_NAMES = {
    "tr": "Turkish",
    "en": "English",
    "es": "Spanish",
}
ASPECTS = ("taste", "atmosphere", "service", "value")
RANDOM_SEED = 42


def load_rows(csv_path: Path, expected_split: str):
    """Bir CSV splitini oku ve temel şemasını doğrula."""
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            raise ValueError(
                f"{csv_path.name} eksik kolonlar içeriyor: {sorted(missing_columns)}"
            )

        rows = list(reader)

    wrong_splits = {row["split"] for row in rows if row["split"] != expected_split}
    if wrong_splits:
        raise ValueError(
            f"{csv_path.name} beklenmeyen split değerleri içeriyor: "
            f"{sorted(wrong_splits)}"
        )

    if any(not row["sentence"].strip() for row in rows):
        raise ValueError(f"{csv_path.name} boş sentence içeriyor.")

    return rows


def build_inputs(rows):
    """Aspect bilgisini cümleyle birlikte model girdisine dönüştür."""
    return [
        f"aspect={row['aspect']} [SEP] {row['sentence']}"
        for row in rows
    ]


def build_model():
    """Basit ve açıklanabilir TF-IDF + Logistic Regression pipeline'ı kur."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def show_confusion_matrix(y_true, y_pred) -> None:
    """Confusion matrix'i sabit sentiment sırasıyla yazdır."""
    matrix = confusion_matrix(y_true, y_pred, labels=SENTIMENTS)
    column_width = 11

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    header = "actual\\pred".ljust(column_width)
    header += "".join(label.rjust(column_width) for label in SENTIMENTS)
    print(header)

    for label, row in zip(SENTIMENTS, matrix):
        values = "".join(str(value).rjust(column_width) for value in row)
        print(label.ljust(column_width) + values)


def show_overall_metrics(y_true, y_pred) -> None:
    """Dev splitinin genel ve sınıf bazlı metriklerini göster."""
    print("\nOverall dev metrics:")
    print(f"Accuracy:        {accuracy_score(y_true, y_pred):.4f}")
    print(
        "Macro precision: "
        f"{precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )
    print(
        "Macro recall:    "
        f"{recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )
    print(
        "Macro F1:        "
        f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )

    print("\nMetrics by sentiment:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=SENTIMENTS,
            digits=4,
            zero_division=0,
        )
    )
    show_confusion_matrix(y_true, y_pred)


def show_group_metrics(rows, predictions, column: str, groups) -> None:
    """Dev performansını language veya aspect gruplarında göster."""
    print(f"\nMetrics by {column}:")

    for group_value, display_name in groups:
        indices = [
            index
            for index, row in enumerate(rows)
            if row[column] == group_value
        ]
        y_true = [rows[index]["sentiment"] for index in indices]
        y_pred = [predictions[index] for index in indices]

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(
            f"{display_name}: n={len(indices)}, "
            f"accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}"
        )


def main() -> None:
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")

    train_inputs = build_inputs(train_rows)
    dev_inputs = build_inputs(dev_rows)
    train_labels = [row["sentiment"] for row in train_rows]
    dev_labels = [row["sentiment"] for row in dev_rows]

    print(f"Train rows: {len(train_rows)}")
    print(f"Dev rows: {len(dev_rows)}")
    print(f"Train sentiment distribution: {dict(Counter(train_labels))}")
    print("Test split is not loaded or evaluated.")

    model = build_model()
    model.fit(train_inputs, train_labels)
    dev_predictions = model.predict(dev_inputs)

    show_overall_metrics(dev_labels, dev_predictions)

    language_groups = list(LANGUAGE_NAMES.items())
    show_group_metrics(dev_rows, dev_predictions, "language", language_groups)

    aspect_groups = [(aspect, aspect) for aspect in ASPECTS]
    show_group_metrics(dev_rows, dev_predictions, "aspect", aspect_groups)


if __name__ == "__main__":
    main()
