"""Mevcut sentiment baseline modelinin dev hatalarını analiz et."""

import csv
import random
from collections import Counter

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from train_sentiment_baseline import (
    ASPECTS,
    DEV_PATH,
    LANGUAGE_NAMES,
    PROJECT_ROOT,
    RANDOM_SEED,
    SENTIMENTS,
    TRAIN_PATH,
    build_inputs,
    build_model,
    load_rows,
)


ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
PREDICTIONS_PATH = ANALYSIS_DIR / "sentiment_predictions.csv"
ERRORS_PATH = ANALYSIS_DIR / "sentiment_errors.csv"
ERROR_COLUMNS = (
    "sentence",
    "language",
    "aspect",
    "actual_sentiment",
    "predicted_sentiment",
)
SAMPLE_SIZE = 15


def build_prediction_records(dev_rows, predictions):
    """Her dev tahminini analiz edilebilir bir kayda dönüştür."""
    return [
        {
            "sentence": row["sentence"],
            "language": row["language"],
            "aspect": row["aspect"],
            "actual_sentiment": row["sentiment"],
            "predicted_sentiment": prediction,
        }
        for row, prediction in zip(dev_rows, predictions)
    ]


def save_records(csv_path, records) -> None:
    """Tahmin kayıtlarını verilen CSV dosyasına yaz."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ERROR_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def save_errors(prediction_records):
    """Yanlış dev tahminlerini ayrı bir CSV dosyasına kaydet."""
    errors = [
        record
        for record in prediction_records
        if record["actual_sentiment"] != record["predicted_sentiment"]
    ]
    save_records(ERRORS_PATH, errors)

    return errors


def show_error_samples(title: str, errors) -> None:
    """Sabit seed ile en fazla 15 yanlış tahmini göster."""
    random_generator = random.Random(RANDOM_SEED)
    sample_size = min(SAMPLE_SIZE, len(errors))
    samples = random_generator.sample(errors, sample_size)

    print(f"\n{title} ({sample_size} examples):")
    for index, error in enumerate(samples, start=1):
        print(f"\nExample {index}")
        print(f"Sentence: {error['sentence']}")
        print(f"Language: {error['language']}")
        print(f"Aspect: {error['aspect']}")
        print(f"Actual: {error['actual_sentiment']}")
        print(f"Predicted: {error['predicted_sentiment']}")


def show_matrix(title: str, records) -> None:
    """Seçilen kayıtlar için confusion matrix göster."""
    actual = [record["actual_sentiment"] for record in records]
    predicted = [record["predicted_sentiment"] for record in records]
    matrix = confusion_matrix(actual, predicted, labels=SENTIMENTS)
    column_width = 11

    print(f"\n{title} confusion matrix (rows=actual, columns=predicted):")
    header = "actual\\pred".ljust(column_width)
    header += "".join(label.rjust(column_width) for label in SENTIMENTS)
    print(header)

    for label, row in zip(SENTIMENTS, matrix):
        values = "".join(str(value).rjust(column_width) for value in row)
        print(label.ljust(column_width) + values)


def show_neutral_analysis(prediction_records) -> None:
    """Gerçek neutral örneklerin tahmin dağılımını ve hatalarını göster."""
    neutral_records = [
        record
        for record in prediction_records
        if record["actual_sentiment"] == "neutral"
    ]
    prediction_counts = Counter(
        record["predicted_sentiment"] for record in neutral_records
    )
    neutral_errors = [
        record
        for record in neutral_records
        if record["predicted_sentiment"] != "neutral"
    ]

    print("\n" + "=" * 80)
    print("NEUTRAL ANALYSIS")
    print(f"Actual neutral examples: {len(neutral_records)}")
    print(f"Predicted positive: {prediction_counts['positive']}")
    print(f"Predicted negative: {prediction_counts['negative']}")
    print(f"Correctly predicted neutral: {prediction_counts['neutral']}")
    show_error_samples("Misclassified neutral samples", neutral_errors)


def show_atmosphere_analysis(prediction_records) -> None:
    """Atmosphere confusion matrix'ini ve yanlış örnekleri göster."""
    atmosphere_records = [
        record
        for record in prediction_records
        if record["aspect"] == "atmosphere"
    ]
    atmosphere_errors = [
        record
        for record in atmosphere_records
        if record["actual_sentiment"] != record["predicted_sentiment"]
    ]

    print("\n" + "=" * 80)
    print("ATMOSPHERE ANALYSIS")
    show_matrix("Atmosphere", atmosphere_records)
    show_error_samples("Misclassified atmosphere samples", atmosphere_errors)


def show_distributions(dev_rows) -> None:
    """Dev verisinin aspect ve dil bazlı gerçek sentiment dağılımlarını göster."""
    aspect_counts = Counter(
        (row["aspect"], row["sentiment"]) for row in dev_rows
    )
    language_counts = Counter(
        (row["language"], row["sentiment"]) for row in dev_rows
    )

    print("\n" + "=" * 80)
    print("ASPECT + SENTIMENT DISTRIBUTION")
    for aspect in ASPECTS:
        print(f"{aspect}:")
        for sentiment in SENTIMENTS:
            print(f"  {sentiment}: {aspect_counts[aspect, sentiment]}")

    print("\n" + "=" * 80)
    print("LANGUAGE + SENTIMENT DISTRIBUTION")
    for language, language_name in LANGUAGE_NAMES.items():
        print(f"{language_name} ({language}):")
        for sentiment in SENTIMENTS:
            print(f"  {sentiment}: {language_counts[language, sentiment]}")


def train_dummy(train_labels, dev_row_count):
    """Most-frequent DummyClassifier tahminlerini üret."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit([[0]] * len(train_labels), train_labels)
    return dummy.predict([[0]] * dev_row_count)


def show_model_comparison(dev_labels, dummy_predictions, model_predictions) -> None:
    """Dummy ve mevcut Logistic Regression sonuçlarını karşılaştır."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")

    print("\nDummy baseline:")
    print(f"Accuracy: {accuracy_score(dev_labels, dummy_predictions):.4f}")
    print(
        "Macro F1: "
        f"{f1_score(dev_labels, dummy_predictions, average='macro', zero_division=0):.4f}"
    )

    print("\nTF-IDF + Logistic Regression:")
    print(f"Accuracy: {accuracy_score(dev_labels, model_predictions):.4f}")
    print(
        "Macro F1: "
        f"{f1_score(dev_labels, model_predictions, average='macro', zero_division=0):.4f}"
    )


def main() -> None:
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")

    train_inputs = build_inputs(train_rows)
    dev_inputs = build_inputs(dev_rows)
    train_labels = [row["sentiment"] for row in train_rows]
    dev_labels = [row["sentiment"] for row in dev_rows]

    model = build_model()
    model.fit(train_inputs, train_labels)
    dev_predictions = model.predict(dev_inputs)

    prediction_records = build_prediction_records(dev_rows, dev_predictions)
    save_records(PREDICTIONS_PATH, prediction_records)
    errors = save_errors(prediction_records)

    print(f"Dev predictions: {len(prediction_records)}")
    print(f"Prediction CSV: {PREDICTIONS_PATH}")
    print(f"Incorrect predictions saved: {len(errors)}")
    print(f"Error CSV: {ERRORS_PATH}")
    print("Test split is not loaded or evaluated.")

    show_neutral_analysis(prediction_records)
    show_atmosphere_analysis(prediction_records)
    show_distributions(dev_rows)

    dummy_predictions = train_dummy(train_labels, len(dev_rows))
    show_model_comparison(dev_labels, dummy_predictions, dev_predictions)


if __name__ == "__main__":
    main()
