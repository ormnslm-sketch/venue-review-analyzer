"""M-ABSA restaurant verisini ilk baseline aspect sistemine dönüştür."""

import csv
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "Multilingual-NLP/M-ABSA"
DOMAIN = "restaurant"
LANGUAGES = ("tr", "en", "es")
SPLITS = ("train", "dev", "test")

CATEGORY_MAPPING = {
    "food quality": "taste",
    "drinks quality": "taste",
    "ambience general": "atmosphere",
    "service general": "service",
    "food prices": "value",
    "restaurant prices": "value",
    "drinks prices": "value",
}

ALLOWED_ASPECTS = {"taste", "atmosphere", "service", "value"}
ALLOWED_SENTIMENTS = {"positive", "neutral", "negative"}
OUTPUT_COLUMNS = (
    "sentence",
    "language",
    "split",
    "aspect",
    "sentiment",
    "source_category",
    "aspect_term",
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TRIPLET_PATTERN = re.compile(
    r"\[\s*(['\"])(.*?)\1\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*\]"
)


def load_language(language_code: str):
    """Bir dilin mevcut train, dev ve test splitlerini yükle."""
    data_files = {
        split: (
            f"hf://datasets/{DATASET_ID}/{DOMAIN}/"
            f"{language_code}/{split}.txt"
        )
        for split in SPLITS
    }
    return load_dataset("text", data_files=data_files)


def parse_row(raw_text: str):
    """Ham satırı cümle ve [term, category, sentiment] tripletlerine ayır."""
    sentence, separator, raw_labels = raw_text.partition("####")
    if not separator:
        raise ValueError(f"Satırda '####' ayırıcısı bulunamadı: {raw_text}")

    if raw_labels.strip() == "[]":
        return sentence, []

    matches = TRIPLET_PATTERN.findall(raw_labels)
    if not matches:
        raise ValueError(f"Label tripletleri okunamadı: {raw_labels}")

    triplets = [
        (term, category, sentiment)
        for _, term, category, sentiment in matches
    ]
    return sentence, triplets


def preprocess_dataset():
    """Mapping'e giren annotation'ları split bazında long formatta üret."""
    rows_by_split = {split: [] for split in SPLITS}

    for language in LANGUAGES:
        dataset = load_language(language)

        for split in SPLITS:
            for raw_row in dataset[split]:
                sentence, triplets = parse_row(raw_row["text"])

                for aspect_term, source_category, sentiment in triplets:
                    aspect = CATEGORY_MAPPING.get(source_category)
                    if aspect is None:
                        continue

                    rows_by_split[split].append(
                        {
                            "sentence": sentence,
                            "language": language,
                            "split": split,
                            "aspect": aspect,
                            "sentiment": sentiment,
                            "source_category": source_category,
                            "aspect_term": aspect_term,
                        }
                    )

    return rows_by_split


def save_csv_files(rows_by_split) -> None:
    """Her mevcut split için üç dili içeren bir CSV dosyası yaz."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, rows in rows_by_split.items():
        output_path = OUTPUT_DIR / f"{split}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


def validate(rows_by_split) -> None:
    """İşlenmiş satırların beklenen baseline şemasına uyduğunu kontrol et."""
    all_rows = [row for rows in rows_by_split.values() for row in rows]

    observed_languages = {row["language"] for row in all_rows}
    observed_aspects = {row["aspect"] for row in all_rows}
    observed_sentiments = {row["sentiment"] for row in all_rows}
    empty_sentence_count = sum(not row["sentence"].strip() for row in all_rows)

    checks = {
        "Only tr/en/es": observed_languages == set(LANGUAGES),
        "Only taste/atmosphere/service/value": (
            observed_aspects == ALLOWED_ASPECTS
        ),
        "Only positive/neutral/negative": (
            observed_sentiments == ALLOWED_SENTIMENTS
        ),
        "No empty sentence": empty_sentence_count == 0,
    }

    print("=" * 80)
    print("VALIDATION")
    for check_name, passed in checks.items():
        print(f"{check_name}: {passed}")

    print(f"Observed languages: {sorted(observed_languages)}")
    print(f"Observed aspects: {sorted(observed_aspects)}")
    print(f"Observed sentiments: {sorted(observed_sentiments)}")
    print(f"Empty sentence count: {empty_sentence_count}")

    print("\nRows per split:")
    for split in SPLITS:
        print(f"{split}: {len(rows_by_split[split])}")

    language_aspect_counts = Counter(
        (row["language"], row["aspect"]) for row in all_rows
    )
    print("\nExamples per language + aspect:")
    for language in LANGUAGES:
        print(f"{language}:")
        for aspect in sorted(ALLOWED_ASPECTS):
            print(f"  {aspect}: {language_aspect_counts[language, aspect]}")

    sentiment_counts = Counter(row["sentiment"] for row in all_rows)
    print("\nSentiment distribution:")
    for sentiment, count in sentiment_counts.most_common():
        print(f"{sentiment}: {count}")

    if not all(checks.values()):
        raise ValueError("Validation kontrollerinden en az biri başarısız oldu.")


def main() -> None:
    rows_by_split = preprocess_dataset()
    save_csv_files(rows_by_split)
    validate(rows_by_split)
    print(f"\nCSV files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
