"""Processed sentiment datasetini baseline eğitimi için temizle."""

import csv
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"
SPLITS = ("train", "dev", "test")
LANGUAGES = ("tr", "en", "es")
ASPECTS = ("taste", "atmosphere", "service", "value")
SENTIMENTS = ("positive", "neutral", "negative")

SOURCE_COLUMNS = (
    "sentence",
    "language",
    "split",
    "aspect",
    "sentiment",
    "source_category",
    "aspect_term",
)
OUTPUT_COLUMNS = SOURCE_COLUMNS + ("source_row_id",)


def exact_key(row):
    """Exact duplicate kontrolünde kullanılan anahtarı döndür."""
    return (
        row["sentence"],
        row["language"],
        row["aspect"],
        row["sentiment"],
    )


def input_key(row):
    """Baseline model girdisini temsil eden anahtarı döndür."""
    return row["sentence"], row["language"], row["aspect"]


def load_split(split):
    """Bir processed CSV'yi oku ve güvenilir kaynak satır kimliği ekle."""
    input_path = INPUT_DIR / f"{split}.csv"
    with input_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError(
                f"{input_path.name} beklenen kolonlara sahip değil: "
                f"{reader.fieldnames}"
            )

        rows = []
        for source_row_number, row in enumerate(reader, start=1):
            if row["split"] != split:
                raise ValueError(
                    f"{input_path.name} içinde beklenmeyen split değeri var."
                )
            row["source_row_id"] = f"{split}:{source_row_number:06d}"
            rows.append(row)
    return rows


def remove_exact_duplicates(rows):
    """Aynı exact key için dosya sırasındaki ilk satırı tut."""
    seen = set()
    cleaned_rows = []

    for row in rows:
        key = exact_key(row)
        if key in seen:
            continue
        seen.add(key)
        cleaned_rows.append(row)

    return cleaned_rows, len(rows) - len(cleaned_rows)


def remove_conflicting_inputs(rows):
    """Birden fazla sentiment taşıyan inputların bütün satırlarını çıkar."""
    labels_by_input = defaultdict(set)
    for row in rows:
        labels_by_input[input_key(row)].add(row["sentiment"])

    conflicting_keys = {
        key for key, labels in labels_by_input.items() if len(labels) > 1
    }
    cleaned_rows = [row for row in rows if input_key(row) not in conflicting_keys]
    return cleaned_rows, len(rows) - len(cleaned_rows), len(conflicting_keys)


def remove_split_leakage(rows_by_split):
    """Train > dev > test önceliğiyle splitler arası ortak inputları çıkar."""
    train_keys = {input_key(row) for row in rows_by_split["train"]}

    dev_before = rows_by_split["dev"]
    clean_dev = [row for row in dev_before if input_key(row) not in train_keys]
    dev_keys = {input_key(row) for row in clean_dev}

    test_before = rows_by_split["test"]
    clean_test = [
        row
        for row in test_before
        if input_key(row) not in train_keys and input_key(row) not in dev_keys
    ]

    cleaned = {
        "train": rows_by_split["train"],
        "dev": clean_dev,
        "test": clean_test,
    }
    removed = {
        "train": 0,
        "dev": len(dev_before) - len(clean_dev),
        "test": len(test_before) - len(clean_test),
    }
    return cleaned, removed


def clean_dataset(rows_by_split):
    """Kuralları sırayla uygula ve temizlik istatistiklerini üret."""
    after_conflicts = {}
    stats = {}

    for split in SPLITS:
        original_rows = rows_by_split[split]
        deduplicated, duplicate_rows = remove_exact_duplicates(original_rows)
        conflict_free, conflict_rows, conflicting_inputs = (
            remove_conflicting_inputs(deduplicated)
        )
        after_conflicts[split] = conflict_free
        stats[split] = {
            "original_rows": len(original_rows),
            "removed_exact_duplicates": duplicate_rows,
            "removed_conflict_rows": conflict_rows,
            "conflicting_inputs": conflicting_inputs,
        }

    cleaned_rows, leakage_rows = remove_split_leakage(after_conflicts)
    for split in SPLITS:
        stats[split]["removed_leakage_rows"] = leakage_rows[split]
        stats[split]["cleaned_rows"] = len(cleaned_rows[split])

    return cleaned_rows, stats


def save_cleaned_files(rows_by_split):
    """Temizlenmiş splitleri yeni data/cleaned klasörüne yaz."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        output_path = OUTPUT_DIR / f"{split}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows_by_split[split])


def count_exact_duplicates(rows):
    """İlk kopya dışındaki exact duplicate satır sayısını hesapla."""
    return len(rows) - len({exact_key(row) for row in rows})


def count_conflicting_inputs(rows):
    """Birden fazla sentiment taşıyan model inputu sayısını hesapla."""
    labels_by_input = defaultdict(set)
    for row in rows:
        labels_by_input[input_key(row)].add(row["sentiment"])
    return sum(len(labels) > 1 for labels in labels_by_input.values())


def print_distribution(title, counts, order):
    """Sabit sırada okunabilir bir dağılım yazdır."""
    print(f"\n{title}:")
    for value in order:
        print(f"  {value}: {counts[value]}")


def source_row_ids_are_valid(cleaned_rows, source_rows):
    """Her source_row_id'nin kaynak processed satırına geri döndüğünü doğrula."""
    for split in SPLITS:
        for row in cleaned_rows[split]:
            id_split, row_number_text = row["source_row_id"].split(":", maxsplit=1)
            if id_split != split:
                return False

            source_index = int(row_number_text) - 1
            if source_index < 0 or source_index >= len(source_rows[split]):
                return False

            original = source_rows[split][source_index]
            if any(row[column] != original[column] for column in SOURCE_COLUMNS):
                return False
    return True


def validate_and_report(cleaned_rows, source_rows, stats):
    """Temizlik özetini, dağılımları ve son kontrolleri göster."""
    print("=" * 72)
    print("CLEANING SUMMARY")
    for split in SPLITS:
        split_stats = stats[split]
        print(f"\n{split.upper()}:")
        print(f"  original rows: {split_stats['original_rows']}")
        print(f"  cleaned rows: {split_stats['cleaned_rows']}")
        print(
            "  removed exact duplicates: "
            f"{split_stats['removed_exact_duplicates']}"
        )
        print(
            "  removed rows from conflicting inputs: "
            f"{split_stats['removed_conflict_rows']}"
        )
        print(
            "  conflicting inputs removed: "
            f"{split_stats['conflicting_inputs']}"
        )
        print(
            "  removed rows from split leakage: "
            f"{split_stats['removed_leakage_rows']}"
        )

    all_rows = [row for split in SPLITS for row in cleaned_rows[split]]
    for split in SPLITS:
        split_rows = cleaned_rows[split]
        print(f"\n--- {split.upper()} CLEANED DISTRIBUTIONS ---")
        print_distribution(
            "Language distribution",
            Counter(row["language"] for row in split_rows),
            LANGUAGES,
        )
        print_distribution(
            "Aspect distribution",
            Counter(row["aspect"] for row in split_rows),
            ASPECTS,
        )
        print_distribution(
            "Sentiment distribution",
            Counter(row["sentiment"] for row in split_rows),
            SENTIMENTS,
        )

    print("\n--- ALL CLEANED DATA ---")
    print_distribution(
        "Language distribution",
        Counter(row["language"] for row in all_rows),
        LANGUAGES,
    )
    print_distribution(
        "Aspect distribution",
        Counter(row["aspect"] for row in all_rows),
        ASPECTS,
    )
    print_distribution(
        "Sentiment distribution",
        Counter(row["sentiment"] for row in all_rows),
        SENTIMENTS,
    )

    atmosphere_rows = [row for row in all_rows if row["aspect"] == "atmosphere"]
    print_distribution(
        "Atmosphere sentiment distribution",
        Counter(row["sentiment"] for row in atmosphere_rows),
        SENTIMENTS,
    )
    neutral_total = sum(row["sentiment"] == "neutral" for row in all_rows)
    print(f"\nNeutral total: {neutral_total}")

    duplicate_counts = {
        split: count_exact_duplicates(cleaned_rows[split]) for split in SPLITS
    }
    conflict_counts = {
        split: count_conflicting_inputs(cleaned_rows[split]) for split in SPLITS
    }
    input_sets = {
        split: {input_key(row) for row in cleaned_rows[split]}
        for split in SPLITS
    }
    overlap_counts = {
        "train/dev": len(input_sets["train"] & input_sets["dev"]),
        "train/test": len(input_sets["train"] & input_sets["test"]),
        "dev/test": len(input_sets["dev"] & input_sets["test"]),
    }

    print("\n" + "=" * 72)
    print("FINAL VALIDATION")
    for split in SPLITS:
        print(f"{split} exact duplicate count: {duplicate_counts[split]}")
        print(f"{split} conflicting input count: {conflict_counts[split]}")
    for pair, count in overlap_counts.items():
        print(f"{pair} overlap: {count}")

    schema_is_valid = all(
        tuple(row) == OUTPUT_COLUMNS
        for split in SPLITS
        for row in cleaned_rows[split]
    )
    source_ids_are_unique = (
        len({row["source_row_id"] for row in all_rows}) == len(all_rows)
    )
    source_ids_are_valid = source_row_ids_are_valid(cleaned_rows, source_rows)
    values_are_valid = (
        {row["language"] for row in all_rows} == set(LANGUAGES)
        and {row["aspect"] for row in all_rows} == set(ASPECTS)
        and {row["sentiment"] for row in all_rows} == set(SENTIMENTS)
        and all(row["sentence"].strip() for row in all_rows)
    )
    all_checks_pass = (
        all(count == 0 for count in duplicate_counts.values())
        and all(count == 0 for count in conflict_counts.values())
        and all(count == 0 for count in overlap_counts.values())
        and schema_is_valid
        and source_ids_are_unique
        and source_ids_are_valid
        and values_are_valid
    )
    print(f"Schema valid: {schema_is_valid}")
    print(f"Values valid: {values_are_valid}")
    print(f"Source row IDs globally unique: {source_ids_are_unique}")
    print(f"Source row IDs resolve to processed rows: {source_ids_are_valid}")
    print(f"All validation checks passed: {all_checks_pass}")

    if not all_checks_pass:
        raise ValueError("Cleaned dataset validation failed.")


def main():
    rows_by_split = {split: load_split(split) for split in SPLITS}
    cleaned_rows, stats = clean_dataset(rows_by_split)
    save_cleaned_files(cleaned_rows)
    validate_and_report(cleaned_rows, rows_by_split, stats)
    print(f"\nCleaned CSV files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
