"""Processed sentiment datasetinde duplicate, conflict ve leakage denetimi yap."""

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "dataset_audit.txt"
SPLITS = ("train", "dev", "test")
LANGUAGES = ("tr", "en", "es")
ASPECTS = ("taste", "atmosphere", "service", "value")
SENTIMENTS = ("positive", "neutral", "negative")
TOP_EXAMPLE_COUNT = 20

REQUIRED_COLUMNS = {
    "sentence",
    "language",
    "split",
    "aspect",
    "sentiment",
    "source_category",
    "aspect_term",
}


def load_split(split: str):
    """Bir processed CSV splitini oku ve şemasını doğrula."""
    csv_path = DATA_DIR / f"{split}.csv"
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            raise ValueError(
                f"{csv_path.name} eksik kolonlar içeriyor: {sorted(missing_columns)}"
            )
        rows = list(reader)

    wrong_split_rows = [row for row in rows if row["split"] != split]
    if wrong_split_rows:
        raise ValueError(f"{csv_path.name} beklenmeyen split değeri içeriyor.")

    return rows


def file_sha256(path: Path) -> str:
    """Raporlanan kaynak dosyanın SHA-256 özetini hesapla."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_key(row):
    """Kullanıcının tanımladığı exact duplicate anahtarını döndür."""
    return (
        row["sentence"],
        row["language"],
        row["aspect"],
        row["sentiment"],
    )


def input_key(row):
    """Model girdisini temsil eden sentence + language + aspect anahtarı."""
    return row["sentence"], row["language"], row["aspect"]


def group_rows(rows, key_function):
    """Satırları verilen anahtara göre grupla."""
    groups = defaultdict(list)
    for row in rows:
        groups[key_function(row)].append(row)
    return groups


def percentage(part: int, total: int) -> str:
    """Güvenli yüzde metni üret."""
    return f"{(100 * part / total):.2f}%" if total else "0.00%"


def sorted_sentiments(values):
    """Sentiment değerlerini sabit ve okunabilir sırada döndür."""
    order = {sentiment: index for index, sentiment in enumerate(SENTIMENTS)}
    return tuple(sorted(values, key=lambda value: order.get(value, 999)))


def annotation_details(rows) -> list[str]:
    """Kaynak category ve aspect term ayrıntılarını sayılarıyla özetle."""
    counts = Counter(
        (row["sentiment"], row["source_category"], row["aspect_term"])
        for row in rows
    )
    return [
        (
            f"sentiment={sentiment}, source_category={source_category}, "
            f"aspect_term={aspect_term}, rows={count}"
        )
        for (sentiment, source_category, aspect_term), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def add_title(report, title: str, marker: str = "=") -> None:
    """Metin raporuna bölüm başlığı ekle."""
    report.extend(["", marker * 88, title, marker * 88])


def exact_duplicate_stats(rows):
    """Exact duplicate istatistiklerini ve tekrar gruplarını hesapla."""
    counts = Counter(exact_key(row) for row in rows)
    duplicate_groups = [
        (key, count) for key, count in counts.items() if count > 1
    ]
    duplicate_groups.sort(key=lambda item: (-item[1], item[0]))
    duplicate_rows = len(rows) - len(counts)
    return {
        "total": len(rows),
        "unique": len(counts),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": percentage(duplicate_rows, len(rows)),
        "groups": duplicate_groups,
    }


def find_conflicts(rows):
    """Aynı model girdisinin birden fazla sentiment taşıdığı grupları bul."""
    groups = group_rows(rows, input_key)
    conflicts = []
    for key, grouped_rows in groups.items():
        labels = sorted_sentiments({row["sentiment"] for row in grouped_rows})
        if len(labels) > 1:
            conflicts.append((key, labels, grouped_rows))
    conflicts.sort(key=lambda item: (item[1], item[0]))
    return conflicts


def report_exact_duplicates(report, split: str, rows) -> None:
    """Bir splitin exact duplicate sonuçlarını raporla."""
    stats = exact_duplicate_stats(rows)
    groups = group_rows(rows, exact_key)

    add_title(report, f"A) EXACT DUPLICATES — {split.upper()}", "-")
    report.append(f"Total rows: {stats['total']}")
    report.append(f"Unique rows: {stats['unique']}")
    report.append(f"Duplicate rows (total - unique): {stats['duplicate_rows']}")
    report.append(f"Duplicate rate: {stats['duplicate_rate']}")
    report.append(f"Duplicate groups: {len(stats['groups'])}")
    report.append(f"Top {TOP_EXAMPLE_COUNT} repeated examples:")

    if not stats["groups"]:
        report.append("None")
        return

    for index, (key, count) in enumerate(
        stats["groups"][:TOP_EXAMPLE_COUNT], start=1
    ):
        sentence, language, aspect, sentiment = key
        report.append(
            f"[{index}] rows={count} | language={language} | "
            f"aspect={aspect} | sentiment={sentiment}"
        )
        report.append(f"    sentence: {sentence}")
        for detail in annotation_details(groups[key]):
            report.append(f"    annotation: {detail}")


def report_conflicts(report, split: str, rows) -> None:
    """Bir splitteki same-input/different-label sonuçlarını raporla."""
    conflicts = find_conflicts(rows)
    combinations = Counter(labels for _, labels, _ in conflicts)

    add_title(report, f"B) SAME INPUT, DIFFERENT LABEL — {split.upper()}", "-")
    report.append(f"Conflicting inputs: {len(conflicts)}")
    report.append("Sentiment combinations:")
    if combinations:
        for labels, count in sorted(
            combinations.items(), key=lambda item: (-item[1], item[0])
        ):
            report.append(f"  {' + '.join(labels)}: {count}")
    else:
        report.append("  None")

    report.append(f"First {TOP_EXAMPLE_COUNT} conflict examples:")
    if not conflicts:
        report.append("None")
        return

    for index, (key, labels, grouped_rows) in enumerate(
        conflicts[:TOP_EXAMPLE_COUNT], start=1
    ):
        sentence, language, aspect = key
        report.append(
            f"[{index}] language={language} | aspect={aspect} | "
            f"labels={' + '.join(labels)}"
        )
        report.append(f"    sentence: {sentence}")
        for detail in annotation_details(grouped_rows):
            report.append(f"    annotation: {detail}")


def report_split_leakage(report, rows_by_split) -> None:
    """Model inputlarının splitler arasındaki kesişimini raporla."""
    add_title(report, "C) SPLIT LEAKAGE")
    input_groups = {
        split: group_rows(rows, input_key)
        for split, rows in rows_by_split.items()
    }
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))

    for left_split, right_split in pairs:
        shared_keys = sorted(
            set(input_groups[left_split]) & set(input_groups[right_split])
        )
        report.append("")
        report.append(
            f"{left_split} vs {right_split} shared inputs: {len(shared_keys)}"
        )

        if not shared_keys:
            report.append("Examples: None")
            continue

        report.append(f"First {TOP_EXAMPLE_COUNT} examples:")
        for index, key in enumerate(shared_keys[:TOP_EXAMPLE_COUNT], start=1):
            sentence, language, aspect = key
            left_labels = sorted_sentiments(
                {row["sentiment"] for row in input_groups[left_split][key]}
            )
            right_labels = sorted_sentiments(
                {row["sentiment"] for row in input_groups[right_split][key]}
            )
            report.append(
                f"[{index}] language={language} | aspect={aspect} | "
                f"{left_split} labels={'+'.join(left_labels)} | "
                f"{right_split} labels={'+'.join(right_labels)}"
            )
            report.append(f"    sentence: {sentence}")

    shared_all = (
        set(input_groups["train"])
        & set(input_groups["dev"])
        & set(input_groups["test"])
    )
    report.append("")
    report.append(f"Inputs shared by train + dev + test: {len(shared_all)}")


def parallel_signature(row):
    """Çeviriden bağımsız olması beklenen annotation imzasını döndür."""
    return row["aspect"], row["sentiment"], row["source_category"]


def report_parallel_structure(report, rows_by_split) -> None:
    """TR/EN/ES satır sırası ve annotation imzalarının hizasını denetle."""
    add_title(report, "D) LANGUAGE-PARALLEL STRUCTURE")
    report.append(
        "Method: within each split, rows are grouped by language in file order. "
        "The same position is compared using aspect + sentiment + source_category."
    )
    report.append(
        "Limitation: processed CSV files have no shared semantic example ID. "
        "Therefore row-order alignment is strong evidence, not proof of translation identity."
    )

    all_splits_aligned = True
    for split, rows in rows_by_split.items():
        by_language = {
            language: [row for row in rows if row["language"] == language]
            for language in LANGUAGES
        }
        lengths = {language: len(items) for language, items in by_language.items()}
        compared_positions = min(lengths.values())
        mismatches = []

        for position in range(compared_positions):
            signatures = {
                language: parallel_signature(by_language[language][position])
                for language in LANGUAGES
            }
            if len(set(signatures.values())) != 1:
                mismatches.append((position, signatures))

        same_lengths = len(set(lengths.values())) == 1
        aligned = same_lengths and not mismatches
        all_splits_aligned = all_splits_aligned and aligned

        report.append("")
        report.append(f"{split.upper()} language rows: {lengths}")
        report.append(f"Compared positions: {compared_positions}")
        report.append(f"Signature mismatches: {len(mismatches)}")
        report.append(f"Inferred within-split parallel alignment: {aligned}")

        if mismatches:
            report.append(f"First {TOP_EXAMPLE_COUNT} mismatches:")
            for position, signatures in mismatches[:TOP_EXAMPLE_COUNT]:
                report.append(f"  position={position}: {signatures}")
        else:
            report.append("Sample inferred parallel rows:")
            for position in range(min(5, compared_positions)):
                signature = parallel_signature(by_language["tr"][position])
                report.append(f"  position={position} | signature={signature}")
                for language in LANGUAGES:
                    sentence = by_language[language][position]["sentence"]
                    report.append(f"    {language}: {sentence}")

    report.append("")
    report.append(f"All splits pass inferred parallel alignment: {all_splits_aligned}")
    if all_splits_aligned:
        report.append(
            "Cross-language split warning: no within-split row-order evidence of "
            "TR/EN/ES counterparts being separated. Definitive cross-split semantic "
            "matching requires a shared source/example ID that is not present."
        )
    else:
        report.append(
            "WARNING: at least one split has unequal language counts or annotation "
            "signature mismatches. Review the listed positions."
        )


def scoped_exact_stats(rows):
    """Split sınırını koruyarak birleşik bir alt kümede duplicate hesapla."""
    keys = [(row["split"],) + exact_key(row) for row in rows]
    unique_count = len(set(keys))
    duplicate_rows = len(rows) - unique_count
    return len(rows), unique_count, duplicate_rows, percentage(duplicate_rows, len(rows))


def scoped_conflicts(rows):
    """Split sınırını koruyarak birleşik bir alt kümede conflict hesapla."""
    groups = group_rows(
        rows,
        lambda row: (row["split"], row["sentence"], row["language"], row["aspect"]),
    )
    return [
        grouped_rows
        for grouped_rows in groups.values()
        if len({row["sentiment"] for row in grouped_rows}) > 1
    ]


def report_focus_audits(report, rows_by_split) -> None:
    """Atmosphere ve neutral için istenen özel denetimleri raporla."""
    all_rows = [row for rows in rows_by_split.values() for row in rows]

    atmosphere_rows = [row for row in all_rows if row["aspect"] == "atmosphere"]
    atmosphere_stats = scoped_exact_stats(atmosphere_rows)
    atmosphere_conflicts = scoped_conflicts(atmosphere_rows)
    atmosphere_sentiments = Counter(row["sentiment"] for row in atmosphere_rows)

    add_title(report, "E1) ATMOSPHERE AUDIT")
    report.append(f"Rows: {atmosphere_stats[0]}")
    report.append(f"Unique exact rows within splits: {atmosphere_stats[1]}")
    report.append(f"Duplicate rows within splits: {atmosphere_stats[2]}")
    report.append(f"Exact duplicate rate: {atmosphere_stats[3]}")
    report.append(f"Conflicting inputs within splits: {len(atmosphere_conflicts)}")
    report.append("Sentiment distribution:")
    for sentiment in SENTIMENTS:
        report.append(f"  {sentiment}: {atmosphere_sentiments[sentiment]}")

    neutral_rows = [row for row in all_rows if row["sentiment"] == "neutral"]
    neutral_stats = scoped_exact_stats(neutral_rows)
    neutral_aspects = Counter(row["aspect"] for row in neutral_rows)

    add_title(report, "E2) NEUTRAL AUDIT")
    report.append(f"Rows: {neutral_stats[0]}")
    report.append(f"Unique exact rows within splits: {neutral_stats[1]}")
    report.append(f"Duplicate rows within splits: {neutral_stats[2]}")
    report.append(f"Exact duplicate rate: {neutral_stats[3]}")
    report.append("Aspect distribution:")
    for aspect in ASPECTS:
        report.append(f"  {aspect}: {neutral_aspects[aspect]}")


def build_report(rows_by_split) -> str:
    """Bütün denetim bölümlerini tek metin raporunda birleştir."""
    report = [
        "M-ABSA PROCESSED SENTIMENT DATASET AUDIT",
        "Scope: train.csv, dev.csv, test.csv",
        "No rows were deleted or changed. No model was trained.",
        "",
        "Definitions:",
        "- Exact duplicate key: sentence + language + aspect + sentiment.",
        "- Duplicate rows: total rows minus unique rows (extra repeated copies).",
        "- Model input key: sentence + language + aspect.",
        "- Conflict: one model input key has more than one sentiment within a split.",
        "- Split leakage: one model input key appears in more than one split.",
        "- Interpretation note: aspect_term and source_category are intentionally not ",
        "  part of the requested duplicate/conflict keys. Different fine-grained ",
        "  annotations can therefore collapse to the same baseline model input; a ",
        "  reported duplicate or conflict is not automatically a source-label error.",
        "",
        "Source files:",
    ]

    for split in SPLITS:
        path = DATA_DIR / f"{split}.csv"
        report.append(
            f"- {path.name}: rows={len(rows_by_split[split])}, "
            f"sha256={file_sha256(path)}"
        )

    for split in SPLITS:
        report_exact_duplicates(report, split, rows_by_split[split])
    for split in SPLITS:
        report_conflicts(report, split, rows_by_split[split])

    report_split_leakage(report, rows_by_split)
    report_parallel_structure(report, rows_by_split)
    report_focus_audits(report, rows_by_split)

    add_title(report, "SUMMARY")
    for split in SPLITS:
        duplicate_stats = exact_duplicate_stats(rows_by_split[split])
        conflicts = find_conflicts(rows_by_split[split])
        report.append(
            f"{split}: rows={duplicate_stats['total']}, "
            f"duplicate_rows={duplicate_stats['duplicate_rows']} "
            f"({duplicate_stats['duplicate_rate']}), conflicts={len(conflicts)}"
        )

    report.append("")
    report.append("Audit only: no automatic remediation was applied.")
    return "\n".join(report) + "\n"


def main() -> None:
    rows_by_split = {split: load_split(split) for split in SPLITS}
    report_text = build_report(rows_by_split)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Dataset audit saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
