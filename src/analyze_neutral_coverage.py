"""Cleaned train splitindeki neutral coverage ve çeşitliliği raporla."""

import csv
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "train.csv"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
DISTRIBUTION_PATH = ANALYSIS_DIR / "neutral_distribution.csv"
GAP_REPORT_PATH = ANALYSIS_DIR / "neutral_gap_report.txt"
EXAMPLES_PATH = ANALYSIS_DIR / "neutral_examples.txt"

LANGUAGES = ("tr", "en", "es")
LANGUAGE_NAMES = {"tr": "TR", "en": "EN", "es": "ES"}
ASPECTS = ("taste", "atmosphere", "service", "value")
TARGET_PER_CELL = 25
MAX_EXAMPLES_PER_CELL = 5

SOURCE_CATEGORY_TO_ASPECT = {
    "food quality": "taste",
    "drinks quality": "taste",
    "ambience general": "atmosphere",
    "service general": "service",
    "food prices": "value",
    "restaurant prices": "value",
    "drinks prices": "value",
}

REQUIRED_COLUMNS = {
    "sentence",
    "language",
    "split",
    "aspect",
    "sentiment",
    "source_category",
    "aspect_term",
    "source_row_id",
}


def load_neutral_rows():
    """Cleaned train.csv içinden yalnız neutral satırları yükle."""
    with INPUT_PATH.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Eksik kolonlar: {sorted(missing)}")
        rows = list(reader)

    if any(row["split"] != "train" for row in rows):
        raise ValueError("train.csv içinde train dışı split bulundu.")
    neutral_rows = [row for row in rows if row["sentiment"] == "neutral"]
    if any(row["language"] not in LANGUAGES for row in neutral_rows):
        raise ValueError("Neutral subset içinde beklenmeyen dil bulundu.")
    if any(row["aspect"] not in ASPECTS for row in neutral_rows):
        raise ValueError("Neutral subset içinde beklenmeyen aspect bulundu.")
    return neutral_rows


def source_row_number(row):
    """train:000123 biçimindeki processed row numarasını döndür."""
    split, separator, number = row["source_row_id"].partition(":")
    if split != "train" or not separator or not number.isdigit():
        raise ValueError(f"Geçersiz source_row_id: {row['source_row_id']}")
    return int(number)


def gap_level(current_count):
    """25 örneklik hedefe göre açık seviyesini sınıflandır."""
    if current_count >= TARGET_PER_CELL:
        return "TARGET MET"
    if current_count <= 5:
        return "CRITICAL GAP"
    if current_count <= 12:
        return "HIGH GAP"
    if current_count <= 19:
        return "MODERATE GAP"
    return "LOW GAP"


def analyze_parallel_structure(neutral_rows):
    """Semantik tahmin yapmadan sıra ve source-row ofsetlerini denetle."""
    ordered = {
        language: sorted(
            (row for row in neutral_rows if row["language"] == language),
            key=source_row_number,
        )
        for language in LANGUAGES
    }
    counts_equal = len({len(rows) for rows in ordered.values()}) == 1
    structural_signatures = {
        language: [
            (row["aspect"], row["source_category"], row["split"])
            for row in rows
        ]
        for language, rows in ordered.items()
    }
    signatures_equal = (
        structural_signatures["tr"]
        == structural_signatures["en"]
        == structural_signatures["es"]
    )

    offset_summary = {}
    offsets_constant = False
    if counts_equal and ordered["tr"]:
        tr_ids = [source_row_number(row) for row in ordered["tr"]]
        for language in ("en", "es"):
            language_ids = [source_row_number(row) for row in ordered[language]]
            offsets = [
                language_id - tr_id
                for tr_id, language_id in zip(tr_ids, language_ids)
            ]
            offset_summary[language] = sorted(set(offsets))
        offsets_constant = all(
            len(offset_summary[language]) == 1 for language in ("en", "es")
        )

    candidate_group_count = (
        len(ordered["tr"])
        if counts_equal and signatures_equal and offsets_constant
        else None
    )
    return {
        "ordered": ordered,
        "counts_equal": counts_equal,
        "signatures_equal": signatures_equal,
        "offsets_constant": offsets_constant,
        "offset_summary": offset_summary,
        "candidate_group_count": candidate_group_count,
    }


def write_distribution_csv(matrix_counts):
    """Her language × aspect hücresinin mevcut ve hedef sayısını yaz."""
    with DISTRIBUTION_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "language",
                "aspect",
                "current_count",
                "target_count",
                "needed_count",
            ),
        )
        writer.writeheader()
        for language in LANGUAGES:
            for aspect in ASPECTS:
                current = matrix_counts[aspect, language]
                writer.writerow(
                    {
                        "language": language,
                        "aspect": aspect,
                        "current_count": current,
                        "target_count": TARGET_PER_CELL,
                        "needed_count": max(0, TARGET_PER_CELL - current),
                    }
                )


def write_examples(neutral_rows):
    """Her aspect × language hücresinden en fazla beş örneği yaz."""
    lines = [
        "NEUTRAL EXAMPLES",
        f"Source: {INPUT_PATH}",
        f"Maximum examples per aspect × language cell: {MAX_EXAMPLES_PER_CELL}",
        "",
    ]
    for aspect in ASPECTS:
        for language in LANGUAGES:
            examples = [
                row
                for row in neutral_rows
                if row["aspect"] == aspect and row["language"] == language
            ][:MAX_EXAMPLES_PER_CELL]
            lines.extend(
                [
                    "=" * 72,
                    f"{aspect} × {LANGUAGE_NAMES[language]} — {len(examples)} shown",
                    "=" * 72,
                ]
            )
            if not examples:
                lines.extend(["No neutral example available.", ""])
                continue
            for index, row in enumerate(examples, start=1):
                lines.extend(
                    [
                        f"Example {index}",
                        f"Sentence: {row['sentence']}",
                        f"Language: {row['language']}",
                        f"Aspect: {row['aspect']}",
                        f"Source category: {row['source_category']}",
                        f"Aspect term: {row['aspect_term']}",
                        f"Sentiment: {row['sentiment']}",
                        "",
                    ]
                )
    EXAMPLES_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def table_line(label, values, widths):
    """Sabit genişlikli metin tablosu satırı oluştur."""
    cells = [f"{label:<{widths[0]}}"]
    cells.extend(
        f"{value:>{width}}" for value, width in zip(values, widths[1:])
    )
    return "  ".join(cells)


def write_gap_report(neutral_rows, matrix_counts, parallel):
    """Neutral dağılımı, yapısal çeşitlilik ve açık analizini yaz."""
    total_neutral = len(neutral_rows)
    aspect_totals = Counter(row["aspect"] for row in neutral_rows)
    language_totals = Counter(row["language"] for row in neutral_rows)
    source_totals = Counter(row["source_category"] for row in neutral_rows)
    duplicate_keys = Counter(
        (row["sentence"], row["aspect"]) for row in neutral_rows
    )
    repeated_keys = {key: count for key, count in duplicate_keys.items() if count > 1}
    duplicate_rows = sum(count - 1 for count in repeated_keys.values())
    source_id_count = len({row["source_row_id"] for row in neutral_rows})
    total_needed = sum(
        max(0, TARGET_PER_CELL - matrix_counts[aspect, language])
        for aspect in ASPECTS
        for language in LANGUAGES
    )

    widths = (14, 7, 7, 7, 7)
    lines = [
        "NEUTRAL DATA EXPANSION v1 — COVERAGE AUDIT",
        "",
        "SOURCE",
        f"Input: {INPUT_PATH}",
        "Split: train",
        "Sentiment filter: neutral",
        "",
        "TOTAL NEUTRAL",
        f"Total neutral rows: {total_neutral}",
        "",
        "ASPECT TOTALS",
    ]
    lines.extend(f"{aspect}: {aspect_totals[aspect]}" for aspect in ASPECTS)
    lines.extend(["", "LANGUAGE TOTALS"])
    lines.extend(
        f"{LANGUAGE_NAMES[language]}: {language_totals[language]}"
        for language in LANGUAGES
    )

    lines.extend(
        [
            "",
            "ASPECT × LANGUAGE MATRIX — COUNTS",
            table_line("aspect", ["TR", "EN", "ES", "TOTAL"], widths),
        ]
    )
    for aspect in ASPECTS:
        values = [matrix_counts[aspect, language] for language in LANGUAGES]
        lines.append(table_line(aspect, values + [sum(values)], widths))
    lines.append(
        table_line(
            "TOTAL",
            [language_totals[language] for language in LANGUAGES]
            + [total_neutral],
            widths,
        )
    )

    lines.extend(
        [
            "",
            "ASPECT × LANGUAGE MATRIX — % OF ALL NEUTRAL ROWS",
            table_line("aspect", ["TR", "EN", "ES", "TOTAL"], widths),
        ]
    )
    for aspect in ASPECTS:
        values = [matrix_counts[aspect, language] for language in LANGUAGES]
        percentages = [f"{100 * value / total_neutral:.1f}%" for value in values]
        percentages.append(f"{100 * sum(values) / total_neutral:.1f}%")
        lines.append(table_line(aspect, percentages, widths))

    lines.extend(["", "SOURCE CATEGORY DISTRIBUTION"])
    for source_category in SOURCE_CATEGORY_TO_ASPECT:
        lines.append(
            f"{source_category}: count={source_totals[source_category]} | "
            f"mapped aspect={SOURCE_CATEGORY_TO_ASPECT[source_category]}"
        )

    candidate_count = parallel["candidate_group_count"]
    candidate_text = "not available" if candidate_count is None else str(candidate_count)
    lines.extend(
        [
            "",
            "UNIQUE / PARALLEL STRUCTURE",
            f"Unique source_row_id values: {source_id_count}/{total_neutral}",
            (
                "Neutral rows per language equal: "
                f"{parallel['counts_equal']} "
                f"({', '.join(f'{language}={language_totals[language]}' for language in LANGUAGES)})"
            ),
            (
                "Ordered (aspect, source_category, split) signatures equal: "
                f"{parallel['signatures_equal']}"
            ),
            f"Constant source-row offsets across all ordinal pairs: {parallel['offsets_constant']}",
            f"EN − TR offsets observed: {parallel['offset_summary'].get('en', [])}",
            f"ES − TR offsets observed: {parallel['offset_summary'].get('es', [])}",
            f"Structurally aligned candidate annotation groups: {candidate_text}",
            "exact unique semantic count cannot be verified",
            (
                "Reason: source_row_id is a processed long-row position, not an "
                "explicit language-independent source annotation ID. The 31 candidate "
                "groups are structural evidence only, not verified semantic labels."
            ),
            "",
            "DUPLICATE CHECK WITHIN NEUTRAL SUBSET",
            "Key: exact (sentence, aspect)",
            f"Repeated keys: {len(repeated_keys)}",
            f"Duplicate rows beyond first: {duplicate_rows}",
            (
                "Near-duplicate or small-wording variants were not classified as "
                "semantic duplicates."
            ),
            "",
            "GAP CLASSIFICATION THRESHOLDS",
            f"Target per aspect × language cell: {TARGET_PER_CELL}",
            "CRITICAL GAP: current 0–5",
            "HIGH GAP: current 6–12",
            "MODERATE GAP: current 13–19",
            "LOW GAP: current 20–24",
            "TARGET MET: current >=25",
            "",
            "GAP ANALYSIS",
        ]
    )
    for aspect in ASPECTS:
        for language in LANGUAGES:
            current = matrix_counts[aspect, language]
            needed = max(0, TARGET_PER_CELL - current)
            lines.append(
                f"{gap_level(current)} — {aspect} / {language}: "
                f"current={current}, target={TARGET_PER_CELL}, needed={needed}"
            )

    lines.extend(
        [
            "",
            "TOTAL GAP",
            f"Minimum balanced target: {len(ASPECTS) * len(LANGUAGES) * TARGET_PER_CELL}",
            f"Current neutral rows: {total_neutral}",
            f"Total additional human-verified neutral examples needed: {total_needed}",
            "",
            "VALIDATION",
            f"Matrix sum equals total neutral: {sum(matrix_counts.values()) == total_neutral}",
            f"Distribution target total: {len(ASPECTS) * len(LANGUAGES) * TARGET_PER_CELL}",
            f"Distribution needed total: {total_needed}",
        ]
    )
    GAP_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    neutral_rows = load_neutral_rows()
    matrix_counts = Counter(
        (row["aspect"], row["language"]) for row in neutral_rows
    )
    parallel = analyze_parallel_structure(neutral_rows)

    write_distribution_csv(matrix_counts)
    write_examples(neutral_rows)
    write_gap_report(neutral_rows, matrix_counts, parallel)

    total_needed = sum(
        max(0, TARGET_PER_CELL - matrix_counts[aspect, language])
        for aspect in ASPECTS
        for language in LANGUAGES
    )
    print(f"Total neutral: {len(neutral_rows)}")
    print(f"Structural candidate groups: {parallel['candidate_group_count']}")
    print("Exact unique semantic count cannot be verified.")
    print(f"Total additional neutral examples needed: {total_needed}")
    print(f"Saved: {DISTRIBUTION_PATH}")
    print(f"Saved: {GAP_REPORT_PATH}")
    print(f"Saved: {EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
