"""Create a deterministic, balanced MAMS human-review batch."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "curation" / "mams_neutral_candidates_en.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "curation" / "review_batches"
CSV_PATH = OUTPUT_DIR / "mams_neutral_batch_01.csv"
MARKDOWN_PATH = OUTPUT_DIR / "mams_neutral_batch_01.md"

RANDOM_SEED = 42
TARGET_PER_ASPECT = {
    "taste": 6,
    "atmosphere": 6,
    "service": 6,
    "value": 6,
}
OVERLAP_MARKER = "EXACT_OVERLAP_EXISTING_DATA"

OUTPUT_FIELDS = [
    "candidate_id",
    "sentence",
    "source_category",
    "mapped_aspect",
    "source_polarity",
    "source_group_id",
    "human_decision",
    "human_final_aspect",
    "human_final_sentiment",
    "review_notes",
]


def load_eligible_candidates() -> list[dict[str, str]]:
    with SOURCE_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    return [
        row
        for row in rows
        if not row["human_decision"].strip()
        and OVERLAP_MARKER not in row["review_notes"]
        and row["mapped_aspect"] in TARGET_PER_ASPECT
    ]


def select_batch(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    shuffled = candidates.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled)

    selected: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    used_source_groups: set[str] = set()

    # First pass enforces one annotation per source sentence.
    for row in shuffled:
        aspect = row["mapped_aspect"]
        group_id = row["source_group_id"]
        if counts[aspect] >= TARGET_PER_ASPECT[aspect]:
            continue
        if group_id in used_source_groups:
            continue
        selected.append(row)
        counts[aspect] += 1
        used_source_groups.add(group_id)
        if counts == Counter(TARGET_PER_ASPECT):
            break

    if counts != Counter(TARGET_PER_ASPECT):
        raise ValueError(
            "Could not satisfy all aspect quotas with unique source_group_id values. "
            f"Selected counts: {dict(counts)}"
        )
    return selected


def validate(batch: list[dict[str, str]]) -> None:
    if len(batch) != 24:
        raise ValueError(f"Expected 24 rows, found {len(batch)}")
    if Counter(row["mapped_aspect"] for row in batch) != Counter(TARGET_PER_ASPECT):
        raise ValueError("Aspect quotas are not balanced")
    if len({row["candidate_id"] for row in batch}) != len(batch):
        raise ValueError("Duplicate candidate_id in batch")
    if len({row["source_group_id"] for row in batch}) != len(batch):
        raise ValueError("Duplicate source_group_id in batch")
    if any(row["human_decision"].strip() for row in batch):
        raise ValueError("A selected row already has a human decision")
    if any(OVERLAP_MARKER in row["review_notes"] for row in batch):
        raise ValueError("An existing-data overlap was included")


def write_csv(batch: list[dict[str, str]]) -> None:
    if CSV_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite review batch: {CSV_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in OUTPUT_FIELDS}
            for row in batch
        )


def write_markdown(batch: list[dict[str, str]]) -> None:
    if MARKDOWN_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite review display: {MARKDOWN_PATH}")

    lines = [
        "# MAMS Neutral Review Batch 01",
        "",
        "Human decisions are intentionally blank. The order is deterministic and aspect-mixed (seed 42).",
        "",
    ]
    for index, row in enumerate(batch, start=1):
        lines.extend(
            [
                f"## {index}",
                "",
                f"Candidate ID: {row['candidate_id']}",
                "",
                f"Mapped aspect: {row['mapped_aspect']}",
                "",
                f"Source category: {row['source_category']}",
                "",
                f"Sentence: {row['sentence']}",
                "",
            ]
        )
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    batch = select_batch(load_eligible_candidates())
    validate(batch)
    write_csv(batch)
    write_markdown(batch)

    print(f"Review batch rows: {len(batch)}")
    for aspect in TARGET_PER_ASPECT:
        count = sum(row["mapped_aspect"] == aspect for row in batch)
        print(f"{aspect}: {count}")
    print(f"Unique source_group_id: {len({row['source_group_id'] for row in batch})}")
    print(f"CSV: {CSV_PATH}")
    print(f"Readable review: {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
