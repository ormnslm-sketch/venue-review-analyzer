"""Prepare an English MAMS neutral candidate pool for human review.

This script only creates curation artifacts. It never changes the processed or
cleaned datasets and never makes ACCEPT/REJECT decisions automatically.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_PATH = PROJECT_ROOT / "data" / "curation" / "mams_neutral_candidates_en.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "analysis" / "mams_neutral_curation_summary.txt"

DATASET_NAME = "justinsiow/mams-acsa"
SOURCE_DATASET = "mams-acsa"
SOURCE_LICENSE = "apache-2.0"
LANGUAGE = "en"
RANDOM_SEED = 42

CATEGORY_TO_ASPECT = {
    "food": "taste",
    "ambience": "atmosphere",
    "service": "service",
    "price": "value",
}

TARGET_COUNTS = {
    "taste": 30,
    "atmosphere": 35,
    "service": 40,
    "value": 35,
}

FIELDNAMES = [
    "candidate_id",
    "sentence",
    "source_dataset",
    "source_license",
    "source_category",
    "mapped_aspect",
    "source_polarity",
    "language",
    "human_decision",
    "human_final_aspect",
    "human_final_sentiment",
    "review_notes",
    "source_group_id",
]


def split_aligned_values(value: str) -> list[str]:
    """Split the comma-separated ACSA annotation fields."""
    return [item.strip() for item in value.split(",")]


def load_existing_sentences() -> dict[str, set[str]]:
    sentences: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        path = PROJECT_ROOT / "data" / "cleaned" / f"{split}.csv"
        with path.open(encoding="utf-8", newline="") as file:
            sentences[split] = {
                row["sentence"]
                for row in csv.DictReader(file)
                if row["sentence"]
            }
    return sentences


def collect_candidates() -> list[dict[str, str]]:
    dataset = load_dataset(DATASET_NAME, split="train")
    by_aspect: dict[str, list[dict[str, str]]] = {
        aspect: [] for aspect in TARGET_COUNTS
    }
    seen_sentence_aspect: set[tuple[str, str]] = set()

    for row_index, row in enumerate(dataset):
        categories = split_aligned_values(row["category"])
        polarities = split_aligned_values(row["polarity"])
        if len(categories) != len(polarities):
            raise ValueError(
                f"Unaligned annotations at MAMS train row {row_index}: "
                f"{len(categories)} categories vs {len(polarities)} polarities"
            )

        sentence = row["text"]
        for category, polarity in zip(categories, polarities):
            if polarity != "neutral" or category not in CATEGORY_TO_ASPECT:
                continue

            aspect = CATEGORY_TO_ASPECT[category]
            duplicate_key = (sentence, aspect)
            if duplicate_key in seen_sentence_aspect:
                continue
            seen_sentence_aspect.add(duplicate_key)

            by_aspect[aspect].append(
                {
                    "candidate_id": f"MAMS-EN-{row_index:06d}-{category.upper()}",
                    "sentence": sentence,
                    "source_dataset": SOURCE_DATASET,
                    "source_license": SOURCE_LICENSE,
                    "source_category": category,
                    "mapped_aspect": aspect,
                    "source_polarity": "neutral",
                    "language": LANGUAGE,
                    "human_decision": "",
                    "human_final_aspect": "",
                    "human_final_sentiment": "",
                    "review_notes": "",
                    "source_group_id": f"MAMS-TRAIN-{row_index:06d}",
                }
            )

    rng = random.Random(RANDOM_SEED)
    selected: list[dict[str, str]] = []
    for aspect, target_count in TARGET_COUNTS.items():
        candidates = by_aspect[aspect]
        rng.shuffle(candidates)
        if len(candidates) < target_count:
            raise ValueError(
                f"Not enough {aspect} candidates: {len(candidates)} < {target_count}"
            )
        selected.extend(candidates[:target_count])

    # The review file must not be grouped by aspect. IDs remain source-based and
    # stable, while this deterministic shuffle only controls display order.
    rng.shuffle(selected)
    return selected


def mark_overlaps(
    candidates: list[dict[str, str]], existing: dict[str, set[str]]
) -> dict[str, int]:
    overlap_counts = {split: 0 for split in ("train", "dev", "test")}
    for candidate in candidates:
        overlaps = [
            split
            for split in ("train", "dev", "test")
            if candidate["sentence"] in existing[split]
        ]
        for split in overlaps:
            overlap_counts[split] += 1
        if overlaps:
            candidate["review_notes"] = "EXACT_OVERLAP_EXISTING_DATA"
    return overlap_counts


def write_candidates(candidates: list[dict[str, str]]) -> None:
    if CURATION_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing human-curation file: {CURATION_PATH}"
        )
    CURATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CURATION_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(candidates)


def write_summary(
    candidates: list[dict[str, str]], overlap_counts: dict[str, int]
) -> None:
    counts = Counter(row["mapped_aspect"] for row in candidates)
    lines = [
        "MAMS NEUTRAL CURATION SUMMARY",
        "================================",
        "",
        "Candidates:",
        f"taste: {counts['taste']}",
        f"atmosphere: {counts['atmosphere']}",
        f"service: {counts['service']}",
        f"value: {counts['value']}",
        f"total: {len(candidates)}",
        "",
        f"Exact overlaps with existing train: {overlap_counts['train']}",
        f"Exact overlaps with existing dev: {overlap_counts['dev']}",
        f"Exact overlaps with existing test: {overlap_counts['test']}",
        "",
        f"License: {SOURCE_LICENSE}",
        f"Source: {DATASET_NAME}",
        "Human review required: YES",
        "",
        "Notes:",
        "- No candidate has been accepted or added to training data.",
        "- Any exact overlap is marked EXACT_OVERLAP_EXISTING_DATA in review_notes.",
        "- Candidate display order uses deterministic random seed 42.",
        "- source_group_id is the stable MAMS train-row identifier.",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(candidates: list[dict[str, str]]) -> None:
    counts = Counter(row["mapped_aspect"] for row in candidates)
    if counts != Counter(TARGET_COUNTS):
        raise ValueError(f"Unexpected aspect counts: {dict(counts)}")
    if len(candidates) != len({row["candidate_id"] for row in candidates}):
        raise ValueError("candidate_id values are not unique")
    if len(candidates) != len(
        {(row["sentence"], row["mapped_aspect"]) for row in candidates}
    ):
        raise ValueError("Duplicate sentence + mapped_aspect pair found")
    for row in candidates:
        if row["human_decision"] or row["human_final_aspect"] or row["human_final_sentiment"]:
            raise ValueError("A human-review field was filled automatically")


def main() -> None:
    candidates = collect_candidates()
    validate(candidates)
    overlap_counts = mark_overlaps(candidates, load_existing_sentences())
    write_candidates(candidates)
    write_summary(candidates, overlap_counts)

    counts = Counter(row["mapped_aspect"] for row in candidates)
    print(f"taste candidates: {counts['taste']}")
    print(f"atmosphere candidates: {counts['atmosphere']}")
    print(f"service candidates: {counts['service']}")
    print(f"value candidates: {counts['value']}")
    print(f"train overlap: {overlap_counts['train']}")
    print(f"dev overlap: {overlap_counts['dev']}")
    print(f"test overlap: {overlap_counts['test']}")
    print(f"CSV path: {CURATION_PATH}")
    print(f"Summary path: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
