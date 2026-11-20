"""Apply Batch 2 human decisions to the main curation pool and batch CSV."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_DIR = PROJECT_ROOT / "data" / "curation"
MAIN_PATH = CURATION_DIR / "mams_neutral_candidates_en.csv"
RANKED_POOL_PATH = CURATION_DIR / "mams_neutral_ranked_pool_en.csv"
BATCH_PATH = CURATION_DIR / "review_batches" / "mams_neutral_batch_02.csv"

# candidate_id: (decision, final_aspect, final_sentiment, review_note)
DECISIONS = {
    "MAMS-EN-001237-PRICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-002555-AMBIENCE": (
        "ACCEPT", "atmosphere", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
    "MAMS-EN-001555-SERVICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-000660-PRICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-001912-AMBIENCE": ("REJECT", "", "", "POSITIVE_NOT_NEUTRAL"),
    "MAMS-EN-002755-SERVICE": ("REJECT", "", "", "WRONG_ASPECT"),
    "MAMS-EN-001747-AMBIENCE": ("UNCERTAIN", "", "", "AMBIGUOUS"),
    "MAMS-EN-001930-PRICE": ("REJECT", "", "", "NO_VALUE_JUDGMENT"),
    "MAMS-EN-001522-SERVICE": ("REJECT", "", "", "WRONG_ASPECT"),
    "MAMS-EN-001712-FOOD": ("REJECT", "", "", "WRONG_ASPECT"),
    "MAMS-EN-000556-AMBIENCE": (
        "ACCEPT", "atmosphere", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
    "MAMS-EN-002395-FOOD": (
        "ACCEPT", "taste", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
    "MAMS-EN-001980-AMBIENCE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-001282-SERVICE": ("REJECT", "", "", "MIXED_NOT_NEUTRAL"),
    "MAMS-EN-001170-AMBIENCE": ("REJECT", "", "", "POSITIVE_NOT_NEUTRAL"),
    "MAMS-EN-000636-SERVICE": (
        "ACCEPT", "service", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
    "MAMS-EN-000178-SERVICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-002379-FOOD": (
        "ACCEPT", "taste", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
    "MAMS-EN-002270-PRICE": ("UNCERTAIN", "", "", "AMBIGUOUS"),
    "MAMS-EN-002214-PRICE": (
        "ACCEPT", "value", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
    "MAMS-EN-001949-FOOD": ("REJECT", "", "", "AMBIGUOUS"),
    "MAMS-EN-002173-FOOD": (
        "ACCEPT", "taste", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
    "MAMS-EN-000341-PRICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-001951-FOOD": (
        "ACCEPT", "taste", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
}

EDITABLE_FIELDS = {
    "human_decision",
    "human_final_aspect",
    "human_final_sentiment",
    "review_notes",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return reader.fieldnames, list(reader)


def atomic_write(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def merge_note(existing: str, new_note: str) -> str:
    notes = [note for note in existing.split(";") if note]
    if new_note not in notes:
        notes.append(new_note)
    return ";".join(notes)


def append_missing_main_rows(
    main_fields: list[str], main_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    _, ranked_rows = read_csv(RANKED_POOL_PATH)
    ranked_by_id = {row["candidate_id"]: row for row in ranked_rows}
    existing_ids = {row["candidate_id"] for row in main_rows}
    missing_ids = set(DECISIONS) - existing_ids

    unavailable = missing_ids - set(ranked_by_id)
    if unavailable:
        raise ValueError(f"Batch 2 candidates missing from ranked pool: {sorted(unavailable)}")

    for candidate_id in sorted(missing_ids):
        source = ranked_by_id[candidate_id]
        row = {field: "" for field in main_fields}
        for field in main_fields:
            if field in source:
                row[field] = source[field]
        main_rows.append(row)
    return main_rows


def apply_to_rows(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    counts = Counter(row["candidate_id"] for row in rows)
    missing = sorted(candidate_id for candidate_id in DECISIONS if counts[candidate_id] == 0)
    repeated = sorted(candidate_id for candidate_id in DECISIONS if counts[candidate_id] > 1)
    if missing or repeated:
        raise ValueError(f"ID validation failed for {path}: missing={missing}, repeated={repeated}")

    originals = {row["candidate_id"]: row.copy() for row in rows}
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id not in DECISIONS:
            continue
        decision, final_aspect, final_sentiment, note = DECISIONS[candidate_id]
        row["human_decision"] = decision
        row["human_final_aspect"] = final_aspect
        row["human_final_sentiment"] = final_sentiment
        row["review_notes"] = merge_note(row["review_notes"], note)

    for row in rows:
        original = originals[row["candidate_id"]]
        for field in fieldnames:
            if field not in EDITABLE_FIELDS and row[field] != original[field]:
                raise ValueError(f"Protected field changed: {path} {row['candidate_id']} {field}")
    return rows


def validate(path: Path) -> None:
    _, rows = read_csv(path)
    by_id = {row["candidate_id"]: row for row in rows}
    for candidate_id, expected in DECISIONS.items():
        decision, aspect, sentiment, note = expected
        row = by_id[candidate_id]
        if (
            row["human_decision"],
            row["human_final_aspect"],
            row["human_final_sentiment"],
        ) != (decision, aspect, sentiment):
            raise ValueError(f"Post-write decision mismatch: {path} {candidate_id}")
        if note not in row["review_notes"].split(";"):
            raise ValueError(f"Post-write note mismatch: {path} {candidate_id}")


def main() -> None:
    main_fields, main_rows = read_csv(MAIN_PATH)
    main_rows = append_missing_main_rows(main_fields, main_rows)
    main_rows = apply_to_rows(MAIN_PATH, main_fields, main_rows)
    atomic_write(MAIN_PATH, main_fields, main_rows)

    batch_fields, batch_rows = read_csv(BATCH_PATH)
    batch_rows = apply_to_rows(BATCH_PATH, batch_fields, batch_rows)
    atomic_write(BATCH_PATH, batch_fields, batch_rows)

    validate(MAIN_PATH)
    validate(BATCH_PATH)

    decision_counts = Counter(value[0] for value in DECISIONS.values())
    accepted = Counter(
        value[1] for value in DECISIONS.values() if value[0] == "ACCEPT"
    )
    print("Batch 2 decisions applied:")
    print(f"ACCEPT: {decision_counts['ACCEPT']}")
    print(f"REJECT: {decision_counts['REJECT']}")
    print(f"UNCERTAIN: {decision_counts['UNCERTAIN']}")
    print("Batch 2 accepted by aspect:")
    for aspect in ("taste", "atmosphere", "service", "value"):
        print(f"{aspect}: {accepted[aspect]}")


if __name__ == "__main__":
    main()
