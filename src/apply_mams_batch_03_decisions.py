"""Apply Batch 3 human decisions to the main curation pool and batch CSV."""

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
BATCH_PATH = CURATION_DIR / "review_batches" / "mams_neutral_batch_03.csv"

# candidate_id: (decision, final_aspect, final_sentiment, review_note)
DECISIONS = {
    "MAMS-EN-001449-PRICE": ("ACCEPT", "value", "neutral", "HUMAN_VERIFIED_NEUTRAL"),
    "MAMS-EN-001278-AMBIENCE": ("REJECT", "", "", "POSITIVE_NOT_NEUTRAL"),
    "MAMS-EN-000477-SERVICE": ("UNCERTAIN", "", "", "AMBIGUOUS"),
    "MAMS-EN-001372-PRICE": ("REJECT", "", "", "NO_VALUE_JUDGMENT"),
    "MAMS-EN-001173-AMBIENCE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-001535-SERVICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-002155-PRICE": ("REJECT", "", "", "NO_VALUE_JUDGMENT"),
    "MAMS-EN-002038-PRICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-001601-SERVICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-002780-SERVICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-001960-AMBIENCE": (
        "ACCEPT", "atmosphere", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
    "MAMS-EN-001428-FOOD": ("ACCEPT", "taste", "neutral", "HUMAN_VERIFIED_NEUTRAL"),
    "MAMS-EN-001134-PRICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-002777-SERVICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-002430-AMBIENCE": ("REJECT", "", "", "POSITIVE_NOT_NEUTRAL"),
    "MAMS-EN-000143-SERVICE": ("REJECT", "", "", "WRONG_ASPECT"),
    "MAMS-EN-000875-SERVICE": ("REJECT", "", "", "POSITIVE_NOT_NEUTRAL"),
    "MAMS-EN-002982-FOOD": ("ACCEPT", "taste", "neutral", "HUMAN_VERIFIED_NEUTRAL"),
    "MAMS-EN-002310-PRICE": ("REJECT", "", "", "NO_VALUE_JUDGMENT"),
    "MAMS-EN-002606-PRICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-000165-FOOD": ("ACCEPT", "taste", "neutral", "HUMAN_VERIFIED_NEUTRAL"),
    "MAMS-EN-001004-FOOD": ("ACCEPT", "taste", "neutral", "HUMAN_VERIFIED_NEUTRAL"),
    "MAMS-EN-000756-PRICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-001895-SERVICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
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


def atomic_write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
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
    fields: list[str], rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    _, ranked_rows = read_csv(RANKED_POOL_PATH)
    ranked_by_id = {row["candidate_id"]: row for row in ranked_rows}
    existing_ids = {row["candidate_id"] for row in rows}
    missing_ids = set(DECISIONS) - existing_ids
    unavailable = missing_ids - set(ranked_by_id)
    if unavailable:
        raise ValueError(f"Candidates missing from ranked pool: {sorted(unavailable)}")
    for candidate_id in sorted(missing_ids):
        source = ranked_by_id[candidate_id]
        rows.append({field: source.get(field, "") for field in fields})
    return rows


def apply_decisions(
    path: Path, fields: list[str], rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    id_counts = Counter(row["candidate_id"] for row in rows)
    missing = sorted(candidate_id for candidate_id in DECISIONS if id_counts[candidate_id] == 0)
    repeated = sorted(candidate_id for candidate_id in DECISIONS if id_counts[candidate_id] > 1)
    if missing or repeated:
        raise ValueError(f"ID validation failed for {path}: missing={missing}, repeated={repeated}")

    originals = {row["candidate_id"]: row.copy() for row in rows}
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id not in DECISIONS:
            continue
        decision, aspect, sentiment, note = DECISIONS[candidate_id]
        row["human_decision"] = decision
        row["human_final_aspect"] = aspect
        row["human_final_sentiment"] = sentiment
        row["review_notes"] = merge_note(row["review_notes"], note)

    for row in rows:
        original = originals[row["candidate_id"]]
        for field in fields:
            if field not in EDITABLE_FIELDS and row[field] != original[field]:
                raise ValueError(f"Protected field changed: {path} {row['candidate_id']} {field}")
    return rows


def validate(path: Path) -> None:
    _, rows = read_csv(path)
    by_id = {row["candidate_id"]: row for row in rows}
    for candidate_id, expected in DECISIONS.items():
        decision, aspect, sentiment, note = expected
        row = by_id[candidate_id]
        actual = (
            row["human_decision"],
            row["human_final_aspect"],
            row["human_final_sentiment"],
        )
        if actual != (decision, aspect, sentiment):
            raise ValueError(f"Post-write mismatch: {path} {candidate_id}")
        if note not in row["review_notes"].split(";"):
            raise ValueError(f"Review note mismatch: {path} {candidate_id}")


def main() -> None:
    main_fields, main_rows = read_csv(MAIN_PATH)
    main_rows = append_missing_main_rows(main_fields, main_rows)
    atomic_write(MAIN_PATH, main_fields, apply_decisions(MAIN_PATH, main_fields, main_rows))

    batch_fields, batch_rows = read_csv(BATCH_PATH)
    atomic_write(BATCH_PATH, batch_fields, apply_decisions(BATCH_PATH, batch_fields, batch_rows))

    validate(MAIN_PATH)
    validate(BATCH_PATH)

    decision_counts = Counter(value[0] for value in DECISIONS.values())
    print("Batch 3 decisions applied:")
    for decision in ("ACCEPT", "REJECT", "UNCERTAIN"):
        print(f"{decision}: {decision_counts[decision]}")


if __name__ == "__main__":
    main()
