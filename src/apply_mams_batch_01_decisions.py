"""Apply the human decisions for MAMS neutral review batch 01."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = PROJECT_ROOT / "data" / "curation" / "mams_neutral_candidates_en.csv"
BATCH_PATH = (
    PROJECT_ROOT
    / "data"
    / "curation"
    / "review_batches"
    / "mams_neutral_batch_01.csv"
)

# candidate_id: (decision, final_aspect, final_sentiment, review_note)
DECISIONS = {
    "MAMS-EN-002712-PRICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-000412-PRICE": ("REJECT", "", "", "NO_VALUE_JUDGMENT"),
    "MAMS-EN-000544-AMBIENCE": (
        "ACCEPT",
        "atmosphere",
        "neutral",
        "HUMAN_VERIFIED_NEUTRAL",
    ),
    "MAMS-EN-002536-SERVICE": ("REJECT", "", "", "MIXED_NOT_NEUTRAL"),
    "MAMS-EN-001341-FOOD": ("REJECT", "", "", "WRONG_ASPECT"),
    "MAMS-EN-001803-FOOD": ("REJECT", "", "", "POSITIVE_NOT_NEUTRAL"),
    "MAMS-EN-001503-AMBIENCE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-000772-PRICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-001026-SERVICE": ("UNCERTAIN", "", "", "AMBIGUOUS"),
    "MAMS-EN-002778-PRICE": ("REJECT", "", "", "POSITIVE_NOT_NEUTRAL"),
    "MAMS-EN-002812-SERVICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-001834-FOOD": ("REJECT", "", "", "WRONG_ASPECT"),
    "MAMS-EN-001551-SERVICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-001604-SERVICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-000990-AMBIENCE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-000678-AMBIENCE": ("REJECT", "", "", "POSITIVE_NOT_NEUTRAL"),
    "MAMS-EN-002960-FOOD": ("REJECT", "", "", "WRONG_ASPECT"),
    "MAMS-EN-000088-PRICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-002473-PRICE": ("REJECT", "", "", "NO_VALUE_JUDGMENT"),
    "MAMS-EN-000956-FOOD": ("REJECT", "", "", "WRONG_ASPECT"),
    "MAMS-EN-001584-AMBIENCE": ("REJECT", "", "", "POSITIVE_NOT_NEUTRAL"),
    "MAMS-EN-000479-SERVICE": ("REJECT", "", "", "NEGATIVE_NOT_NEUTRAL"),
    "MAMS-EN-000038-AMBIENCE": ("REJECT", "", "", "POSITIVE_NOT_NEUTRAL"),
    "MAMS-EN-001508-FOOD": ("REJECT", "", "", "WRONG_ASPECT"),
}

EDITABLE_FIELDS = {
    "human_decision",
    "human_final_aspect",
    "human_final_sentiment",
    "review_notes",
}


def merge_note(existing: str, new_note: str) -> str:
    notes = [note for note in existing.split(";") if note]
    if new_note not in notes:
        notes.append(new_note)
    return ";".join(notes)


def update_csv(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Missing header: {path}")
        rows = list(reader)

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
        decision, final_aspect, final_sentiment, note = DECISIONS[candidate_id]
        row["human_decision"] = decision
        row["human_final_aspect"] = final_aspect
        row["human_final_sentiment"] = final_sentiment
        row["review_notes"] = merge_note(row["review_notes"], note)

    # Prove that candidate/source fields and all non-review fields are unchanged.
    for row in rows:
        original = originals[row["candidate_id"]]
        for field in fieldnames:
            if field not in EDITABLE_FIELDS and row[field] != original[field]:
                raise ValueError(f"Protected field changed: {path} {row['candidate_id']} {field}")

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def validate_results() -> None:
    for path in (POOL_PATH, BATCH_PATH):
        with path.open(encoding="utf-8", newline="") as file:
            rows = {
                row["candidate_id"]: row
                for row in csv.DictReader(file)
                if row["candidate_id"] in DECISIONS
            }
        if set(rows) != set(DECISIONS):
            raise ValueError(f"Post-write ID mismatch: {path}")
        for candidate_id, expected in DECISIONS.items():
            decision, final_aspect, final_sentiment, note = expected
            row = rows[candidate_id]
            actual = (
                row["human_decision"],
                row["human_final_aspect"],
                row["human_final_sentiment"],
            )
            if actual != (decision, final_aspect, final_sentiment):
                raise ValueError(f"Decision mismatch: {path} {candidate_id}")
            if note not in row["review_notes"].split(";"):
                raise ValueError(f"Review note mismatch: {path} {candidate_id}")


def main() -> None:
    update_csv(POOL_PATH)
    update_csv(BATCH_PATH)
    validate_results()

    counts = Counter(decision[0] for decision in DECISIONS.values())
    accepted_aspects = Counter(
        final_aspect
        for decision, final_aspect, _, _ in DECISIONS.values()
        if decision == "ACCEPT"
    )
    print("Batch 1:")
    print(f"ACCEPT: {counts['ACCEPT']}")
    print(f"REJECT: {counts['REJECT']}")
    print(f"UNCERTAIN: {counts['UNCERTAIN']}")
    print("Accepted by aspect:")
    for aspect in ("taste", "atmosphere", "service", "value"):
        print(f"{aspect}: {accepted_aspects[aspect]}")


if __name__ == "__main__":
    main()
