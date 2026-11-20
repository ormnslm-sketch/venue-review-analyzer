"""Apply the final strict MAMS decisions and close MAMS curation."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_DIR = PROJECT_ROOT / "data" / "curation"
MAIN_PATH = CURATION_DIR / "mams_neutral_candidates_en.csv"
TARGETED_PATH = CURATION_DIR / "mams_targeted_neutral_candidates.csv"

DECISIONS = {
    "MAMS-EN-002585-AMBIENCE": ("REJECT", "", "", "AMBIGUOUS"),
    "MAMS-EN-000793-FOOD": (
        "ACCEPT", "taste", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
    "MAMS-EN-001950-FOOD": (
        "ACCEPT", "taste", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
    "MAMS-EN-002497-FOOD": (
        "ACCEPT", "taste", "neutral", "HUMAN_VERIFIED_NEUTRAL"
    ),
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header: {path}")
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


def main() -> None:
    targeted_fields, targeted_rows = read_csv(TARGETED_PATH)
    targeted_by_id = {row["candidate_id"]: row for row in targeted_rows}
    if set(targeted_by_id) != set(DECISIONS):
        raise ValueError("Targeted candidate IDs do not match final decision IDs")

    for candidate_id, (decision, _, _, _) in DECISIONS.items():
        targeted_by_id[candidate_id]["human_decision"] = decision
    atomic_write(TARGETED_PATH, targeted_fields, targeted_rows)

    main_fields, main_rows = read_csv(MAIN_PATH)
    main_by_id = {row["candidate_id"]: row for row in main_rows}
    for candidate_id, source in targeted_by_id.items():
        if candidate_id not in main_by_id:
            new_row = {field: "" for field in main_fields}
            new_row.update(
                {
                    "candidate_id": candidate_id,
                    "sentence": source["sentence"],
                    "source_dataset": "mams-acsa",
                    "source_license": "apache-2.0",
                    "source_category": source["source_category"],
                    "mapped_aspect": source["mapped_aspect"],
                    "source_polarity": "neutral",
                    "language": "en",
                    "source_group_id": source["source_group_id"],
                }
            )
            main_rows.append(new_row)
            main_by_id[candidate_id] = new_row

    for candidate_id, (decision, aspect, sentiment, note) in DECISIONS.items():
        row = main_by_id[candidate_id]
        row["human_decision"] = decision
        row["human_final_aspect"] = aspect
        row["human_final_sentiment"] = sentiment
        existing_notes = [value for value in row["review_notes"].split(";") if value]
        if note not in existing_notes:
            existing_notes.append(note)
        row["review_notes"] = ";".join(existing_notes)

    atomic_write(MAIN_PATH, main_fields, main_rows)
    print("Final MAMS targeted decisions applied: ACCEPT=3, REJECT=1")


if __name__ == "__main__":
    main()
