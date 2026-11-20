"""Apply all-ACCEPT decisions for the final English neutral review batch."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = PROJECT_ROOT / "data" / "curation" / "human_authored_neutral_candidates_en.csv"
BATCH_PATH = (
    PROJECT_ROOT
    / "data"
    / "curation"
    / "review_batches"
    / "human_authored_neutral_en_final_batch.csv"
)

ACCEPT_IDS = {
    "HA-CAND-EN-VALUE-0056",
    "HA-CAND-EN-VALUE-0055",
    "HA-CAND-EN-SERVICE-0063",
    "HA-CAND-EN-VALUE-0052",
    "HA-CAND-EN-SERVICE-0064",
    "HA-CAND-EN-VALUE-0050",
    "HA-CAND-EN-VALUE-0059",
    "HA-CAND-EN-SERVICE-0065",
    "HA-CAND-EN-VALUE-0053",
    "HA-CAND-EN-VALUE-0058",
    "HA-CAND-EN-SERVICE-0060",
    "HA-CAND-EN-VALUE-0054",
    "HA-CAND-EN-SERVICE-0061",
    "HA-CAND-EN-VALUE-0057",
    "HA-CAND-EN-VALUE-0060",
    "HA-CAND-EN-VALUE-0051",
    "HA-CAND-EN-SERVICE-0059",
    "HA-CAND-EN-SERVICE-0062",
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


def apply(path: Path) -> None:
    fields, rows = read_csv(path)
    by_id = {row["candidate_id"]: row for row in rows}
    missing = ACCEPT_IDS - set(by_id)
    if missing:
        raise ValueError(f"Missing candidate IDs in {path}: {sorted(missing)}")
    for candidate_id in ACCEPT_IDS:
        row = by_id[candidate_id]
        row["human_decision"] = "ACCEPT"
        row["human_final_aspect"] = row["mapped_aspect"]
        row["human_final_sentiment"] = "neutral"
        row["human_verified"] = "true"
        row["review_notes"] = "HUMAN_VERIFIED_NEUTRAL"
        if "source_type" in row and row["source_type"] != "synthetic_candidate":
            raise ValueError(f"source_type changed unexpectedly: {candidate_id}")
    atomic_write(path, fields, rows)


def main() -> None:
    apply(POOL_PATH)
    apply(BATCH_PATH)
    print(f"Final English decisions applied: ACCEPT={len(ACCEPT_IDS)}")


if __name__ == "__main__":
    main()
