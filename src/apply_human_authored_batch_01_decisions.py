"""Apply human-authored English neutral Review Batch 1 decisions."""

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
    / "human_authored_neutral_en_batch_01.csv"
)

ACCEPT_IDS = {
    "HA-CAND-EN-SERVICE-0028",
    "HA-CAND-EN-ATMOSPHERE-0016",
    "HA-CAND-EN-VALUE-0009",
    "HA-CAND-EN-SERVICE-0021",
    "HA-CAND-EN-SERVICE-0009",
    "HA-CAND-EN-VALUE-0005",
    "HA-CAND-EN-SERVICE-0013",
    "HA-CAND-EN-SERVICE-0024",
    "HA-CAND-EN-VALUE-0001",
    "HA-CAND-EN-ATMOSPHERE-0004",
    "HA-CAND-EN-SERVICE-0017",
    "HA-CAND-EN-VALUE-0020",
    "HA-CAND-EN-ATMOSPHERE-0019",
    "HA-CAND-EN-SERVICE-0005",
    "HA-CAND-EN-ATMOSPHERE-0007",
    "HA-CAND-EN-ATMOSPHERE-0013",
    "HA-CAND-EN-ATMOSPHERE-0022",
    "HA-CAND-EN-SERVICE-0001",
    "HA-CAND-EN-ATMOSPHERE-0001",
    "HA-CAND-EN-ATMOSPHERE-0010",
    "HA-CAND-EN-VALUE-0017",
}

REJECT_IDS = {
    "HA-CAND-EN-VALUE-0027",
    "HA-CAND-EN-VALUE-0023",
    "HA-CAND-EN-VALUE-0013",
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


def apply(path: Path) -> None:
    fields, rows = read_csv(path)
    by_id = {row["candidate_id"]: row for row in rows}
    expected = ACCEPT_IDS | REJECT_IDS
    missing = expected - set(by_id)
    if missing:
        raise ValueError(f"Missing candidate IDs in {path}: {sorted(missing)}")

    for candidate_id in ACCEPT_IDS:
        row = by_id[candidate_id]
        original_source_type = row.get("source_type")
        row["human_decision"] = "ACCEPT"
        row["human_final_aspect"] = row["mapped_aspect"]
        row["human_final_sentiment"] = "neutral"
        row["human_verified"] = "true"
        row["review_notes"] = "HUMAN_VERIFIED_NEUTRAL"
        if original_source_type is not None and row["source_type"] != "synthetic_candidate":
            raise ValueError(f"source_type changed unexpectedly: {candidate_id}")

    for candidate_id in REJECT_IDS:
        row = by_id[candidate_id]
        row["human_decision"] = "REJECT"
        row["human_final_aspect"] = ""
        row["human_final_sentiment"] = ""
        row["human_verified"] = "false"
        row["review_notes"] = "HUMAN_REJECTED"

    atomic_write(path, fields, rows)


def main() -> None:
    apply(POOL_PATH)
    apply(BATCH_PATH)
    print(f"Batch 1 decisions applied: ACCEPT={len(ACCEPT_IDS)}, REJECT={len(REJECT_IDS)}")


if __name__ == "__main__":
    main()
