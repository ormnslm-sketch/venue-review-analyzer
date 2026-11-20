"""Apply human-authored English neutral Review Batch 2 decisions."""

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
    / "human_authored_neutral_en_batch_02.csv"
)

ACCEPT_IDS = {
    "HA-CAND-EN-VALUE-0035",
    "HA-CAND-EN-ATMOSPHERE-0034",
    "HA-CAND-EN-VALUE-0041",
    "HA-CAND-EN-SERVICE-0054",
    "HA-CAND-EN-SERVICE-0049",
    "HA-CAND-EN-VALUE-0039",
    "HA-CAND-EN-SERVICE-0051",
    "HA-CAND-EN-SERVICE-0055",
    "HA-CAND-EN-VALUE-0037",
    "HA-CAND-EN-ATMOSPHERE-0030",
    "HA-CAND-EN-SERVICE-0052",
    "HA-CAND-EN-SERVICE-0043",
    "HA-CAND-EN-SERVICE-0047",
    "HA-CAND-EN-ATMOSPHERE-0031",
    "HA-CAND-EN-ATMOSPHERE-0033",
    "HA-CAND-EN-VALUE-0043",
    "HA-CAND-EN-SERVICE-0045",
    "HA-CAND-EN-SERVICE-0046",
    "HA-CAND-EN-ATMOSPHERE-0029",
    "HA-CAND-EN-ATMOSPHERE-0032",
}

REJECT_IDS = {
    "HA-CAND-EN-VALUE-0048",
    "HA-CAND-EN-VALUE-0045",
    "HA-CAND-EN-VALUE-0047",
    "HA-CAND-EN-VALUE-0044",
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
    missing = (ACCEPT_IDS | REJECT_IDS) - set(by_id)
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

    for candidate_id in REJECT_IDS:
        row = by_id[candidate_id]
        row["human_decision"] = "REJECT"
        row["human_final_aspect"] = ""
        row["human_final_sentiment"] = ""
        row["human_verified"] = "false"
        row["review_notes"] = "HUMAN_REJECTED"
        if "source_type" in row and row["source_type"] != "synthetic_candidate":
            raise ValueError(f"source_type changed unexpectedly: {candidate_id}")

    atomic_write(path, fields, rows)


def main() -> None:
    apply(POOL_PATH)
    apply(BATCH_PATH)
    print(f"Batch 2 decisions applied: ACCEPT={len(ACCEPT_IDS)}, REJECT={len(REJECT_IDS)}")


if __name__ == "__main__":
    main()
