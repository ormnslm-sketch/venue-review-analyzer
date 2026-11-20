"""Apply final TR decisions and freeze neutral expansion v1."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_DIR = PROJECT_ROOT / "data" / "curation"
MASTER_PATH = CURATION_DIR / "translated_neutral_candidates_tr.csv"
FINAL_BATCH_PATH = (
    CURATION_DIR / "review_batches" / "neutral_tr_translation_final_batch.csv"
)
SUMMARY_PATH = PROJECT_ROOT / "data" / "analysis" / "tr_neutral_expansion_v1_summary.txt"

FINAL_IDS = {
    "HA-CAND-TR-ATMOSPHERE-0033",
    "MAMS-CAND-TR-ATMOSPHERE-001960",
    "MAMS-CAND-TR-ATMOSPHERE-002555",
    "HA-CAND-TR-SERVICE-0061",
    "HA-CAND-TR-SERVICE-0062",
    "HA-CAND-TR-SERVICE-0063",
    "HA-CAND-TR-SERVICE-0065",
    "HA-CAND-TR-VALUE-0058",
    "HA-CAND-TR-VALUE-0059",
    "HA-CAND-TR-VALUE-0060",
}

REVISIONS = {
    "HA-CAND-TR-SERVICE-0065": "Tezgahta sipariş verdik, basılı numarayı sakladık ve numaramız ekranda görününce tepsimizi aldık.",
    "HA-CAND-TR-VALUE-0058": "Yakındaki restoranların dışarıda sergilediği menülere göre öğle yemeği fiyatları bölge ortalaması civarındaydı.",
    "HA-CAND-TR-VALUE-0059": "Bu noodle yemeğinin fiyatı, başka yerlerde aynı tür yemek için normalde ödediğim kadardı.",
    "HA-CAND-TR-VALUE-0060": "Alakart fiyatlar, mahalledeki benzer restoranların menü fiyatlarıyla aynı seviyedeydi.",
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


def apply_final(path: Path) -> None:
    fields, rows = read_csv(path)
    by_id = {row["candidate_id"]: row for row in rows}
    missing = FINAL_IDS - set(by_id)
    if missing:
        raise ValueError(f"Missing final IDs in {path}: {sorted(missing)}")
    for candidate_id in FINAL_IDS:
        row = by_id[candidate_id]
        if candidate_id in REVISIONS:
            row["translated_sentence"] = REVISIONS[candidate_id]
            row["review_notes"] = "REVISED_BEFORE_ACCEPT;HUMAN_VERIFIED_TRANSLATION"
        else:
            row["review_notes"] = "HUMAN_VERIFIED_TRANSLATION"
        row["human_decision"] = "ACCEPT"
        row["translation_human_verified"] = "true"
        row["human_final_aspect"] = row["mapped_aspect"]
        row["human_final_sentiment"] = "neutral"
    atomic_write(path, fields, rows)


def main() -> None:
    apply_final(MASTER_PATH)
    apply_final(FINAL_BATCH_PATH)

    _, master_rows = read_csv(MASTER_PATH)
    _, train_rows = read_csv(PROJECT_ROOT / "data" / "cleaned" / "train.csv")
    aspects = ("taste", "atmosphere", "service", "value")
    original = Counter(
        row["aspect"]
        for row in train_rows
        if row["language"] == "tr" and row["sentiment"] == "neutral"
    )
    verified_rows = [
        row
        for row in master_rows
        if row["human_decision"] == "ACCEPT"
        and row["translation_human_verified"].lower() == "true"
        and row["human_final_sentiment"] == "neutral"
    ]
    additions = Counter(row["human_final_aspect"] for row in verified_rows)
    final = {aspect: original[aspect] + additions[aspect] for aspect in aspects}
    if not all(final[aspect] >= 25 for aspect in aspects):
        raise ValueError(f"TR target not reached: {final}")

    revised_count = sum(
        "REVISED_BEFORE_ACCEPT" in row["review_notes"] for row in verified_rows
    )
    direct_count = len(verified_rows) - revised_count
    source_counts = Counter(row["original_source_type"] for row in verified_rows)
    summary = [
        "TR_NEUTRAL_EXPANSION_V1 = COMPLETE",
        "",
        "Original M-ABSA TR neutral counts:",
        *[f"{aspect}: {original[aspect]}" for aspect in aspects],
        "",
        "Human-verified translated additions:",
        *[f"{aspect}: {additions[aspect]}" for aspect in aspects],
        f"total: {len(verified_rows)}",
        "",
        "Final verified TR neutral coverage:",
        *[f"{aspect}: {final[aspect]}" for aspect in aspects],
        "",
        f"Direct accepts: {direct_count}",
        f"Revised-then-accepted translations: {revised_count}",
        "",
        "Provenance:",
        f"human-authored English sources translated to TR: {source_counts['synthetic_candidate']}",
        f"human-reviewed MAMS gold sources translated to TR: {source_counts['mams_gold_annotation']}",
        "Parallel source-group IDs and original source/license fields were preserved.",
        "No automatically accepted translations were added.",
        "The cleaned training dataset was not modified.",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
