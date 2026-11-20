"""Apply TR translation Batch 1 decisions and prepare diverse Batch 2."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_DIR = PROJECT_ROOT / "data" / "curation"
MASTER_PATH = CURATION_DIR / "translated_neutral_candidates_tr.csv"
BATCH1_PATH = CURATION_DIR / "review_batches" / "neutral_tr_translation_batch_01.csv"
BATCH2_PATH = CURATION_DIR / "review_batches" / "neutral_tr_translation_batch_02.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "analysis" / "tr_translation_batch_01_summary.txt"

ACCEPT_IDS = {
    "HA-CAND-TR-VALUE-0001",
    "HA-CAND-TR-SERVICE-0028",
    "HA-CAND-TR-ATMOSPHERE-0004",
    "HA-CAND-TR-SERVICE-0009",
    "HA-CAND-TR-ATMOSPHERE-0007",
    "HA-CAND-TR-VALUE-0005",
    "HA-CAND-TR-VALUE-0017",
    "HA-CAND-TR-ATMOSPHERE-0013",
    "HA-CAND-TR-SERVICE-0024",
    "HA-CAND-TR-VALUE-0009",
    "HA-CAND-TR-SERVICE-0013",
    "MAMS-CAND-TR-TASTE-002395",
    "HA-CAND-TR-SERVICE-0017",
    "MAMS-CAND-TR-TASTE-002173",
    "HA-CAND-TR-VALUE-0052",
    "HA-CAND-TR-ATMOSPHERE-0019",
    "HA-CAND-TR-SERVICE-0001",
    "MAMS-CAND-TR-TASTE-001004",
    "HA-CAND-TR-ATMOSPHERE-0001",
}

REVISIONS = {
    "HA-CAND-TR-VALUE-0017": "Öğle setinin fiyatı, yakındaki diğer iki çeşit içeren menülerle benzer seviyede.",
    "HA-CAND-TR-VALUE-0009": "Listelenen fiyatlar, orta segment bir restoran için beklenen aralıkla örtüşüyor.",
}

BATCH2_TRANSLATION_REVISIONS = {
    "MAMS-CAND-TR-TASTE-000793": "Kötü servis, ortalama yemekler ve sıradan menü seçenekleri olmasa fiyat o kadar da kötü sayılmazdı.",
    "MAMS-CAND-TR-TASTE-002497": "Yönetim her zaman kabaydı ama iyi personel bunu dengeliyordu; o personelin desteği olmayınca Jesse’s, zayıf dekoru ve ortalama yemekleriyle sıradan bir pubdan ibaret.",
    "HA-CAND-TR-SERVICE-0005": "Listeye adımızı yazdırmamızla masaya oturmamız arasında yaklaşık on dakika bekledik.",
    "HA-CAND-TR-SERVICE-0059": "Yan tezgahta teslim alma numaramı söyledim ve numara çağrılınca paketimi teslim aldım.",
    "HA-CAND-TR-VALUE-0020": "Bu noodle yemeğinin fiyatı, bu tür yemekler için tipik olan aralıkta.",
    "HA-CAND-TR-VALUE-0057": "Kahve ve tatlının fiyatları, aynı meydandaki kafelerdeki fiyatlara benzerdi.",
}

# Diverse unreviewed sources: taste 4, atmosphere 5, service 6, value 5.
BATCH2_IDS = [
    "MAMS-CAND-TR-TASTE-000165",
    "MAMS-CAND-TR-TASTE-000793",
    "MAMS-CAND-TR-TASTE-001951",
    "MAMS-CAND-TR-TASTE-002497",
    "HA-CAND-TR-ATMOSPHERE-0010",
    "HA-CAND-TR-ATMOSPHERE-0016",
    "HA-CAND-TR-ATMOSPHERE-0022",
    "HA-CAND-TR-ATMOSPHERE-0029",
    "HA-CAND-TR-ATMOSPHERE-0031",
    "HA-CAND-TR-SERVICE-0005",
    "HA-CAND-TR-SERVICE-0021",
    "HA-CAND-TR-SERVICE-0043",
    "HA-CAND-TR-SERVICE-0045",
    "HA-CAND-TR-SERVICE-0052",
    "HA-CAND-TR-SERVICE-0059",
    "HA-CAND-TR-VALUE-0020",
    "HA-CAND-TR-VALUE-0039",
    "HA-CAND-TR-VALUE-0043",
    "HA-CAND-TR-VALUE-0050",
    "HA-CAND-TR-VALUE-0057",
]


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
        raise ValueError(f"Missing Batch 1 IDs in {path}: {sorted(missing)}")

    for candidate_id in ACCEPT_IDS:
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


def calculate_coverage(master_rows: list[dict[str, str]]) -> dict[str, int]:
    original = Counter()
    with (PROJECT_ROOT / "data" / "cleaned" / "train.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file):
            if row["language"] == "tr" and row["sentiment"] == "neutral":
                original[row["aspect"]] += 1
    verified = Counter(
        row["human_final_aspect"]
        for row in master_rows
        if row["human_decision"] == "ACCEPT"
        and row["translation_human_verified"] == "true"
    )
    return {
        aspect: original[aspect] + verified[aspect]
        for aspect in ("taste", "atmosphere", "service", "value")
    }


def main() -> None:
    apply(MASTER_PATH)
    apply(BATCH1_PATH)

    fields, master_rows = read_csv(MASTER_PATH)
    by_id = {row["candidate_id"]: row for row in master_rows}
    for candidate_id, revised_sentence in BATCH2_TRANSLATION_REVISIONS.items():
        by_id[candidate_id]["translated_sentence"] = revised_sentence
    atomic_write(MASTER_PATH, fields, master_rows)
    unavailable = set(BATCH2_IDS) - set(by_id)
    if unavailable:
        raise ValueError(f"Batch 2 IDs missing from master: {sorted(unavailable)}")
    batch2 = [by_id[candidate_id].copy() for candidate_id in BATCH2_IDS]
    if any(row["human_decision"] for row in batch2):
        raise ValueError("Batch 2 contains an already reviewed translation")
    counts = Counter(row["mapped_aspect"] for row in batch2)
    if counts != Counter({"taste": 4, "atmosphere": 5, "service": 6, "value": 5}):
        raise ValueError(f"Batch 2 quotas invalid: {dict(counts)}")
    if len({row["parallel_source_group_id"] for row in batch2}) != len(batch2):
        raise ValueError("Batch 2 contains duplicate semantic source groups")
    atomic_write(BATCH2_PATH, fields, batch2)

    coverage = calculate_coverage(master_rows)
    remaining = {aspect: max(0, 25 - coverage[aspect]) for aspect in coverage}
    summary = [
        "Verified TR coverage:",
        *[f"{aspect}: {coverage[aspect]}" for aspect in coverage],
        "Remaining TR gap:",
        *[f"{aspect}: {remaining[aspect]}" for aspect in remaining],
        f"TR Batch 2 size: {len(batch2)}",
        "Batch 2 human decisions assigned: 0",
        "training rows added: 0",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
