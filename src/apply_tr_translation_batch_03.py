"""Apply TR translation Batch 3 decisions and prepare the final TR batch."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_DIR = PROJECT_ROOT / "data" / "curation"
MASTER_PATH = CURATION_DIR / "translated_neutral_candidates_tr.csv"
BATCH3_PATH = CURATION_DIR / "review_batches" / "neutral_tr_translation_batch_03.csv"
FINAL_BATCH_PATH = (
    CURATION_DIR / "review_batches" / "neutral_tr_translation_final_batch.csv"
)
SUMMARY_PATH = PROJECT_ROOT / "data" / "analysis" / "tr_translation_batch_03_summary.txt"

BATCH3_IDS = {
    "MAMS-CAND-TR-TASTE-001428",
    "MAMS-CAND-TR-TASTE-001950",
    "MAMS-CAND-TR-TASTE-002982",
    "HA-CAND-TR-ATMOSPHERE-0030",
    "HA-CAND-TR-ATMOSPHERE-0032",
    "HA-CAND-TR-ATMOSPHERE-0034",
    "MAMS-CAND-TR-ATMOSPHERE-000544",
    "MAMS-CAND-TR-ATMOSPHERE-000556",
    "HA-CAND-TR-SERVICE-0046",
    "HA-CAND-TR-SERVICE-0049",
    "HA-CAND-TR-SERVICE-0054",
    "HA-CAND-TR-SERVICE-0055",
    "HA-CAND-TR-SERVICE-0060",
    "HA-CAND-TR-SERVICE-0064",
    "HA-CAND-TR-VALUE-0041",
    "HA-CAND-TR-VALUE-0051",
    "HA-CAND-TR-VALUE-0053",
    "HA-CAND-TR-VALUE-0054",
    "HA-CAND-TR-VALUE-0055",
    "HA-CAND-TR-VALUE-0056",
}

REVISIONS = {
    "MAMS-CAND-TR-TASTE-001428": "Ortalama yemekler, kabarık bir hesap ve servisten pek hoş olmayan tavırlar bekleyin.",
    "MAMS-CAND-TR-TASTE-002982": "Yemekler ortalama; servis ise ortalamanın çok altında. Garson özel yemekleri anlatırken sürekli takılıyor ve gecenin büyük kısmında bardaklarımız boş kalıyordu; ortam da müşteri kitlesi de oldukça yapaydı.",
    "HA-CAND-TR-ATMOSPHERE-0030": "Akşam yemeği kalabalığı geldikçe salonun daha hareketli hâle geldiğini seslerden anlayabiliyorduk.",
    "HA-CAND-TR-SERVICE-0046": "Karşılama bankosunda adımı söyledim; bizi masaya götürmeden önce ismimi rezervasyondaki kayıtla eşleştirdiler.",
    "HA-CAND-TR-VALUE-0041": "Bu tür bir noodle yemeği için genelde ödediğim seviyede bir fiyat ödedim.",
    "HA-CAND-TR-VALUE-0051": "Ana yemek fiyatları, şehrin o bölgesindeki ortalama fiyatlar civarındaydı.",
    "HA-CAND-TR-VALUE-0053": "Menü, bu tarz rahat deniz ürünleri restoranlarında gördüğüm olağan fiyat aralığındaydı.",
    "HA-CAND-TR-VALUE-0056": "Akşam yemeği, orta segment ve üç çeşit sunan bir restorandan beklediğim fiyat aralığıyla örtüştü.",
}

FINAL_BATCH_IDS = [
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
]

FINAL_TRANSLATION_REVISIONS = {
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


def apply_batch3(path: Path) -> None:
    fields, rows = read_csv(path)
    by_id = {row["candidate_id"]: row for row in rows}
    missing = BATCH3_IDS - set(by_id)
    if missing:
        raise ValueError(f"Missing Batch 3 IDs in {path}: {sorted(missing)}")
    for candidate_id in BATCH3_IDS:
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


def verified_english_source_ids() -> set[str]:
    verified: set[str] = set()
    for path in (
        CURATION_DIR / "mams_neutral_candidates_en.csv",
        CURATION_DIR / "human_authored_neutral_candidates_en.csv",
    ):
        _, rows = read_csv(path)
        for row in rows:
            if row.get("human_decision") == "ACCEPT" and (
                row.get("human_verified", "true").lower() == "true"
            ):
                verified.add(row["candidate_id"])
    return verified


def existing_tr_sentences() -> set[str]:
    sentences: set[str] = set()
    for split in ("train", "dev", "test"):
        _, rows = read_csv(PROJECT_ROOT / "data" / "cleaned" / f"{split}.csv")
        sentences.update(
            row["sentence"].strip() for row in rows if row["language"] == "tr"
        )
    return sentences


def calculate_coverage(master_rows: list[dict[str, str]]) -> dict[str, int]:
    _, train_rows = read_csv(PROJECT_ROOT / "data" / "cleaned" / "train.csv")
    original = Counter(
        row["aspect"]
        for row in train_rows
        if row["language"] == "tr" and row["sentiment"] == "neutral"
    )
    translated = Counter(
        row["human_final_aspect"]
        for row in master_rows
        if row["human_decision"] == "ACCEPT"
        and row["translation_human_verified"].lower() == "true"
    )
    return {
        aspect: original[aspect] + translated[aspect]
        for aspect in ("taste", "atmosphere", "service", "value")
    }


def main() -> None:
    apply_batch3(MASTER_PATH)
    apply_batch3(BATCH3_PATH)

    fields, master_rows = read_csv(MASTER_PATH)
    by_id = {row["candidate_id"]: row for row in master_rows}
    missing = set(FINAL_BATCH_IDS) - set(by_id)
    if missing:
        raise ValueError(f"Final batch IDs missing from master: {sorted(missing)}")

    for candidate_id, revised_sentence in FINAL_TRANSLATION_REVISIONS.items():
        by_id[candidate_id]["translated_sentence"] = revised_sentence
    atomic_write(MASTER_PATH, fields, master_rows)

    verified_sources = verified_english_source_ids()
    overlaps = existing_tr_sentences()
    final_batch = [by_id[candidate_id].copy() for candidate_id in FINAL_BATCH_IDS]
    for row in final_batch:
        if row["human_decision"]:
            raise ValueError(f"Already reviewed final candidate: {row['candidate_id']}")
        if row["english_source_id"] not in verified_sources:
            raise ValueError(f"Unverified English source: {row['english_source_id']}")
        if row["translated_sentence"].strip() in overlaps:
            raise ValueError(f"Existing split overlap: {row['candidate_id']}")

    expected = Counter({"atmosphere": 3, "service": 4, "value": 3})
    counts = Counter(row["mapped_aspect"] for row in final_batch)
    if counts != expected:
        raise ValueError(f"Final batch quotas invalid: {dict(counts)}")
    if len({row["parallel_source_group_id"] for row in final_batch}) != len(final_batch):
        raise ValueError("Final batch contains duplicate semantic source groups")
    atomic_write(FINAL_BATCH_PATH, fields, final_batch)

    coverage = calculate_coverage(master_rows)
    remaining = {aspect: max(0, 25 - count) for aspect, count in coverage.items()}
    if remaining != {"taste": 0, "atmosphere": 3, "service": 4, "value": 3}:
        raise ValueError(f"Unexpected remaining gap: {remaining}")

    summary = [
        "Verified TR coverage:",
        *[f"{aspect}: {coverage[aspect]}" for aspect in coverage],
        "Remaining TR gap:",
        *[f"{aspect}: {remaining[aspect]}" for aspect in ("atmosphere", "service", "value")],
        "Final TR review batch:",
        *[f"{aspect}: {counts[aspect]}" for aspect in ("atmosphere", "service", "value")],
        f"total: {len(final_batch)}",
        "Final batch human decisions assigned: 0",
        "training rows added: 0",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
