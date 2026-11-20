"""Apply ES translation Batch 1 decisions and prepare diverse Batch 2."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_DIR = PROJECT_ROOT / "data" / "curation"
MASTER_PATH = CURATION_DIR / "translated_neutral_candidates_es.csv"
BATCH1_PATH = CURATION_DIR / "review_batches" / "neutral_es_translation_batch_01.csv"
BATCH2_PATH = CURATION_DIR / "review_batches" / "neutral_es_translation_batch_02.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "analysis" / "es_translation_batch_01_summary.txt"

BATCH1_IDS = {
    "HA-CAND-ES-VALUE-0001",
    "HA-CAND-ES-SERVICE-0028",
    "HA-CAND-ES-ATMOSPHERE-0004",
    "HA-CAND-ES-SERVICE-0009",
    "HA-CAND-ES-ATMOSPHERE-0007",
    "HA-CAND-ES-VALUE-0005",
    "HA-CAND-ES-VALUE-0017",
    "HA-CAND-ES-ATMOSPHERE-0013",
    "HA-CAND-ES-SERVICE-0024",
    "HA-CAND-ES-VALUE-0009",
    "HA-CAND-ES-SERVICE-0013",
    "MAMS-CAND-ES-TASTE-002395",
    "HA-CAND-ES-SERVICE-0017",
    "MAMS-CAND-ES-TASTE-002173",
    "HA-CAND-ES-VALUE-0052",
    "HA-CAND-ES-ATMOSPHERE-0019",
    "HA-CAND-ES-SERVICE-0001",
    "MAMS-CAND-ES-TASTE-001004",
    "HA-CAND-ES-ATMOSPHERE-0001",
}

REVISIONS = {
    "HA-CAND-ES-VALUE-0005": "Los precios de los platos principales están cerca de la media de esta zona.",
    "HA-CAND-ES-VALUE-0017": "El precio del menú de almuerzo es comparable al de otros menús de dos platos de restaurantes cercanos.",
    "HA-CAND-ES-VALUE-0009": "Los precios indicados están dentro del rango esperado para un restaurante de precio medio.",
    "HA-CAND-ES-SERVICE-0013": "La reserva se confirmó en la entrada usando el nombre.",
    "MAMS-CAND-ES-TASTE-002173": "La comida estaba bien, aunque los precios eran altos se mirara por donde se mirara: una cena para cuatro, sin bebidas, superó los 300 dólares.",
    "HA-CAND-ES-SERVICE-0001": "El camarero toma primero el pedido de bebidas de la mesa y después anota el pedido de comida.",
    "HA-CAND-ES-ATMOSPHERE-0001": "Durante el almuerzo, el comedor utiliza luces de techo.",
}

BATCH2_IDS = [
    "MAMS-CAND-ES-TASTE-000165",
    "MAMS-CAND-ES-TASTE-000793",
    "MAMS-CAND-ES-TASTE-001951",
    "MAMS-CAND-ES-TASTE-002497",
    "HA-CAND-ES-ATMOSPHERE-0010",
    "HA-CAND-ES-ATMOSPHERE-0016",
    "HA-CAND-ES-ATMOSPHERE-0022",
    "HA-CAND-ES-ATMOSPHERE-0029",
    "HA-CAND-ES-ATMOSPHERE-0031",
    "HA-CAND-ES-SERVICE-0005",
    "HA-CAND-ES-SERVICE-0021",
    "HA-CAND-ES-SERVICE-0043",
    "HA-CAND-ES-SERVICE-0045",
    "HA-CAND-ES-SERVICE-0052",
    "HA-CAND-ES-SERVICE-0059",
    "HA-CAND-ES-VALUE-0020",
    "HA-CAND-ES-VALUE-0039",
    "HA-CAND-ES-VALUE-0043",
    "HA-CAND-ES-VALUE-0050",
    "HA-CAND-ES-VALUE-0057",
]

BATCH2_TRANSLATION_REVISIONS = {
    "HA-CAND-ES-SERVICE-0045": "Nos presentamos en la entrada y, tras confirmar el número de personas, nos acompañaron a la mesa.",
    "HA-CAND-ES-SERVICE-0052": "Pedimos la cuenta y pagamos en la mesa con el datáfono.",
    "HA-CAND-ES-SERVICE-0059": "Di el número de recogida en el mostrador lateral y recogí la bolsa cuando llamaron mi número.",
    "HA-CAND-ES-VALUE-0020": "El precio de este plato de fideos está dentro del rango habitual para este tipo de plato.",
    "HA-CAND-ES-VALUE-0043": "Comparamos las cartas de varios restaurantes cercanos y los menús de almuerzo estaban en un rango de precios parecido.",
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


def apply_batch1(path: Path) -> None:
    fields, rows = read_csv(path)
    by_id = {row["candidate_id"]: row for row in rows}
    missing = BATCH1_IDS - set(by_id)
    if missing:
        raise ValueError(f"Missing Batch 1 IDs in {path}: {sorted(missing)}")
    for candidate_id in BATCH1_IDS:
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


def existing_es_sentences() -> set[str]:
    sentences: set[str] = set()
    for split in ("train", "dev", "test"):
        _, rows = read_csv(PROJECT_ROOT / "data" / "cleaned" / f"{split}.csv")
        sentences.update(
            row["sentence"].strip() for row in rows if row["language"] == "es"
        )
    return sentences


def calculate_coverage(master_rows: list[dict[str, str]]) -> dict[str, int]:
    _, train_rows = read_csv(PROJECT_ROOT / "data" / "cleaned" / "train.csv")
    original = Counter(
        row["aspect"]
        for row in train_rows
        if row["language"] == "es" and row["sentiment"] == "neutral"
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
    apply_batch1(MASTER_PATH)
    apply_batch1(BATCH1_PATH)

    fields, master_rows = read_csv(MASTER_PATH)
    by_id = {row["candidate_id"]: row for row in master_rows}
    missing = set(BATCH2_IDS) - set(by_id)
    if missing:
        raise ValueError(f"Batch 2 IDs missing from master: {sorted(missing)}")
    for candidate_id, revised_sentence in BATCH2_TRANSLATION_REVISIONS.items():
        by_id[candidate_id]["translated_sentence"] = revised_sentence
    atomic_write(MASTER_PATH, fields, master_rows)

    verified_sources = verified_english_source_ids()
    overlaps = existing_es_sentences()
    batch2 = [by_id[candidate_id].copy() for candidate_id in BATCH2_IDS]
    for row in batch2:
        if row["human_decision"]:
            raise ValueError(f"Already reviewed Batch 2 candidate: {row['candidate_id']}")
        if row["english_source_id"] not in verified_sources:
            raise ValueError(f"Unverified English source: {row['english_source_id']}")
        if row["translated_sentence"].strip() in overlaps:
            raise ValueError(f"Existing split overlap: {row['candidate_id']}")

    expected = Counter({"taste": 4, "atmosphere": 5, "service": 6, "value": 5})
    counts = Counter(row["mapped_aspect"] for row in batch2)
    if counts != expected:
        raise ValueError(f"Batch 2 quotas invalid: {dict(counts)}")
    if len({row["parallel_source_group_id"] for row in batch2}) != len(batch2):
        raise ValueError("Batch 2 contains duplicate semantic source groups")
    atomic_write(BATCH2_PATH, fields, batch2)

    coverage = calculate_coverage(master_rows)
    remaining = {aspect: max(0, 25 - count) for aspect, count in coverage.items()}
    summary = [
        "Verified ES coverage:",
        *[f"{aspect}: {coverage[aspect]}" for aspect in coverage],
        "Remaining ES gap:",
        *[f"{aspect}: {remaining[aspect]}" for aspect in remaining],
        "ES Batch 2:",
        *[f"{aspect}: {counts[aspect]}" for aspect in coverage],
        f"total: {len(batch2)}",
        "Batch 2 human decisions assigned: 0",
        "training rows added: 0",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
