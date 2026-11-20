"""Apply ES translation Batch 3 decisions and prepare the final ES batch."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_DIR = PROJECT_ROOT / "data" / "curation"
MASTER_PATH = CURATION_DIR / "translated_neutral_candidates_es.csv"
BATCH3_PATH = CURATION_DIR / "review_batches" / "neutral_es_translation_batch_03.csv"
FINAL_BATCH_PATH = (
    CURATION_DIR / "review_batches" / "neutral_es_translation_final_batch.csv"
)
SUMMARY_PATH = PROJECT_ROOT / "data" / "analysis" / "es_translation_batch_03_summary.txt"

BATCH3_IDS = {
    "MAMS-CAND-ES-TASTE-001428",
    "MAMS-CAND-ES-TASTE-001950",
    "MAMS-CAND-ES-TASTE-002982",
    "HA-CAND-ES-ATMOSPHERE-0030",
    "HA-CAND-ES-ATMOSPHERE-0032",
    "HA-CAND-ES-ATMOSPHERE-0034",
    "MAMS-CAND-ES-ATMOSPHERE-000544",
    "MAMS-CAND-ES-ATMOSPHERE-000556",
    "HA-CAND-ES-SERVICE-0046",
    "HA-CAND-ES-SERVICE-0049",
    "HA-CAND-ES-SERVICE-0054",
    "HA-CAND-ES-SERVICE-0055",
    "HA-CAND-ES-SERVICE-0060",
    "HA-CAND-ES-SERVICE-0064",
    "HA-CAND-ES-VALUE-0041",
    "HA-CAND-ES-VALUE-0051",
    "HA-CAND-ES-VALUE-0053",
    "HA-CAND-ES-VALUE-0054",
    "HA-CAND-ES-VALUE-0055",
    "HA-CAND-ES-VALUE-0056",
}

REVISIONS = {
    "HA-CAND-ES-ATMOSPHERE-0030": "Se notaba por el ruido que la sala estaba cada vez más concurrida a medida que llegaba la gente para cenar.",
    "MAMS-CAND-ES-ATMOSPHERE-000544": "Sonaban al mismo tiempo música house de la radio y una banda en directo.",
    "MAMS-CAND-ES-ATMOSPHERE-000556": "Una decoración nostálgica de estilo refugio de montaña adorna el comedor delantero y la barra, más concurridos; la sala trasera, con luz tenue y para no fumadores, tiene vidrieras y sillas modernas de estilo Mission.",
    "HA-CAND-ES-SERVICE-0049": "Primero llegaron los entrantes y, después de retirar los platos, trajeron los platos principales.",
    "HA-CAND-ES-VALUE-0053": "La carta estaba en el rango de precios habitual que suelo ver en restaurantes informales de marisco.",
    "HA-CAND-ES-VALUE-0056": "El precio de la cena estuvo dentro del rango que esperaba para una cena de tres platos en un restaurante de precio medio.",
}

FINAL_BATCH_IDS = [
    "HA-CAND-ES-ATMOSPHERE-0033",
    "MAMS-CAND-ES-ATMOSPHERE-001960",
    "MAMS-CAND-ES-ATMOSPHERE-002555",
    "HA-CAND-ES-SERVICE-0061",
    "HA-CAND-ES-SERVICE-0062",
    "HA-CAND-ES-SERVICE-0063",
    "HA-CAND-ES-SERVICE-0065",
    "HA-CAND-ES-VALUE-0058",
    "HA-CAND-ES-VALUE-0059",
    "HA-CAND-ES-VALUE-0060",
]

FINAL_TRANSLATION_REVISIONS = {
    "HA-CAND-ES-ATMOSPHERE-0033": "En la zona de comedor hay bancos corridos junto a una pared y pequeñas mesas para dos.",
    "MAMS-CAND-ES-ATMOSPHERE-001960": "Hay una barra larga con espacio suficiente para mucha gente y pantallas grandes en las paredes para ver partidos, aunque no dominan el ambiente. En la parte delantera hay una amplia zona de estar con varios bancos y una chimenea en funcionamiento.",
    "HA-CAND-ES-SERVICE-0061": "Retiraron los platos de los entrantes antes de traer los platos principales.",
    "HA-CAND-ES-SERVICE-0063": "El camarero tomó primero el pedido de bebidas, volvió para anotar el pedido de comida y trajo la cuenta cuando se la pedimos.",
    "HA-CAND-ES-VALUE-0058": "Según las cartas expuestas en los restaurantes cercanos, los precios del menú de mediodía estaban alrededor de la media de la zona.",
    "HA-CAND-ES-VALUE-0059": "Este plato de fideos costó más o menos lo que suelo pagar por el mismo tipo de plato en otros sitios.",
    "HA-CAND-ES-VALUE-0060": "Los precios a la carta estaban en línea con los de restaurantes similares del barrio.",
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
    overlaps = existing_es_sentences()
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
        "Verified ES coverage:",
        *[f"{aspect}: {coverage[aspect]}" for aspect in coverage],
        "Remaining ES gap:",
        *[f"{aspect}: {remaining[aspect]}" for aspect in ("atmosphere", "service", "value")],
        "Final ES review batch:",
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
