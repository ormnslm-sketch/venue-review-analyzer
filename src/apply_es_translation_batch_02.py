"""Apply ES translation Batch 2 decisions and prepare diverse Batch 3."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_DIR = PROJECT_ROOT / "data" / "curation"
MASTER_PATH = CURATION_DIR / "translated_neutral_candidates_es.csv"
BATCH2_PATH = CURATION_DIR / "review_batches" / "neutral_es_translation_batch_02.csv"
BATCH3_PATH = CURATION_DIR / "review_batches" / "neutral_es_translation_batch_03.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "analysis" / "es_translation_batch_02_summary.txt"

BATCH2_IDS = {
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
}

REVISIONS = {
    "HA-CAND-ES-SERVICE-0045": "Nos presentamos en la entrada y, tras confirmar el número de personas, nos acompañaron a la mesa.",
    "HA-CAND-ES-SERVICE-0052": "Pedimos la cuenta y pagamos en la mesa con el datáfono.",
    "HA-CAND-ES-SERVICE-0059": "Di el número de recogida en el mostrador lateral y recogí la bolsa cuando llamaron mi número.",
    "HA-CAND-ES-VALUE-0020": "El precio de este plato de fideos está dentro del rango habitual para este tipo de plato.",
    "HA-CAND-ES-VALUE-0043": "Comparamos las cartas de varios restaurantes cercanos y los menús de almuerzo estaban en un rango de precios parecido.",
}

BATCH3_IDS = [
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
]

BATCH3_TRANSLATION_REVISIONS = {
    "MAMS-CAND-ES-TASTE-001428": "Espera comida normal, una cuenta elevada y un trato poco agradable por parte del servicio.",
    "HA-CAND-ES-ATMOSPHERE-0030": "Se notaba por el ruido que la sala estaba cada vez más concurrida a medida que llegaba la gente para cenar.",
    "MAMS-CAND-ES-ATMOSPHERE-000544": "Sonaban al mismo tiempo música house de la radio y una banda en directo.",
    "MAMS-CAND-ES-ATMOSPHERE-000556": "Una decoración nostálgica de estilo refugio de montaña adorna el comedor delantero y la barra, más concurridos; la sala trasera, con luz tenue y para no fumadores, tiene vidrieras y sillas modernas de estilo Mission.",
    "HA-CAND-ES-SERVICE-0049": "Primero llegaron los entrantes y, después de retirar los platos, trajeron los platos principales.",
    "HA-CAND-ES-SERVICE-0060": "Apuntamos nuestro nombre en la lista de la entrada y esperamos a que el personal de recepción nos llamara.",
    "HA-CAND-ES-SERVICE-0064": "Acerqué la tarjeta al datáfono y el recibo se imprimió en la mesa.",
    "HA-CAND-ES-VALUE-0041": "Pagué más o menos lo que suelo pagar por un plato de fideos de este tipo.",
    "HA-CAND-ES-VALUE-0051": "Los precios de los platos principales estaban alrededor de la media local de esa zona.",
    "HA-CAND-ES-VALUE-0053": "La carta estaba en el rango de precios habitual que suelo ver en restaurantes informales de marisco.",
    "HA-CAND-ES-VALUE-0054": "Los precios eran comparables a los de otros menús similares que vimos cerca.",
    "HA-CAND-ES-VALUE-0056": "El precio de la cena estuvo dentro del rango que esperaba para una cena de tres platos en un restaurante de precio medio.",
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


def apply_batch2(path: Path) -> None:
    fields, rows = read_csv(path)
    by_id = {row["candidate_id"]: row for row in rows}
    missing = BATCH2_IDS - set(by_id)
    if missing:
        raise ValueError(f"Missing Batch 2 IDs in {path}: {sorted(missing)}")
    for candidate_id in BATCH2_IDS:
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
    apply_batch2(MASTER_PATH)
    apply_batch2(BATCH2_PATH)

    fields, master_rows = read_csv(MASTER_PATH)
    by_id = {row["candidate_id"]: row for row in master_rows}
    missing = set(BATCH3_IDS) - set(by_id)
    if missing:
        raise ValueError(f"Batch 3 IDs missing from master: {sorted(missing)}")
    for candidate_id, revised_sentence in BATCH3_TRANSLATION_REVISIONS.items():
        by_id[candidate_id]["translated_sentence"] = revised_sentence
    atomic_write(MASTER_PATH, fields, master_rows)

    verified_sources = verified_english_source_ids()
    overlaps = existing_es_sentences()
    batch3 = [by_id[candidate_id].copy() for candidate_id in BATCH3_IDS]
    for row in batch3:
        if row["human_decision"]:
            raise ValueError(f"Already reviewed Batch 3 candidate: {row['candidate_id']}")
        if row["english_source_id"] not in verified_sources:
            raise ValueError(f"Unverified English source: {row['english_source_id']}")
        if row["translated_sentence"].strip() in overlaps:
            raise ValueError(f"Existing split overlap: {row['candidate_id']}")

    expected = Counter({"taste": 3, "atmosphere": 5, "service": 6, "value": 6})
    counts = Counter(row["mapped_aspect"] for row in batch3)
    if counts != expected:
        raise ValueError(f"Batch 3 quotas invalid: {dict(counts)}")
    if len({row["parallel_source_group_id"] for row in batch3}) != len(batch3):
        raise ValueError("Batch 3 contains duplicate semantic source groups")
    atomic_write(BATCH3_PATH, fields, batch3)

    coverage = calculate_coverage(master_rows)
    remaining = {aspect: max(0, 25 - count) for aspect, count in coverage.items()}
    summary = [
        "Verified ES coverage:",
        *[f"{aspect}: {coverage[aspect]}" for aspect in coverage],
        "Remaining ES gap:",
        *[f"{aspect}: {remaining[aspect]}" for aspect in remaining],
        "ES Batch 3:",
        *[f"{aspect}: {counts[aspect]}" for aspect in coverage],
        f"total: {len(batch3)}",
        "Batch 3 human decisions assigned: 0",
        "training rows added: 0",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
