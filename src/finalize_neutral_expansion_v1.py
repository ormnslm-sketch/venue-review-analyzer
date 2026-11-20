"""Finalize ES review, build balanced neutral expansion v1, and audit it."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURATION = ROOT / "data" / "curation"
CLEANED = ROOT / "data" / "cleaned"
OUTPUT = ROOT / "data" / "expanded_neutral_v1"
ANALYSIS = ROOT / "data" / "analysis"

TR_PATH = CURATION / "translated_neutral_candidates_tr.csv"
ES_PATH = CURATION / "translated_neutral_candidates_es.csv"
ES_FINAL_BATCH_PATH = CURATION / "review_batches" / "neutral_es_translation_final_batch.csv"
MAMS_EN_PATH = CURATION / "mams_neutral_candidates_en.csv"
HA_EN_PATH = CURATION / "human_authored_neutral_candidates_en.csv"
AUDIT_PATH = ANALYSIS / "neutral_expansion_v1_audit.txt"
ES_SUMMARY_PATH = ANALYSIS / "es_neutral_expansion_v1_summary.txt"

ASPECTS = ("taste", "atmosphere", "service", "value")
LANGUAGES = ("tr", "en", "es")
REQUIRED_GROUPS = {"taste": 10, "atmosphere": 18, "service": 22, "value": 19}

ES_FINAL_IDS = {
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
}

ES_FINAL_REVISIONS = {
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
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def append_note(existing: str, token: str) -> str:
    parts = [part for part in existing.split(";") if part and part != token]
    parts.append(token)
    return ";".join(parts)


def remove_note(existing: str, token: str) -> str:
    return ";".join(part for part in existing.split(";") if part and part != token)


def apply_es_final(path: Path) -> None:
    fields, rows = read_csv(path)
    by_id = {row["candidate_id"]: row for row in rows}
    missing = ES_FINAL_IDS - set(by_id)
    if missing:
        raise ValueError(f"Missing final ES IDs in {path}: {sorted(missing)}")
    for candidate_id in ES_FINAL_IDS:
        row = by_id[candidate_id]
        if candidate_id in ES_FINAL_REVISIONS:
            row["translated_sentence"] = ES_FINAL_REVISIONS[candidate_id]
            row["review_notes"] = "REVISED_BEFORE_ACCEPT;HUMAN_VERIFIED_TRANSLATION"
        else:
            row["review_notes"] = "HUMAN_VERIFIED_TRANSLATION"
        row["human_decision"] = "ACCEPT"
        row["translation_human_verified"] = "true"
        row["human_final_aspect"] = row["mapped_aspect"]
        row["human_final_sentiment"] = "neutral"
    atomic_write_csv(path, fields, rows)


def verified_translation_rows(path: Path, language: str) -> dict[str, dict[str, str]]:
    _, rows = read_csv(path)
    verified = {}
    for row in rows:
        if (
            row["language"] == language
            and row["human_decision"] == "ACCEPT"
            and row["translation_human_verified"].lower() == "true"
            and row["human_final_sentiment"] == "neutral"
            and row["human_final_aspect"] == row["mapped_aspect"]
        ):
            group = row["parallel_source_group_id"]
            if group in verified:
                raise ValueError(f"Duplicate verified group for {language}: {group}")
            verified[group] = row
    return verified


def verified_english_sources() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    source_kind: dict[str, str] = {}

    _, mams_rows = read_csv(MAMS_EN_PATH)
    for row in mams_rows:
        if (
            row["human_decision"] == "ACCEPT"
            and row["human_final_sentiment"] == "neutral"
            and row["human_final_aspect"] == row["mapped_aspect"]
        ):
            sources[row["candidate_id"]] = row
            source_kind[row["candidate_id"]] = "mams"

    _, ha_rows = read_csv(HA_EN_PATH)
    for row in ha_rows:
        if (
            row["human_decision"] == "ACCEPT"
            and row["human_verified"].lower() == "true"
            and row["human_final_sentiment"] == "neutral"
            and row["human_final_aspect"] == row["mapped_aspect"]
        ):
            sources[row["candidate_id"]] = row
            source_kind[row["candidate_id"]] = "human_authored"
    return sources, source_kind


def mark_english_reserve(selected_source_ids: set[str]) -> list[str]:
    reserve: list[str] = []
    for path, require_human_verified in ((MAMS_EN_PATH, False), (HA_EN_PATH, True)):
        fields, rows = read_csv(path)
        for row in rows:
            verified = (
                row["human_decision"] == "ACCEPT"
                and row["human_final_sentiment"] == "neutral"
                and row["human_final_aspect"] == row["mapped_aspect"]
                and (
                    not require_human_verified
                    or row.get("human_verified", "").lower() == "true"
                )
            )
            if not verified:
                continue
            if row["candidate_id"] in selected_source_ids:
                row["review_notes"] = remove_note(
                    row["review_notes"], "reserve_for_future_experiments"
                )
            else:
                row["review_notes"] = append_note(
                    row["review_notes"], "reserve_for_future_experiments"
                )
                reserve.append(row["candidate_id"])
        atomic_write_csv(path, fields, rows)
    return sorted(reserve)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_count(rows: list[dict[str, str]]) -> int:
    key_counts = Counter(
        (row["sentence"], row["language"], row["aspect"], row["sentiment"])
        for row in rows
    )
    return sum(count - 1 for count in key_counts.values() if count > 1)


def conflict_count(rows: list[dict[str, str]]) -> int:
    labels: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        labels[(row["sentence"], row["language"], row["aspect"])].add(
            row["sentiment"]
        )
    return sum(len(values) > 1 for values in labels.values())


def input_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {(row["sentence"], row["language"], row["aspect"]) for row in rows}


def main() -> None:
    apply_es_final(ES_PATH)
    apply_es_final(ES_FINAL_BATCH_PATH)

    tr_by_group = verified_translation_rows(TR_PATH, "tr")
    es_by_group = verified_translation_rows(ES_PATH, "es")
    english_sources, english_kinds = verified_english_sources()

    common_groups = sorted(set(tr_by_group) & set(es_by_group))
    groups_by_aspect: dict[str, list[str]] = {aspect: [] for aspect in ASPECTS}
    for group in common_groups:
        tr_row = tr_by_group[group]
        es_row = es_by_group[group]
        if tr_row["mapped_aspect"] != es_row["mapped_aspect"]:
            raise ValueError(f"Cross-language aspect mismatch: {group}")
        if tr_row["english_source_id"] != es_row["english_source_id"]:
            raise ValueError(f"Cross-language source mismatch: {group}")
        if tr_row["english_source_id"] not in english_sources:
            raise ValueError(f"English source is not verified: {group}")
        groups_by_aspect[tr_row["mapped_aspect"]].append(group)

    actual_common_counts = {
        aspect: len(groups_by_aspect[aspect]) for aspect in ASPECTS
    }
    if actual_common_counts != REQUIRED_GROUPS:
        raise ValueError(
            "Verified parallel groups do not match the required matrix: "
            f"actual={actual_common_counts}, required={REQUIRED_GROUPS}"
        )
    selected_groups = [group for aspect in ASPECTS for group in groups_by_aspect[aspect]]
    selected_source_ids = {
        tr_by_group[group]["english_source_id"] for group in selected_groups
    }
    reserve_ids = mark_english_reserve(selected_source_ids)

    base_fields, base_train = read_csv(CLEANED / "train.csv")
    _, base_dev = read_csv(CLEANED / "dev.csv")
    _, base_test = read_csv(CLEANED / "test.csv")
    required_training_fields = {
        "sentence",
        "language",
        "split",
        "aspect",
        "sentiment",
        "source_category",
        "aspect_term",
    }
    if not required_training_fields.issubset(base_fields):
        raise ValueError(f"Clean train schema missing fields: {required_training_fields - set(base_fields)}")

    added_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    for group in selected_groups:
        tr_row = tr_by_group[group]
        es_row = es_by_group[group]
        english_id = tr_row["english_source_id"]
        en_row = english_sources[english_id]
        aspect = tr_row["mapped_aspect"]
        kind = english_kinds[english_id]
        if en_row["human_final_aspect"] != aspect:
            raise ValueError(f"English aspect mismatch: {group}")

        if kind == "mams":
            source_category = en_row["source_category"]
            source_type = "mams_gold_annotation"
            source_dataset = "justinsiow/mams-acsa"
            source_license = en_row["source_license"]
            english_provenance = "human-reviewed MAMS gold English annotation"
        else:
            source_category = f"human_authored_{aspect}"
            source_type = en_row["source_type"]
            source_dataset = "human_authored_neutral_candidates_en"
            source_license = "not_applicable_project_authored"
            english_provenance = "human-authored English candidate; human-verified"

        language_records = (
            (
                "en",
                english_id,
                en_row["sentence"],
                "not_applicable",
                english_provenance,
            ),
            (
                "tr",
                tr_row["candidate_id"],
                tr_row["translated_sentence"],
                "true",
                "human-verified TR translation of a human-verified English source",
            ),
            (
                "es",
                es_row["candidate_id"],
                es_row["translated_sentence"],
                "true",
                "human-verified ES translation of a human-verified English source",
            ),
        )
        for language, candidate_id, sentence, translation_verified, provenance in language_records:
            training_row = {field: "" for field in base_fields}
            training_row.update(
                {
                    "sentence": sentence,
                    "language": language,
                    "split": "train",
                    "aspect": aspect,
                    "sentiment": "neutral",
                    "source_category": source_category,
                    "aspect_term": "",
                }
            )
            if "source_row_id" in training_row:
                training_row["source_row_id"] = f"expanded_neutral_v1:{candidate_id}"
            added_rows.append(training_row)
            manifest_rows.append(
                {
                    "sentence": sentence,
                    "language": language,
                    "split": "train",
                    "aspect": aspect,
                    "sentiment": "neutral",
                    "candidate_id": candidate_id,
                    "parallel_source_group_id": group,
                    "source_type": source_type,
                    "original_source_dataset": source_dataset,
                    "original_source_license": source_license,
                    "human_verified": "true",
                    "translation_human_verified": translation_verified,
                    "provenance": provenance,
                }
            )

    added_matrix = Counter((row["language"], row["aspect"]) for row in added_rows)
    required_additions = {
        (language, aspect): REQUIRED_GROUPS[aspect]
        for language in LANGUAGES
        for aspect in ASPECTS
    }
    if len(added_rows) != 207 or added_matrix != Counter(required_additions):
        raise ValueError(
            f"Added-row matrix mismatch: rows={len(added_rows)}, matrix={dict(added_matrix)}"
        )

    final_train = base_train + added_rows
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(OUTPUT / "train.csv", base_fields, final_train)
    shutil.copyfile(CLEANED / "dev.csv", OUTPUT / "dev.csv")
    shutil.copyfile(CLEANED / "test.csv", OUTPUT / "test.csv")
    manifest_fields = [
        "sentence",
        "language",
        "split",
        "aspect",
        "sentiment",
        "candidate_id",
        "parallel_source_group_id",
        "source_type",
        "original_source_dataset",
        "original_source_license",
        "human_verified",
        "translation_human_verified",
        "provenance",
    ]
    atomic_write_csv(OUTPUT / "added_neutral_manifest.csv", manifest_fields, manifest_rows)

    _, out_train = read_csv(OUTPUT / "train.csv")
    _, out_dev = read_csv(OUTPUT / "dev.csv")
    _, out_test = read_csv(OUTPUT / "test.csv")
    split_rows = {"train": out_train, "dev": out_dev, "test": out_test}

    duplicates = {split: duplicate_count(rows) for split, rows in split_rows.items()}
    conflicts = {split: conflict_count(rows) for split, rows in split_rows.items()}
    split_keys = {split: input_keys(rows) for split, rows in split_rows.items()}
    leakage = {
        "train-dev": len(split_keys["train"] & split_keys["dev"]),
        "train-test": len(split_keys["train"] & split_keys["test"]),
        "dev-test": len(split_keys["dev"] & split_keys["test"]),
    }

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        group_splits[row["parallel_source_group_id"]].add(row["split"])
    group_leakage = sum(len(splits) > 1 for splits in group_splits.values())

    sentiment_counts = {
        split: Counter(row["sentiment"] for row in rows)
        for split, rows in split_rows.items()
    }
    neutral_matrix = Counter(
        (row["language"], row["aspect"])
        for row in out_train
        if row["sentiment"] == "neutral"
    )
    expected_neutral_matrix = {
        (language, aspect): 25 for language in LANGUAGES for aspect in ASPECTS
    }

    provenance_by_type = Counter(row["source_type"] for row in manifest_rows)
    provenance_by_dataset = Counter(
        row["original_source_dataset"] for row in manifest_rows
    )
    provenance_by_language = Counter(row["language"] for row in manifest_rows)
    provenance_by_aspect = Counter(row["aspect"] for row in manifest_rows)
    unverified_rows = sum(
        row["human_verified"].lower() != "true"
        or (
            row["language"] in {"tr", "es"}
            and row["translation_human_verified"].lower() != "true"
        )
        or (
            row["language"] == "en"
            and row["translation_human_verified"] != "not_applicable"
        )
        for row in manifest_rows
    )

    dev_unchanged = hash_file(CLEANED / "dev.csv") == hash_file(OUTPUT / "dev.csv")
    test_unchanged = hash_file(CLEANED / "test.csv") == hash_file(OUTPUT / "test.csv")
    expected_sentiments = Counter({"positive": 2173, "neutral": 300, "negative": 1050})
    audit_pass = all(
        (
            len(base_train) == 3316,
            len(added_rows) == 207,
            len(out_train) == 3523,
            sentiment_counts["train"] == expected_sentiments,
            neutral_matrix == Counter(expected_neutral_matrix),
            all(value == 0 for value in duplicates.values()),
            all(value == 0 for value in conflicts.values()),
            all(value == 0 for value in leakage.values()),
            group_leakage == 0,
            unverified_rows == 0,
            dev_unchanged,
            test_unchanged,
            len(selected_groups) == 69,
            len(reserve_ids) == 8,
        )
    )

    group_counts = Counter(tr_by_group[group]["mapped_aspect"] for group in selected_groups)
    report = [
        "NEUTRAL EXPANSION V1 AUDIT",
        "",
        "ES_NEUTRAL_EXPANSION_V1 = COMPLETE",
        "",
        f"Base train rows: {len(base_train)}",
        f"Added rows: {len(added_rows)}",
        f"Final train rows: {len(out_train)}",
        "",
        "Sentiments (train):",
        f"positive: {sentiment_counts['train']['positive']}",
        f"neutral: {sentiment_counts['train']['neutral']}",
        f"negative: {sentiment_counts['train']['negative']}",
        "",
        "Sentiment distribution by split:",
        *[
            f"{split}: positive={sentiment_counts[split]['positive']}, "
            f"neutral={sentiment_counts[split]['neutral']}, "
            f"negative={sentiment_counts[split]['negative']}, total={len(split_rows[split])}"
            for split in ("train", "dev", "test")
        ],
        "",
        "Neutral matrix by language/aspect:",
        "language,taste,atmosphere,service,value,total",
        *[
            f"{language},{neutral_matrix[(language, 'taste')]},"
            f"{neutral_matrix[(language, 'atmosphere')]},"
            f"{neutral_matrix[(language, 'service')]},"
            f"{neutral_matrix[(language, 'value')]},"
            f"{sum(neutral_matrix[(language, aspect)] for aspect in ASPECTS)}"
            for language in LANGUAGES
        ],
        "",
        "Selected semantic groups:",
        *[f"{aspect}: {group_counts[aspect]}" for aspect in ASPECTS],
        f"total: {len(selected_groups)}",
        "",
        f"English verified reserve count: {len(reserve_ids)}",
        "English reserve IDs:",
        *[f"- {candidate_id}" for candidate_id in reserve_ids],
        "",
        "Provenance counts by source_type:",
        *[f"{key}: {value}" for key, value in sorted(provenance_by_type.items())],
        "Provenance counts by original_source_dataset:",
        *[f"{key}: {value}" for key, value in sorted(provenance_by_dataset.items())],
        "Provenance counts by language:",
        *[f"{key}: {value}" for key, value in sorted(provenance_by_language.items())],
        "Provenance counts by aspect:",
        *[f"{key}: {value}" for key, value in sorted(provenance_by_aspect.items())],
        "",
        "Duplicates:",
        *[f"{split}: {duplicates[split]}" for split in ("train", "dev", "test")],
        "Conflicts:",
        *[f"{split}: {conflicts[split]}" for split in ("train", "dev", "test")],
        f"Train-dev leakage: {leakage['train-dev']}",
        f"Train-test leakage: {leakage['train-test']}",
        f"Dev-test leakage: {leakage['dev-test']}",
        f"Parallel-group split leakage: {group_leakage}",
        f"Unverified added rows: {unverified_rows}",
        f"Dev unchanged: {dev_unchanged}",
        f"Test unchanged: {test_unchanged}",
        "",
        "Limitation:",
        "The original M-ABSA rows do not contain a reliable language-independent "
        "source_example_id, so zero cross-language semantic leakage cannot be claimed "
        "for the original dataset. For the 207 newly added rows, "
        "parallel_source_group_id confirms that every EN/TR/ES version is assigned only to train.",
        "",
        f"Overall audit: {'PASS' if audit_pass else 'FAIL'}",
    ]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    es_final = Counter()
    for row in out_train:
        if row["language"] == "es" and row["sentiment"] == "neutral":
            es_final[row["aspect"]] += 1
    es_summary = [
        "ES_NEUTRAL_EXPANSION_V1 = COMPLETE",
        *[f"{aspect}: {es_final[aspect]}" for aspect in ASPECTS],
        "No automatically accepted translations were added.",
        "The cleaned training dataset was not overwritten.",
    ]
    ES_SUMMARY_PATH.write_text("\n".join(es_summary) + "\n", encoding="utf-8")

    print("\n".join(report))
    if not audit_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
