"""Create the final English service/value neutral human-review batch."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from difflib import SequenceMatcher
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
MARKDOWN_PATH = BATCH_PATH.with_suffix(".md")
DIAGNOSTICS_PATH = BATCH_PATH.with_name("human_authored_neutral_en_final_batch_diagnostics.csv")
SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "analysis" / "human_authored_neutral_en_final_batch_summary.txt"
)

POOL_FIELDS = [
    "candidate_id",
    "source_group_id",
    "sentence",
    "language",
    "mapped_aspect",
    "proposed_sentiment",
    "source_type",
    "generation_batch",
    "human_decision",
    "human_final_aspect",
    "human_final_sentiment",
    "human_verified",
    "review_notes",
]

BATCH_FIELDS = [
    "candidate_id",
    "source_group_id",
    "mapped_aspect",
    "semantic_subtopic",
    "sentence",
    "human_decision",
    "human_final_aspect",
    "human_final_sentiment",
    "human_verified",
    "review_notes",
]

DIAGNOSTIC_FIELDS = [
    "candidate_id",
    "exact_duplicate",
    "high_lexical_overlap",
    "closest_lexical_ratio",
    "diagnostic_notes",
]

# (semantic_subtopic, sentence)
SERVICE = [
    ("takeaway_pickup", "I gave the pickup number at the side counter and collected the takeaway bag when it was called."),
    ("queue_procedure", "We wrote our name on the walk-in list and waited for the host to call it."),
    ("course_timing", "Our starter plates were cleared before the main courses were brought out."),
    ("normal_wait_process", "From joining the list to reaching the table, the process took roughly a quarter of an hour."),
    ("table_service_sequence", "The server took the drinks first, returned for the food order, and brought the check after we asked."),
    ("payment_process_variant", "I tapped my card on the portable reader, and the receipt was printed at the table."),
    ("counter_process_variant", "We ordered at the counter, kept the printed number, and collected the tray when that number appeared on the screen."),
]

VALUE = [
    ("nearby_price_comparison", "The prices were about the same as the other Italian restaurants we visited nearby."),
    ("local_average", "The main courses seemed to sit around the local average for that part of town."),
    ("usual_payment_comparison", "I paid about what I usually do for a meal at this kind of neighborhood restaurant."),
    ("usual_category_range", "The menu was in the usual price range I see at casual seafood places."),
    ("comparable_nearby_menus", "Prices were comparable with the similar set menus we checked nearby."),
    ("same_as_area", "Our meal came to roughly the same as at the other places we tried in the area."),
    ("expected_range", "The dinner matched the price range I expected for a mid-range three-course restaurant."),
    ("nearby_cafe_comparison", "Coffee and dessert were priced similarly to the cafés around the same square."),
    ("local_lunch_average", "Based on the menus outside nearby restaurants, the lunch prices were around the local average."),
    ("usual_dish_comparison", "The noodle bowl cost about what I normally pay for the same type of dish elsewhere."),
    ("neighborhood_menu_comparison", "The à la carte prices lined up with comparable restaurant menus in the neighborhood."),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def read_pool() -> list[dict[str, str]]:
    with POOL_PATH.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_rows(existing: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, str]]:
    existing_texts = {normalize(row["sentence"]) for row in existing}
    rows: list[dict[str, str]] = []
    subtopics: dict[str, str] = {}
    for aspect, start, candidates in (
        ("service", 59, SERVICE),
        ("value", 50, VALUE),
    ):
        for offset, (subtopic, sentence) in enumerate(candidates):
            if normalize(sentence) in existing_texts:
                raise ValueError(f"Exact duplicate candidate: {sentence}")
            existing_texts.add(normalize(sentence))
            number = start + offset
            token = aspect.upper()
            candidate_id = f"HA-CAND-EN-{token}-{number:04d}"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "source_group_id": f"HA-EN-{token}-{number:04d}",
                    "sentence": sentence,
                    "language": "en",
                    "mapped_aspect": aspect,
                    "proposed_sentiment": "neutral",
                    "source_type": "synthetic_candidate",
                    "generation_batch": "human_authored_v3_final_en",
                    "human_decision": "",
                    "human_final_aspect": "",
                    "human_final_sentiment": "",
                    "human_verified": "false",
                    "review_notes": "",
                }
            )
            subtopics[candidate_id] = subtopic
    return rows, subtopics


def diagnose(
    existing: list[dict[str, str]], new_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], int]:
    comparison = existing + new_rows
    diagnostics: list[dict[str, str]] = []
    warnings = 0
    for row in new_rows:
        closest = 0.0
        for other in comparison:
            if other["candidate_id"] == row["candidate_id"]:
                continue
            ratio = SequenceMatcher(
                None, normalize(row["sentence"]), normalize(other["sentence"])
            ).ratio()
            closest = max(closest, ratio)
        high_overlap = closest >= 0.84
        warnings += high_overlap
        diagnostics.append(
            {
                "candidate_id": row["candidate_id"],
                "exact_duplicate": "false",
                "high_lexical_overlap": str(high_overlap).lower(),
                "closest_lexical_ratio": f"{closest:.3f}",
                "diagnostic_notes": "HIGH_LEXICAL_OVERLAP" if high_overlap else "",
            }
        )
    return diagnostics, warnings


def write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    *,
    allow_preserved_reviews: bool = False,
) -> None:
    if path.exists() and not allow_preserved_reviews:
        with path.open(encoding="utf-8", newline="") as file:
            old_rows = list(csv.DictReader(file))
        if any(row.get("human_decision", "") for row in old_rows):
            raise FileExistsError(f"Refusing to overwrite reviewed file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def coverage(all_human_rows: list[dict[str, str]]) -> dict[str, int]:
    original = Counter()
    with (PROJECT_ROOT / "data" / "cleaned" / "train.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file):
            if row["language"] == "en" and row["sentiment"] == "neutral":
                original[row["aspect"]] += 1

    mams = Counter()
    with (PROJECT_ROOT / "data" / "curation" / "mams_neutral_candidates_en.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file):
            if row["human_decision"] == "ACCEPT":
                mams[row["human_final_aspect"]] += 1

    authored = Counter(
        row["human_final_aspect"]
        for row in all_human_rows
        if row["human_decision"] == "ACCEPT" and row["human_verified"] == "true"
    )
    return {
        aspect: original[aspect] + mams[aspect] + authored[aspect]
        for aspect in ("taste", "atmosphere", "service", "value")
    }


def main() -> None:
    existing = read_pool()
    old_v3 = [row for row in existing if row["generation_batch"] == "human_authored_v3_final_en"]
    if any(row["human_decision"] for row in old_v3):
        raise FileExistsError("Refusing to replace reviewed final-batch candidates")
    existing = [row for row in existing if row["generation_batch"] != "human_authored_v3_final_en"]

    new_rows, subtopics = build_rows(existing)
    diagnostics, overlap_warnings = diagnose(existing, new_rows)
    counts = Counter(row["mapped_aspect"] for row in new_rows)
    if counts != Counter({"service": 7, "value": 11}):
        raise ValueError(f"Final generation quotas invalid: {dict(counts)}")

    all_rows = existing + new_rows
    write_csv(POOL_PATH, POOL_FIELDS, all_rows, allow_preserved_reviews=True)
    write_csv(DIAGNOSTICS_PATH, DIAGNOSTIC_FIELDS, diagnostics)

    random.Random(42).shuffle(new_rows)
    batch_rows = [
        {
            "candidate_id": row["candidate_id"],
            "source_group_id": row["source_group_id"],
            "mapped_aspect": row["mapped_aspect"],
            "semantic_subtopic": subtopics[row["candidate_id"]],
            "sentence": row["sentence"],
            "human_decision": "",
            "human_final_aspect": "",
            "human_final_sentiment": "",
            "human_verified": "false",
            "review_notes": "",
        }
        for row in new_rows
    ]
    write_csv(BATCH_PATH, BATCH_FIELDS, batch_rows)

    markdown = [
        "# Human-Authored Neutral — Final English Review Batch",
        "",
        "All 18 candidates require human review. No decision has been assigned.",
        "",
    ]
    for index, row in enumerate(batch_rows, start=1):
        markdown.extend(
            [
                f"## {index}",
                "",
                f"Candidate ID: {row['candidate_id']}",
                "",
                f"Source group ID: {row['source_group_id']}",
                "",
                f"Mapped aspect: {row['mapped_aspect']}",
                "",
                f"Semantic subtopic: {row['semantic_subtopic']}",
                "",
                f"Sentence: {row['sentence']}",
                "",
            ]
        )
    MARKDOWN_PATH.write_text("\n".join(markdown), encoding="utf-8")

    verified = coverage(all_rows)
    remaining = {aspect: max(0, 25 - verified[aspect]) for aspect in verified}
    summary = [
        "FINAL ENGLISH NEUTRAL CANDIDATE BATCH",
        "=====================================",
        "Verified coverage:",
        *[f"{aspect}: {verified[aspect]}" for aspect in verified],
        "Remaining EN gap:",
        f"service: {remaining['service']}",
        f"value: {remaining['value']}",
        "Final review batch:",
        f"service candidates: {counts['service']}",
        f"value candidates: {counts['value']}",
        f"total: {len(new_rows)}",
        "exact duplicates: 0",
        f"high lexical overlap warnings: {overlap_warnings}",
        "human decisions assigned: 0",
        "training rows added: 0",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
