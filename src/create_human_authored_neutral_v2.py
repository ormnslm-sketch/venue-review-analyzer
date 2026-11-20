"""Append natural-style English neutral candidates and create Review Batch 2."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = PROJECT_ROOT / "data" / "curation" / "human_authored_neutral_candidates_en.csv"
DIAGNOSTICS_PATH = (
    PROJECT_ROOT / "data" / "curation" / "human_authored_neutral_candidates_en_v2_diagnostics.csv"
)
BATCH_PATH = (
    PROJECT_ROOT
    / "data"
    / "curation"
    / "review_batches"
    / "human_authored_neutral_en_batch_02.csv"
)
MARKDOWN_PATH = BATCH_PATH.with_suffix(".md")
SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "analysis" / "human_authored_neutral_en_batch_02_summary.txt"
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

DIAGNOSTIC_FIELDS = [
    "candidate_id",
    "semantic_subtopic",
    "style_type",
    "exact_duplicate",
    "high_lexical_overlap",
    "repeated_opening_phrase",
    "template_similarity_warning",
    "diagnostic_notes",
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

# (semantic_subtopic, style_type, sentence)
ATMOSPHERE = [
    ("lighting", "first_person", "I noticed the lights were kept fairly low from the time we sat down until dessert."),
    ("noise_level", "first_person", "We could hear the room getting busier as the dinner crowd came in."),
    ("background_music", "first_person", "The music stayed at about the same level throughout our meal."),
    ("crowd_level", "third_person", "By seven, most of the tables were occupied, with a few seats still open near the bar."),
    ("seating_layout", "third_person", "The dining area mixes wall booths with small tables for two."),
    ("room_layout", "multi_aspect", "Our drinks came from the bar, and the dining room wrapped around an open kitchen."),
    ("indoor_outdoor", "multi_aspect", "The menu was the same in both sections, with indoor tables at the front and patio seating behind the building."),
    ("temperature", "third_person", "Ceiling fans ran above the dining area during the evening."),
]

SERVICE = [
    ("order_taking", "first_person", "We looked over the menu for a few minutes, then gave our drink and food orders together."),
    ("order_taking", "third_person", "At each table, drink orders are taken before the food order is entered."),
    ("seating_process", "first_person", "We checked in at the front and were shown to our table after the party size was confirmed."),
    ("reservation_process", "first_person", "I gave my name at the host stand, and they matched it to the reservation before seating us."),
    ("normal_wait", "first_person", "We waited around ten minutes before being shown to our table."),
    ("normal_wait", "multi_aspect", "The room was already half full, and we waited about twelve minutes from check-in to seating."),
    ("food_delivery", "first_person", "Our starters came first, and the main dishes followed after the plates were cleared."),
    ("food_delivery", "multi_aspect", "The pasta was well seasoned, and the other main dishes arrived in the next round from the kitchen."),
    ("drink_delivery", "third_person", "Water is set down first, with the ordered drinks brought afterward."),
    ("payment_check", "first_person", "We asked for the check and paid at the table with the card terminal."),
    ("payment_check", "third_person", "The receipt is printed at the counter after payment is completed."),
    ("self_service", "first_person", "We picked up our own cutlery and returned the trays to the collection area."),
    ("counter_ordering", "first_person", "I ordered at the counter, took a table number, and waited for the number to be called."),
    ("takeaway_pickup", "third_person", "Pickup customers give their order number at a separate desk beside the entrance."),
    ("course_timing", "multi_aspect", "The dessert was rich, while coffee was brought after the dessert plates were removed."),
    ("queue_procedure", "first_person", "We added our name to the list at the entrance and waited for it to be called."),
]

VALUE = [
    ("nearby_price_comparison", "first_person", "The prices looked pretty similar to the other restaurants we tried nearby."),
    ("nearby_price_comparison", "multi_aspect", "The curry had plenty of spice, and its price was in line with similar dishes along the same street."),
    ("area_average_price", "first_person", "From what I saw on nearby menus, the main courses sat around the area average."),
    ("area_average_price", "third_person", "Dinner prices fall within the range commonly listed by restaurants in this neighborhood."),
    ("expected_price_range", "first_person", "The set menu was in the price range I expected for a three-course dinner."),
    ("expected_price_range", "multi_aspect", "Service followed a counter-ordering setup, and the prices matched the range I expected for that format."),
    ("typical_category_price", "first_person", "I paid about what I normally see for this type of noodle bowl."),
    ("typical_category_price", "third_person", "The coffee is priced within the usual range for specialty cafés in the area."),
    ("comparable_menu_price", "first_person", "We compared a few menus nearby, and the lunch sets were in a similar price range."),
    ("standard_menu_pricing", "third_person", "The menu uses separate standard price bands for starters, mains, and desserts."),
    ("lunch_dinner_pricing", "first_person", "I noticed that lunch and dinner had separate price lists for the same dining room."),
    ("lunch_dinner_pricing", "multi_aspect", "Reservations were handled online, and the lunch and dinner prices appeared on different menu pages."),
    ("portion_price_context", "first_person", "The menu showed small and large portions with a separate price for each size."),
    ("fixed_price_menu", "first_person", "We chose the fixed menu, which listed one set price for a starter, main course, and dessert."),
    ("usual_price_range", "multi_aspect", "The room used shared tables, while the menu prices stayed in the usual range for this kind of restaurant."),
]

REVIEW_QUOTAS = {"atmosphere": 6, "service": 9, "value": 9}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def opening_phrase(text: str, word_count: int = 4) -> str:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return " ".join(words[:word_count])


def read_pool() -> list[dict[str, str]]:
    with POOL_PATH.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_new_rows(
    existing_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], int]:
    existing_texts = {normalize(row["sentence"]) for row in existing_rows}
    rows: list[dict[str, str]] = []
    metadata: dict[str, dict[str, str]] = {}
    exact_removed = 0

    starts = {"atmosphere": 29, "service": 43, "value": 35}
    for aspect, candidates in (
        ("atmosphere", ATMOSPHERE),
        ("service", SERVICE),
        ("value", VALUE),
    ):
        for offset, (subtopic, style_type, sentence) in enumerate(candidates):
            number = starts[aspect] + offset
            normalized = normalize(sentence)
            if normalized in existing_texts:
                exact_removed += 1
                continue
            existing_texts.add(normalized)
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
                    "generation_batch": "human_authored_v2",
                    "human_decision": "",
                    "human_final_aspect": "",
                    "human_final_sentiment": "",
                    "human_verified": "false",
                    "review_notes": "",
                }
            )
            metadata[candidate_id] = {
                "semantic_subtopic": subtopic,
                "style_type": style_type,
            }
    return rows, metadata, exact_removed


def create_diagnostics(
    existing_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
    metadata: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], int, int, int]:
    all_comparison_rows = existing_rows + new_rows
    opening_counts = Counter(opening_phrase(row["sentence"]) for row in new_rows)
    diagnostics: list[dict[str, str]] = []
    high_overlap_count = 0
    repeated_opening_count = 0
    warning_count = 0

    for row in new_rows:
        normalized = normalize(row["sentence"])
        high_overlap = False
        closest_ratio = 0.0
        for other in all_comparison_rows:
            if other["candidate_id"] == row["candidate_id"]:
                continue
            ratio = SequenceMatcher(None, normalized, normalize(other["sentence"])).ratio()
            closest_ratio = max(closest_ratio, ratio)
            if ratio >= 0.84:
                high_overlap = True
        opening = opening_phrase(row["sentence"])
        repeated_opening = opening_counts[opening] > 1
        warning = high_overlap or repeated_opening
        high_overlap_count += high_overlap
        repeated_opening_count += repeated_opening
        warning_count += warning
        notes: list[str] = []
        if high_overlap:
            notes.append(f"HIGH_LEXICAL_OVERLAP_MAX_RATIO={closest_ratio:.3f}")
        if repeated_opening:
            notes.append(f"REPEATED_OPENING={opening}")
        info = metadata[row["candidate_id"]]
        diagnostics.append(
            {
                "candidate_id": row["candidate_id"],
                "semantic_subtopic": info["semantic_subtopic"],
                "style_type": info["style_type"],
                "exact_duplicate": "false",
                "high_lexical_overlap": str(high_overlap).lower(),
                "repeated_opening_phrase": str(repeated_opening).lower(),
                "template_similarity_warning": str(warning).lower(),
                "diagnostic_notes": ";".join(notes),
            }
        )
    return diagnostics, high_overlap_count, repeated_opening_count, warning_count


def select_batch(
    rows: list[dict[str, str]],
    metadata: dict[str, dict[str, str]],
    diagnostics: list[dict[str, str]],
) -> list[dict[str, str]]:
    warning_by_id = {
        row["candidate_id"]: row["template_similarity_warning"] == "true"
        for row in diagnostics
    }
    selected: list[dict[str, str]] = []
    for aspect, quota in REVIEW_QUOTAS.items():
        aspect_rows = [row for row in rows if row["mapped_aspect"] == aspect]
        aspect_rows.sort(key=lambda row: warning_by_id[row["candidate_id"]])
        used_subtopics: set[str] = set()
        for row in aspect_rows:
            subtopic = metadata[row["candidate_id"]]["semantic_subtopic"]
            if subtopic in used_subtopics:
                continue
            selected.append(row)
            used_subtopics.add(subtopic)
            if len(used_subtopics) == quota:
                break
        if len(used_subtopics) != quota:
            raise ValueError(f"Could not fill diverse review quota for {aspect}")
    random.Random(42).shuffle(selected)
    return selected


def write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    *,
    preserve_reviewed_pool: bool = False,
) -> None:
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            existing = list(csv.DictReader(file))
        if (
            not preserve_reviewed_pool
            and any(row.get("human_decision", "").strip() for row in existing)
        ):
            raise FileExistsError(f"Refusing to overwrite human-reviewed file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calculate_coverage(all_human_rows: list[dict[str, str]]) -> dict[str, int]:
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

    human_authored = Counter(
        row["human_final_aspect"]
        for row in all_human_rows
        if row["human_decision"] == "ACCEPT" and row["human_verified"] == "true"
    )
    return {
        aspect: original[aspect] + mams[aspect] + human_authored[aspect]
        for aspect in ("taste", "atmosphere", "service", "value")
    }


def main() -> None:
    existing_rows = read_pool()
    # Safe reruns may replace unreviewed v2 rows, but never reviewed ones.
    old_v2 = [row for row in existing_rows if row["generation_batch"] == "human_authored_v2"]
    if any(row["human_decision"] for row in old_v2):
        raise FileExistsError("Refusing to replace reviewed human_authored_v2 rows")
    existing_rows = [row for row in existing_rows if row["generation_batch"] != "human_authored_v2"]

    new_rows, metadata, exact_removed = build_new_rows(existing_rows)
    diagnostics, overlap_count, opening_count, warning_count = create_diagnostics(
        existing_rows, new_rows, metadata
    )
    batch = select_batch(new_rows, metadata, diagnostics)

    counts = Counter(row["mapped_aspect"] for row in new_rows)
    styles = Counter(metadata[row["candidate_id"]]["style_type"] for row in new_rows)
    if counts != Counter({"atmosphere": 8, "service": 16, "value": 15}):
        raise ValueError(f"Generation quotas invalid: {dict(counts)}")
    if styles != Counter({"first_person": 20, "third_person": 10, "multi_aspect": 9}):
        raise ValueError(f"Style distribution invalid: {dict(styles)}")
    if Counter(row["mapped_aspect"] for row in batch) != Counter(REVIEW_QUOTAS):
        raise ValueError("Review Batch 2 quotas invalid")

    all_rows = existing_rows + new_rows
    write_csv(POOL_PATH, POOL_FIELDS, all_rows, preserve_reviewed_pool=True)
    write_csv(DIAGNOSTICS_PATH, DIAGNOSTIC_FIELDS, diagnostics)

    batch_rows: list[dict[str, str]] = []
    for row in batch:
        batch_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "source_group_id": row["source_group_id"],
                "mapped_aspect": row["mapped_aspect"],
                "semantic_subtopic": metadata[row["candidate_id"]]["semantic_subtopic"],
                "sentence": row["sentence"],
                "human_decision": "",
                "human_final_aspect": "",
                "human_final_sentiment": "",
                "human_verified": "false",
                "review_notes": "",
            }
        )
    write_csv(BATCH_PATH, BATCH_FIELDS, batch_rows)

    markdown = [
        "# Human-Authored English Neutral Review Batch 02",
        "",
        "All decisions are blank. Candidates remain synthetic_candidate until separately human-verified.",
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

    coverage = calculate_coverage(all_rows)
    remaining = {aspect: max(0, 25 - count) for aspect, count in coverage.items()}
    summary = [
        "HUMAN-AUTHORED NEUTRAL EN — NATURAL BATCH 2",
        "=============================================",
        "Verified coverage:",
        *[f"{aspect}: {coverage[aspect]}" for aspect in coverage],
        "Remaining EN gap:",
        *[f"{aspect}: {remaining[aspect]}" for aspect in remaining],
        "Generated candidates:",
        f"atmosphere: {counts['atmosphere']}",
        f"service: {counts['service']}",
        f"value: {counts['value']}",
        f"total: {len(new_rows)}",
        "Style distribution:",
        f"first_person: {styles['first_person']}",
        f"third_person: {styles['third_person']}",
        f"multi_aspect: {styles['multi_aspect']}",
        f"exact duplicates removed: {exact_removed}",
        f"high lexical overlap warnings: {overlap_count}",
        f"repeated opening warnings: {opening_count}",
        f"template similarity warnings: {warning_count}",
        f"Review Batch 2 size: {len(batch_rows)}",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
