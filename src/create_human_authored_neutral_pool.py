"""Create English human-authored neutral candidates for human review only."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = PROJECT_ROOT / "data" / "curation" / "human_authored_neutral_candidates_en.csv"
DIAGNOSTICS_PATH = (
    PROJECT_ROOT / "data" / "curation" / "human_authored_neutral_candidates_en_diagnostics.csv"
)
BATCH_PATH = (
    PROJECT_ROOT
    / "data"
    / "curation"
    / "review_batches"
    / "human_authored_neutral_en_batch_01.csv"
)
MARKDOWN_PATH = BATCH_PATH.with_suffix(".md")
SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "analysis" / "human_authored_neutral_candidates_en_summary.txt"
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
    "is_multi_aspect",
    "possible_duplicate",
    "possible_strong_valence",
    "possible_wrong_aspect",
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

# (semantic_subtopic, is_multi_aspect, sentence)
ATMOSPHERE = [
    ("lighting", False, "The dining room uses overhead lighting during lunch."),
    ("lighting", True, "The soup arrived hot, while the dining room switched to lower lighting after sunset."),
    ("lighting", False, "Wall lamps provide the lighting along the window side of the room."),
    ("background_music", False, "Instrumental music plays in the background during dinner."),
    ("background_music", True, "Orders are placed at the counter, and recorded jazz plays through the ceiling speakers."),
    ("background_music", False, "The background playlist changes from acoustic tracks at lunch to instrumental tracks in the evening."),
    ("noise_level", False, "Conversation noise increases as more tables become occupied."),
    ("noise_level", True, "The pasta had a strong garlic flavor, while the room carried a steady level of conversation noise."),
    ("noise_level", False, "Kitchen sounds are audible from tables near the open pass."),
    ("seating_layout", False, "Booths line one wall, with two-person tables arranged in the center."),
    ("seating_layout", True, "Drinks are collected from the bar, and the dining area uses shared tables near the entrance."),
    ("seating_layout", False, "The seating plan places larger tables at the back and smaller tables by the windows."),
    ("indoor_outdoor", False, "The patio opens for service when weather permits, while the main dining room remains indoors."),
    ("indoor_outdoor", True, "Reservations cover the indoor tables, and walk-in diners may be seated on the patio."),
    ("indoor_outdoor", False, "The restaurant has an indoor room and a separate covered outdoor section."),
    ("crowd_level", False, "About half of the tables were occupied at six o'clock."),
    ("crowd_level", True, "Dessert service began as the dining room moved from half occupied to nearly full."),
    ("crowd_level", False, "The crowd builds gradually between the lunch and dinner periods."),
    ("room_size", False, "The main room contains twelve tables and a six-seat counter."),
    ("room_size", True, "The menu lists regional dishes, and the dining room seats approximately forty guests."),
    ("room_size", False, "A smaller side room is used for groups of up to ten people."),
    ("temperature", False, "Ceiling fans operate in the dining room during the afternoon."),
    ("temperature", True, "The drinks remained chilled, while the indoor temperature was maintained by ceiling vents."),
    ("table_spacing", False, "Tables are arranged in two rows with a central aisle."),
    ("table_spacing", True, "Servers use the center aisle, and the tables are spaced at regular intervals."),
    ("general_environment", False, "The open kitchen is visible from the main dining area."),
    ("general_environment", True, "Payment is handled near the exit, and the dining area extends behind the front counter."),
    ("general_environment", True, "A partition separates the entrance from the seating area, while takeaway orders are collected at the front counter."),
]

SERVICE = [
    ("order_taking", False, "A server takes the table's drink order before recording the food order."),
    ("order_taking", True, "The curry was highly seasoned, and the food order was recorded after the drinks were listed."),
    ("order_taking", False, "Each table receives a menu before the order is entered into the register."),
    ("order_taking", True, "The dining room uses paper menus, while orders are entered on a handheld device."),
    ("normal_wait", False, "We waited about ten minutes between joining the list and being seated."),
    ("normal_wait", True, "The pastries were fresh, and we waited about eight minutes for a table."),
    ("normal_wait", False, "The quoted wait for a table was approximately fifteen minutes."),
    ("normal_wait", True, "The room was half full, and the wait from check-in to seating was about twelve minutes."),
    ("seating_process", False, "The host assigns a table after confirming the party size."),
    ("seating_process", True, "The lighting was dim, and the host directed our group to a table near the back."),
    ("seating_process", False, "Walk-in parties receive a table number before being shown to their seats."),
    ("seating_process", True, "Coats were stored by the entrance before the party was taken to its table."),
    ("reservation_process", False, "The reservation was confirmed by name at the entrance."),
    ("reservation_process", True, "The tasting menu had five courses, and the reservation time was checked before seating."),
    ("reservation_process", False, "Online reservations receive a confirmation code for check-in."),
    ("reservation_process", True, "The patio uses separate seating, while reservations are managed from the indoor host desk."),
    ("food_delivery", False, "Starters are brought to the table before the main dishes."),
    ("food_delivery", True, "The fish was excellent, and the remaining main dishes arrived in the next table-service round."),
    ("food_delivery", False, "Main dishes are delivered after the starter plates are cleared."),
    ("food_delivery", True, "The table shared two appetizers, followed by separately plated main courses."),
    ("drink_delivery", False, "Water is placed on the table before other drinks are delivered."),
    ("drink_delivery", True, "The dessert was disappointing, while coffee was delivered after the plates were cleared."),
    ("drink_delivery", False, "Bottled drinks are brought unopened and poured at the table."),
    ("payment_check", False, "The check is brought after the table requests it."),
    ("payment_check", True, "The meal included three courses, and payment was taken at the table after the check arrived."),
    ("payment_check", False, "Card payments are processed with a portable terminal at the table."),
    ("payment_check", True, "Takeaway items were packed separately, and the receipt was issued at the counter."),
    ("self_service", False, "Diners collect trays and utensils from the self-service station."),
    ("self_service", True, "The dining room has shared seating, and customers return used trays to the collection point."),
    ("self_service", False, "Water and napkins are available from the self-service area."),
    ("counter_ordering", False, "Customers choose a table number after placing their order at the counter."),
    ("counter_ordering", True, "The menu is displayed above the counter, and completed orders are called by number."),
    ("counter_ordering", False, "Payment is completed at the counter when the order is placed."),
    ("table_service_sequence", False, "Menus are provided first, followed by drink orders and then food orders."),
    ("table_service_sequence", True, "Bread was placed on the table, and the server returned later to record the main courses."),
    ("table_service_sequence", False, "Used starter plates are cleared before the next course is served."),
    ("takeaway_pickup", False, "Takeaway orders are collected from a shelf beside the counter."),
    ("takeaway_pickup", True, "The dining room remained occupied, while pickup customers waited in a separate marked area."),
    ("takeaway_pickup", False, "Customers provide the order number at the pickup desk."),
    ("queue_procedure", False, "The entrance queue is organized by names written on a waiting list."),
    ("queue_procedure", False, "Walk-ins join a separate queue from reservation check-ins."),
    ("queue_procedure", False, "A display shows the next order number for counter collection."),
]

VALUE = [
    ("nearby_price_comparison", False, "Menu prices are similar to those at nearby restaurants serving the same cuisine."),
    ("nearby_price_comparison", True, "The noodles were spicy, while their price matched comparable dishes at nearby restaurants."),
    ("nearby_price_comparison", False, "The set menu is in the same price range as other restaurants on the street."),
    ("nearby_price_comparison", True, "The dining room uses counter service, and its prices follow the range used by nearby cafés."),
    ("area_average_price", False, "Main-course prices are close to the average listed by restaurants in this area."),
    ("area_average_price", True, "The soup was mild, and its price was within the area's usual lunch range."),
    ("area_average_price", False, "The restaurant's dinner prices fall around the neighborhood average."),
    ("area_average_price", True, "Tables are available indoors and outside, while menu prices stay within the local average range."),
    ("expected_price_range", False, "The listed prices match the expected range for a mid-range restaurant."),
    ("expected_price_range", True, "Orders are placed at the counter, and the dishes are priced within the expected range for this category."),
    ("expected_price_range", False, "The tasting menu sits within the expected price band for a multi-course dinner."),
    ("expected_price_range", True, "The portions use individual plates, and their prices remain within the stated mid-range bracket."),
    ("standard_menu_pricing", False, "Starters, mains, and desserts follow the restaurant's standard menu price structure."),
    ("standard_menu_pricing", True, "The menu changes seasonally, while each course keeps the standard pricing shown for its section."),
    ("standard_menu_pricing", False, "Weekend dishes use the same standard pricing format as the weekday menu."),
    ("standard_menu_pricing", True, "Drinks are ordered separately, and food items retain the standard prices printed on the menu."),
    ("comparable_price", False, "The lunch set has a price comparable to other two-course menus nearby."),
    ("comparable_price", True, "The room seats forty guests, and the fixed menu has a comparable price to similar local menus."),
    ("comparable_price", False, "Prices for the vegetarian and meat set menus are comparable."),
    ("typical_category_price", False, "The bowl is priced within the typical range for this type of noodle dish."),
    ("typical_category_price", True, "The sandwich includes two side items, and its price is typical for the category."),
    ("typical_category_price", False, "Coffee prices follow the usual range for specialty cafés in the area."),
    ("lunch_dinner_pricing", False, "Lunch and dinner use different price lists, with both ranges printed on the menu."),
    ("lunch_dinner_pricing", True, "The lunch menu has fewer courses, while the dinner menu uses a separate price range."),
    ("lunch_dinner_pricing", False, "The same dish appears at lunch and dinner under their respective menu prices."),
    ("lunch_dinner_pricing", True, "Reservations are handled online, and lunch and dinner prices are displayed on separate pages."),
    ("portion_price_context", False, "The menu lists portion sizes beside the corresponding price categories."),
    ("portion_price_context", True, "The plate contains two side dishes, and its portion and price are shown together on the menu."),
    ("portion_price_context", False, "Small and large portions are listed with separate standard prices."),
    ("fixed_price_menu", False, "The fixed-price menu includes three courses at the listed set price."),
    ("fixed_price_menu", False, "The fixed menu retains the same stated price for any starter and main combination."),
    ("fixed_price_menu", False, "A separate fixed-price option is offered alongside individually priced dishes."),
    ("usual_price_range", False, "Prices remain within the usual range shown for restaurants of this type."),
    ("usual_price_range", True, "The service follows a counter-ordering format, and prices stay in the usual range for the category."),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def create_rows() -> tuple[list[dict[str, str]], dict[str, dict[str, object]]]:
    rows: list[dict[str, str]] = []
    metadata: dict[str, dict[str, object]] = {}
    exact_seen: set[str] = set()
    exact_removed = 0

    for aspect, candidates in (
        ("atmosphere", ATMOSPHERE),
        ("service", SERVICE),
        ("value", VALUE),
    ):
        for number, (subtopic, is_multi, sentence) in enumerate(candidates, start=1):
            normalized = normalize(sentence)
            if normalized in exact_seen:
                exact_removed += 1
                continue
            exact_seen.add(normalized)
            aspect_token = aspect.upper()
            candidate_id = f"HA-CAND-EN-{aspect_token}-{number:04d}"
            source_group_id = f"HA-EN-{aspect_token}-{number:04d}"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "source_group_id": source_group_id,
                    "sentence": sentence,
                    "language": "en",
                    "mapped_aspect": aspect,
                    "proposed_sentiment": "neutral",
                    "source_type": "synthetic_candidate",
                    "generation_batch": "human_authored_v1",
                    "human_decision": "",
                    "human_final_aspect": "",
                    "human_final_sentiment": "",
                    "human_verified": "false",
                    "review_notes": "",
                }
            )
            metadata[candidate_id] = {
                "semantic_subtopic": subtopic,
                "is_multi_aspect": is_multi,
            }
    metadata["_exact_removed"] = {"count": exact_removed}
    return rows, metadata


def diagnostic_rows(
    rows: list[dict[str, str]], metadata: dict[str, dict[str, object]]
) -> tuple[list[dict[str, str]], int, int, int]:
    near_duplicate_ids: set[str] = set()
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            ratio = SequenceMatcher(
                None, normalize(left["sentence"]), normalize(right["sentence"])
            ).ratio()
            if ratio >= 0.88:
                near_duplicate_ids.update([left["candidate_id"], right["candidate_id"]])

    strong_terms = {
        "beautiful", "amazing", "lovely", "cozy", "terrible", "awful",
        "uncomfortable", "perfect", "horrible", "excellent", "disappointing",
        "friendly", "rude", "helpful", "kind", "impolite", "slow", "fast",
        "cheap", "expensive", "overpriced", "bargain", "worth it", "not worth",
    }
    aspect_signals = {
        "atmosphere": {
            "lighting", "lamps", "music", "playlist", "speakers", "noise", "sounds",
            "tables", "seating", "patio", "room", "crowd", "temperature", "fans",
            "aisle", "kitchen", "dining area", "partition",
        },
        "service": {
            "order", "menu", "wait", "seated", "host", "table", "reservation",
            "delivered", "brought", "payment", "check", "receipt", "self-service",
            "counter", "server", "cleared", "pickup", "queue", "waiting list",
            "trays", "collection point", "utensils",
        },
        "value": {
            "price", "prices", "priced", "range", "average", "pricing", "comparable",
            "typical", "cost", "fixed-price", "fixed menu",
        },
    }

    diagnostics: list[dict[str, str]] = []
    strong_count = 0
    wrong_count = 0
    for row in rows:
        text = normalize(row["sentence"])
        strong_hits = sorted(term for term in strong_terms if term in text)
        wrong_aspect = not any(
            signal in text for signal in aspect_signals[row["mapped_aspect"]]
        )
        possible_duplicate = row["candidate_id"] in near_duplicate_ids
        strong_count += bool(strong_hits)
        wrong_count += wrong_aspect
        notes: list[str] = []
        if possible_duplicate:
            notes.append("NEAR_DUPLICATE_LEXICAL_SIMILARITY")
        if strong_hits:
            notes.append("STRONG_VALENCE_TERMS=" + "|".join(strong_hits))
        if wrong_aspect:
            notes.append("NO_CLEAR_TARGET_ASPECT_SIGNAL")
        info = metadata[row["candidate_id"]]
        diagnostics.append(
            {
                "candidate_id": row["candidate_id"],
                "semantic_subtopic": str(info["semantic_subtopic"]),
                "is_multi_aspect": str(info["is_multi_aspect"]).lower(),
                "possible_duplicate": str(possible_duplicate).lower(),
                "possible_strong_valence": str(bool(strong_hits)).lower(),
                "possible_wrong_aspect": str(wrong_aspect).lower(),
                "diagnostic_notes": ";".join(notes),
            }
        )
    return diagnostics, len(near_duplicate_ids), strong_count, wrong_count


def select_review_batch(
    rows: list[dict[str, str]], metadata: dict[str, dict[str, object]]
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for aspect in ("atmosphere", "service", "value"):
        used_subtopics: set[str] = set()
        for row in rows:
            if row["mapped_aspect"] != aspect:
                continue
            subtopic = str(metadata[row["candidate_id"]]["semantic_subtopic"])
            if subtopic in used_subtopics:
                continue
            selected.append(row)
            used_subtopics.add(subtopic)
            if len(used_subtopics) == 8:
                break
    random.Random(42).shuffle(selected)
    return selected


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            existing = list(csv.DictReader(file))
        if any(row.get("human_decision", "").strip() for row in existing):
            raise FileExistsError(f"Refusing to overwrite human-reviewed file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, metadata = create_rows()
    diagnostics, near_count, strong_count, wrong_count = diagnostic_rows(rows, metadata)
    batch = select_review_batch(rows, metadata)

    counts = Counter(row["mapped_aspect"] for row in rows)
    multi_count = sum(bool(metadata[row["candidate_id"]]["is_multi_aspect"]) for row in rows)
    if counts != Counter({"atmosphere": 28, "service": 42, "value": 34}):
        raise ValueError(f"Candidate quotas are invalid: {dict(counts)}")
    if len(rows) != 104 or multi_count != 42:
        raise ValueError(f"Expected 104 rows and 42 multi-aspect rows: {len(rows)}, {multi_count}")
    if Counter(row["mapped_aspect"] for row in batch) != Counter(
        {"atmosphere": 8, "service": 8, "value": 8}
    ):
        raise ValueError("Review batch quotas are invalid")

    write_csv(POOL_PATH, POOL_FIELDS, rows)
    write_csv(DIAGNOSTICS_PATH, DIAGNOSTIC_FIELDS, diagnostics)

    batch_rows = []
    for row in batch:
        batch_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "source_group_id": row["source_group_id"],
                "mapped_aspect": row["mapped_aspect"],
                "semantic_subtopic": str(metadata[row["candidate_id"]]["semantic_subtopic"]),
                "sentence": row["sentence"],
                "human_decision": "",
                "human_final_aspect": "",
                "human_final_sentiment": "",
                "human_verified": "false",
                "review_notes": "",
            }
        )
    write_csv(BATCH_PATH, BATCH_FIELDS, batch_rows)

    markdown_lines = [
        "# Human-Authored English Neutral Review Batch 01",
        "",
        "All candidates require human review. No decision has been assigned.",
        "",
    ]
    for index, row in enumerate(batch_rows, start=1):
        markdown_lines.extend(
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
    MARKDOWN_PATH.write_text("\n".join(markdown_lines), encoding="utf-8")

    exact_removed = int(metadata["_exact_removed"]["count"])
    summary = [
        "HUMAN-AUTHORED NEUTRAL CANDIDATES — ENGLISH",
        "================================================",
        f"atmosphere: {counts['atmosphere']}",
        f"service: {counts['service']}",
        f"value: {counts['value']}",
        f"total: {len(rows)}",
        f"single-aspect: {len(rows) - multi_count}",
        f"multi-aspect: {multi_count}",
        f"exact duplicates removed: {exact_removed}",
        f"near-duplicate warnings: {near_count}",
        f"strong-valence warnings: {strong_count}",
        f"wrong-aspect warnings: {wrong_count}",
        f"review batch 1 size: {len(batch_rows)}",
        "human decisions assigned: 0",
        "training rows added: 0",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
