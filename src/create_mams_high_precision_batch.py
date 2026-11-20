"""Rank MAMS neutral annotations and create a high-precision review batch.

The score is only a transparent human-review priority. It is never a label or
an automatic ACCEPT/REJECT/UNCERTAIN decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_PATH = PROJECT_ROOT / "data" / "curation" / "mams_neutral_candidates_en.csv"
RANKED_POOL_PATH = PROJECT_ROOT / "data" / "curation" / "mams_neutral_ranked_pool_en.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "curation" / "review_batches"
BATCH_PATH = OUTPUT_DIR / "mams_neutral_batch_03.csv"
MARKDOWN_PATH = OUTPUT_DIR / "mams_neutral_batch_03.md"

DATASET_NAME = "justinsiow/mams-acsa"
RANDOM_SEED = 42
OVERLAP_MARKER = "EXACT_OVERLAP_EXISTING_DATA"
CATEGORY_TO_ASPECT = {
    "food": "taste",
    "ambience": "atmosphere",
    "service": "service",
    "price": "value",
}
TARGET_PER_ASPECT = {
    "taste": 4,
    "atmosphere": 4,
    "service": 8,
    "value": 8,
}

NEUTRAL_CUES = [
    "average",
    "standard",
    "typical",
    "ordinary",
    "okay",
    "fine",
    "neither good nor bad",
    "nothing special",
    "as expected",
]
STRONG_EVALUATIVE_CUES = [
    "amazing",
    "beautiful",
    "lovely",
    "terrible",
    "awful",
    "horrible",
    "excellent",
    "fantastic",
    "wonderful",
    "best",
    "worst",
    "delicious",
    "disgusting",
    "great",
    "bad",
]

BATCH_FIELDS = [
    "candidate_id",
    "sentence",
    "source_category",
    "mapped_aspect",
    "source_polarity",
    "review_priority_score",
    "priority_reason",
    "source_group_id",
    "human_decision",
    "human_final_aspect",
    "human_final_sentiment",
    "review_notes",
]

RANKED_POOL_FIELDS = [
    *BATCH_FIELDS,
    "source_dataset",
    "source_license",
    "language",
]


def contains_term(text: str, term: str) -> bool:
    if " " in term or "-" in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def find_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if contains_term(text, term)]


def add_reason(reasons: list[str], label: str, hits: list[str]) -> None:
    if hits:
        reasons.append(f"{label}: {', '.join(hits[:3])}")


def score_candidate(sentence: str, aspect: str) -> tuple[int, str]:
    text = sentence.lower()
    score = 0
    reasons: list[str] = []

    neutral_hits = find_terms(text, NEUTRAL_CUES)
    if neutral_hits:
        score += 4 + min(len(neutral_hits) - 1, 2)
        add_reason(reasons, "neutral/descriptive cue", neutral_hits)

    if aspect == "taste":
        quality_hits = find_terms(
            text,
            ["taste", "flavor", "quality", "texture", "cooked", "preparation"],
        )
        wrong_aspect_hits = find_terms(
            text,
            [
                "portion",
                "portions",
                "menu variety",
                "delivery",
                "delivered",
                "order",
                "ordering",
                "price",
                "bill",
                "waiter",
                "staff",
                "presentation",
                "menu",
                "owner",
                "customer",
                "patrons",
            ],
        )
        score += min(2 * len(quality_hits), 6)
        score -= min(3 * len(wrong_aspect_hits), 9)
        add_reason(reasons, "explicit taste/quality cue", quality_hits)
        add_reason(reasons, "possible wrong-aspect cue", wrong_aspect_hits)

        # A neutral cue is much more useful when it is tied directly to food,
        # a dish, a meal, or cuisine rather than another aspect in the sentence.
        food_subject_pattern = r"\b(?:food|dish(?:es)?|meal(?:s)?|cuisine|plate(?:s)?)\b"
        neutral_pattern = "|".join(re.escape(cue) for cue in NEUTRAL_CUES)
        tied_neutral_food = bool(
            re.search(rf"(?:{neutral_pattern}).{{0,35}}{food_subject_pattern}", text)
            or re.search(rf"{food_subject_pattern}.{{0,35}}(?:{neutral_pattern})", text)
        )
        if tied_neutral_food:
            score += 5
            reasons.append("neutral cue tied to food/dish/meal")
        explicit_neutral_food_hits = find_terms(
            text,
            [
                "food was average",
                "food is average",
                "food was fine",
                "food is fine",
                "meal was average",
                "meal was fine",
            ],
        )
        score += min(6 * len(explicit_neutral_food_hits), 6)
        add_reason(reasons, "explicit neutral food-quality statement", explicit_neutral_food_hits)
        if "standard dishes" in text or "standard italian dishes" in text:
            score -= 6
            reasons.append("food-style/menu description rather than taste quality")
        if "below average" in text:
            score -= 5
            reasons.append("below-average wording is evaluative rather than neutral")
        if find_terms(text, ["asking patrons", "asked customers", "asking customers"]):
            score -= 5
            reasons.append("reported owner/customer question rather than quality assertion")
        if re.search(r"\b\d+\s*%", text):
            score -= 4
            reasons.append("percentage context may create a false food-quality match")

    elif aspect == "atmosphere":
        atmosphere_hits = find_terms(
            text,
            [
                "music",
                "lighting",
                "lights",
                "noise",
                "seating",
                "chairs",
                "crowd",
                "crowded",
                "room",
                "environment",
                "setting",
                "atmosphere",
                "ambience",
                "decor",
                "interior",
            ],
        )
        score += min(2 * len(atmosphere_hits), 8)
        add_reason(reasons, "explicit atmosphere cue", atmosphere_hits)

        descriptive_atmosphere_hits = find_terms(
            text,
            [
                "dim lighting",
                "bright lighting",
                "music is",
                "dining room",
                "decorated with",
                "seating area",
                "background music",
            ],
        )
        score += min(3 * len(descriptive_atmosphere_hits), 6)
        add_reason(reasons, "descriptive atmosphere pattern", descriptive_atmosphere_hits)

        atmosphere_evaluative_hits = find_terms(
            text,
            [
                "amazing",
                "beautiful",
                "lovely",
                "terrible",
                "awful",
                "cozy",
                "romantic",
                "horrible",
                "uncomfortable",
                "nice",
                "blaring",
                "negated",
                "need to change",
                "want to stay",
                "doesn't make up",
                "does not make up",
                "dated",
                "old-fashioned",
                "dreary",
            ],
        )
        score -= min(3 * len(atmosphere_evaluative_hits), 9)
        add_reason(reasons, "strong evaluative atmosphere cue", atmosphere_evaluative_hits)
        if "place setting" in text:
            score -= 5
            reasons.append("possible wrong-aspect cue: place setting")

    elif aspect == "service":
        process_hits = find_terms(
            text,
            [
                "wait time",
                "waiting",
                "waited",
                "minutes",
                "order taking",
                "order",
                "ordering",
                "reservation",
                "reservations",
                "seated",
                "seating",
                "served",
                "serving",
                "arrived",
                "arriving",
                "checkout",
                "check",
            ],
        )
        score += min(len(process_hits), 5)
        add_reason(reasons, "service-process cue", process_hits)

        neutral_process_hits = find_terms(
            text,
            [
                "we were seated",
                "was served",
                "were served",
                "takes reservations",
                "take reservations",
                "food arrived",
                "order was taken",
                "orders were taken",
            ],
        )
        score += min(3 * len(neutral_process_hits), 6)
        add_reason(reasons, "factual service-process pattern", neutral_process_hits)
        neutral_pattern = "|".join(re.escape(cue) for cue in NEUTRAL_CUES)
        neutral_service_tied = bool(
            re.search(rf"\bservice\b.{{0,25}}(?:{neutral_pattern})", text)
            or re.search(rf"(?:{neutral_pattern}).{{0,25}}\bservice\b", text)
        )
        if neutral_service_tied:
            score += 5
            reasons.append("neutral cue tied to service")
        if neutral_hits and not neutral_process_hits and not neutral_service_tied:
            score -= 4
            reasons.append("neutral cue not clearly tied to service process")

        staff_behavior_hits = find_terms(
            text,
            [
                "rude",
                "friendly",
                "kind",
                "helpful",
                "attitude",
                "surly",
                "attentive",
                "accommodating",
                "smile",
            ],
        )
        process_valence_hits = find_terms(
            text,
            [
                "delay",
                "delayed",
                "late",
                "wrong order",
                "fast service",
                "quick service",
                "prompt service",
                "slow service",
                "never arrived",
                "forgot",
                "hour",
                "long wait",
                "angry",
                "inefficient",
                "wrong",
                "not even",
                "could not",
                "no one",
                "told us no",
                "refused",
                "downhill",
                "painless",
                "fastly",
                "quickly",
                "efficient",
                "slow",
                "overpriced",
                "agonizing",
                "promptly",
                "short wait",
                "didn't come",
                "don't come",
                "not prepared",
                "varies",
                "great to poor",
                "poor service",
                "good service",
                "great service",
                "yelled",
                "carelessly",
                "carlessly",
                "no waitress",
                "had to request",
                "wait forever",
            ],
        )
        score -= min(3 * len(staff_behavior_hits), 9)
        score -= min(4 * len(process_valence_hits), 12)
        add_reason(reasons, "staff-behavior cue", staff_behavior_hits)
        add_reason(reasons, "strong service-valence cue", process_valence_hits)
        if re.search(r"\b\d+\s*(?:minutes?|hours?)\b", text) and find_terms(
            text, ["wait", "waiting", "waited"]
        ):
            score -= 4
            reasons.append("explicit timed wait suggests strong service valence")
        if find_terms(
            text,
            [
                "one minute wait",
                "two minute wait",
                "five minute wait",
                "ten minute wait",
                "fifteen minute wait",
                "twenty minute wait",
                "thirty minute wait",
                "half an hour wait",
            ],
        ):
            score -= 4
            reasons.append("written-out timed wait suggests strong service valence")
        food_presentation_service = bool(
            re.search(r"\bserved\s+(?:in|with|on|alongside)\b", text)
            or re.search(
                r"\b(?:food|dish|tea|coffee|bread|drink|drinks|appetizer|scallops?|meal)\b.{0,35}\bserved\b",
                text,
            )
        )
        if food_presentation_service:
            score -= 6
            reasons.append("served likely describes food presentation, not service process")

    elif aspect == "value":
        price_hits = find_terms(text, ["price", "prices", "pricing", "cost", "value"])
        score += min(2 * len(price_hits), 4)
        add_reason(reasons, "explicit price/value cue", price_hits)

        neutral_value_phrases = [
            "average price",
            "average prices",
            "similar price",
            "similar prices",
            "typical price",
            "typical prices",
            "comparable price",
            "comparable prices",
            "standard pricing",
            "standard price",
            "standard prices",
            "normal pricing",
            "normal price",
            "normal prices",
            "average pricing",
            "average prices",
            "comparable pricing",
            "neither cheap nor expensive",
            "price is about the same",
            "prices are about the same",
        ]
        neutral_value_hits = find_terms(text, neutral_value_phrases)
        score += min(5 * len(neutral_value_hits), 10)
        add_reason(reasons, "neutral price comparison", neutral_value_hits)
        if neutral_hits and not price_hits and not neutral_value_hits:
            score -= 4
            reasons.append("neutral cue not tied to price/value")

        strong_value_hits = find_terms(
            text,
            [
                "expensive",
                "cheap",
                "overpriced",
                "bargain",
                "great deal",
                "worth it",
                "not worth",
                "good value",
                "bad value",
                "for the price i expected",
                "for the price paid, i expected",
                "better elsewhere",
                "much better",
                "good for the price",
                "perfect for the price",
                "small for the price",
                "decent for the price",
                "price shock",
                "worth every",
                "plenty of food for the price",
                "twice the cost",
                "included in their price",
            ],
        )
        score -= min(4 * len(strong_value_hits), 12)
        add_reason(reasons, "strong value judgement", strong_value_hits)
        if "same price" in text and "better" in text:
            score -= 5
            reasons.append("better-elsewhere comparison at the same price")

        bill_hits = find_terms(text, ["bill", "check"])
        if bill_hits and not price_hits and not neutral_value_hits:
            score -= 3
            add_reason(reasons, "bill/check mention without value judgement", bill_hits)
        if re.search(r"\$\s*\d|\b\d+\s*(?:dollars?|bucks?)\b", text) and not neutral_value_hits:
            score -= 2
            reasons.append("amount mention without neutral comparison")

    # Generic evaluative language is a weaker penalty than aspect-specific
    # penalties; it may refer to another annotation in the same MAMS sentence.
    generic_evaluative_hits = find_terms(text, STRONG_EVALUATIVE_CUES)
    if generic_evaluative_hits:
        score -= min(2 * len(generic_evaluative_hits), 6)
        add_reason(reasons, "strong evaluative language", generic_evaluative_hits)

    if not reasons:
        reasons.append("no strong high-precision cue")
    return score, "; ".join(reasons)


def load_reviewed_ids() -> set[str]:
    with CURATION_PATH.open(encoding="utf-8", newline="") as file:
        return {
            row["candidate_id"]
            for row in csv.DictReader(file)
            if row["human_decision"].strip()
        }


def load_existing_sentences() -> set[str]:
    sentences: set[str] = set()
    for split in ("train", "dev", "test"):
        path = PROJECT_ROOT / "data" / "cleaned" / f"{split}.csv"
        with path.open(encoding="utf-8", newline="") as file:
            sentences.update(
                row["sentence"] for row in csv.DictReader(file) if row["sentence"]
            )
    return sentences


def collect_ranked_pool() -> list[dict[str, str]]:
    reviewed_ids = load_reviewed_ids()
    existing_sentences = load_existing_sentences()
    dataset = load_dataset(DATASET_NAME, split="train")
    rows: list[dict[str, str]] = []
    seen_sentence_aspect: set[tuple[str, str]] = set()

    for row_index, source_row in enumerate(dataset):
        categories = [value.strip() for value in source_row["category"].split(",")]
        polarities = [value.strip() for value in source_row["polarity"].split(",")]
        if len(categories) != len(polarities):
            raise ValueError(f"Unaligned MAMS annotations at train row {row_index}")

        sentence = source_row["text"]
        if sentence in existing_sentences:
            continue

        for category, polarity in zip(categories, polarities):
            if polarity != "neutral" or category not in CATEGORY_TO_ASPECT:
                continue
            aspect = CATEGORY_TO_ASPECT[category]
            duplicate_key = (sentence, aspect)
            if duplicate_key in seen_sentence_aspect:
                continue
            seen_sentence_aspect.add(duplicate_key)

            candidate_id = f"MAMS-EN-{row_index:06d}-{category.upper()}"
            if candidate_id in reviewed_ids:
                continue
            score, reason = score_candidate(sentence, aspect)
            tie_break = hashlib.sha256(
                f"{RANDOM_SEED}:{candidate_id}".encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "sentence": sentence,
                    "source_category": category,
                    "mapped_aspect": aspect,
                    "source_polarity": "neutral",
                    "review_priority_score": str(score),
                    "priority_reason": reason,
                    "source_group_id": f"MAMS-TRAIN-{row_index:06d}",
                    "human_decision": "",
                    "human_final_aspect": "",
                    "human_final_sentiment": "",
                    "review_notes": "",
                    "source_dataset": "mams-acsa",
                    "source_license": "apache-2.0",
                    "language": "en",
                    "_tie_break": tie_break,
                }
            )

    rows.sort(
        key=lambda row: (
            row["mapped_aspect"],
            -int(row["review_priority_score"]),
            row["_tie_break"],
        )
    )
    return rows


def select_batch(ranked_pool: list[dict[str, str]]) -> list[dict[str, str]]:
    by_aspect = {
        aspect: [row for row in ranked_pool if row["mapped_aspect"] == aspect]
        for aspect in TARGET_PER_ASPECT
    }
    selected: list[dict[str, str]] = []
    used_groups: set[str] = set()

    # Round-robin selection prevents an earlier aspect from unnecessarily
    # monopolizing a source sentence that has more than one neutral annotation.
    for rank in range(max(TARGET_PER_ASPECT.values())):
        for aspect in TARGET_PER_ASPECT:
            if rank >= TARGET_PER_ASPECT[aspect]:
                continue
            candidate = next(
                (
                    row
                    for row in by_aspect[aspect]
                    if row["source_group_id"] not in used_groups
                    and row not in selected
                ),
                None,
            )
            if candidate is None:
                raise ValueError(f"Could not fill Batch 2 quota for {aspect}")
            selected.append(candidate)
            used_groups.add(candidate["source_group_id"])

    random.Random(RANDOM_SEED).shuffle(selected)
    return selected


def clean_row(row: dict[str, str], fields: list[str]) -> dict[str, str]:
    return {field: row[field] for field in fields}


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    if path.exists():
        with path.open(encoding="utf-8", newline="") as existing_file:
            existing_rows = list(csv.DictReader(existing_file))
        if any(row.get("human_decision", "").strip() for row in existing_rows):
            raise FileExistsError(
                f"Refusing to overwrite a file containing human decisions: {path}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean_row(row, fields) for row in rows)


def write_markdown(batch: list[dict[str, str]]) -> None:
    lines = [
        "# MAMS Neutral High-Precision Review Batch 03",
        "",
        "Scores are review priorities only. All human decision fields are blank.",
        "",
    ]
    for index, row in enumerate(batch, start=1):
        lines.extend(
            [
                f"## {index}",
                "",
                f"Candidate ID: {row['candidate_id']}",
                "",
                f"Mapped aspect: {row['mapped_aspect']}",
                "",
                f"Priority score: {row['review_priority_score']}",
                "",
                f"Priority reason: {row['priority_reason']}",
                "",
                f"Sentence: {row['sentence']}",
                "",
            ]
        )
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")


def validate(ranked_pool: list[dict[str, str]], batch: list[dict[str, str]]) -> None:
    if Counter(row["mapped_aspect"] for row in batch) != Counter(TARGET_PER_ASPECT):
        raise ValueError("Batch 3 aspect quotas are invalid")
    if len(batch) != len({row["candidate_id"] for row in batch}):
        raise ValueError("Duplicate candidate ID in Batch 3")
    if len(batch) != len({row["source_group_id"] for row in batch}):
        raise ValueError("Duplicate source_group_id in Batch 3")
    if len(ranked_pool) != len(
        {(row["sentence"], row["mapped_aspect"]) for row in ranked_pool}
    ):
        raise ValueError("Duplicate sentence + mapped_aspect in ranked pool")
    if any(row["human_decision"] for row in ranked_pool):
        raise ValueError("Heuristic ranking filled a human decision")


def print_preview(ranked_pool: list[dict[str, str]], limit: int = 10) -> None:
    for aspect in TARGET_PER_ASPECT:
        print(f"\n=== {aspect.upper()} TOP {limit} ===")
        rows = [row for row in ranked_pool if row["mapped_aspect"] == aspect][:limit]
        for row in rows:
            print(f"[{row['review_priority_score']}] {row['candidate_id']}")
            print(row["priority_reason"])
            print(row["sentence"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    ranked_pool = collect_ranked_pool()
    if args.preview:
        print_preview(ranked_pool)
        return

    batch = select_batch(ranked_pool)
    validate(ranked_pool, batch)
    write_csv(RANKED_POOL_PATH, ranked_pool, RANKED_POOL_FIELDS)
    write_csv(BATCH_PATH, batch, BATCH_FIELDS)
    write_markdown(batch)

    pool_counts = Counter(row["mapped_aspect"] for row in ranked_pool)
    batch_counts = Counter(row["mapped_aspect"] for row in batch)
    print("Eligible neutral pool before Batch 3:")
    for aspect in TARGET_PER_ASPECT:
        print(f"{aspect}: {pool_counts[aspect]}")
    print("Remaining outside Batch 3:")
    for aspect in TARGET_PER_ASPECT:
        print(f"{aspect}: {pool_counts[aspect] - batch_counts[aspect]}")
    print(f"Batch 3: {BATCH_PATH}")
    print(f"Readable review: {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
