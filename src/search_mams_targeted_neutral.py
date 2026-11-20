"""Find unreviewed, high-precision MAMS neutral candidates by explicit patterns."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "data" / "curation" / "mams_neutral_candidates_en.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "curation" / "mams_targeted_neutral_candidates.csv"
DATASET_NAME = "justinsiow/mams-acsa"

CATEGORY_TO_ASPECT = {
    "food": "taste",
    "ambience": "atmosphere",
    "service": "service",
    "price": "value",
}

OUTPUT_FIELDS = [
    "candidate_id",
    "sentence",
    "mapped_aspect",
    "source_category",
    "matched_neutral_pattern",
    "source_group_id",
    "human_decision",
]

SERVICE_DIRECT_PATTERNS = [
    r"\bservice (?:was|is) average\b",
    r"\bservice (?:was|is) (?:okay|ok|fine|normal|standard|adequate|acceptable)\b",
    r"\b(?:average|normal|okay|ok|fine|standard|adequate|acceptable) service\b",
    r"\bservice was neither fast nor slow\b",
    r"\b(?:usual|normal|standard) wait\b",
]
SERVICE_WAIT_PATTERN = re.compile(
    r"\bwaited about (?:\d+|one|two|three|five|ten|fifteen|twenty|thirty) "
    r"(?:minutes?|hours?)\b"
)
SERVICE_WAIT_VALENCE = [
    "too long",
    "only",
    "finally",
    "ignored",
    "rude",
    "friendly",
    "slow",
    "fast",
    "quick",
    "forever",
    "terrible",
    "awful",
    "horrible",
    "excellent",
    "great",
    "bad",
]

VALUE_PATTERNS = [
    r"\baverage price(?:s)?\b",
    r"\bprices? (?:was|were|is|are) average\b",
    r"\bnormal price(?:s)?\b",
    r"\bstandard price(?:s)?\b",
    r"\btypical price(?:s)?\b",
    r"\bsimilar price(?:s)?\b",
    r"\bcomparable price(?:s)?\b",
    r"\babout the same price\b",
    r"\bprices? (?:is|are|was|were) about the same\b",
    r"\breasonably priced\b",
]
VALUE_EXCLUSIONS = [
    "cheap",
    "expensive",
    "overpriced",
    "bargain",
    "great deal",
    "worth it",
    "not worth",
    "better elsewhere",
    "better food",
    "much better",
]

ATMOSPHERE_VALENCE = [
    "beautiful",
    "lovely",
    "cozy",
    "romantic",
    "terrible",
    "awful",
    "uncomfortable",
    "amazing",
    "great",
    "nice",
    "excellent",
    "quaint",
    "blaring",
    "comfy",
    "spectacular",
    "thank goodness",
    "need to change",
    "negated",
    "good time",
    "tasty",
]
ATMOSPHERE_PATTERNS = [
    (r"\b(?:lighting|lights)\b", "factual lighting description"),
    (r"\b(?:music|soundtrack)\b", "factual music description"),
    (
        r"\b(?:dining room|room layout|bar area|lounge area|walls?|ceiling|booths?|fireplace|interior|decor)\b",
        "room/layout description",
    ),
    (r"\b(?:crowd|crowded|crowds)\b", "crowd description"),
    (r"\b(?:seating|environment|atmosphere|ambience)\b", "seating/environment description"),
]

TASTE_PATTERNS = [
    r"\bfood (?:was|is) average\b",
    r"\baverage food\b",
    r"\bfood (?:was|is) fine\b",
    r"\bfine but unremarkable\b",
    r"\bstandard taste\b",
    r"\bordinary food\b",
    r"\bnothing special\b",
]


def first_regex_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def match_targeted_pattern(sentence: str, aspect: str) -> str | None:
    text = sentence.lower()
    if aspect == "service":
        direct = first_regex_match(text, SERVICE_DIRECT_PATTERNS)
        if direct:
            return f"explicit neutral service phrase: {direct}"
        wait_match = SERVICE_WAIT_PATTERN.search(text)
        if wait_match and not any(cue in text for cue in SERVICE_WAIT_VALENCE):
            return f"neutral timed-wait phrase: {wait_match.group(0)}"
        return None

    if aspect == "value":
        if any(cue in text for cue in VALUE_EXCLUSIONS):
            return None
        match = first_regex_match(text, VALUE_PATTERNS)
        return f"explicit neutral price phrase: {match}" if match else None

    if aspect == "atmosphere":
        if any(cue in text for cue in ATMOSPHERE_VALENCE):
            return None
        matches = [label for pattern, label in ATMOSPHERE_PATTERNS if re.search(pattern, text)]
        if not matches:
            return None
        # Require two environment signals, or one strongly specific signal.
        specific = {"factual lighting description", "factual music description", "room/layout description"}
        if len(matches) >= 2 or any(label in specific for label in matches):
            return " + ".join(matches)
        return None

    if aspect == "taste":
        match = first_regex_match(text, TASTE_PATTERNS)
        return f"explicit neutral taste/quality phrase: {match}" if match else None
    return None


def load_reviewed_ids() -> set[str]:
    with MAIN_PATH.open(encoding="utf-8", newline="") as file:
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
            sentences.update(row["sentence"] for row in csv.DictReader(file))
    return sentences


def collect_candidates() -> list[dict[str, str]]:
    reviewed_ids = load_reviewed_ids()
    existing_sentences = load_existing_sentences()
    dataset = load_dataset(DATASET_NAME, split="train")
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row_index, row in enumerate(dataset):
        sentence = row["text"]
        if sentence in existing_sentences:
            continue
        categories = [value.strip() for value in row["category"].split(",")]
        polarities = [value.strip() for value in row["polarity"].split(",")]
        if len(categories) != len(polarities):
            raise ValueError(f"Unaligned MAMS annotations at train row {row_index}")

        for category, polarity in zip(categories, polarities):
            if polarity != "neutral" or category not in CATEGORY_TO_ASPECT:
                continue
            aspect = CATEGORY_TO_ASPECT[category]
            key = (sentence, aspect)
            if key in seen:
                continue
            seen.add(key)
            candidate_id = f"MAMS-EN-{row_index:06d}-{category.upper()}"
            if candidate_id in reviewed_ids:
                continue
            matched_pattern = match_targeted_pattern(sentence, aspect)
            if not matched_pattern:
                continue
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "sentence": sentence,
                    "mapped_aspect": aspect,
                    "source_category": category,
                    "matched_neutral_pattern": matched_pattern,
                    "source_group_id": f"MAMS-TRAIN-{row_index:06d}",
                    "human_decision": "",
                }
            )

    candidates.sort(key=lambda row: (row["mapped_aspect"], row["candidate_id"]))
    return candidates


def main() -> None:
    candidates = collect_candidates()
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(encoding="utf-8", newline="") as file:
            existing = list(csv.DictReader(file))
        if any(row["human_decision"].strip() for row in existing):
            raise FileExistsError("Refusing to overwrite targeted candidates with human decisions")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(candidates)

    counts = Counter(row["mapped_aspect"] for row in candidates)
    print(f"Targeted candidates written: {OUTPUT_PATH}")
    for aspect in ("taste", "atmosphere", "service", "value"):
        print(f"{aspect}: {counts[aspect]}")


if __name__ == "__main__":
    main()
