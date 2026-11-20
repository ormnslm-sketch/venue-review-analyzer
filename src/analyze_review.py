"""Sentence-aware review analysis: aspect presence, evidence, then sentiment."""

from __future__ import annotations

import argparse
import json
import math
import re
from typing import Any

from aspect_presence import EXPECTED_ASPECTS, detect_aspects
from inference import LABEL2ID, predict_sentiment


ASPECT_ORDER = EXPECTED_ASPECTS
SENTIMENT_ORDER = ("negative", "neutral", "positive")
FIDELITY_TOLERANCE = 1e-6

# A full stop is not a boundary when it is between digits, so values such as
# 3.5 remain intact. Euro amounts such as 6€ and 1€ are unaffected.
_SENTENCE_BOUNDARY = re.compile(r"(?<!\d)\.(?!\d)|[!?]+|(?:\r\n|\r|\n)+")


def _validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return text.strip()


def segment_review(text: str) -> list[str]:
    """Split a TR/EN/ES review on simple punctuation and newline boundaries."""
    normalized_text = _validate_text(text)
    sentences: list[str] = []
    start = 0
    for boundary in _SENTENCE_BOUNDARY.finditer(normalized_text):
        segment = normalized_text[start : boundary.end()].strip()
        if segment:
            sentences.append(segment)
        start = boundary.end()
    tail = normalized_text[start:].strip()
    if tail:
        sentences.append(tail)
    if not sentences:
        raise RuntimeError("sentence segmentation returned no content")
    return sentences


def _validate_presence_result(
    presence: dict[str, dict[str, bool | float]],
) -> None:
    if tuple(presence) != ASPECT_ORDER:
        raise RuntimeError(
            f"Unexpected aspect order: {tuple(presence)}; expected {ASPECT_ORDER}"
        )
    for aspect in ASPECT_ORDER:
        probability = float(presence[aspect]["probability"])
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(
                f"Presence probability for {aspect} is outside [0, 1]: "
                f"{probability}"
            )


def _validate_sentiment_result(
    sentiment: dict[str, Any],
    expected_aspect: str,
) -> None:
    if sentiment.get("aspect") != expected_aspect:
        raise RuntimeError(
            f"Sentiment result aspect mismatch: {sentiment.get('aspect')} != "
            f"{expected_aspect}"
        )
    label = sentiment.get("sentiment")
    if label not in LABEL2ID:
        raise RuntimeError(f"Unexpected sentiment label: {label}")
    probabilities = sentiment.get("probabilities")
    if not isinstance(probabilities, dict) or tuple(probabilities) != SENTIMENT_ORDER:
        raise RuntimeError(
            "Unexpected sentiment probability order: "
            f"{tuple(probabilities) if isinstance(probabilities, dict) else None}"
        )
    probability_sum = sum(float(value) for value in probabilities.values())
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError(
            f"Sentiment probabilities for {expected_aspect} do not sum to 1: "
            f"{probability_sum}"
        )


def _aggregate_sentiment(
    evidence_sentences: list[dict[str, Any]],
) -> tuple[str, float, dict[str, float]]:
    total_weight = sum(
        float(evidence["presence_probability"]) for evidence in evidence_sentences
    )
    if total_weight <= 0.0:
        raise RuntimeError("evidence sentence weights must sum to a positive value")

    probabilities = {
        label: sum(
            float(evidence["presence_probability"])
            * float(evidence["sentiment_probabilities"][label])
            for evidence in evidence_sentences
        )
        / total_weight
        for label in SENTIMENT_ORDER
    }
    # The weighted mean should already sum to one. Normalize once to remove only
    # floating-point drift, not to change the aggregation rule.
    probability_sum = sum(probabilities.values())
    probabilities = {
        label: probabilities[label] / probability_sum for label in SENTIMENT_ORDER
    }
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-8):
        raise RuntimeError("aggregated sentiment probabilities do not sum to 1")
    label = max(SENTIMENT_ORDER, key=probabilities.__getitem__)
    return label, probabilities[label], probabilities


def analyze_review(text: str) -> dict[str, Any]:
    """Analyze a review hierarchically using sentence-level model inputs."""
    normalized_text = _validate_text(text)
    sentences = segment_review(normalized_text)

    sentence_presence: list[dict[str, Any]] = []
    for sentence_index, sentence in enumerate(sentences):
        presence = detect_aspects(sentence)
        _validate_presence_result(presence)
        sentence_presence.append(
            {
                "sentence_index": sentence_index,
                "text": sentence,
                "probabilities": {
                    aspect: float(presence[aspect]["probability"])
                    for aspect in ASPECT_ORDER
                },
                "present": {
                    aspect: bool(presence[aspect]["present"])
                    for aspect in ASPECT_ORDER
                },
            }
        )

    aspects: dict[str, dict[str, Any]] = {}
    sentiment_calls = 0
    for aspect in ASPECT_ORDER:
        review_presence_probability = max(
            item["probabilities"][aspect] for item in sentence_presence
        )
        evidence_sentences: list[dict[str, Any]] = []
        for item in sentence_presence:
            if not item["present"][aspect]:
                continue
            sentiment = predict_sentiment(item["text"], aspect)
            _validate_sentiment_result(sentiment, aspect)
            sentiment_calls += 1
            evidence_sentences.append(
                {
                    "sentence_index": item["sentence_index"],
                    "text": item["text"],
                    "presence_probability": item["probabilities"][aspect],
                    "sentiment": sentiment["sentiment"],
                    "sentiment_confidence": float(sentiment["confidence"]),
                    "sentiment_probabilities": {
                        label: float(sentiment["probabilities"][label])
                        for label in SENTIMENT_ORDER
                    },
                }
            )

        if not evidence_sentences:
            aspects[aspect] = {
                "present": False,
                "presence_probability": review_presence_probability,
                "sentiment": None,
                "sentiment_confidence": None,
                "sentiment_probabilities": None,
                "evidence_sentences": [],
            }
            continue

        label, confidence, probabilities = _aggregate_sentiment(evidence_sentences)
        aspects[aspect] = {
            "present": True,
            "presence_probability": review_presence_probability,
            "sentiment": label,
            "sentiment_confidence": confidence,
            "sentiment_probabilities": probabilities,
            "evidence_sentences": evidence_sentences,
        }

    return {
        "text": normalized_text,
        "sentence_count": len(sentences),
        "sentences": sentences,
        "sentence_aspect_probabilities": [
            {
                "sentence_index": item["sentence_index"],
                "text": item["text"],
                "probabilities": item["probabilities"],
            }
            for item in sentence_presence
        ],
        "aspects": aspects,
        "no_target_aspects_detected": not any(
            item["present"] for item in aspects.values()
        ),
        "sentiment_calls": sentiment_calls,
    }


def _fidelity_check(text: str, result: dict[str, Any]) -> dict[str, Any]:
    """Compare every evidence sentiment with a direct public API call."""
    del text  # Evidence carries the exact segmented model input.
    checked = 0
    maximum_difference = 0.0
    for aspect in ASPECT_ORDER:
        for evidence in result["aspects"][aspect]["evidence_sentences"]:
            direct = predict_sentiment(evidence["text"], aspect)
            if direct["sentiment"] != evidence["sentiment"]:
                raise RuntimeError(
                    f"Sentiment fidelity failed for {aspect}: "
                    f"pipeline={evidence['sentiment']} "
                    f"direct={direct['sentiment']}"
                )
            for sentiment in SENTIMENT_ORDER:
                difference = abs(
                    evidence["sentiment_probabilities"][sentiment]
                    - direct["probabilities"][sentiment]
                )
                maximum_difference = max(maximum_difference, difference)
            checked += 1
    if maximum_difference >= FIDELITY_TOLERANCE:
        raise RuntimeError(
            "Sentiment fidelity exceeded tolerance: "
            f"max_difference={maximum_difference} tolerance={FIDELITY_TOLERANCE}"
        )
    return {
        "checked_evidence_sentences": checked,
        "max_probability_difference": maximum_difference,
        "tolerance": FIDELITY_TOLERANCE,
        "status": "PASS",
    }


def _single_sentence_regression(text: str) -> dict[str, Any]:
    """Verify that one sentence retains direct detector/sentiment behavior."""
    result = analyze_review(text)
    if result["sentence_count"] != 1:
        raise RuntimeError("single-sentence regression input was split unexpectedly")
    direct_presence = detect_aspects(text)
    _validate_presence_result(direct_presence)
    maximum_presence_difference = 0.0
    maximum_sentiment_difference = 0.0
    for aspect in ASPECT_ORDER:
        item = result["aspects"][aspect]
        direct_probability = float(direct_presence[aspect]["probability"])
        maximum_presence_difference = max(
            maximum_presence_difference,
            abs(item["presence_probability"] - direct_probability),
        )
        if item["present"] != bool(direct_presence[aspect]["present"]):
            raise RuntimeError(f"single-sentence presence mismatch for {aspect}")
        if item["present"]:
            direct_sentiment = predict_sentiment(text, aspect)
            if item["sentiment"] != direct_sentiment["sentiment"]:
                raise RuntimeError(f"single-sentence sentiment mismatch for {aspect}")
            for label in SENTIMENT_ORDER:
                maximum_sentiment_difference = max(
                    maximum_sentiment_difference,
                    abs(
                        item["sentiment_probabilities"][label]
                        - direct_sentiment["probabilities"][label]
                    ),
                )
    fidelity = _fidelity_check(text, result)
    if maximum_presence_difference >= FIDELITY_TOLERANCE:
        raise RuntimeError("single-sentence presence probability mismatch")
    if maximum_sentiment_difference >= FIDELITY_TOLERANCE:
        raise RuntimeError("single-sentence aggregated sentiment mismatch")
    return {
        "status": "PASS",
        "max_presence_probability_difference": maximum_presence_difference,
        "max_sentiment_probability_difference": maximum_sentiment_difference,
        "sentiment_fidelity": fidelity,
    }


def run_smoke_tests() -> dict[str, Any]:
    multilingual_cases = {
        "tr": "Kahve çok güzeldi ama servis oldukça yavaştı.",
        "en": "The coffee was excellent but the service was very slow.",
        "es": "El café estaba muy bueno, pero el servicio fue muy lento.",
    }
    multilingual: dict[str, Any] = {}
    for language, text in multilingual_cases.items():
        result = analyze_review(text)
        multilingual[language] = {
            "result": result,
            "fidelity": _fidelity_check(text, result),
        }

    all_zero_text = "Finally a reliable Chinese restaurant !"
    all_zero_result = analyze_review(all_zero_text)
    if all_zero_result["no_target_aspects_detected"]:
        if all_zero_result["sentiment_calls"] != 0:
            raise RuntimeError("All-zero result unexpectedly called sentiment")

    problem_reviews = {
        "review_1_es": (
            "Pido una tostada de aguacate con pavo y cherrys ya que en la foto "
            "de la entrada parecía muy apetecible. El chico cuando nos la sirve, "
            "el aguacate brilla por su ausencia y en lugar de ser aguacate "
            "chafado es guacamole del Mercadona restregado con desgana. El pavo "
            "de una calidad pésima y los cherrys bien.\n6€ nos cobran por una "
            "tostada que no vale ni 1€.\nNo vuelvo más."
        ),
        "review_2_en": (
            "A big illy fan, so suffered a big illy let down. Coffee beans were "
            "over-extracted, so the taste wasn't up to a good espresso standard. "
            "Place has an ok vibe for the tourist spot it is in, however you may "
            "as well buy the beans in your local supermarket and make it at home."
        ),
    }
    diagnostics: dict[str, Any] = {}
    for name, text in problem_reviews.items():
        result = analyze_review(text)
        diagnostics[name] = {
            "result": result,
            "fidelity": _fidelity_check(text, result),
        }

    return {
        "multilingual": multilingual,
        "all_zero": {
            "source_type": "natural_raw_mabsa_all_zero",
            "result": all_zero_result,
        },
        "problem_reviews": diagnostics,
        "single_sentence_regression": _single_sentence_regression(
            "The coffee was excellent but the service was very slow."
        ),
    }


def _print_readable(result: dict[str, Any]) -> None:
    print(f"Review: {result['text']}")
    print(f"Sentences: {result['sentence_count']}")
    print("\nDetected aspects:")
    present_count = 0
    for aspect in ASPECT_ORDER:
        item = result["aspects"][aspect]
        if not item["present"]:
            continue
        present_count += 1
        print(f"\n{aspect}")
        print(f"  presence: {item['presence_probability']:.6f}")
        print(f"  sentiment: {item['sentiment']}")
        print(f"  sentiment confidence: {item['sentiment_confidence']:.6f}")
        print("  evidence:")
        for evidence in item["evidence_sentences"]:
            print(
                f"    [{evidence['sentence_index']}] "
                f"{evidence['presence_probability']:.6f} "
                f"{evidence['sentiment']}: {evidence['text']}"
            )
    if present_count == 0:
        print("  none")

    absent = [
        aspect for aspect in ASPECT_ORDER if not result["aspects"][aspect]["present"]
    ]
    print("\nAbsent:")
    if absent:
        for aspect in absent:
            probability = result["aspects"][aspect]["presence_probability"]
            print(f"  {aspect} ({probability:.6f})")
    else:
        print("  none")
    print(f"\nSentiment calls: {result['sentiment_calls']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Restaurant review to analyze")
    group.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of the readable view",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        print(json.dumps(run_smoke_tests(), ensure_ascii=False, indent=2))
        return
    result = analyze_review(args.text)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_readable(result)


if __name__ == "__main__":
    main()
