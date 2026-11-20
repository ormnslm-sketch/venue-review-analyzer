"""Inference API for the classical Aspect Presence v2 detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "aspect_presence_logreg_v2"
EXPECTED_ASPECTS = ("taste", "atmosphere", "service", "value")
EXPECTED_ARTIFACT_VERSION = "aspect_presence_logreg_v2"
DEFAULT_THRESHOLD = 0.5

_DETECTOR_CACHE: dict[str, Any] | None = None


def load_detector(model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    """Load the trusted local vectorizer and four classifiers once per process."""
    global _DETECTOR_CACHE
    resolved_dir = model_dir.resolve()
    if (
        _DETECTOR_CACHE is not None
        and _DETECTOR_CACHE["model_dir"] == resolved_dir
    ):
        return _DETECTOR_CACHE

    metadata_path = resolved_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Aspect detector metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("artifact_version") != EXPECTED_ARTIFACT_VERSION:
        raise ValueError(
            "Unexpected aspect-detector artifact version: "
            f"{metadata.get('artifact_version')}"
        )

    aspects = tuple(metadata.get("aspects", ()))
    if aspects != EXPECTED_ASPECTS:
        raise ValueError(
            f"Unexpected aspect order in metadata: {aspects}; "
            f"expected {EXPECTED_ASPECTS}"
        )
    if float(metadata.get("threshold")) != DEFAULT_THRESHOLD:
        raise ValueError("Artifact legacy/default threshold is not 0.5")
    if metadata.get("input") != "sentence only":
        raise ValueError("Artifact input representation is not sentence-only")

    calibrated_thresholds = metadata.get("thresholds")
    if not isinstance(calibrated_thresholds, dict):
        calibrated_thresholds = {
            aspect: DEFAULT_THRESHOLD for aspect in EXPECTED_ASPECTS
        }
    calibrated_thresholds = _validate_thresholds(calibrated_thresholds)

    vectorizer = joblib.load(resolved_dir / "tfidf_vectorizer.joblib")
    classifiers = {
        aspect: joblib.load(resolved_dir / f"{aspect}_classifier.joblib")
        for aspect in EXPECTED_ASPECTS
    }
    for aspect, classifier in classifiers.items():
        if 1 not in classifier.classes_:
            raise ValueError(f"{aspect} classifier has no present=1 class")

    _DETECTOR_CACHE = {
        "model_dir": resolved_dir,
        "metadata": metadata,
        "vectorizer": vectorizer,
        "classifiers": classifiers,
        "thresholds": calibrated_thresholds,
    }
    return _DETECTOR_CACHE


def _validate_thresholds(
    thresholds: dict[str, float],
) -> dict[str, float]:
    if set(thresholds) != set(EXPECTED_ASPECTS):
        raise ValueError(
            "thresholds must contain exactly: " + ", ".join(EXPECTED_ASPECTS)
        )
    result = {aspect: float(thresholds[aspect]) for aspect in EXPECTED_ASPECTS}
    if any(not 0.0 <= value <= 1.0 for value in result.values()):
        raise ValueError("Every threshold must be within [0, 1]")
    return result


def _resolve_thresholds(
    detector: dict[str, Any],
    override: float | dict[str, float] | None,
) -> dict[str, float]:
    if override is None:
        return detector["thresholds"]
    if isinstance(override, (int, float)):
        value = float(override)
        return _validate_thresholds({aspect: value for aspect in EXPECTED_ASPECTS})
    if isinstance(override, dict):
        return _validate_thresholds(override)
    raise TypeError("thresholds must be None, a float, or a four-aspect dictionary")


def detect_aspects(
    text: str,
    *,
    thresholds: float | dict[str, float] | None = None,
) -> dict[str, dict[str, bool | float]]:
    """Return present/absent decisions and present probabilities for four aspects."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    detector = load_detector()
    decision_thresholds = _resolve_thresholds(detector, thresholds)
    features = detector["vectorizer"].transform([text.strip()])
    result: dict[str, dict[str, bool | float]] = {}
    for aspect in EXPECTED_ASPECTS:
        classifier = detector["classifiers"][aspect]
        present_index = list(classifier.classes_).index(1)
        probability = float(classifier.predict_proba(features)[0, present_index])
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Invalid probability for {aspect}: {probability}")
        result[aspect] = {
            "present": probability >= decision_thresholds[aspect],
            "probability": probability,
        }
    return result


def run_smoke_test() -> None:
    examples = (
        # Raw M-ABSA train all-zero: source category = restaurant general.
        "Finally a reliable Chinese restaurant !",
        # Raw M-ABSA train target-containing: food quality + service general.
        "We have gone for dinner only a few times but the same great quality and service is given .",
    )
    for text in examples:
        predictions = detect_aspects(text)
        if tuple(predictions) != EXPECTED_ASPECTS:
            raise RuntimeError("Smoke test returned an invalid aspect set")
        print(f"Text: {text}")
        print(json.dumps(predictions, ensure_ascii=False, indent=2))
    print("ASPECT PRESENCE SMOKE TEST: PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Restaurant review to classify")
    group.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
        return
    print(json.dumps(detect_aspects(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
