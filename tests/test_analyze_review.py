"""Lightweight regression tests for sentence-aware review analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import call, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import analyze_review as unified  # noqa: E402


def _presence(
    *,
    probabilities: dict[str, float] | None = None,
    present: set[str] | None = None,
) -> dict[str, dict[str, bool | float]]:
    probabilities = probabilities or {}
    present = present or set()
    return {
        aspect: {
            "present": aspect in present,
            "probability": probabilities.get(aspect, 0.1),
        }
        for aspect in unified.ASPECT_ORDER
    }


def _sentiment(
    text: str,
    aspect: str,
    probabilities: dict[str, float] | None = None,
) -> dict[str, object]:
    del text
    probabilities = probabilities or {
        "negative": 0.1,
        "neutral": 0.2,
        "positive": 0.7,
    }
    label = max(probabilities, key=probabilities.__getitem__)
    return {
        "aspect": aspect,
        "sentiment": label,
        "confidence": probabilities[label],
        "probabilities": probabilities,
    }


class SentenceSegmentationTests(unittest.TestCase):
    def test_single_sentence_segmentation(self) -> None:
        self.assertEqual(
            unified.segment_review("Coffee was good."),
            ["Coffee was good."],
        )

    def test_multisentence_segmentation_preserves_decimals(self) -> None:
        self.assertEqual(
            unified.segment_review("It cost 3.5€. Fine!\nService?"),
            ["It cost 3.5€.", "Fine!", "Service?"],
        )

    def test_barbican_fragment_is_not_split_from_its_sentence(self) -> None:
        review = (
            "Barbican Kitchen is a good option if you want a quick and casual "
            "meal while exploring the Barbican Centre. It’s self-service."
        )
        sentences = unified.segment_review(review)
        self.assertEqual(len(sentences), 2)
        self.assertIn("exploring the Barbican Centre", sentences[0])
        self.assertNotIn("exploring the Barbican Centre.", sentences[1:])


class SentenceAwarePipelineTests(unittest.TestCase):
    def test_absent_aspects_do_not_call_sentiment(self) -> None:
        with (
            patch.object(unified, "detect_aspects", return_value=_presence()),
            patch.object(unified, "predict_sentiment") as sentiment_mock,
        ):
            result = unified.analyze_review("Only a general restaurant comment.")

        sentiment_mock.assert_not_called()
        self.assertTrue(result["no_target_aspects_detected"])
        self.assertEqual(result["sentiment_calls"], 0)
        for item in result["aspects"].values():
            self.assertFalse(item["present"])
            self.assertEqual(item["evidence_sentences"], [])
            self.assertIsNone(item["sentiment"])

    def test_multiple_evidence_sentences_use_presence_weighted_mean(self) -> None:
        first_presence = _presence(
            probabilities={"taste": 0.6}, present={"taste"}
        )
        second_presence = _presence(
            probabilities={"taste": 0.9}, present={"taste"}
        )

        def sentiment_side_effect(text: str, aspect: str) -> dict[str, object]:
            probabilities = (
                {"negative": 0.8, "neutral": 0.1, "positive": 0.1}
                if text == "First."
                else {"negative": 0.2, "neutral": 0.2, "positive": 0.6}
            )
            return _sentiment(text, aspect, probabilities)

        with (
            patch.object(
                unified,
                "detect_aspects",
                side_effect=[first_presence, second_presence],
            ),
            patch.object(
                unified,
                "predict_sentiment",
                side_effect=sentiment_side_effect,
            ) as sentiment_mock,
        ):
            result = unified.analyze_review("First. Second.")

        taste = result["aspects"]["taste"]
        self.assertEqual(
            sentiment_mock.call_args_list,
            [call("First.", "taste"), call("Second.", "taste")],
        )
        self.assertEqual(result["sentiment_calls"], 2)
        self.assertEqual(taste["presence_probability"], 0.9)
        self.assertEqual(taste["sentiment"], "negative")
        self.assertAlmostEqual(taste["sentiment_probabilities"]["negative"], 0.44)
        self.assertAlmostEqual(taste["sentiment_probabilities"]["neutral"], 0.16)
        self.assertAlmostEqual(taste["sentiment_probabilities"]["positive"], 0.40)
        self.assertAlmostEqual(sum(taste["sentiment_probabilities"].values()), 1.0)

    def test_multisentence_aspects_use_separate_sentence_evidence(self) -> None:
        text = "The coffee was excellent. The service was slow."
        taste_presence = _presence(
            probabilities={"taste": 0.8}, present={"taste"}
        )
        service_presence = _presence(
            probabilities={"service": 0.9}, present={"service"}
        )
        with (
            patch.object(
                unified,
                "detect_aspects",
                side_effect=[taste_presence, service_presence],
            ),
            patch.object(
                unified,
                "predict_sentiment",
                side_effect=lambda sentence, aspect: _sentiment(sentence, aspect),
            ) as sentiment_mock,
        ):
            result = unified.analyze_review(text)

        self.assertEqual(
            sentiment_mock.call_args_list,
            [
                call("The coffee was excellent.", "taste"),
                call("The service was slow.", "service"),
            ],
        )
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(
            result["aspects"]["taste"]["evidence_sentences"][0]["sentence_index"],
            0,
        )
        self.assertEqual(
            result["aspects"]["service"]["evidence_sentences"][0]["sentence_index"],
            1,
        )
        self.assertEqual(result["sentiment_calls"], 2)
        self.assertNotIn("clause_candidate_count", result)
        self.assertNotIn("clause_candidates", result)
        self.assertNotIn("clause_diagnostics", result)

    def test_single_sentence_output_and_sentiment_fidelity(self) -> None:
        text = "The coffee was excellent."
        presence = _presence(
            probabilities={"taste": 0.8}, present={"taste"}
        )
        with (
            patch.object(unified, "detect_aspects", return_value=presence),
            patch.object(
                unified,
                "predict_sentiment",
                side_effect=lambda sentence, aspect: _sentiment(sentence, aspect),
            ),
        ):
            result = unified.analyze_review(text)
            fidelity = unified._fidelity_check(text, result)

        self.assertEqual(result["sentence_count"], 1)
        self.assertEqual(result["sentences"], [text])
        self.assertEqual(result["aspects"]["taste"]["presence_probability"], 0.8)
        self.assertEqual(fidelity["status"], "PASS")
        self.assertEqual(fidelity["max_probability_difference"], 0.0)

    def test_every_aggregated_probability_vector_sums_to_one(self) -> None:
        presence = _presence(
            probabilities={"taste": 0.8, "service": 0.7},
            present={"taste", "service"},
        )
        with (
            patch.object(unified, "detect_aspects", return_value=presence),
            patch.object(
                unified,
                "predict_sentiment",
                side_effect=lambda sentence, aspect: _sentiment(sentence, aspect),
            ),
        ):
            result = unified.analyze_review("Coffee and service were good.")

        for item in result["aspects"].values():
            if item["present"]:
                self.assertAlmostEqual(
                    sum(item["sentiment_probabilities"].values()), 1.0
                )

    def test_empty_text_is_rejected_before_model_calls(self) -> None:
        for invalid in (None, "", "   "):
            with self.subTest(invalid=invalid):
                with (
                    patch.object(unified, "detect_aspects") as detector_mock,
                    patch.object(unified, "predict_sentiment") as sentiment_mock,
                ):
                    with self.assertRaises(ValueError):
                        unified.analyze_review(invalid)  # type: ignore[arg-type]
                detector_mock.assert_not_called()
                sentiment_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
