"""One-time final TEST evaluation for the frozen classical aspect detector.

This script never fits a model, searches thresholds, or loads the experimental
transformer. It evaluates the existing Aspect Presence V2 Logistic Regression
artifacts with the thresholds selected previously on DEV.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "aspect_presence_v2"
TRAIN_PATH = DATA_DIR / "train.csv"
DEV_PATH = DATA_DIR / "dev.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "aspect_presence_logreg_v2"
REPORT_PATH = (
    PROJECT_ROOT / "data" / "analysis" / "aspect_presence_final_test_evaluation.txt"
)

ASPECTS = ("taste", "atmosphere", "service", "value")
LANGUAGES = ("tr", "en", "es")
EXPECTED_COLUMNS = ("sentence", "language", "split", *ASPECTS)
EXPECTED_TEST_ROWS = 1623
EXPECTED_TEST_ALL_ZERO = 496
EXPECTED_THRESHOLDS = {
    "taste": 0.50,
    "atmosphere": 0.55,
    "service": 0.50,
    "value": 0.60,
}
DEV_REFERENCE = {
    "macro_f1": 0.7158,
    "micro_f1": 0.7705,
    "exact_match": 0.6507,
    "hamming_loss": 0.1056,
    "all_zero_exact_accuracy": 0.7783,
    "aspect_f1": {
        "taste": 0.8211,
        "atmosphere": 0.6810,
        "service": 0.7914,
        "value": 0.5697,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_hashes() -> dict[str, str]:
    paths = [MODEL_DIR / "tfidf_vectorizer.joblib"] + [
        MODEL_DIR / f"{aspect}_classifier.joblib" for aspect in ASPECTS
    ]
    return {path.name: file_sha256(path) for path in paths}


def load_rows(path: Path, expected_split: str) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected schema: {path}")
        rows: list[dict[str, object]] = list(reader)
    for row in rows:
        if row["split"] != expected_split:
            raise ValueError(f"Unexpected split in {path}")
        if not str(row["sentence"]).strip():
            raise ValueError(f"Empty sentence in {path}")
        if row["language"] not in LANGUAGES:
            raise ValueError(f"Unsupported language in {path}")
        for aspect in ASPECTS:
            if row[aspect] not in {"0", "1"}:
                raise ValueError(f"Non-binary {aspect} label in {path}")
            row[aspect] = int(row[aspect])
    return rows


def label_array(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [[int(row[aspect]) for aspect in ASPECTS] for row in rows],
        dtype=np.int64,
    )


def exact_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["sentence"]), str(row["language"])


def leakage_audit(train_rows, dev_rows, test_rows) -> dict[str, int]:
    train_keys = {exact_key(row) for row in train_rows}
    dev_keys = {exact_key(row) for row in dev_rows}
    test_keys = [exact_key(row) for row in test_rows]
    test_counter = Counter(test_keys)
    return {
        "train_test": len(train_keys & set(test_keys)),
        "dev_test": len(dev_keys & set(test_keys)),
        "test_duplicate_excess_rows": sum(
            count - 1 for count in test_counter.values() if count > 1
        ),
    }


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int | float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int((y_true == 1).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    by_aspect = {
        aspect: binary_counts(y_true[:, index], y_pred[:, index])
        for index, aspect in enumerate(ASPECTS)
    }
    zero_mask = y_true.sum(axis=1) == 0
    target_mask = ~zero_mask
    zero_predictions = y_pred[zero_mask]
    target_true = y_true[target_mask]
    target_pred = y_pred[target_mask]
    zero_correct = int(np.all(zero_predictions == 0, axis=1).sum())
    return {
        "macro_f1": float(np.mean([item["f1"] for item in by_aspect.values()])),
        "micro_f1": float(micro_f1(y_true, y_pred)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(np.mean(y_true != y_pred)),
        "by_aspect": by_aspect,
        "all_zero_support": int(zero_mask.sum()),
        "all_zero_correct": zero_correct,
        "all_zero_exact_accuracy": zero_correct / int(zero_mask.sum()),
        "all_zero_false_positive_total": int(zero_predictions.sum()),
        "all_zero_false_positive_by_aspect": {
            aspect: int(zero_predictions[:, index].sum())
            for index, aspect in enumerate(ASPECTS)
        },
        "target_support": int(target_mask.sum()),
        "target_macro_f1": float(
            np.mean(
                [
                    binary_counts(target_true[:, index], target_pred[:, index])["f1"]
                    for index in range(len(ASPECTS))
                ]
            )
        ),
        "target_micro_f1": float(micro_f1(target_true, target_pred)),
        "target_exact_match": float(
            np.mean(np.all(target_true == target_pred, axis=1))
        ),
    }


def load_frozen_predictions(test_rows):
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("artifact_version") != "aspect_presence_logreg_v2":
        raise RuntimeError("Unexpected model artifact version")
    if tuple(metadata.get("aspects", ())) != ASPECTS:
        raise RuntimeError("Unexpected model aspect order")
    thresholds = {
        aspect: float(metadata.get("thresholds", {}).get(aspect))
        for aspect in ASPECTS
    }
    if thresholds != EXPECTED_THRESHOLDS:
        raise RuntimeError(
            f"Frozen thresholds changed: expected {EXPECTED_THRESHOLDS}, got {thresholds}"
        )
    if metadata.get("threshold_selection_split") != "dev":
        raise RuntimeError("Threshold provenance is not DEV")
    if metadata.get("threshold_selection_test_used") is not False:
        raise RuntimeError("Metadata indicates TEST was used for threshold selection")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    features = vectorizer.transform([str(row["sentence"]) for row in test_rows])
    probabilities = np.zeros((len(test_rows), len(ASPECTS)), dtype=np.float64)
    for index, aspect in enumerate(ASPECTS):
        classifier = joblib.load(MODEL_DIR / f"{aspect}_classifier.joblib")
        present_index = list(classifier.classes_).index(1)
        probabilities[:, index] = classifier.predict_proba(features)[:, present_index]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite TEST probability")
    threshold_array = np.asarray([thresholds[aspect] for aspect in ASPECTS])
    predictions = (probabilities >= threshold_array).astype(np.int64)
    return metadata, thresholds, predictions


def language_breakdown(test_rows, y_true, y_pred) -> dict[str, dict[str, object]]:
    output = {}
    for language in LANGUAGES:
        indices = np.asarray(
            [index for index, row in enumerate(test_rows) if row["language"] == language]
        )
        output[language] = evaluate(y_true[indices], y_pred[indices])
        output[language]["rows"] = int(len(indices))
    return output


def format_report(counts, leakage, thresholds, metrics, language, hashes_unchanged):
    lines = [
        "ASPECT PRESENCE — FINAL TEST EVALUATION",
        "=" * 88,
        "",
        "FINAL MODEL",
        "Model: Aspect Presence V2 Classical Calibrated",
        "Artifact: models/aspect_presence_logreg_v2/",
        "Vectorizer/classifiers refit: NO",
        "Threshold tuning on TEST: NO",
        "Transformer TEST evaluation: NO",
        f"Model artifact hashes unchanged after evaluation: {'YES' if hashes_unchanged else 'NO'}",
        "",
        "FINAL THRESHOLDS — SELECTED ON DEV",
        *[f"{aspect}: {thresholds[aspect]:.2f}" for aspect in ASPECTS],
        "",
        "TEST DATA SANITY",
        f"Total rows: {counts['total']}",
        f"TR rows: {counts['languages']['tr']}",
        f"EN rows: {counts['languages']['en']}",
        f"ES rows: {counts['languages']['es']}",
        *[
            f"{aspect} positive: {counts['aspect_positive'][aspect]}"
            for aspect in ASPECTS
        ],
        f"All-zero rows: {counts['all_zero']}",
        "",
        "LEAKAGE AND DUPLICATES — EXACT SENTENCE + LANGUAGE",
        f"Train-Test exact leakage: {leakage['train_test']}",
        f"Dev-Test exact leakage: {leakage['dev_test']}",
        f"Test duplicate excess rows: {leakage['test_duplicate_excess_rows']}",
        "",
        "TEST OVERALL",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        f"Micro F1: {metrics['micro_f1']:.4f}",
        f"Exact Match: {metrics['exact_match']:.4f}",
        f"Hamming Loss: {metrics['hamming_loss']:.4f}",
        "",
        "ASPECT METRICS",
    ]
    for aspect in ASPECTS:
        item = metrics["by_aspect"][aspect]
        lines.append(
            f"{aspect}: P={item['precision']:.4f} | R={item['recall']:.4f} | "
            f"F1={item['f1']:.4f} | support={item['support']} | "
            f"TP={item['tp']} FP={item['fp']} FN={item['fn']} TN={item['tn']}"
        )
    lines.extend(
        [
            "",
            "ALL-ZERO TEST",
            f"Support: {metrics['all_zero_support']}",
            f"Correct 0/0/0/0: {metrics['all_zero_correct']}",
            f"Exact accuracy: {metrics['all_zero_exact_accuracy']:.4f}",
            f"False-positive aspect total: {metrics['all_zero_false_positive_total']}",
            *[
                f"{aspect} false positive: "
                f"{metrics['all_zero_false_positive_by_aspect'][aspect]}"
                for aspect in ASPECTS
            ],
            "",
            "TARGET-CONTAINING TEST",
            f"Support: {metrics['target_support']}",
            f"Macro F1: {metrics['target_macro_f1']:.4f}",
            f"Micro F1: {metrics['target_micro_f1']:.4f}",
            f"Exact Match: {metrics['target_exact_match']:.4f}",
            "",
            "LANGUAGE BREAKDOWN",
        ]
    )
    for code in LANGUAGES:
        item = language[code]
        lines.append(
            f"{code.upper()}: rows={item['rows']} | Macro F1={item['macro_f1']:.4f} | "
            f"Micro F1={item['micro_f1']:.4f} | Exact Match={item['exact_match']:.4f} | "
            f"All-zero exact={item['all_zero_exact_accuracy']:.4f} "
            f"(support={item['all_zero_support']})"
        )
    lines.extend(
        [
            "",
            "TEST MINUS DEV",
            f"Macro F1 delta: {metrics['macro_f1'] - DEV_REFERENCE['macro_f1']:+.4f}",
            f"Micro F1 delta: {metrics['micro_f1'] - DEV_REFERENCE['micro_f1']:+.4f}",
            f"Exact Match delta: {metrics['exact_match'] - DEV_REFERENCE['exact_match']:+.4f}",
            f"Hamming Loss delta: {metrics['hamming_loss'] - DEV_REFERENCE['hamming_loss']:+.4f}",
            "All-zero exact delta: "
            f"{metrics['all_zero_exact_accuracy'] - DEV_REFERENCE['all_zero_exact_accuracy']:+.4f}",
        ]
    )
    for aspect in ASPECTS:
        lines.append(
            f"{aspect} F1 delta: "
            f"{metrics['by_aspect'][aspect]['f1'] - DEV_REFERENCE['aspect_f1'][aspect]:+.4f}"
        )
    lines.extend(
        [
            "",
            "LIMITATIONS",
            "- All-zero rejection is not perfect.",
            "- Value detection may remain weaker than other aspects.",
            "- Multilingual performance varies by language.",
            "- The source ontology can be ambiguous for some natural sentences.",
            "- Thresholds were selected on DEV.",
            "- TEST was used only for this final unbiased estimate, not tuning.",
            "",
            "POST-TEST POLICY",
            "Model, thresholds, training configuration, and selection remain frozen.",
            "No TEST-driven tuning or model change was performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    before_hashes = model_hashes()
    train_rows = load_rows(TRAIN_PATH, "train")
    dev_rows = load_rows(DEV_PATH, "dev")
    test_rows = load_rows(TEST_PATH, "test")
    y_test = label_array(test_rows)

    counts = {
        "total": len(test_rows),
        "languages": dict(Counter(str(row["language"]) for row in test_rows)),
        "aspect_positive": {
            aspect: int(y_test[:, index].sum())
            for index, aspect in enumerate(ASPECTS)
        },
        "all_zero": int((y_test.sum(axis=1) == 0).sum()),
    }
    if counts["total"] != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_TEST_ROWS} TEST rows, got {counts['total']}")
    if counts["all_zero"] != EXPECTED_TEST_ALL_ZERO:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_ALL_ZERO} all-zero TEST rows, got {counts['all_zero']}"
        )

    leakage = leakage_audit(train_rows, dev_rows, test_rows)
    if any(leakage.values()):
        raise RuntimeError(f"Final TEST leakage/duplicate audit failed: {leakage}")

    _, thresholds, predictions = load_frozen_predictions(test_rows)
    metrics = evaluate(y_test, predictions)
    language = language_breakdown(test_rows, y_test, predictions)
    after_hashes = model_hashes()
    hashes_unchanged = before_hashes == after_hashes
    if not hashes_unchanged:
        raise RuntimeError("Frozen model artifact changed during final evaluation")

    report = format_report(
        counts,
        leakage,
        thresholds,
        metrics,
        language,
        hashes_unchanged,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
