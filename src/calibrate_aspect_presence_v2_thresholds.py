"""Calibrate per-aspect thresholds for the frozen Aspect Presence v2 model.

This script performs no fit, training, or test-set access. It loads the existing
TF-IDF vectorizer and four Logistic Regression classifiers, computes DEV
probabilities once, exhaustively evaluates the requested threshold grid, updates
metadata, and writes a diagnostic report.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_PATH = PROJECT_ROOT / "data" / "aspect_presence_v2" / "dev.csv"
V1_DEV_PATH = PROJECT_ROOT / "data" / "aspect_presence_v1" / "dev.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "aspect_presence_logreg_v2"
METADATA_PATH = MODEL_DIR / "metadata.json"
REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "aspect_presence_v2_threshold_calibration.txt"
)

ASPECTS = ("taste", "atmosphere", "service", "value")
LANGUAGES = ("tr", "en", "es")
THRESHOLD_CANDIDATES = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
DEFAULT_THRESHOLDS = {aspect: 0.50 for aspect in ASPECTS}
MACRO_F1_GUARDRAIL = 0.7141
EXPECTED_BASELINE_MACRO_F1 = 0.7241
EXPECTED_BASELINE_EXACT_MATCH = 0.6433
V1_REFERENCE = {"macro_f1": 0.7542, "exact_match": 0.6548}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_file_hashes() -> dict[str, str]:
    files = [MODEL_DIR / "tfidf_vectorizer.joblib"] + [
        MODEL_DIR / f"{aspect}_classifier.joblib" for aspect in ASPECTS
    ]
    return {path.name: file_sha256(path) for path in files}


def load_presence_rows(path: Path, expected_split: str):
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        expected_columns = ("sentence", "language", "split", *ASPECTS)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(f"Unexpected presence schema: {path}")
        rows = list(reader)
    for row in rows:
        if row["split"] != expected_split:
            raise ValueError(f"Unexpected split value in {path}")
        for aspect in ASPECTS:
            row[aspect] = int(row[aspect])
    return rows


def label_array(rows) -> np.ndarray:
    return np.asarray(
        [[int(row[aspect]) for aspect in ASPECTS] for row in rows],
        dtype=np.int64,
    )


def load_frozen_probabilities(rows):
    """Load frozen artifacts and return positive-class DEV probabilities."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("artifact_version") != "aspect_presence_logreg_v2":
        raise ValueError("Unexpected artifact version")
    if metadata.get("test_evaluated") is not False:
        raise ValueError("Artifact metadata does not preserve untouched TEST status")

    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    features = vectorizer.transform([row["sentence"] for row in rows])
    probabilities = np.zeros((len(rows), len(ASPECTS)), dtype=float)
    for column, aspect in enumerate(ASPECTS):
        classifier = joblib.load(MODEL_DIR / f"{aspect}_classifier.joblib")
        present_index = list(classifier.classes_).index(1)
        probabilities[:, column] = classifier.predict_proba(features)[:, present_index]
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite DEV probability")
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise ValueError("DEV probability outside [0, 1]")
    return metadata, probabilities


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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    by_aspect = {
        aspect: binary_counts(y_true[:, column], y_pred[:, column])
        for column, aspect in enumerate(ASPECTS)
    }
    tp_total = sum(item["tp"] for item in by_aspect.values())
    fp_total = sum(item["fp"] for item in by_aspect.values())
    fn_total = sum(item["fn"] for item in by_aspect.values())
    micro_precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )
    cardinality = y_true.sum(axis=1)
    zero_mask = cardinality == 0
    target_mask = cardinality > 0
    zero_predictions = y_pred[zero_mask]
    target_true = y_true[target_mask]
    target_pred = y_pred[target_mask]
    return {
        "macro_f1": float(np.mean([item["f1"] for item in by_aspect.values()])),
        "micro_f1": float(micro_f1),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(np.mean(y_true != y_pred)),
        "by_aspect": by_aspect,
        "all_zero_support": int(zero_mask.sum()),
        "all_zero_exact_accuracy": (
            float(np.mean(np.all(zero_predictions == 0, axis=1)))
            if zero_predictions.shape[0] else None
        ),
        "all_zero_false_positive_total": int(zero_predictions.sum()),
        "all_zero_false_positive_by_aspect": {
            aspect: int(zero_predictions[:, column].sum())
            for column, aspect in enumerate(ASPECTS)
        },
        "target_support": int(target_mask.sum()),
        "target_macro_f1": float(
            np.mean(
                [
                    binary_counts(target_true[:, column], target_pred[:, column])["f1"]
                    for column in range(len(ASPECTS))
                ]
            )
        ),
        "target_micro_f1": float(
            _micro_f1_for_subset(target_true, target_pred)
        ),
        "target_exact_match": float(
            np.mean(np.all(target_true == target_pred, axis=1))
        ),
    }


def _micro_f1_for_subset(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def predictions_for(probabilities: np.ndarray, thresholds: dict[str, float]):
    threshold_array = np.asarray([thresholds[aspect] for aspect in ASPECTS])
    return (probabilities >= threshold_array).astype(np.int64)


def single_aspect_analysis(y_true, probabilities):
    zero_mask = y_true.sum(axis=1) == 0
    analysis = {}
    for column, aspect in enumerate(ASPECTS):
        rows = []
        for threshold in THRESHOLD_CANDIDATES:
            prediction = (probabilities[:, column] >= threshold).astype(np.int64)
            metrics = binary_counts(y_true[:, column], prediction)
            metrics.update(
                {
                    "threshold": threshold,
                    "all_zero_fp": int(prediction[zero_mask].sum()),
                }
            )
            rows.append(metrics)
        analysis[aspect] = rows
    return analysis


def joint_search(y_true, probabilities):
    results = []
    for values in itertools.product(THRESHOLD_CANDIDATES, repeat=len(ASPECTS)):
        thresholds = dict(zip(ASPECTS, values))
        predictions = predictions_for(probabilities, thresholds)
        metrics = evaluate_predictions(y_true, predictions)
        results.append({"thresholds": thresholds, "metrics": metrics})

    eligible = [
        result
        for result in results
        if result["metrics"]["macro_f1"] >= MACRO_F1_GUARDRAIL
    ]
    if not eligible:
        raise RuntimeError("No threshold combination passed the Macro F1 guardrail")

    max_exact = max(result["metrics"]["exact_match"] for result in eligible)
    one_dev_example = 1.0 / len(y_true)
    near_best_exact = [
        result
        for result in eligible
        if result["metrics"]["exact_match"] >= max_exact - one_dev_example - 1e-12
    ]
    max_zero_exact = max(
        result["metrics"]["all_zero_exact_accuracy"]
        for result in near_best_exact
    )
    zero_best = [
        result for result in near_best_exact
        if abs(
            result["metrics"]["all_zero_exact_accuracy"] - max_zero_exact
        ) < 1e-12
    ]
    max_macro = max(result["metrics"]["macro_f1"] for result in zero_best)
    macro_best = [
        result for result in zero_best
        if abs(result["metrics"]["macro_f1"] - max_macro) < 1e-12
    ]

    def conservative_key(result):
        distance = sum(
            abs(result["thresholds"][aspect] - 0.50) for aspect in ASPECTS
        )
        threshold_tuple = tuple(result["thresholds"][aspect] for aspect in ASPECTS)
        return distance, threshold_tuple

    selected = min(macro_best, key=conservative_key)
    return selected, {
        "combination_count": len(results),
        "eligible_count": len(eligible),
        "max_exact_match": max_exact,
        "near_best_exact_tolerance": one_dev_example,
        "near_best_exact_count": len(near_best_exact),
    }


def language_breakdown(rows, y_true, y_pred):
    result = {}
    for language in LANGUAGES:
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["language"] == language]
        )
        result[language] = evaluate_predictions(y_true[indices], y_pred[indices])
    return result


def common_subset_metrics(dev_rows, probabilities, default_predictions, calibrated_predictions):
    v1_rows = load_presence_rows(V1_DEV_PATH, "dev")
    index = {(row["sentence"], row["language"]): i for i, row in enumerate(dev_rows)}
    common_indices = np.asarray(
        [index[(row["sentence"], row["language"])] for row in v1_rows]
    )
    y_common = label_array(v1_rows)
    v2_labels = label_array([dev_rows[i] for i in common_indices])
    if not np.array_equal(y_common, v2_labels):
        raise ValueError("V1 and V2 labels differ on the common DEV subset")
    return {
        "rows": len(v1_rows),
        "v1_reference": V1_REFERENCE,
        "v2_default": evaluate_predictions(
            y_common, default_predictions[common_indices]
        ),
        "v2_calibrated": evaluate_predictions(
            y_common, calibrated_predictions[common_indices]
        ),
    }


def update_metadata(metadata, selected_thresholds, search_info, default_metrics, calibrated_metrics):
    metadata["default_threshold"] = 0.5
    metadata["thresholds"] = selected_thresholds
    metadata["threshold_selection_split"] = "dev"
    metadata["threshold_selection_test_used"] = False
    metadata["threshold_selection_candidates"] = list(THRESHOLD_CANDIDATES)
    metadata["threshold_selection_guardrail_macro_f1"] = MACRO_F1_GUARDRAIL
    metadata["threshold_selection_rule"] = (
        "Macro F1 guardrail; Exact Match within one DEV example of maximum; "
        "then highest all-zero exact accuracy; then Macro F1; then minimum "
        "total distance from 0.50"
    )
    metadata["threshold_selection_combination_count"] = search_info[
        "combination_count"
    ]
    metadata["dev_metrics_default_thresholds"] = {
        key: default_metrics[key]
        for key in ("macro_f1", "micro_f1", "exact_match", "hamming_loss")
    }
    metadata["dev_metrics_calibrated_thresholds"] = {
        key: calibrated_metrics[key]
        for key in ("macro_f1", "micro_f1", "exact_match", "hamming_loss")
    }
    metadata["classifier"]["default_probability_threshold"] = 0.5
    metadata["classifier"]["probability_thresholds"] = selected_thresholds
    metadata["classifier"].pop("probability_threshold", None)
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def decision_label(default_metrics, calibrated_metrics):
    macro_delta = calibrated_metrics["macro_f1"] - default_metrics["macro_f1"]
    exact_delta = calibrated_metrics["exact_match"] - default_metrics["exact_match"]
    zero_delta = (
        calibrated_metrics["all_zero_exact_accuracy"]
        - default_metrics["all_zero_exact_accuracy"]
    )
    fp_delta = (
        calibrated_metrics["all_zero_false_positive_total"]
        - default_metrics["all_zero_false_positive_total"]
    )
    atmosphere_delta = (
        calibrated_metrics["by_aspect"]["atmosphere"]["f1"]
        - default_metrics["by_aspect"]["atmosphere"]["f1"]
    )
    value_delta = (
        calibrated_metrics["by_aspect"]["value"]["f1"]
        - default_metrics["by_aspect"]["value"]["f1"]
    )
    if (
        macro_delta >= -0.005
        and exact_delta >= 0.02
        and zero_delta >= 0.05
        and fp_delta <= -10
        and atmosphere_delta >= -0.01
        and value_delta >= -0.01
    ):
        return "YES"
    if exact_delta > 0 or zero_delta > 0 or fp_delta < 0:
        return "MAYBE"
    return "NO"


def write_report(
    single_analysis,
    selected,
    search_info,
    default_metrics,
    calibrated_metrics,
    language,
    common,
    weights_unchanged,
):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    thresholds = selected["thresholds"]
    decision = decision_label(default_metrics, calibrated_metrics)
    lines = [
        "ASPECT PRESENCE V2 — PER-ASPECT THRESHOLD CALIBRATION",
        "=" * 96,
        "",
        "SAFETY AND REPRODUCIBILITY",
        "Model refit/training: NO",
        "DEV probabilities generated from existing v2 artifacts: YES",
        "TEST loaded or evaluated: NO",
        f"Model/vectorizer hashes unchanged: {'YES' if weights_unchanged else 'NO'}",
        "Threshold selection split: DEV",
        (
            "Warning: thresholds were selected on DEV and must not be changed "
            "after TEST is opened."
        ),
        "",
        "SEARCH",
        "Candidates per aspect: " + ", ".join(f"{x:.2f}" for x in THRESHOLD_CANDIDATES),
        f"Joint combinations evaluated: {search_info['combination_count']}",
        f"Macro F1 guardrail: {MACRO_F1_GUARDRAIL:.4f}",
        f"Combinations passing guardrail: {search_info['eligible_count']}",
        f"Maximum eligible Exact Match: {search_info['max_exact_match']:.4f}",
        (
            "Near-best Exact Match tolerance: one DEV example = "
            f"{search_info['near_best_exact_tolerance']:.6f}"
        ),
        f"Near-best Exact Match combinations: {search_info['near_best_exact_count']}",
        (
            "Selection order: Macro F1 guardrail -> Exact Match within one DEV "
            "example of maximum -> all-zero Exact Accuracy -> Macro F1 -> "
            "minimum distance from 0.50."
        ),
        "",
        "THRESHOLDS",
        "Default: " + ", ".join(f"{a}=0.50" for a in ASPECTS),
        "Selected: " + ", ".join(f"{a}={thresholds[a]:.2f}" for a in ASPECTS),
        "",
        "DEFAULT VS CALIBRATED",
        "Metric                              Default    Calibrated    Delta",
    ]
    for key, label in (
        ("macro_f1", "Macro F1"),
        ("micro_f1", "Micro F1"),
        ("exact_match", "Exact Match"),
        ("hamming_loss", "Hamming Loss"),
        ("all_zero_exact_accuracy", "All-zero Exact Accuracy"),
        ("target_exact_match", "Target-containing Exact Match"),
    ):
        lines.append(
            f"{label:<35} {default_metrics[key]:>8.4f}    "
            f"{calibrated_metrics[key]:>8.4f}    "
            f"{calibrated_metrics[key] - default_metrics[key]:>+8.4f}"
        )
    lines.append(
        f"{'All-zero FP total':<35} "
        f"{default_metrics['all_zero_false_positive_total']:>8d}    "
        f"{calibrated_metrics['all_zero_false_positive_total']:>8d}    "
        f"{calibrated_metrics['all_zero_false_positive_total'] - default_metrics['all_zero_false_positive_total']:>+8d}"
    )
    lines.extend(["", "ASPECT F1 — DEFAULT VS CALIBRATED"])
    for aspect in ASPECTS:
        before = default_metrics["by_aspect"][aspect]["f1"]
        after = calibrated_metrics["by_aspect"][aspect]["f1"]
        lines.append(
            f"{aspect}: {before:.4f} -> {after:.4f} (delta={after - before:+.4f})"
        )

    lines.extend(["", "SINGLE-ASPECT THRESHOLD ANALYSIS"])
    for aspect in ASPECTS:
        lines.extend(
            [
                "",
                aspect.upper(),
                "Threshold  Precision  Recall    F1        FP    FN    All-zero FP",
            ]
        )
        for row in single_analysis[aspect]:
            lines.append(
                f"{row['threshold']:<10.2f} {row['precision']:<10.4f} "
                f"{row['recall']:<9.4f} {row['f1']:<9.4f} "
                f"{row['fp']:<5d} {row['fn']:<5d} {row['all_zero_fp']:<5d}"
            )

    lines.extend(
        [
            "",
            "ALL-ZERO ANALYSIS",
            f"Support: {calibrated_metrics['all_zero_support']}",
            (
                "False positives by aspect (default): "
                + ", ".join(
                    f"{aspect}={default_metrics['all_zero_false_positive_by_aspect'][aspect]}"
                    for aspect in ASPECTS
                )
            ),
            (
                "False positives by aspect (calibrated): "
                + ", ".join(
                    f"{aspect}={calibrated_metrics['all_zero_false_positive_by_aspect'][aspect]}"
                    for aspect in ASPECTS
                )
            ),
            "",
            "LANGUAGE BREAKDOWN — CALIBRATED",
        ]
    )
    for code in LANGUAGES:
        item = language[code]
        lines.append(
            f"{code}: macro_f1={item['macro_f1']:.4f}, "
            f"exact_match={item['exact_match']:.4f}, "
            f"all_zero_exact_accuracy={item['all_zero_exact_accuracy']:.4f}"
        )

    lines.extend(
        [
            "",
            "COMMON V1 DEV SUBSET — THRESHOLDS SELECTED ON FULL V2 DEV",
            f"Rows: {common['rows']}",
            (
                "V1 reference: "
                f"macro_f1={common['v1_reference']['macro_f1']:.4f}, "
                f"exact_match={common['v1_reference']['exact_match']:.4f}"
            ),
            (
                "V2 default: "
                f"macro_f1={common['v2_default']['macro_f1']:.4f}, "
                f"exact_match={common['v2_default']['exact_match']:.4f}"
            ),
            (
                "V2 calibrated: "
                f"macro_f1={common['v2_calibrated']['macro_f1']:.4f}, "
                f"exact_match={common['v2_calibrated']['exact_match']:.4f}"
            ),
            "",
            "FINAL EVIDENCE-BASED DECISION",
            f"CLASSICAL ASPECT DETECTOR SUFFICIENT: {decision}",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main():
    before_hashes = model_file_hashes()
    dev_rows = load_presence_rows(DEV_PATH, "dev")
    y_dev = label_array(dev_rows)
    metadata, probabilities = load_frozen_probabilities(dev_rows)

    default_predictions = predictions_for(probabilities, DEFAULT_THRESHOLDS)
    default_metrics = evaluate_predictions(y_dev, default_predictions)
    if round(default_metrics["macro_f1"], 4) != EXPECTED_BASELINE_MACRO_F1:
        raise RuntimeError(
            "Default threshold Macro F1 did not reproduce V2 baseline: "
            f"{default_metrics['macro_f1']:.8f}"
        )
    if round(default_metrics["exact_match"], 4) != EXPECTED_BASELINE_EXACT_MATCH:
        raise RuntimeError(
            "Default threshold Exact Match did not reproduce V2 baseline: "
            f"{default_metrics['exact_match']:.8f}"
        )

    single = single_aspect_analysis(y_dev, probabilities)
    selected, search_info = joint_search(y_dev, probabilities)
    selected_thresholds = selected["thresholds"]
    calibrated_predictions = predictions_for(probabilities, selected_thresholds)
    calibrated_metrics = evaluate_predictions(y_dev, calibrated_predictions)
    calibrated_language = language_breakdown(
        dev_rows, y_dev, calibrated_predictions
    )
    common = common_subset_metrics(
        dev_rows,
        probabilities,
        default_predictions,
        calibrated_predictions,
    )

    update_metadata(
        metadata,
        selected_thresholds,
        search_info,
        default_metrics,
        calibrated_metrics,
    )
    after_hashes = model_file_hashes()
    weights_unchanged = before_hashes == after_hashes
    if not weights_unchanged:
        raise RuntimeError("A model/vectorizer artifact changed during calibration")
    decision = write_report(
        single,
        selected,
        search_info,
        default_metrics,
        calibrated_metrics,
        calibrated_language,
        common,
        weights_unchanged,
    )

    print("ASPECT PRESENCE V2 THRESHOLD CALIBRATION COMPLETE")
    print("Training/refit: NO")
    print("TEST used: NO")
    print(
        "Selected thresholds: "
        + ", ".join(
            f"{aspect}={selected_thresholds[aspect]:.2f}" for aspect in ASPECTS
        )
    )
    print(
        f"Macro F1: {default_metrics['macro_f1']:.4f} -> "
        f"{calibrated_metrics['macro_f1']:.4f}"
    )
    print(
        f"Exact Match: {default_metrics['exact_match']:.4f} -> "
        f"{calibrated_metrics['exact_match']:.4f}"
    )
    print(
        "All-zero Exact Accuracy: "
        f"{default_metrics['all_zero_exact_accuracy']:.4f} -> "
        f"{calibrated_metrics['all_zero_exact_accuracy']:.4f}"
    )
    print(
        "All-zero FP total: "
        f"{default_metrics['all_zero_false_positive_total']} -> "
        f"{calibrated_metrics['all_zero_false_positive_total']}"
    )
    print(f"Classical detector sufficient: {decision}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
