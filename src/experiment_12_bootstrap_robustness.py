"""Experiment 12: paired bootstrap robustness checks on the dev split.

The test split is deliberately never loaded. All bootstrap replicates use the
same sampled dev indices for all three models.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from train_sentiment_baseline import SENTIMENTS, build_inputs, build_model, load_rows


ROOT = Path(__file__).resolve().parents[1]
CLEAN_TRAIN_PATH = ROOT / "data" / "cleaned" / "train.csv"
EXPANDED_TRAIN_PATH = ROOT / "data" / "expanded_neutral_v1" / "train.csv"
DEV_PATH = ROOT / "data" / "cleaned" / "dev.csv"
MANIFEST_PATH = ROOT / "data" / "expanded_neutral_v1" / "added_neutral_manifest.csv"
REPORT_PATH = ROOT / "data" / "analysis" / "experiment_12_bootstrap_robustness.txt"
CSV_PATH = ROOT / "data" / "analysis" / "experiment_12_bootstrap_deltas.csv"

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
CHUNK_SIZE = 500
LABEL_TO_ID = {label: index for index, label in enumerate(SENTIMENTS)}
METRICS = ("accuracy", "macro_f1", "neutral_precision", "neutral_recall", "neutral_f1")

FIXED_OLD_WEIGHTS = {
    "positive": 0.5086669734621875,
    "neutral": 11.885304659498209,
    "negative": 1.0526984126984127,
}

EXPECTED = {
    "A": {"macro_f1": 0.5891, "neutral_f1": 0.2529},
    "B": {"macro_f1": 0.5920, "neutral_f1": 0.2529},
    "C": {
        "macro_f1": 0.5913,
        "neutral_precision": 0.2268,
        "neutral_recall": 0.3860,
        "neutral_f1": 0.2857,
    },
}


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def row_key(row):
    return (row["sentence"], row["language"], row["aspect"], row["sentiment"])


def build_added_mask(expanded_rows, manifest):
    if len(expanded_rows) != 3523 or len(manifest) != 207:
        raise ValueError("Expected 3523 expanded rows and 207 manifest rows")
    manifest_keys = {row_key(row) for row in manifest}
    if len(manifest_keys) != 207:
        raise ValueError("Manifest keys are not unique")
    mask = np.asarray([row_key(row) in manifest_keys for row in expanded_rows], dtype=bool)
    if int(mask.sum()) != 207:
        raise ValueError(f"Expected 207 matched added rows, found {int(mask.sum())}")
    if {row_key(row) for row, selected in zip(expanded_rows, mask) if selected} != manifest_keys:
        raise ValueError("Expanded-train added-row matching differs from manifest")
    return mask


def ordinary_metrics(y_true, y_pred):
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true, y_pred, labels=SENTIMENTS, zero_division=0
    )
    classes = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1_values[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(SENTIMENTS)
    }
    true_neutral_correct = sum(
        actual == "neutral" and predicted == "neutral"
        for actual, predicted in zip(y_true, y_pred)
    )
    false_neutral = sum(
        actual != "neutral" and predicted == "neutral"
        for actual, predicted in zip(y_true, y_pred)
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "neutral_precision": classes["neutral"]["precision"],
        "neutral_recall": classes["neutral"]["recall"],
        "neutral_f1": classes["neutral"]["f1"],
        "true_neutral_correct": true_neutral_correct,
        "false_neutral": false_neutral,
        "classes": classes,
        "predicted": Counter(y_pred),
    }


def train_predictions(clean_rows, expanded_rows, dev_rows, added_mask):
    dev_inputs = build_inputs(dev_rows)

    model_a = build_model()
    model_a.fit(build_inputs(clean_rows), [row["sentiment"] for row in clean_rows])
    predictions_a = model_a.predict(dev_inputs).tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    expanded_inputs = build_inputs(expanded_rows)
    expanded_matrix = vectorizer.fit_transform(expanded_inputs)
    dev_matrix = vectorizer.transform(dev_inputs)
    labels = [row["sentiment"] for row in expanded_rows]

    predictions = {"A": predictions_a}
    for model_name, alpha in (("B", 0.0), ("C", 0.4)):
        classifier = LogisticRegression(
            class_weight=FIXED_OLD_WEIGHTS,
            max_iter=1000,
            random_state=42,
        )
        sample_weights = np.where(added_mask, alpha, 1.0)
        classifier.fit(expanded_matrix, labels, sample_weight=sample_weights)
        predictions[model_name] = classifier.predict(dev_matrix).tolist()
    return predictions


def verify_reproduction(actual_metrics):
    differences = {}
    for model_name, expected in EXPECTED.items():
        differences[model_name] = {
            metric: abs(actual_metrics[model_name][metric] - value)
            for metric, value in expected.items()
        }
        if any(difference > 0.00015 for difference in differences[model_name].values()):
            raise RuntimeError(
                f"Model {model_name} reproduction failed: "
                f"actual={actual_metrics[model_name]}, differences={differences[model_name]}"
            )
    return differences


def metrics_from_confusions(flat_confusions):
    confusion = flat_confusions.reshape(-1, 3, 3).astype(np.float64)
    diagonal = np.diagonal(confusion, axis1=1, axis2=2)
    actual_totals = confusion.sum(axis=2)
    predicted_totals = confusion.sum(axis=1)
    precision = np.divide(
        diagonal,
        predicted_totals,
        out=np.zeros_like(diagonal),
        where=predicted_totals != 0,
    )
    recall = np.divide(
        diagonal,
        actual_totals,
        out=np.zeros_like(diagonal),
        where=actual_totals != 0,
    )
    f1_values = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) != 0,
    )
    sample_sizes = confusion.sum(axis=(1, 2))
    return {
        "accuracy": diagonal.sum(axis=1) / sample_sizes,
        "macro_f1": f1_values.mean(axis=1),
        "neutral_precision": precision[:, LABEL_TO_ID["neutral"]],
        "neutral_recall": recall[:, LABEL_TO_ID["neutral"]],
        "neutral_f1": f1_values[:, LABEL_TO_ID["neutral"]],
    }


def paired_bootstrap(y_true_ids, prediction_ids):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    row_count = len(y_true_ids)
    pair_codes = {
        model_name: y_true_ids * 3 + model_predictions
        for model_name, model_predictions in prediction_ids.items()
    }
    output = {
        model_name: {metric: np.empty(BOOTSTRAP_REPLICATES) for metric in METRICS}
        for model_name in prediction_ids
    }

    for start in range(0, BOOTSTRAP_REPLICATES, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, BOOTSTRAP_REPLICATES)
        chunk_size = stop - start
        sampled_indices = rng.integers(0, row_count, size=(chunk_size, row_count))
        sample_rows = np.repeat(np.arange(chunk_size), row_count)
        for model_name, codes in pair_codes.items():
            flat_confusions = np.zeros((chunk_size, 9), dtype=np.int32)
            sampled_codes = codes[sampled_indices].reshape(-1)
            np.add.at(flat_confusions, (sample_rows, sampled_codes), 1)
            chunk_metrics = metrics_from_confusions(flat_confusions)
            for metric in METRICS:
                output[model_name][metric][start:stop] = chunk_metrics[metric]
    return output


def delta_summary(first, second, metric, observed_metrics, bootstrap_metrics):
    delta = bootstrap_metrics[first][metric] - bootstrap_metrics[second][metric]
    return {
        "comparison": f"{first}-{second}",
        "metric": metric,
        "observed_delta": observed_metrics[first][metric] - observed_metrics[second][metric],
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "ci_lower_2_5": float(np.percentile(delta, 2.5)),
        "ci_upper_97_5": float(np.percentile(delta, 97.5)),
        "p_delta_gt_0": float(np.mean(delta > 0)),
        "p_delta_gt_minus_0_01": (
            float(np.mean(delta > -0.01))
            if first == "C" and second == "A" and metric == "macro_f1"
            else ""
        ),
    }


def save_csv(rows):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.8f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def find_delta(rows, comparison, metric):
    return next(
        row
        for row in rows
        if row["comparison"] == comparison and row["metric"] == metric
    )


def main():
    clean_rows = load_rows(CLEAN_TRAIN_PATH, expected_split="train")
    expanded_rows = load_rows(EXPANDED_TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")
    manifest = read_csv(MANIFEST_PATH)
    added_mask = build_added_mask(expanded_rows, manifest)

    predictions = train_predictions(clean_rows, expanded_rows, dev_rows, added_mask)
    y_true = [row["sentiment"] for row in dev_rows]
    actual_metrics = {
        model_name: ordinary_metrics(y_true, model_predictions)
        for model_name, model_predictions in predictions.items()
    }
    reproduction_differences = verify_reproduction(actual_metrics)

    y_true_ids = np.asarray([LABEL_TO_ID[label] for label in y_true], dtype=np.int8)
    prediction_ids = {
        model_name: np.asarray(
            [LABEL_TO_ID[label] for label in model_predictions], dtype=np.int8
        )
        for model_name, model_predictions in predictions.items()
    }
    bootstrap_metrics = paired_bootstrap(y_true_ids, prediction_ids)

    delta_rows = [
        delta_summary(first, second, metric, actual_metrics, bootstrap_metrics)
        for first, second in (("C", "A"), ("C", "B"), ("B", "A"))
        for metric in METRICS
    ]
    save_csv(delta_rows)

    c_vs_a = {
        metric: find_delta(delta_rows, "C-A", metric) for metric in METRICS
    }
    macro_noninferiority_probability = c_vs_a["macro_f1"]["p_delta_gt_minus_0_01"]
    additional_true_neutral = (
        actual_metrics["C"]["true_neutral_correct"]
        - actual_metrics["A"]["true_neutral_correct"]
    )
    additional_false_neutral = (
        actual_metrics["C"]["false_neutral"] - actual_metrics["A"]["false_neutral"]
    )

    macro_delta = c_vs_a["macro_f1"]
    neutral_f1_delta = c_vs_a["neutral_f1"]
    neutral_recall_delta = c_vs_a["neutral_recall"]
    if (
        macro_delta["ci_lower_2_5"] > 0
        and neutral_f1_delta["ci_lower_2_5"] > 0
    ):
        verdict = "ROBUSTLY BETTER"
    elif (
        macro_noninferiority_probability >= 0.95
        and neutral_recall_delta["ci_lower_2_5"] > 0
        and neutral_f1_delta["observed_delta"] > 0
    ):
        verdict = "PROMISING / MACRO-F1 EQUIVALENT"
    elif macro_delta["ci_upper_97_5"] < -0.01:
        verdict = "WORSE"
    else:
        verdict = "INCONCLUSIVE"

    report = [
        "EXPERIMENT 12 — PAIRED BOOTSTRAP ROBUSTNESS CHECK",
        "",
        "Test split was not loaded or evaluated.",
        f"Dev rows: {len(dev_rows)}",
        f"Paired bootstrap replicates: {BOOTSTRAP_REPLICATES}",
        f"Random seed: {BOOTSTRAP_SEED}",
        "The same sampled dev indices were used for Models A, B, and C in every replicate.",
        "",
        "MODEL DEFINITIONS",
        "A: cleaned train, inline representation, class_weight='balanced'.",
        "B: expanded train, one shared expanded TF-IDF, fixed old weights, added-row alpha=0.00.",
        "C: expanded train, same shared expanded TF-IDF, fixed old weights, added-row alpha=0.40.",
        "",
        "REPRODUCTION CHECK",
    ]
    for model_name in ("A", "B", "C"):
        metrics = actual_metrics[model_name]
        report.extend(
            [
                f"Model {model_name}:",
                f"  Accuracy: {metrics['accuracy']:.4f}",
                f"  Macro F1: {metrics['macro_f1']:.4f}",
                f"  Neutral P/R/F1: {metrics['neutral_precision']:.4f} / "
                f"{metrics['neutral_recall']:.4f} / {metrics['neutral_f1']:.4f}",
                f"  True neutral predicted neutral: {metrics['true_neutral_correct']}",
                f"  False neutral: {metrics['false_neutral']}",
                f"  Maximum rounded-reference difference: "
                f"{max(reproduction_differences[model_name].values()):.8f}",
            ]
        )
    report.append("Reproduction status: PASS")

    report.extend(
        [
            "",
            "PAIRED BOOTSTRAP DELTAS",
            "Comparison | Metric | Observed | Mean | Median | 95% CI lower | "
            "95% CI upper | P(delta > 0)",
        ]
    )
    for row in delta_rows:
        report.append(
            f"{row['comparison']} | {row['metric']} | {row['observed_delta']:+.6f} | "
            f"{row['mean_delta']:+.6f} | {row['median_delta']:+.6f} | "
            f"{row['ci_lower_2_5']:+.6f} | {row['ci_upper_97_5']:+.6f} | "
            f"{row['p_delta_gt_0']:.4f}"
        )

    report.extend(
        [
            "",
            "PRIMARY COMPARISON: C (ALPHA 0.40) VS A (CLEANED)",
        ]
    )
    for metric in ("macro_f1", "neutral_f1", "neutral_precision", "neutral_recall", "accuracy"):
        row = c_vs_a[metric]
        report.append(
            f"{metric}: observed={row['observed_delta']:+.6f}, "
            f"95% CI=[{row['ci_lower_2_5']:+.6f}, {row['ci_upper_97_5']:+.6f}], "
            f"P(delta>0)={row['p_delta_gt_0']:.4f}"
        )
    report.extend(
        [
            f"P(Macro F1 delta > -0.01): {macro_noninferiority_probability:.4f}",
            "This is a bootstrap diagnostic, not a formal non-inferiority test.",
            "",
            "NEUTRAL CONFUSION TRADE-OFF",
            f"Cleaned baseline true-neutral recovered: {actual_metrics['A']['true_neutral_correct']}",
            f"Cleaned baseline false-neutral: {actual_metrics['A']['false_neutral']}",
            f"Alpha 0.40 true-neutral recovered: {actual_metrics['C']['true_neutral_correct']}",
            f"Alpha 0.40 false-neutral: {actual_metrics['C']['false_neutral']}",
            f"Additional true neutral recovered: {additional_true_neutral:+d}",
            f"Additional false-neutral introduced: {additional_false_neutral:+d}",
            "",
            f"ALPHA 0.40 TRADE-OFF: {verdict}",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("ALPHA 0.40 vs CLEANED BASELINE")
    for metric, label in (
        ("macro_f1", "Macro F1"),
        ("neutral_f1", "Neutral F1"),
        ("neutral_precision", "Neutral Precision"),
        ("neutral_recall", "Neutral Recall"),
    ):
        row = c_vs_a[metric]
        print(f"\nObserved {label} delta: {row['observed_delta']:+.4f}")
        print(
            f"95% bootstrap CI: [{row['ci_lower_2_5']:+.4f}, "
            f"{row['ci_upper_97_5']:+.4f}]"
        )
        if metric in {"macro_f1", "neutral_f1"}:
            print(f"P(delta > 0): {row['p_delta_gt_0']:.4f}")
    print(f"\nP(Macro F1 delta > -0.01): {macro_noninferiority_probability:.4f}")
    print("\nNeutral trade-off:")
    print(f"Additional true neutral recovered: {additional_true_neutral:+d}")
    print(f"Additional false-neutral introduced: {additional_false_neutral:+d}")
    print(f"\nALPHA 0.40 TRADE-OFF: {verdict}")
    print(f"Report: {REPORT_PATH}")
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
