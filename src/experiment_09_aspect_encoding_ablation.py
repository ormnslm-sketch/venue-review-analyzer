"""Experiment 09: inline ve ayrı aspect encoding ablation deneyi.

Bu script yalnız train/dev kullanır. Test splitini yüklemez.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import OneHotEncoder

from train_sentiment_baseline import (
    ASPECTS,
    SENTIMENTS,
    build_inputs,
    build_model,
    load_rows,
)


ROOT = Path(__file__).resolve().parents[1]
OLD_TRAIN_PATH = ROOT / "data" / "cleaned" / "train.csv"
EXPANDED_TRAIN_PATH = ROOT / "data" / "expanded_neutral_v1" / "train.csv"
DEV_PATH = ROOT / "data" / "cleaned" / "dev.csv"
REPORT_PATH = ROOT / "data" / "analysis" / "experiment_09_aspect_encoding_ablation.txt"
CSV_PATH = ROOT / "data" / "analysis" / "experiment_09_comparison.csv"

FIXED_CLASS_WEIGHTS = {
    "positive": 0.54042031,
    "neutral": 11.88530466,
    "negative": 1.11841270,
}

EXPECTED_OLD_TRAIN = Counter({"positive": 2173, "neutral": 93, "negative": 1050})
EXPECTED_EXPANDED_TRAIN = Counter(
    {"positive": 2173, "neutral": 300, "negative": 1050}
)
EXPECTED_DEV = Counter({"positive": 586, "neutral": 57, "negative": 258})

LANGUAGES = ("tr", "en", "es")
REFERENCE_C = {
    "accuracy": 0.7159,
    "macro_f1": 0.5805,
    "neutral_precision": 0.2066,
    "neutral_recall": 0.4386,
    "neutral_f1": 0.2809,
    "pred_neutral": 121,
}


def validate_distribution(rows, expected, name):
    actual = Counter(row["sentiment"] for row in rows)
    if actual != expected:
        raise ValueError(f"{name} distribution mismatch: expected={expected}, actual={actual}")


def make_one_hot_encoder():
    """Aspect kolonları her koşulda aynı sırada olsun."""
    return OneHotEncoder(
        categories=[list(ASPECTS)],
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float64,
    )


def class_metrics(y_true, y_pred):
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=SENTIMENTS,
        zero_division=0,
    )
    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1_values[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(SENTIMENTS)
    }


def false_neutral_breakdown(dev_rows, predictions):
    false_indices = [
        index
        for index, (row, prediction) in enumerate(zip(dev_rows, predictions))
        if prediction == "neutral" and row["sentiment"] != "neutral"
    ]
    positive_to_neutral = sum(
        dev_rows[index]["sentiment"] == "positive" for index in false_indices
    )
    negative_to_neutral = sum(
        dev_rows[index]["sentiment"] == "negative" for index in false_indices
    )
    return {
        "positive_to_neutral": positive_to_neutral,
        "negative_to_neutral": negative_to_neutral,
        "total": len(false_indices),
        "by_language": Counter(dev_rows[index]["language"] for index in false_indices),
        "by_aspect": Counter(dev_rows[index]["aspect"] for index in false_indices),
    }


def evaluate(dev_rows, predictions):
    y_true = [row["sentiment"] for row in dev_rows]
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "classes": class_metrics(y_true, predictions),
        "predicted": Counter(predictions),
        "matrix": confusion_matrix(y_true, predictions, labels=SENTIMENTS),
        "false_neutral": false_neutral_breakdown(dev_rows, predictions),
    }


def neutral_coefficients(classifier, feature_names, limit=30):
    neutral_index = list(classifier.classes_).index("neutral")
    coefficients = classifier.coef_[neutral_index]
    top_indices = np.argsort(coefficients)[::-1][:limit]
    return [(str(feature_names[index]), float(coefficients[index])) for index in top_indices]


def run_inline(train_rows, dev_rows):
    """Mevcut baseline pipeline'ını yalnız sabit weights ile çalıştır."""
    model = build_model()
    model.set_params(classifier__class_weight=FIXED_CLASS_WEIGHTS)
    model.fit(
        build_inputs(train_rows),
        [row["sentiment"] for row in train_rows],
    )
    predictions = model.predict(build_inputs(dev_rows)).tolist()
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    return {
        "metrics": evaluate(dev_rows, predictions),
        "top_neutral_features": neutral_coefficients(
            classifier, vectorizer.get_feature_names_out()
        ),
    }


def run_separate(train_rows, dev_rows):
    """Sentence TF-IDF ile aspect one-hot matrislerini sparse biçimde birleştir."""
    train_sentences = [row["sentence"] for row in train_rows]
    dev_sentences = [row["sentence"] for row in dev_rows]
    train_aspects = np.array([[row["aspect"]] for row in train_rows], dtype=object)
    dev_aspects = np.array([[row["aspect"]] for row in dev_rows], dtype=object)

    # Yapay aspect prefix'i veya [SEP] tokenı TF-IDF girdisine hiç eklenmez.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    encoder = make_one_hot_encoder()
    train_text = vectorizer.fit_transform(train_sentences)
    dev_text = vectorizer.transform(dev_sentences)
    train_aspect = encoder.fit_transform(train_aspects)
    dev_aspect = encoder.transform(dev_aspects)

    train_matrix = hstack([train_text, train_aspect], format="csr")
    dev_matrix = hstack([dev_text, dev_aspect], format="csr")
    classifier = LogisticRegression(
        class_weight=FIXED_CLASS_WEIGHTS,
        max_iter=1000,
        random_state=42,
    )
    classifier.fit(train_matrix, [row["sentiment"] for row in train_rows])
    predictions = classifier.predict(dev_matrix).tolist()

    text_feature_names = vectorizer.get_feature_names_out()
    neutral_index = list(classifier.classes_).index("neutral")
    neutral_coefficients_all = classifier.coef_[neutral_index]
    text_coefficients = neutral_coefficients_all[: len(text_feature_names)]
    top_text_indices = np.argsort(text_coefficients)[::-1][:30]
    top_text_features = [
        (str(text_feature_names[index]), float(text_coefficients[index]))
        for index in top_text_indices
    ]
    aspect_feature_names = encoder.get_feature_names_out(["aspect"])
    aspect_coefficients = neutral_coefficients_all[len(text_feature_names) :]
    aspect_features = [
        (name.replace("aspect_", "aspect="), float(coefficient))
        for name, coefficient in zip(aspect_feature_names, aspect_coefficients)
    ]

    return {
        "metrics": evaluate(dev_rows, predictions),
        "top_neutral_sentence_features": top_text_features,
        "neutral_aspect_coefficients": aspect_features,
        "tfidf_input_verified_sentence_only": True,
    }


def assert_condition_c_reproduced(metrics):
    checks = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "neutral_precision": metrics["classes"]["neutral"]["precision"],
        "neutral_recall": metrics["classes"]["neutral"]["recall"],
        "neutral_f1": metrics["classes"]["neutral"]["f1"],
    }
    differences = {
        key: abs(value - REFERENCE_C[key]) for key, value in checks.items()
    }
    if any(difference > 0.00015 for difference in differences.values()):
        raise RuntimeError(
            f"Condition C failed Experiment 07 reproduction: {checks}; "
            f"differences={differences}"
        )
    if metrics["predicted"]["neutral"] != REFERENCE_C["pred_neutral"]:
        raise RuntimeError(
            "Condition C failed predicted-neutral reproduction: "
            f"expected={REFERENCE_C['pred_neutral']}, "
            f"actual={metrics['predicted']['neutral']}"
        )


def metric_delta(new_metrics, old_metrics):
    return {
        "accuracy": new_metrics["accuracy"] - old_metrics["accuracy"],
        "macro_f1": new_metrics["macro_f1"] - old_metrics["macro_f1"],
        "neutral_precision": (
            new_metrics["classes"]["neutral"]["precision"]
            - old_metrics["classes"]["neutral"]["precision"]
        ),
        "neutral_recall": (
            new_metrics["classes"]["neutral"]["recall"]
            - old_metrics["classes"]["neutral"]["recall"]
        ),
        "neutral_f1": (
            new_metrics["classes"]["neutral"]["f1"]
            - old_metrics["classes"]["neutral"]["f1"]
        ),
        "pred_neutral": (
            new_metrics["predicted"]["neutral"] - old_metrics["predicted"]["neutral"]
        ),
        "false_neutral": (
            new_metrics["false_neutral"]["total"]
            - old_metrics["false_neutral"]["total"]
        ),
        "atmosphere_false_neutral": (
            new_metrics["false_neutral"]["by_aspect"]["atmosphere"]
            - old_metrics["false_neutral"]["by_aspect"]["atmosphere"]
        ),
    }


def format_matrix(matrix):
    lines = ["                 positive  neutral  negative"]
    for label, row in zip(SENTIMENTS, matrix):
        lines.append(f"{label:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}")
    return lines


def append_condition(report, label, description, metrics):
    report.extend(
        [
            "",
            f"CONDITION {label}: {description}",
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Macro F1: {metrics['macro_f1']:.4f}",
        ]
    )
    for sentiment in SENTIMENTS:
        item = metrics["classes"][sentiment]
        report.append(
            f"{sentiment}: P={item['precision']:.4f} | R={item['recall']:.4f} | "
            f"F1={item['f1']:.4f} | support={item['support']}"
        )
    report.extend(
        [
            "Predicted: "
            f"positive={metrics['predicted']['positive']}, "
            f"neutral={metrics['predicted']['neutral']}, "
            f"negative={metrics['predicted']['negative']}",
            "Confusion matrix (rows=actual, columns=predicted):",
            *format_matrix(metrics["matrix"]),
            "False-neutral: "
            f"positive->neutral={metrics['false_neutral']['positive_to_neutral']}, "
            f"negative->neutral={metrics['false_neutral']['negative_to_neutral']}, "
            f"total={metrics['false_neutral']['total']}",
            "False-neutral by language: "
            + ", ".join(
                f"{language.upper()}={metrics['false_neutral']['by_language'][language]}"
                for language in LANGUAGES
            ),
            "False-neutral by aspect: "
            + ", ".join(
                f"{aspect}={metrics['false_neutral']['by_aspect'][aspect]}"
                for aspect in ASPECTS
            ),
        ]
    )


def append_delta(report, title, delta):
    report.extend(
        [
            "",
            title,
            f"Delta Accuracy: {delta['accuracy']:+.4f}",
            f"Delta Macro F1: {delta['macro_f1']:+.4f}",
            f"Delta Neutral precision: {delta['neutral_precision']:+.4f}",
            f"Delta Neutral recall: {delta['neutral_recall']:+.4f}",
            f"Delta Neutral F1: {delta['neutral_f1']:+.4f}",
            f"Delta Pred Neutral: {delta['pred_neutral']:+d}",
            f"Delta False Neutral: {delta['false_neutral']:+d}",
            "Delta atmosphere false-neutral: "
            f"{delta['atmosphere_false_neutral']:+d}",
        ]
    )


def artifact_conclusion(c_metrics, d_metrics):
    favorable = {
        "neutral_precision_higher": (
            d_metrics["classes"]["neutral"]["precision"]
            > c_metrics["classes"]["neutral"]["precision"]
        ),
        "false_neutral_lower": (
            d_metrics["false_neutral"]["total"]
            < c_metrics["false_neutral"]["total"]
        ),
        "atmosphere_false_neutral_lower": (
            d_metrics["false_neutral"]["by_aspect"]["atmosphere"]
            < c_metrics["false_neutral"]["by_aspect"]["atmosphere"]
        ),
        "neutral_recall_not_lower": (
            d_metrics["classes"]["neutral"]["recall"]
            >= c_metrics["classes"]["neutral"]["recall"]
        ),
        "macro_f1_not_lower": d_metrics["macro_f1"] >= c_metrics["macro_f1"],
    }
    if all(favorable.values()):
        verdict = "YES"
    elif any(favorable.values()):
        verdict = "MIXED"
    else:
        verdict = "NO"
    return verdict, favorable


def save_csv(results):
    fields = [
        "condition",
        "dataset",
        "aspect_representation",
        "accuracy",
        "macro_f1",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "neutral_precision",
        "neutral_recall",
        "neutral_f1",
        "negative_precision",
        "negative_recall",
        "negative_f1",
        "pred_positive",
        "pred_neutral",
        "pred_negative",
        "positive_to_neutral",
        "negative_to_neutral",
        "false_neutral_total",
        "atmosphere_false_neutral",
    ]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for condition in ("A", "B", "C", "D"):
            result = results[condition]
            metrics = result["metrics"]
            writer.writerow(
                {
                    "condition": condition,
                    "dataset": "old" if condition in {"A", "B"} else "expanded",
                    "aspect_representation": (
                        "inline" if condition in {"A", "C"} else "separate_one_hot"
                    ),
                    "accuracy": f"{metrics['accuracy']:.8f}",
                    "macro_f1": f"{metrics['macro_f1']:.8f}",
                    **{
                        f"{sentiment}_{metric}": (
                            f"{metrics['classes'][sentiment][metric]:.8f}"
                        )
                        for sentiment in SENTIMENTS
                        for metric in ("precision", "recall", "f1")
                    },
                    "pred_positive": metrics["predicted"]["positive"],
                    "pred_neutral": metrics["predicted"]["neutral"],
                    "pred_negative": metrics["predicted"]["negative"],
                    "positive_to_neutral": metrics["false_neutral"][
                        "positive_to_neutral"
                    ],
                    "negative_to_neutral": metrics["false_neutral"][
                        "negative_to_neutral"
                    ],
                    "false_neutral_total": metrics["false_neutral"]["total"],
                    "atmosphere_false_neutral": metrics["false_neutral"][
                        "by_aspect"
                    ]["atmosphere"],
                }
            )


def main():
    old_train = load_rows(OLD_TRAIN_PATH, expected_split="train")
    expanded_train = load_rows(EXPANDED_TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")
    validate_distribution(old_train, EXPECTED_OLD_TRAIN, "old train")
    validate_distribution(expanded_train, EXPECTED_EXPANDED_TRAIN, "expanded train")
    validate_distribution(dev_rows, EXPECTED_DEV, "dev")

    print("Running A: old data + inline aspect")
    condition_a = run_inline(old_train, dev_rows)
    print("Running B: old data + separate one-hot aspect")
    condition_b = run_separate(old_train, dev_rows)
    print("Running C: expanded data + inline aspect")
    condition_c = run_inline(expanded_train, dev_rows)
    assert_condition_c_reproduced(condition_c["metrics"])
    print("Condition C reproduction: PASS")
    print("Running D: expanded data + separate one-hot aspect")
    condition_d = run_separate(expanded_train, dev_rows)

    results = {
        "A": condition_a,
        "B": condition_b,
        "C": condition_c,
        "D": condition_d,
    }
    save_csv(results)

    report = [
        "EXPERIMENT 09 — ASPECT ENCODING ABLATION",
        "",
        "Test split was not loaded or evaluated.",
        f"Fixed class weights: {FIXED_CLASS_WEIGHTS}",
        "TF-IDF: ngram_range=(1, 2), min_df=2; other settings are sklearn defaults.",
        "LogisticRegression: max_iter=1000, random_state=42; default solver retained.",
        "Separate encoding verification: TF-IDF receives sentence text only; "
        "aspect is appended as four sparse one-hot columns.",
        "Condition C Experiment 07 reproduction: PASS.",
    ]
    descriptions = {
        "A": "old data + inline aspect",
        "B": "old data + separate one-hot aspect",
        "C": "expanded data + inline aspect",
        "D": "expanded data + separate one-hot aspect",
    }
    for condition in ("A", "B", "C", "D"):
        append_condition(
            report,
            condition,
            descriptions[condition],
            results[condition]["metrics"],
        )

    delta_d_c = metric_delta(condition_d["metrics"], condition_c["metrics"])
    delta_b_a = metric_delta(condition_b["metrics"], condition_a["metrics"])
    append_delta(report, "C -> D (expanded: separate minus inline)", delta_d_c)
    append_delta(report, "A -> B (old: separate minus inline)", delta_b_a)

    report.extend(["", "INLINE TOP 30 NEUTRAL COEFFICIENTS (Condition C)"])
    report.extend(
        f"{rank:>2}. {feature}: {coefficient:+.6f}"
        for rank, (feature, coefficient) in enumerate(
            condition_c["top_neutral_features"], start=1
        )
    )
    report.extend(["", "SEPARATE TOP 30 NEUTRAL SENTENCE FEATURES (Condition D)"])
    report.extend(
        f"{rank:>2}. {feature}: {coefficient:+.6f}"
        for rank, (feature, coefficient) in enumerate(
            condition_d["top_neutral_sentence_features"], start=1
        )
    )
    report.extend(["", "SEPARATE NEUTRAL ASPECT COEFFICIENTS (Condition D)"])
    report.extend(
        f"{feature}: {coefficient:+.6f}"
        for feature, coefficient in condition_d["neutral_aspect_coefficients"]
    )

    verdict, evidence = artifact_conclusion(
        condition_c["metrics"], condition_d["metrics"]
    )
    report.extend(
        [
            "",
            f"DOES SEPARATE ASPECT ENCODING REDUCE THE ARTIFACT? {verdict}",
            "Evidence criteria (Condition D vs C):",
            *[f"{name}: {value}" for name, value in evidence.items()],
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nCondition | Data | Aspect representation | Accuracy | Macro F1 | "
          "Neutral P | Neutral R | Neutral F1 | Pred Neutral | False Neutral")
    for condition in ("A", "B", "C", "D"):
        metrics = results[condition]["metrics"]
        neutral = metrics["classes"]["neutral"]
        print(
            f"{condition} | "
            f"{'old' if condition in {'A', 'B'} else 'expanded'} | "
            f"{'inline' if condition in {'A', 'C'} else 'separate'} | "
            f"{metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{neutral['precision']:.4f} | {neutral['recall']:.4f} | "
            f"{neutral['f1']:.4f} | {metrics['predicted']['neutral']} | "
            f"{metrics['false_neutral']['total']}"
        )
    print("\nC -> D:")
    for name, value in delta_d_c.items():
        print(f"{name}: {value:+.4f}" if isinstance(value, float) else f"{name}: {value:+d}")
    print("\nA -> B:")
    for name, value in delta_b_a.items():
        print(f"{name}: {value:+.4f}" if isinstance(value, float) else f"{name}: {value:+d}")
    print(f"\nDOES SEPARATE ASPECT ENCODING REDUCE THE ARTIFACT? {verdict}")
    print(f"Report: {REPORT_PATH}")
    print(f"Comparison CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
