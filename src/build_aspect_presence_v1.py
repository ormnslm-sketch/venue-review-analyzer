"""Build and evaluate the Aspect Presence v1 classical baseline.

The source is data/processed, not data/cleaned. Processed rows retain reliable
aspect annotations that may later have been removed because of sentiment-label
conflicts. Sentiment values are deliberately ignored when creating presence
labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "aspect_presence_v1"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
MODEL_DIR = PROJECT_ROOT / "models" / "aspect_presence_logreg_v1"

SPLITS = ("train", "dev", "test")
LANGUAGES = ("tr", "en", "es")
ASPECTS = ("taste", "atmosphere", "service", "value")
THRESHOLD = 0.5
RANDOM_SEED = 42

SOURCE_CATEGORY_MAPPING = {
    "food quality": "taste",
    "drinks quality": "taste",
    "ambience general": "atmosphere",
    "service general": "service",
    "food prices": "value",
    "restaurant prices": "value",
    "drinks prices": "value",
}

SOURCE_COLUMNS = {
    "sentence",
    "language",
    "split",
    "aspect",
    "sentiment",
    "source_category",
    "aspect_term",
}
OUTPUT_COLUMNS = ("sentence", "language", "split", *ASPECTS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_processed_split(split: str) -> list[dict[str, str]]:
    path = SOURCE_DIR / f"{split}.csv"
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing = SOURCE_COLUMNS - columns
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        rows = list(reader)

    for row in rows:
        if row["split"] != split:
            raise ValueError(f"{path} contains a non-{split} split value")
        if row["language"] not in LANGUAGES:
            raise ValueError(f"Unexpected language: {row['language']}")
        if not row["sentence"].strip():
            raise ValueError(f"{path} contains an empty sentence")

        expected_aspect = SOURCE_CATEGORY_MAPPING.get(row["source_category"])
        if expected_aspect is None:
            raise ValueError(
                f"Unexpected/unmapped source category: {row['source_category']}"
            )
        if row["aspect"] != expected_aspect:
            raise ValueError(
                "Processed aspect disagrees with source-category mapping: "
                f"{row['source_category']} -> {row['aspect']}"
            )
    return rows


def build_sentence_rows(
    rows_by_split: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Collapse long annotations to sentence-level binary presence labels."""
    grouped_by_split: dict[str, list[dict[str, object]]] = {}
    duplicate_annotation_excess = {}
    conflict_aspect_keys = {}
    conflict_sentence_keys = {}

    for split in SPLITS:
        rows = rows_by_split[split]
        grouped: dict[tuple[str, str], dict[str, object]] = {}
        annotation_counts = Counter()
        sentiments_by_aspect: dict[tuple[str, str, str], set[str]] = defaultdict(set)

        for source_row in rows:
            sentence_key = (source_row["sentence"], source_row["language"])
            aspect = SOURCE_CATEGORY_MAPPING[source_row["source_category"]]
            annotation_counts[(*sentence_key, aspect)] += 1
            sentiments_by_aspect[(*sentence_key, aspect)].add(
                source_row["sentiment"]
            )

            if sentence_key not in grouped:
                grouped[sentence_key] = {
                    "sentence": source_row["sentence"],
                    "language": source_row["language"],
                    "split": split,
                    **{name: 0 for name in ASPECTS},
                }
            grouped[sentence_key][aspect] = 1

        grouped_by_split[split] = list(grouped.values())
        duplicate_annotation_excess[split] = sum(
            max(0, count - 1) for count in annotation_counts.values()
        )
        conflicts = {
            key for key, sentiments in sentiments_by_aspect.items()
            if len(sentiments) > 1
        }
        conflict_aspect_keys[split] = len(conflicts)
        conflict_sentence_keys[split] = len({key[:2] for key in conflicts})

    # Presence is sentence-level, so leakage is checked by sentence + language.
    before_key_sets = {
        split: {
            (row["sentence"], row["language"])
            for row in grouped_by_split[split]
        }
        for split in SPLITS
    }
    before_leakage = {
        "train-dev": len(before_key_sets["train"] & before_key_sets["dev"]),
        "train-test": len(before_key_sets["train"] & before_key_sets["test"]),
        "dev-test": len(before_key_sets["dev"] & before_key_sets["test"]),
    }

    train_keys = before_key_sets["train"]
    retained_dev = [
        row
        for row in grouped_by_split["dev"]
        if (row["sentence"], row["language"]) not in train_keys
    ]
    retained_dev_keys = {
        (row["sentence"], row["language"]) for row in retained_dev
    }
    retained_test = [
        row
        for row in grouped_by_split["test"]
        if (row["sentence"], row["language"]) not in train_keys
        and (row["sentence"], row["language"]) not in retained_dev_keys
    ]
    cleaned = {
        "train": grouped_by_split["train"],
        "dev": retained_dev,
        "test": retained_test,
    }

    removed_for_leakage = {
        "train": 0,
        "dev": len(grouped_by_split["dev"]) - len(retained_dev),
        "test": len(grouped_by_split["test"]) - len(retained_test),
    }
    return cleaned, {
        "duplicate_annotation_excess": duplicate_annotation_excess,
        "conflict_aspect_keys_preserved": conflict_aspect_keys,
        "conflict_sentence_keys_preserved": conflict_sentence_keys,
        "before_leakage": before_leakage,
        "removed_for_leakage": removed_for_leakage,
    }


def save_dataset(rows_by_split: dict[str, list[dict[str, object]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = OUTPUT_DIR / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows_by_split[split])


def dataset_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    aspect_positive = {
        aspect: sum(int(row[aspect]) for row in rows) for aspect in ASPECTS
    }
    cardinality = Counter(
        sum(int(row[aspect]) for aspect in ASPECTS) for row in rows
    )
    keys = [(row["sentence"], row["language"]) for row in rows]
    return {
        "rows": total,
        "language_counts": Counter(row["language"] for row in rows),
        "aspect_positive": aspect_positive,
        "aspect_prevalence": {
            aspect: (aspect_positive[aspect] / total if total else 0.0)
            for aspect in ASPECTS
        },
        "mean_cardinality": (
            sum(cardinality * count for cardinality, count in cardinality.items())
            / total
            if total else 0.0
        ),
        "cardinality_counts": cardinality,
        "all_zero": cardinality[0],
        "duplicate_sentence_language": len(keys) - len(set(keys)),
    }


def leakage_counts(rows_by_split: dict[str, list[dict[str, object]]]) -> dict[str, int]:
    key_sets = {
        split: {(row["sentence"], row["language"]) for row in rows}
        for split, rows in rows_by_split.items()
    }
    return {
        "train-dev": len(key_sets["train"] & key_sets["dev"]),
        "train-test": len(key_sets["train"] & key_sets["test"]),
        "dev-test": len(key_sets["dev"] & key_sets["test"]),
    }


def write_audit(
    rows_by_split: dict[str, list[dict[str, object]]],
    source_rows: dict[str, list[dict[str, str]]],
    build_info: dict[str, object],
) -> Path:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ANALYSIS_DIR / "aspect_presence_v1_audit.txt"
    stats = {split: dataset_stats(rows_by_split[split]) for split in SPLITS}
    after_leakage = leakage_counts(rows_by_split)

    lines = [
        "ASPECT PRESENCE V1 DATASET AUDIT",
        "=" * 88,
        "",
        "SOURCE DECISION",
        "Source: data/processed/{train,dev,test}.csv",
        (
            "Reason: processed data retains reliable mapped aspect annotations "
            "before sentiment-conflict removal. Presence labels ignore sentiment."
        ),
        (
            "data/cleaned was not used because its conflict removal can erase a "
            "present aspect. data/expanded_neutral_v1 was not used because added "
            "target-aspect rows do not establish reliable absent labels for the "
            "other aspects."
        ),
        "Reliable category mapping:",
    ]
    for category, aspect in SOURCE_CATEGORY_MAPPING.items():
        lines.append(f"  {category} -> {aspect}")

    lines.extend(["", "SOURCE FILES"])
    for split in SPLITS:
        source_path = SOURCE_DIR / f"{split}.csv"
        lines.append(
            f"{split}: rows={len(source_rows[split])}, sha256={file_sha256(source_path)}"
        )

    lines.extend(["", "SENTIMENT-CONFLICT PRESENCE PRESERVATION"])
    for split in SPLITS:
        lines.append(
            f"{split}: conflicting sentence+language+aspect keys preserved as present="
            f"{build_info['conflict_aspect_keys_preserved'][split]}; "
            f"affected sentence+language keys="
            f"{build_info['conflict_sentence_keys_preserved'][split]}"
        )

    lines.extend(["", "SPLIT SUMMARY"])
    for split in SPLITS:
        item = stats[split]
        lines.extend(
            [
                "",
                f"{split.upper()}",
                f"Sentence count: {item['rows']}",
                "Language counts: "
                + ", ".join(
                    f"{language}={item['language_counts'][language]}"
                    for language in LANGUAGES
                ),
                "Aspect positive counts: "
                + ", ".join(
                    f"{aspect}={item['aspect_positive'][aspect]}"
                    for aspect in ASPECTS
                ),
                "Aspect prevalence: "
                + ", ".join(
                    f"{aspect}={item['aspect_prevalence'][aspect]:.4f} "
                    f"({item['aspect_prevalence'][aspect] * 100:.2f}%)"
                    for aspect in ASPECTS
                ),
                f"Mean multi-label cardinality: {item['mean_cardinality']:.4f}",
                "Cardinality counts: "
                + ", ".join(
                    f"{count} aspect={item['cardinality_counts'][count]}"
                    for count in range(5)
                ),
                f"All-zero count: {item['all_zero']}",
                f"Duplicate sentence+language rows: {item['duplicate_sentence_language']}",
                "Collapsed duplicate annotation excess: "
                f"{build_info['duplicate_annotation_excess'][split]}",
            ]
        )

    lines.extend(
        [
            "",
            "EXACT SAME-LANGUAGE SENTENCE LEAKAGE",
            "Before priority cleanup: "
            + ", ".join(
                f"{pair}={count}"
                for pair, count in build_info["before_leakage"].items()
            ),
            "Removed with train > dev > test priority: "
            + ", ".join(
                f"{split}={build_info['removed_for_leakage'][split]}"
                for split in SPLITS
            ),
            "Final leakage: "
            + ", ".join(f"{pair}={count}" for pair, count in after_leakage.items()),
            "",
            "LIMITATIONS",
            (
                "All-zero samples are absent because the repository's processed "
                "source contains only annotations mapped to the four target aspects. "
                "No synthetic all-zero negatives were added."
            ),
            (
                "Cross-language semantic leakage cannot be fully verified because "
                "the source files do not contain a reliable language-independent "
                "source_example_id. Only exact sentence+language leakage is proven zero."
            ),
            "",
            "AUDIT RESULT",
            "PASS" if all(count == 0 for count in after_leakage.values()) else "FAIL",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(np.sum(y_true == 1)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def labels_for(row_values: np.ndarray) -> list[str]:
    return [aspect for aspect, value in zip(ASPECTS, row_values) if int(value) == 1]


def diagnostic_examples(
    rows: list[dict[str, object]], y_true: np.ndarray, y_pred: np.ndarray
) -> list[dict[str, object]]:
    categories = {
        "fully correct": [],
        "contains false positive": [],
        "contains false negative": [],
        "multi-label": [],
    }
    for index, (actual, predicted) in enumerate(zip(y_true, y_pred)):
        fp = [
            aspect for aspect, a, p in zip(ASPECTS, actual, predicted)
            if a == 0 and p == 1
        ]
        fn = [
            aspect for aspect, a, p in zip(ASPECTS, actual, predicted)
            if a == 1 and p == 0
        ]
        if np.array_equal(actual, predicted):
            categories["fully correct"].append(index)
        if fp:
            categories["contains false positive"].append(index)
        if fn:
            categories["contains false negative"].append(index)
        if int(actual.sum()) > 1:
            categories["multi-label"].append(index)

    selected = []
    used = set()
    # Three per diagnostic category, while avoiding repeated examples.
    for category, indices in categories.items():
        added = 0
        for index in indices:
            if index in used:
                continue
            actual = y_true[index]
            predicted = y_pred[index]
            selected.append(
                {
                    "category": category,
                    "language": rows[index]["language"],
                    "sentence": str(rows[index]["sentence"])[:240],
                    "actual": labels_for(actual),
                    "predicted": labels_for(predicted),
                    "false_positive": [
                        aspect
                        for aspect, a, p in zip(ASPECTS, actual, predicted)
                        if a == 0 and p == 1
                    ],
                    "false_negative": [
                        aspect
                        for aspect, a, p in zip(ASPECTS, actual, predicted)
                        if a == 1 and p == 0
                    ],
                }
            )
            used.add(index)
            added += 1
            if added == 3:
                break
    return selected


def train_and_evaluate(
    train_rows: list[dict[str, object]], dev_rows: list[dict[str, object]]
) -> tuple[dict[str, object], Path]:
    train_text = [str(row["sentence"]) for row in train_rows]
    dev_text = [str(row["sentence"]) for row in dev_rows]
    y_train = np.asarray(
        [[int(row[aspect]) for aspect in ASPECTS] for row in train_rows],
        dtype=np.int64,
    )
    y_dev = np.asarray(
        [[int(row[aspect]) for aspect in ASPECTS] for row in dev_rows],
        dtype=np.int64,
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(train_text)
    x_dev = vectorizer.transform(dev_text)

    classifiers = {}
    probabilities = np.zeros_like(y_dev, dtype=float)
    for column, aspect in enumerate(ASPECTS):
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        classifier.fit(x_train, y_train[:, column])
        present_index = list(classifier.classes_).index(1)
        probabilities[:, column] = classifier.predict_proba(x_dev)[:, present_index]
        classifiers[aspect] = classifier

    y_pred = (probabilities >= THRESHOLD).astype(np.int64)
    by_aspect = {
        aspect: metric_dict(y_dev[:, column], y_pred[:, column])
        for column, aspect in enumerate(ASPECTS)
    }
    overall = {
        "macro_f1": float(f1_score(y_dev, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_dev, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_dev, y_pred)),
        "exact_match": float(accuracy_score(y_dev, y_pred)),
    }

    language_metrics = {}
    for language in LANGUAGES:
        indices = np.asarray(
            [index for index, row in enumerate(dev_rows) if row["language"] == language]
        )
        language_metrics[language] = {
            "count": int(len(indices)),
            "macro_f1": float(
                f1_score(
                    y_dev[indices], y_pred[indices], average="macro", zero_division=0
                )
            ),
            "aspect_f1": {
                aspect: float(
                    f1_score(
                        y_dev[indices, column],
                        y_pred[indices, column],
                        zero_division=0,
                    )
                )
                for column, aspect in enumerate(ASPECTS)
            },
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    for aspect, classifier in classifiers.items():
        joblib.dump(classifier, MODEL_DIR / f"{aspect}_classifier.joblib")

    metadata = {
        "artifact_version": "aspect_presence_logreg_v1",
        "task": "multi-label aspect presence detection",
        "aspects": list(ASPECTS),
        "threshold": THRESHOLD,
        "input": "sentence only",
        "source_dataset": "data/aspect_presence_v1/train.csv",
        "test_evaluated": False,
        "random_seed": RANDOM_SEED,
        "tfidf": {"ngram_range": [1, 2], "min_df": 2},
        "classifier": {
            "type": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
            "probability_threshold": THRESHOLD,
        },
        "train_sentence_count": len(train_rows),
        "dev_sentence_count": len(dev_rows),
        "sklearn_version": sklearn.__version__,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    results = {
        "overall": overall,
        "by_aspect": by_aspect,
        "by_language": language_metrics,
        "diagnostic_examples": diagnostic_examples(dev_rows, y_dev, y_pred),
    }
    report_path = write_baseline_report(results, metadata)
    return results, report_path


def write_baseline_report(results: dict[str, object], metadata: dict[str, object]) -> Path:
    report_path = ANALYSIS_DIR / "aspect_presence_v1_baseline.txt"
    overall = results["overall"]
    lines = [
        "ASPECT PRESENCE V1 — TF-IDF + LOGISTIC REGRESSION DEV BASELINE",
        "=" * 88,
        "",
        f"Train sentence count: {metadata['train_sentence_count']}",
        f"Dev sentence count: {metadata['dev_sentence_count']}",
        "Test evaluated: NO",
        "Representation: sentence only",
        "TF-IDF: word unigram + bigram, min_df=2",
        "Classifier: four independent LogisticRegression models",
        "class_weight: balanced",
        f"Default threshold: {THRESHOLD:.1f}",
        "",
        "OVERALL DEV METRICS",
        f"Macro F1: {overall['macro_f1']:.4f}",
        f"Micro F1: {overall['micro_f1']:.4f}",
        f"Exact Match Ratio: {overall['exact_match']:.4f}",
        f"Hamming Loss: {overall['hamming_loss']:.4f}",
        "",
        "ASPECT METRICS",
        "Aspect       Precision  Recall  F1      Support  TP   FP   FN   TN",
    ]
    for aspect in ASPECTS:
        item = results["by_aspect"][aspect]
        lines.append(
            f"{aspect:<12} {item['precision']:<10.4f} {item['recall']:<7.4f} "
            f"{item['f1']:<7.4f} {item['support']:<8d} "
            f"{item['tp']:<4d} {item['fp']:<4d} {item['fn']:<4d} {item['tn']:<4d}"
        )

    lines.extend(["", "LANGUAGE BREAKDOWN"])
    for language in LANGUAGES:
        item = results["by_language"][language]
        lines.append(
            f"{language}: n={item['count']}, macro_f1={item['macro_f1']:.4f}; "
            + ", ".join(
                f"{aspect}_f1={item['aspect_f1'][aspect]:.4f}"
                for aspect in ASPECTS
            )
        )

    lines.extend(["", "DEV DIAGNOSTIC EXAMPLES"])
    for index, example in enumerate(results["diagnostic_examples"], start=1):
        lines.extend(
            [
                f"[{index}] {example['category']} | language={example['language']}",
                f"sentence: {example['sentence']}",
                f"actual: {example['actual']}",
                f"predicted: {example['predicted']}",
                f"false_positive: {example['false_positive']}",
                f"false_negative: {example['false_negative']}",
                "",
            ]
        )
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def validate_outputs(rows_by_split: dict[str, list[dict[str, object]]]) -> None:
    after_leakage = leakage_counts(rows_by_split)
    if any(after_leakage.values()):
        raise ValueError(f"Final split leakage is non-zero: {after_leakage}")
    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError(f"{split} is empty")
        for row in rows:
            if tuple(row) != OUTPUT_COLUMNS:
                raise ValueError(f"{split} output schema/order is invalid")
            if any(int(row[aspect]) not in (0, 1) for aspect in ASPECTS):
                raise ValueError(f"{split} contains a non-binary label")


def main() -> None:
    source_rows = {split: load_processed_split(split) for split in SPLITS}
    rows_by_split, build_info = build_sentence_rows(source_rows)
    validate_outputs(rows_by_split)
    save_dataset(rows_by_split)
    audit_path = write_audit(rows_by_split, source_rows, build_info)
    results, baseline_path = train_and_evaluate(
        rows_by_split["train"], rows_by_split["dev"]
    )

    print("ASPECT PRESENCE V1 COMPLETE")
    print("Test split was created but NOT evaluated.")
    for split in SPLITS:
        print(f"{split}: {len(rows_by_split[split])} sentences")
    print(f"Macro F1: {results['overall']['macro_f1']:.4f}")
    print(f"Micro F1: {results['overall']['micro_f1']:.4f}")
    print(f"Exact Match: {results['overall']['exact_match']:.4f}")
    print(f"Hamming Loss: {results['overall']['hamming_loss']:.4f}")
    for aspect in ASPECTS:
        print(f"{aspect} F1: {results['by_aspect'][aspect]['f1']:.4f}")
    for language in LANGUAGES:
        print(
            f"{language} Macro F1: "
            f"{results['by_language'][language]['macro_f1']:.4f}"
        )
    print(f"Audit: {audit_path}")
    print(f"Baseline report: {baseline_path}")
    print(f"Model artifacts: {MODEL_DIR}")


if __name__ == "__main__":
    main()
