"""Build Aspect Presence v2 from raw M-ABSA annotations and train its baseline.

V2 differs from v1 only in dataset coverage: raw source sentences whose complete
annotation set contains none of the four target categories become natural
all-zero examples. Sentiment polarity is ignored. The test split is built and
audited but is never transformed by the model or evaluated.
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
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

from build_aspect_presence_v1 import metric_dict
from preprocess_dataset import parse_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "aspect_presence_v2"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
MODEL_DIR = PROJECT_ROOT / "models" / "aspect_presence_logreg_v2"
V1_DATA_DIR = PROJECT_ROOT / "data" / "aspect_presence_v1"
V1_MODEL_DIR = PROJECT_ROOT / "models" / "aspect_presence_logreg_v1"

DATASET_ID = "Multilingual-NLP/M-ABSA"
DOMAIN = "restaurant"
SPLITS = ("train", "dev", "test")
LANGUAGES = ("tr", "en", "es")
ASPECTS = ("taste", "atmosphere", "service", "value")
OUTPUT_COLUMNS = ("sentence", "language", "split", *ASPECTS)
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
NON_TARGET_CATEGORIES = {
    "restaurant general",
    "restaurant miscellaneous",
    "food style_options",
    "drinks style_options",
    "location general",
    "food general",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_raw_snapshot() -> Path:
    """Resolve the exact locally cached Hugging Face dataset snapshot."""
    path = Path(
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_files_only=True,
        )
    )
    for split in SPLITS:
        for language in LANGUAGES:
            expected = path / DOMAIN / language / f"{split}.txt"
            if not expected.is_file():
                raise FileNotFoundError(f"Missing raw M-ABSA source: {expected}")
    return path


def load_raw_source(snapshot: Path):
    """Read every raw sentence and its complete source annotation set."""
    records_by_split: dict[str, list[dict[str, object]]] = {
        split: [] for split in SPLITS
    }
    source_files = {}
    observed_categories = set()

    for split in SPLITS:
        for language in LANGUAGES:
            path = snapshot / DOMAIN / language / f"{split}.txt"
            source_files[(split, language)] = path
            for source_line, raw_text in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                sentence, triplets = parse_row(raw_text)
                if not sentence.strip():
                    raise ValueError(f"Empty sentence at {path}:{source_line}")
                if not triplets:
                    raise ValueError(
                        f"Annotation-free raw sentence at {path}:{source_line}"
                    )
                categories = {category for _, category, _ in triplets}
                observed_categories.update(categories)
                records_by_split[split].append(
                    {
                        "sentence": sentence,
                        "language": language,
                        "split": split,
                        "categories": categories,
                        "triplets": triplets,
                        "source_line": source_line,
                    }
                )

    expected_categories = set(SOURCE_CATEGORY_MAPPING) | NON_TARGET_CATEGORIES
    unknown = observed_categories - expected_categories
    if unknown:
        raise ValueError(f"Unreviewed raw source categories: {sorted(unknown)}")
    return records_by_split, source_files, observed_categories


def build_sentence_dataset(records_by_split):
    """Union duplicate raw lines and map complete category sets to binary labels."""
    grouped_by_split = {}
    metadata_by_split = {}
    raw_duplicate_excess = {}

    for split in SPLITS:
        grouped = {}
        for record in records_by_split[split]:
            key = (record["sentence"], record["language"])
            if key not in grouped:
                grouped[key] = {
                    "sentence": record["sentence"],
                    "language": record["language"],
                    "split": split,
                    "categories": set(),
                    "source_lines": [],
                }
            grouped[key]["categories"].update(record["categories"])
            grouped[key]["source_lines"].append(record["source_line"])

        raw_duplicate_excess[split] = len(records_by_split[split]) - len(grouped)
        rows = []
        metadata = {}
        for key, item in grouped.items():
            categories = set(item["categories"])
            labels = {aspect: 0 for aspect in ASPECTS}
            for category in categories:
                aspect = SOURCE_CATEGORY_MAPPING.get(category)
                if aspect is not None:
                    labels[aspect] = 1
            rows.append(
                {
                    "sentence": item["sentence"],
                    "language": item["language"],
                    "split": split,
                    **labels,
                }
            )
            metadata[key] = {
                "categories": categories,
                "source_lines": tuple(item["source_lines"]),
            }
        grouped_by_split[split] = rows
        metadata_by_split[split] = metadata

    before_sets = {
        split: {(row["sentence"], row["language"]) for row in rows}
        for split, rows in grouped_by_split.items()
    }
    before_leakage = {
        "train-dev": len(before_sets["train"] & before_sets["dev"]),
        "train-test": len(before_sets["train"] & before_sets["test"]),
        "dev-test": len(before_sets["dev"] & before_sets["test"]),
    }

    train_keys = before_sets["train"]
    dev_rows = [
        row for row in grouped_by_split["dev"]
        if (row["sentence"], row["language"]) not in train_keys
    ]
    dev_keys = {(row["sentence"], row["language"]) for row in dev_rows}
    test_rows = [
        row for row in grouped_by_split["test"]
        if (row["sentence"], row["language"]) not in train_keys
        and (row["sentence"], row["language"]) not in dev_keys
    ]
    retained = {
        "train": grouped_by_split["train"],
        "dev": dev_rows,
        "test": test_rows,
    }
    retained_metadata = {
        split: {
            (row["sentence"], row["language"]): metadata_by_split[split][
                (row["sentence"], row["language"])
            ]
            for row in retained[split]
        }
        for split in SPLITS
    }
    removed_for_leakage = {
        "train": 0,
        "dev": len(grouped_by_split["dev"]) - len(dev_rows),
        "test": len(grouped_by_split["test"]) - len(test_rows),
    }
    return retained, retained_metadata, {
        "raw_duplicate_excess": raw_duplicate_excess,
        "before_leakage": before_leakage,
        "removed_for_leakage": removed_for_leakage,
    }


def save_dataset(rows_by_split) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with (OUTPUT_DIR / f"{split}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows_by_split[split])


def load_presence_csv(path: Path, expected_split: str):
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise ValueError(f"Unexpected presence schema: {path}")
        rows = list(reader)
    for row in rows:
        if row["split"] != expected_split:
            raise ValueError(f"Unexpected split in {path}")
        for aspect in ASPECTS:
            row[aspect] = int(row[aspect])
    return rows


def row_key(row):
    return row["sentence"], row["language"]


def label_array(rows):
    return np.asarray(
        [[int(row[aspect]) for aspect in ASPECTS] for row in rows],
        dtype=np.int64,
    )


def split_stats(rows):
    labels = label_array(rows)
    cardinality = labels.sum(axis=1)
    return {
        "rows": len(rows),
        "language": Counter(row["language"] for row in rows),
        "positive": {
            aspect: int(labels[:, column].sum())
            for column, aspect in enumerate(ASPECTS)
        },
        "prevalence": {
            aspect: float(labels[:, column].mean())
            for column, aspect in enumerate(ASPECTS)
        },
        "cardinality": Counter(int(value) for value in cardinality),
        "mean_cardinality": float(cardinality.mean()),
        "all_zero": int((cardinality == 0).sum()),
        "duplicate": len(rows) - len({row_key(row) for row in rows}),
    }


def leakage_counts(rows_by_split):
    sets = {split: {row_key(row) for row in rows} for split, rows in rows_by_split.items()}
    return {
        "train-dev": len(sets["train"] & sets["dev"]),
        "train-test": len(sets["train"] & sets["test"]),
        "dev-test": len(sets["dev"] & sets["test"]),
    }


def verify_v1_positive_preservation(rows_by_split):
    results = {}
    for split in SPLITS:
        v1_rows = load_presence_csv(V1_DATA_DIR / f"{split}.csv", split)
        v2_by_key = {row_key(row): row for row in rows_by_split[split]}
        missing = []
        mismatched = []
        for v1_row in v1_rows:
            key = row_key(v1_row)
            if key not in v2_by_key:
                missing.append(key)
                continue
            v2_row = v2_by_key[key]
            if any(v1_row[aspect] != v2_row[aspect] for aspect in ASPECTS):
                mismatched.append(key)
        results[split] = {
            "v1_rows": len(v1_rows),
            "missing": len(missing),
            "label_mismatch": len(mismatched),
        }
        if missing or mismatched:
            raise ValueError(
                f"V1 positive preservation failed for {split}: "
                f"missing={len(missing)}, mismatch={len(mismatched)}"
            )
    return results


def all_zero_category_distribution(rows, metadata):
    """Sentence-set incidence: each category counts at most once per sentence."""
    counts = Counter()
    by_language = Counter()
    for row in rows:
        if sum(int(row[aspect]) for aspect in ASPECTS) != 0:
            continue
        by_language[row["language"]] += 1
        counts.update(metadata[row_key(row)]["categories"])
    return counts, by_language


def select_diagnostic_examples(rows, metadata, limit=20):
    """Select readable non-test all-zero examples for manual source audit."""
    selected = []
    for row in rows:
        if row["language"] != "en":
            continue
        if sum(int(row[aspect]) for aspect in ASPECTS) == 0:
            selected.append(
                {
                    "sentence": row["sentence"],
                    "categories": sorted(metadata[row_key(row)]["categories"]),
                }
            )
        if len(selected) == limit:
            break
    return selected


def write_audit(
    snapshot,
    source_files,
    observed_categories,
    rows_by_split,
    metadata_by_split,
    build_info,
    v1_preservation,
):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / "aspect_presence_v2_audit.txt"
    stats = {split: split_stats(rows_by_split[split]) for split in SPLITS}
    final_leakage = leakage_counts(rows_by_split)
    diagnostics = select_diagnostic_examples(
        rows_by_split["train"], metadata_by_split["train"], limit=20
    )

    lines = [
        "ASPECT PRESENCE V2 DATASET AUDIT",
        "=" * 92,
        "",
        "RAW SOURCE",
        f"Dataset: {DATASET_ID}",
        f"Local snapshot commit: {snapshot.name}",
        "Domain: restaurant",
        "Network used: NO (local_files_only=True)",
        "Source representation: sentence####[term, category, polarity] triplets",
        "All source categories observed: " + ", ".join(sorted(observed_categories)),
        "",
        "TARGET CATEGORY MAPPING",
    ]
    for category, aspect in SOURCE_CATEGORY_MAPPING.items():
        lines.append(f"  {category} -> {aspect}")
    lines.extend(
        [
            "Non-target categories remain unmapped: "
            + ", ".join(sorted(NON_TARGET_CATEGORIES)),
            "Sentiment polarity and aspect-term explicitness are not used.",
            "",
            "SOURCE FILE HASHES",
        ]
    )
    for split in SPLITS:
        for language in LANGUAGES:
            source = source_files[(split, language)]
            lines.append(
                f"{split}/{language}: sha256={file_sha256(source)}"
            )

    lines.extend(["", "V1 TARGET-POSITIVE PRESERVATION"])
    for split in SPLITS:
        item = v1_preservation[split]
        lines.append(
            f"{split}: checked={item['v1_rows']}, missing={item['missing']}, "
            f"label_mismatch={item['label_mismatch']}"
        )

    lines.extend(["", "SPLIT SUMMARY"])
    for split in SPLITS:
        item = stats[split]
        category_counts, zero_by_language = all_zero_category_distribution(
            rows_by_split[split], metadata_by_split[split]
        )
        lines.extend(
            [
                "",
                split.upper(),
                f"Sentence count: {item['rows']}",
                "Language counts: "
                + ", ".join(
                    f"{language}={item['language'][language]}"
                    for language in LANGUAGES
                ),
                "Aspect positive counts: "
                + ", ".join(
                    f"{aspect}={item['positive'][aspect]}" for aspect in ASPECTS
                ),
                "Aspect prevalence: "
                + ", ".join(
                    f"{aspect}={item['prevalence'][aspect]:.4f} "
                    f"({item['prevalence'][aspect] * 100:.2f}%)"
                    for aspect in ASPECTS
                ),
                f"All-zero count: {item['all_zero']}",
                f"All-zero prevalence: {item['all_zero'] / item['rows']:.4f} "
                f"({item['all_zero'] / item['rows'] * 100:.2f}%)",
                "All-zero by language: "
                + ", ".join(
                    f"{language}={zero_by_language[language]}"
                    for language in LANGUAGES
                ),
                (
                    "All-zero source-category sentence incidence "
                    "(each category counts once per sentence; multi-category "
                    "sentences contribute to multiple categories): "
                    + ", ".join(
                        f"{category}={count}"
                        for category, count in sorted(category_counts.items())
                    )
                ),
                f"Mean multi-label cardinality: {item['mean_cardinality']:.4f}",
                "Label-count distribution: "
                + ", ".join(
                    f"{count} aspect={item['cardinality'][count]}"
                    for count in range(5)
                ),
                f"Output duplicate sentence+language rows: {item['duplicate']}",
                "Raw duplicate line excess collapsed: "
                f"{build_info['raw_duplicate_excess'][split]}",
            ]
        )

    lines.extend(
        [
            "",
            "SPLIT LEAKAGE",
            "Before cleanup: "
            + ", ".join(
                f"{pair}={count}"
                for pair, count in build_info["before_leakage"].items()
            ),
            "Removed with train > dev > test priority: "
            + ", ".join(
                f"{split}={build_info['removed_for_leakage'][split]}"
                for split in SPLITS
            ),
            "Final exact sentence+language leakage: "
            + ", ".join(f"{pair}={count}" for pair, count in final_leakage.items()),
            "",
            "MANUAL DIAGNOSTIC — 20 NATURAL ALL-ZERO TRAIN EXAMPLES",
            (
                "Selection: first 20 English train all-zero sentences in raw source "
                "order. This is a mapping audit, not model evaluation."
            ),
        ]
    )
    for index, example in enumerate(diagnostics, start=1):
        lines.append(
            f"[{index}] categories={example['categories']} | {example['sentence']}"
        )
    lines.extend(
        [
            "",
            "Manual diagnostic finding:",
            (
                "Most examples express restaurant-general, miscellaneous, location, "
                "or menu/style content outside the target ontology. Four examples "
                "show plausible lexical/semantic overlap with target concepts "
                "(menu simplicity, burgers falling apart, preference for gnocchi, "
                "and a window-table request). They remain all-zero because their "
                "gold source categories are explicitly excluded by the requested "
                "mapping. No automatic relabeling was performed."
            ),
            "",
            "LIMITATION",
            (
                "Cross-language semantic leakage cannot be fully verified because "
                "the raw dataset has no reliable language-independent source_example_id. "
                "Only exact sentence+language leakage is proven zero."
            ),
            "",
            "AUDIT RESULT",
            "PASS" if all(value == 0 for value in final_leakage.values()) else "FAIL",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def fit_models(train_rows, dev_rows):
    train_text = [row["sentence"] for row in train_rows]
    dev_text = [row["sentence"] for row in dev_rows]
    y_train = label_array(train_rows)
    y_dev = label_array(dev_rows)

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
    predictions = (probabilities >= THRESHOLD).astype(np.int64)
    return vectorizer, classifiers, y_dev, predictions


def predict_with_artifact(model_dir, rows):
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    features = vectorizer.transform([row["sentence"] for row in rows])
    predictions = np.zeros((len(rows), len(ASPECTS)), dtype=np.int64)
    for column, aspect in enumerate(ASPECTS):
        classifier = joblib.load(model_dir / f"{aspect}_classifier.joblib")
        present_index = list(classifier.classes_).index(1)
        probabilities = classifier.predict_proba(features)[:, present_index]
        predictions[:, column] = (probabilities >= THRESHOLD).astype(np.int64)
    return predictions


def multi_label_metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def evaluate_v2(dev_rows, y_dev, y_pred):
    overall = multi_label_metrics(y_dev, y_pred)
    by_aspect = {
        aspect: metric_dict(y_dev[:, column], y_pred[:, column])
        for column, aspect in enumerate(ASPECTS)
    }
    cardinality = y_dev.sum(axis=1)
    zero_indices = np.flatnonzero(cardinality == 0)
    target_indices = np.flatnonzero(cardinality > 0)
    zero_predictions = y_pred[zero_indices]
    all_zero = {
        "support": int(len(zero_indices)),
        "correct": int((zero_predictions.sum(axis=1) == 0).sum()),
        "exact_accuracy": float((zero_predictions.sum(axis=1) == 0).mean()),
        "false_positive_total": int(zero_predictions.sum()),
        "false_positive_by_aspect": {
            aspect: int(zero_predictions[:, column].sum())
            for column, aspect in enumerate(ASPECTS)
        },
    }
    target_metrics = multi_label_metrics(y_dev[target_indices], y_pred[target_indices])

    language = {}
    for code in LANGUAGES:
        indices = np.asarray(
            [index for index, row in enumerate(dev_rows) if row["language"] == code]
        )
        lang_true = y_dev[indices]
        lang_pred = y_pred[indices]
        lang_zero = np.flatnonzero(lang_true.sum(axis=1) == 0)
        language[code] = {
            **multi_label_metrics(lang_true, lang_pred),
            "all_zero_support": int(len(lang_zero)),
            "all_zero_exact_accuracy": (
                float((lang_pred[lang_zero].sum(axis=1) == 0).mean())
                if len(lang_zero) else None
            ),
        }
    return {
        "overall": overall,
        "by_aspect": by_aspect,
        "all_zero": all_zero,
        "target_containing": target_metrics,
        "language": language,
    }


def common_subset_comparison(dev_rows, v2_predictions):
    v1_rows = load_presence_csv(V1_DATA_DIR / "dev.csv", "dev")
    v2_index = {row_key(row): index for index, row in enumerate(dev_rows)}
    if any(row_key(row) not in v2_index for row in v1_rows):
        raise ValueError("A V1 DEV row is missing from V2 DEV")
    common_indices = np.asarray([v2_index[row_key(row)] for row in v1_rows])
    y_common = label_array(v1_rows)
    v2_labels = label_array([dev_rows[index] for index in common_indices])
    if not np.array_equal(y_common, v2_labels):
        raise ValueError("V1 and V2 labels differ on common DEV rows")
    v1_predictions = predict_with_artifact(V1_MODEL_DIR, v1_rows)
    v1_metrics = multi_label_metrics(y_common, v1_predictions)
    v2_metrics = multi_label_metrics(y_common, v2_predictions[common_indices])
    preserved = (
        v2_metrics["macro_f1"] >= v1_metrics["macro_f1"] - 0.01
        and v2_metrics["exact_match"] >= v1_metrics["exact_match"] - 0.01
    )
    return {
        "rows": len(v1_rows),
        "v1": v1_metrics,
        "v2": v2_metrics,
        "preservation_rule": (
            "V2 Macro F1 and Exact Match are each no more than 0.01 below V1 "
            "on identical DEV rows"
        ),
        "preserved": preserved,
    }


def save_artifacts(snapshot, vectorizer, classifiers, train_rows, dev_rows):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    for aspect, classifier in classifiers.items():
        joblib.dump(classifier, MODEL_DIR / f"{aspect}_classifier.joblib")
    metadata = {
        "artifact_version": "aspect_presence_logreg_v2",
        "task": "multi-label aspect presence detection with natural all-zero coverage",
        "aspects": list(ASPECTS),
        "threshold": THRESHOLD,
        "input": "sentence only",
        "source_dataset": "data/aspect_presence_v2/train.csv",
        "raw_source": DATASET_ID,
        "raw_snapshot_commit": snapshot.name,
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
    return metadata


def write_baseline_report(results, common, metadata):
    path = ANALYSIS_DIR / "aspect_presence_v2_baseline.txt"
    overall = results["overall"]
    zero = results["all_zero"]
    target = results["target_containing"]
    lines = [
        "ASPECT PRESENCE V2 — NATURAL ALL-ZERO CLASSICAL DEV BASELINE",
        "=" * 92,
        "",
        f"Train sentence count: {metadata['train_sentence_count']}",
        f"Dev sentence count: {metadata['dev_sentence_count']}",
        "Test evaluated: NO",
        "TF-IDF: word unigram + bigram, min_df=2",
        "Classifiers: four independent LogisticRegression models",
        "class_weight: balanced",
        f"Default threshold: {THRESHOLD:.1f}",
        "",
        "OVERALL DEV",
        f"Macro F1: {overall['macro_f1']:.4f}",
        f"Micro F1: {overall['micro_f1']:.4f}",
        f"Exact Match: {overall['exact_match']:.4f}",
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
    lines.extend(
        [
            "",
            "TRUE ALL-ZERO DEV SUBSET",
            f"Support: {zero['support']}",
            f"Correct 0/0/0/0 predictions: {zero['correct']}",
            f"All-zero Exact Accuracy: {zero['exact_accuracy']:.4f}",
            f"False-positive aspect count: {zero['false_positive_total']}",
            "False positives by aspect: "
            + ", ".join(
                f"{aspect}={zero['false_positive_by_aspect'][aspect]}"
                for aspect in ASPECTS
            ),
            "",
            "TARGET-CONTAINING DEV SUBSET",
            f"Macro F1: {target['macro_f1']:.4f}",
            f"Micro F1: {target['micro_f1']:.4f}",
            f"Exact Match: {target['exact_match']:.4f}",
            "",
            "LANGUAGE BREAKDOWN",
        ]
    )
    for language in LANGUAGES:
        item = results["language"][language]
        zero_accuracy = (
            f"{item['all_zero_exact_accuracy']:.4f}"
            if item["all_zero_exact_accuracy"] is not None else "N/A"
        )
        lines.append(
            f"{language}: macro_f1={item['macro_f1']:.4f}, "
            f"exact_match={item['exact_match']:.4f}, "
            f"all_zero_support={item['all_zero_support']}, "
            f"all_zero_exact_accuracy={zero_accuracy}"
        )
    lines.extend(
        [
            "",
            "V1 VS V2 ON IDENTICAL V1 DEV SENTENCES",
            f"Common rows: {common['rows']}",
            f"Rule: {common['preservation_rule']}",
            (
                "V1: "
                f"macro_f1={common['v1']['macro_f1']:.4f}, "
                f"micro_f1={common['v1']['micro_f1']:.4f}, "
                f"exact_match={common['v1']['exact_match']:.4f}, "
                f"hamming_loss={common['v1']['hamming_loss']:.4f}"
            ),
            (
                "V2: "
                f"macro_f1={common['v2']['macro_f1']:.4f}, "
                f"micro_f1={common['v2']['micro_f1']:.4f}, "
                f"exact_match={common['v2']['exact_match']:.4f}, "
                f"hamming_loss={common['v2']['hamming_loss']:.4f}"
            ),
            f"Existing target-sentence performance preserved: {'YES' if common['preserved'] else 'NO'}",
            "",
            "INTERPRETATION NOTE",
            (
                "V1 and V2 overall DEV metrics use different DEV populations. "
                "The common-subset comparison above is the controlled check for "
                "existing target-containing sentence behavior."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def validate_outputs(rows_by_split):
    leakage = leakage_counts(rows_by_split)
    if any(leakage.values()):
        raise ValueError(f"Final leakage is non-zero: {leakage}")
    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError(f"Empty split: {split}")
        if len(rows) != len({row_key(row) for row in rows}):
            raise ValueError(f"Duplicate sentence+language row in {split}")
        for row in rows:
            if tuple(row) != OUTPUT_COLUMNS:
                raise ValueError(f"Output schema/order mismatch in {split}")
            if any(int(row[aspect]) not in (0, 1) for aspect in ASPECTS):
                raise ValueError(f"Non-binary label in {split}")


def main():
    snapshot = resolve_raw_snapshot()
    records_by_split, source_files, categories = load_raw_source(snapshot)
    rows_by_split, metadata_by_split, build_info = build_sentence_dataset(
        records_by_split
    )
    validate_outputs(rows_by_split)
    v1_preservation = verify_v1_positive_preservation(rows_by_split)
    save_dataset(rows_by_split)
    audit_path = write_audit(
        snapshot,
        source_files,
        categories,
        rows_by_split,
        metadata_by_split,
        build_info,
        v1_preservation,
    )

    vectorizer, classifiers, y_dev, v2_predictions = fit_models(
        rows_by_split["train"], rows_by_split["dev"]
    )
    results = evaluate_v2(rows_by_split["dev"], y_dev, v2_predictions)
    metadata = save_artifacts(
        snapshot,
        vectorizer,
        classifiers,
        rows_by_split["train"],
        rows_by_split["dev"],
    )
    common = common_subset_comparison(rows_by_split["dev"], v2_predictions)
    baseline_path = write_baseline_report(results, common, metadata)

    stats = {split: split_stats(rows_by_split[split]) for split in SPLITS}
    print("ASPECT PRESENCE V2 COMPLETE")
    print("TEST split was created and audited but NOT evaluated.")
    for split in SPLITS:
        print(
            f"{split}: rows={stats[split]['rows']}, "
            f"all_zero={stats[split]['all_zero']}"
        )
    print(f"DEV Macro F1: {results['overall']['macro_f1']:.4f}")
    print(f"DEV Micro F1: {results['overall']['micro_f1']:.4f}")
    print(f"DEV Exact Match: {results['overall']['exact_match']:.4f}")
    print(f"DEV Hamming Loss: {results['overall']['hamming_loss']:.4f}")
    print(
        "DEV All-zero Exact Accuracy: "
        f"{results['all_zero']['exact_accuracy']:.4f}"
    )
    print(
        "Target-containing Exact Match: "
        f"{results['target_containing']['exact_match']:.4f}"
    )
    print(
        "V1 common-subset performance preserved: "
        f"{'YES' if common['preserved'] else 'NO'}"
    )
    print(f"Audit: {audit_path}")
    print(f"Baseline: {baseline_path}")
    print(f"Artifacts: {MODEL_DIR}")


if __name__ == "__main__":
    main()
