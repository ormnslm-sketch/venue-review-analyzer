"""Analyze neutral false positives and source effects on the dev split only."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.metrics.pairwise import cosine_similarity

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
DEV_PATH = ROOT / "data" / "expanded_neutral_v1" / "dev.csv"
MANIFEST_PATH = ROOT / "data" / "expanded_neutral_v1" / "added_neutral_manifest.csv"
REPORT_PATH = ROOT / "data" / "analysis" / "experiment_08_neutral_error_analysis.txt"
FALSE_NEUTRAL_PATH = (
    ROOT / "data" / "analysis" / "experiment_08_false_neutral_predictions.csv"
)

FIXED_WEIGHTS = {
    "positive": 0.54042031,
    "neutral": 11.88530466,
    "negative": 1.11841270,
}

EXPECTED_EXPANDED = {
    "accuracy": 0.7159,
    "macro_f1": 0.5805,
    "neutral_precision": 0.2066,
    "neutral_recall": 0.4386,
    "neutral_f1": 0.2809,
    "pred_neutral": 121,
}

LANGUAGES = (("tr", "TR"), ("en", "EN"), ("es", "ES"))

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)
STOPWORDS = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "they", "this", "to", "was",
    "we", "were", "with", "you", "your",
    # Turkish
    "ama", "bir", "bu", "da", "de", "daha", "gibi", "için", "ile", "mi",
    "mı", "mu", "mü", "o", "ve", "ya", "çok", "şu", "ise", "olan",
    # Spanish
    "a", "al", "con", "como", "de", "del", "el", "ella", "en", "era",
    "es", "esta", "fue", "la", "las", "lo", "los", "me", "mi", "para",
    "pero", "por", "que", "se", "sin", "su", "sus", "un", "una", "y",
}

CUE_TERMS = (
    "average", "normal", "fine", "similar", "same", "expected", "usual",
    "typical", "around", "standard", "minutes", "table", "menu", "price",
    "restaurant", "service", "waited", "ordered",
)


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def tokenize(text: str):
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def content_tokens(text: str):
    return [token for token in tokenize(text) if len(token) > 1 and token not in STOPWORDS]


def ngrams(tokens, size):
    return [" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


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


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classes": class_metrics(y_true, y_pred),
        "predicted": Counter(y_pred),
        "matrix": confusion_matrix(y_true, y_pred, labels=SENTIMENTS),
    }


def build_fixed_model(train_rows):
    model = build_model()
    model.set_params(classifier__class_weight=FIXED_WEIGHTS)
    model.fit(build_inputs(train_rows), [row["sentiment"] for row in train_rows])
    return model


def reproduce_or_stop(metrics):
    neutral = metrics["classes"]["neutral"]
    observed = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "neutral_precision": neutral["precision"],
        "neutral_recall": neutral["recall"],
        "neutral_f1": neutral["f1"],
        "pred_neutral": metrics["predicted"]["neutral"],
    }
    differences = {
        key: abs(observed[key] - expected)
        for key, expected in EXPECTED_EXPANDED.items()
        if key != "pred_neutral"
    }
    if max(differences.values()) > 0.00015 or observed["pred_neutral"] != 121:
        raise RuntimeError(
            "Experiment 07 reproduction failed; diagnostic analysis stopped. "
            f"Observed={observed}"
        )


def manifest_to_train_rows(rows):
    return [
        {
            "sentence": row["sentence"],
            "language": row["language"],
            "split": "train",
            "aspect": row["aspect"],
            "sentiment": "neutral",
        }
        for row in rows
    ]


def validate_manifest(manifest):
    if len(manifest) != 207:
        raise ValueError(f"Expected 207 manifest rows, found {len(manifest)}")
    invalid = []
    for row in manifest:
        human_ok = row["human_verified"].lower() == "true"
        translation_ok = (
            row["language"] == "en"
            or row["translation_human_verified"].lower() == "true"
        )
        if not human_ok or not translation_ok or row["sentiment"] != "neutral":
            invalid.append(row["candidate_id"])
    if invalid:
        raise ValueError(f"Unverified manifest rows: {invalid[:10]}")


def feature_rankings(model, top_n=50):
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    rankings = {}
    for class_name in SENTIMENTS:
        class_index = list(classifier.classes_).index(class_name)
        coefficients = classifier.coef_[class_index]
        top_indices = np.argsort(coefficients)[::-1][:top_n]
        rankings[class_name] = [
            (feature_names[index], float(coefficients[index])) for index in top_indices
        ]
    return rankings


def example_neutral_contributions(model, transformed_row, top_n=8):
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    neutral_index = list(classifier.classes_).index("neutral")
    coefficients = classifier.coef_[neutral_index]
    contributions = []
    for feature_index, value in zip(transformed_row.indices, transformed_row.data):
        contribution = float(value * coefficients[feature_index])
        if contribution > 0:
            contributions.append((feature_names[feature_index], contribution))
    contributions.sort(key=lambda item: item[1], reverse=True)
    return contributions[:top_n]


def lexical_stats(rows):
    sentences = [row["sentence"] for row in rows]
    token_lists = [tokenize(sentence) for sentence in sentences]
    content_lists = [content_tokens(sentence) for sentence in sentences]
    lengths = [len(tokens) for tokens in token_lists]
    all_vocab = {token for tokens in token_lists for token in tokens}
    unigram_counter = Counter(token for tokens in content_lists for token in tokens)
    bigram_counter = Counter(
        bigram for tokens in content_lists for bigram in ngrams(tokens, 2)
    )
    return {
        "row_count": len(rows),
        "average_length": mean(lengths) if lengths else 0.0,
        "median_length": median(lengths) if lengths else 0.0,
        "vocabulary_size": len(all_vocab),
        "unique_unigrams": len(unigram_counter),
        "unique_bigrams": len(bigram_counter),
        "top_unigrams": unigram_counter.most_common(20),
        "top_bigrams": bigram_counter.most_common(20),
    }


def format_counter_items(items):
    return ", ".join(f"{term} ({count})" for term, count in items)


def opening_patterns(rows, token_count=3, limit=15):
    counter = Counter()
    for row in rows:
        tokens = tokenize(row["sentence"])
        if tokens:
            counter[" ".join(tokens[:token_count])] += 1
    return counter.most_common(limit)


def repeated_ngrams(rows, size, limit=20):
    counter = Counter()
    for row in rows:
        tokens = content_tokens(row["sentence"])
        counter.update(ngrams(tokens, size))
    return [(term, count) for term, count in counter.most_common(limit) if count > 1]


def near_duplicate_clusters(rows, threshold=0.78):
    sentences = [row["sentence"] for row in rows]
    if len(sentences) < 2:
        return []
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(sentences)
    similarities = cosine_similarity(matrix)
    parent = list(range(len(rows)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if similarities[left, right] >= threshold:
                union(left, right)

    grouped = defaultdict(list)
    for index in range(len(rows)):
        grouped[find(index)].append(index)

    clusters = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        max_similarity = max(
            similarities[left, right]
            for position, left in enumerate(indices)
            for right in indices[position + 1 :]
        )
        clusters.append(
            {
                "indices": indices,
                "size": len(indices),
                "max_similarity": float(max_similarity),
            }
        )
    clusters.sort(key=lambda item: (item["size"], item["max_similarity"]), reverse=True)
    return clusters


def append_model_metrics(report, title, metrics):
    neutral = metrics["classes"]["neutral"]
    report.extend(
        [
            title,
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Macro F1: {metrics['macro_f1']:.4f}",
            f"Neutral precision: {neutral['precision']:.4f}",
            f"Neutral recall: {neutral['recall']:.4f}",
            f"Neutral F1: {neutral['f1']:.4f}",
            f"Pred neutral: {metrics['predicted']['neutral']}",
        ]
    )


def save_false_neutral_csv(rows):
    FALSE_NEUTRAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sentence",
        "language",
        "aspect",
        "true_sentiment",
        "predicted_sentiment",
        "old_baseline_prediction",
        "neutral_probability_or_score",
        "top_neutral_features",
    ]
    with FALSE_NEUTRAL_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main():
    old_train_rows = load_rows(OLD_TRAIN_PATH, expected_split="train")
    expanded_train_rows = load_rows(EXPANDED_TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")
    manifest = read_csv(MANIFEST_PATH)
    validate_manifest(manifest)

    y_true = [row["sentiment"] for row in dev_rows]
    dev_inputs = build_inputs(dev_rows)

    expanded_model = build_fixed_model(expanded_train_rows)
    expanded_predictions = expanded_model.predict(dev_inputs).tolist()
    expanded_probabilities = expanded_model.predict_proba(dev_inputs)
    expanded_metrics = evaluate(y_true, expanded_predictions)
    reproduce_or_stop(expanded_metrics)

    old_model = build_model()
    old_model.fit(
        build_inputs(old_train_rows),
        [row["sentiment"] for row in old_train_rows],
    )
    old_predictions = old_model.predict(dev_inputs).tolist()
    old_metrics = evaluate(y_true, old_predictions)

    transformed_dev = expanded_model.named_steps["tfidf"].transform(dev_inputs)
    neutral_class_index = list(
        expanded_model.named_steps["classifier"].classes_
    ).index("neutral")

    groups = Counter()
    false_neutral_indices = []
    for index, (actual, predicted) in enumerate(zip(y_true, expanded_predictions)):
        if actual == "neutral" and predicted == "neutral":
            groups["true_neutral"] += 1
        elif actual == "positive" and predicted == "neutral":
            groups["positive_to_neutral"] += 1
            false_neutral_indices.append(index)
        elif actual == "negative" and predicted == "neutral":
            groups["negative_to_neutral"] += 1
            false_neutral_indices.append(index)
        elif actual == "neutral" and predicted == "positive":
            groups["neutral_to_positive"] += 1
        elif actual == "neutral" and predicted == "negative":
            groups["neutral_to_negative"] += 1

    language_false = Counter(dev_rows[index]["language"] for index in false_neutral_indices)
    aspect_false = Counter(dev_rows[index]["aspect"] for index in false_neutral_indices)
    aspect_label_false = Counter(
        (dev_rows[index]["aspect"], dev_rows[index]["sentiment"])
        for index in false_neutral_indices
    )

    false_neutral_rows = []
    shared_feature_counter = Counter()
    shared_feature_contribution = Counter()

    human_manifest = [row for row in manifest if row["source_type"] == "synthetic_candidate"]
    mams_manifest = [row for row in manifest if row["source_type"] == "mams_gold_annotation"]
    human_inputs = [
        f"aspect={row['aspect']} [SEP] {row['sentence']}" for row in human_manifest
    ]
    human_matrix = expanded_model.named_steps["tfidf"].transform(human_inputs)
    human_document_frequency = np.asarray((human_matrix > 0).sum(axis=0)).ravel()
    feature_names = expanded_model.named_steps["tfidf"].get_feature_names_out()
    feature_to_index = {feature: index for index, feature in enumerate(feature_names)}

    for index in false_neutral_indices:
        contributions = example_neutral_contributions(
            expanded_model,
            transformed_dev[index],
            top_n=10,
        )
        for feature, contribution in contributions:
            feature_index = feature_to_index[feature]
            if human_document_frequency[feature_index] > 0:
                shared_feature_counter[feature] += 1
                shared_feature_contribution[feature] += contribution
        false_neutral_rows.append(
            {
                "sentence": dev_rows[index]["sentence"],
                "language": dev_rows[index]["language"],
                "aspect": dev_rows[index]["aspect"],
                "true_sentiment": y_true[index],
                "predicted_sentiment": expanded_predictions[index],
                "old_baseline_prediction": old_predictions[index],
                "neutral_probability_or_score": (
                    f"{expanded_probabilities[index, neutral_class_index]:.8f}"
                ),
                "top_neutral_features": "; ".join(
                    f"{feature}:{contribution:.6f}"
                    for feature, contribution in contributions
                ),
                "_probability": float(
                    expanded_probabilities[index, neutral_class_index]
                ),
            }
        )
    false_neutral_rows.sort(key=lambda row: row["_probability"], reverse=True)
    save_false_neutral_csv(false_neutral_rows)

    rankings = feature_rankings(expanded_model, top_n=50)
    neutral_rank_lookup = {
        feature: (rank, coefficient)
        for rank, (feature, coefficient) in enumerate(rankings["neutral"], start=1)
    }

    old_neutral = [row for row in old_train_rows if row["sentiment"] == "neutral"]
    lexical_groups = {
        "OLD_NEUTRAL": old_neutral,
        "NEW_MAMS_NEUTRAL": mams_manifest,
        "NEW_HUMAN_AUTHORED_NEUTRAL": human_manifest,
    }
    lexical_results = {
        name: lexical_stats(rows) for name, rows in lexical_groups.items()
    }

    openings = opening_patterns(human_manifest, token_count=3, limit=20)
    repeated_trigrams = repeated_ngrams(human_manifest, 3, limit=25)
    repeated_fourgrams = repeated_ngrams(human_manifest, 4, limit=25)
    similarity_threshold = 0.78
    duplicate_clusters = near_duplicate_clusters(
        human_manifest,
        threshold=similarity_threshold,
    )

    old_correct_expanded_wrong_neutral = sum(
        old_predictions[index] == y_true[index]
        and expanded_predictions[index] == "neutral"
        and y_true[index] != "neutral"
        for index in range(len(dev_rows))
    )
    old_wrong_expanded_correct_neutral = sum(
        old_predictions[index] != y_true[index]
        and expanded_predictions[index] == y_true[index] == "neutral"
        for index in range(len(dev_rows))
    )
    old_wrong_neutral_expanded_correct = sum(
        old_predictions[index] == "neutral"
        and old_predictions[index] != y_true[index]
        and expanded_predictions[index] == y_true[index]
        for index in range(len(dev_rows))
    )

    mams_train = old_train_rows + manifest_to_train_rows(mams_manifest)
    human_train = old_train_rows + manifest_to_train_rows(human_manifest)
    mams_model = build_fixed_model(mams_train)
    human_model = build_fixed_model(human_train)
    mams_predictions = mams_model.predict(dev_inputs).tolist()
    human_predictions = human_model.predict(dev_inputs).tolist()
    mams_metrics = evaluate(y_true, mams_predictions)
    human_metrics = evaluate(y_true, human_predictions)

    report = [
        "EXPERIMENT 08 — NEUTRAL FALSE POSITIVE & DOMAIN ANALYSIS",
        "",
        "CONTROLLED SETTINGS",
        "Main train: data/expanded_neutral_v1/train.csv",
        "Dev: data/expanded_neutral_v1/dev.csv",
        "Test: not loaded or evaluated",
        "Pipeline: unchanged TF-IDF + Logistic Regression",
        f"Fixed weights: {FIXED_WEIGHTS}",
        "",
        "1. EXPERIMENT 07 REPRODUCTION",
    ]
    append_model_metrics(report, "Expanded best configuration:", expanded_metrics)
    report.append("Reproduction check: PASS")
    report.extend(["", "Historical cleaned baseline reproduction:"])
    append_model_metrics(report, "Old cleaned dynamic-balanced model:", old_metrics)

    report.extend(
        [
            "",
            "2. NEUTRAL PREDICTION GROUPS",
            f"All predicted neutral: {expanded_metrics['predicted']['neutral']}",
            f"A) True neutral -> neutral: {groups['true_neutral']}",
            f"B) Positive -> neutral: {groups['positive_to_neutral']}",
            f"C) Negative -> neutral: {groups['negative_to_neutral']}",
            f"D) Neutral -> positive: {groups['neutral_to_positive']}",
            f"E) Neutral -> negative: {groups['neutral_to_negative']}",
            "",
            "3. FALSE-NEUTRAL BREAKDOWN",
            "Language x false-neutral count:",
            *[f"{display}: {language_false[code]}" for code, display in LANGUAGES],
            "",
            "Aspect x false-neutral count:",
            *[f"{aspect}: {aspect_false[aspect]}" for aspect in ASPECTS],
            "",
            "Aspect x true-label x false-neutral count:",
        ]
    )
    for aspect in ASPECTS:
        report.append(
            f"{aspect}: positive={aspect_label_false[(aspect, 'positive')]}, "
            f"negative={aspect_label_false[(aspect, 'negative')]}"
        )

    report.extend(["", "4. MODEL FEATURE ANALYSIS"])
    for class_name in SENTIMENTS:
        report.append(f"Top 50 positive coefficients for {class_name}:")
        report.append(
            ", ".join(
                f"{feature} ({coefficient:.4f})"
                for feature, coefficient in rankings[class_name]
            )
        )
    report.extend(["", "Requested cue terms in neutral top-50:"])
    for cue in CUE_TERMS:
        if cue in neutral_rank_lookup:
            rank, coefficient = neutral_rank_lookup[cue]
            report.append(f"{cue}: rank={rank}, coefficient={coefficient:.4f}")
        else:
            report.append(f"{cue}: not in neutral top-50")

    report.extend(["", "5. OLD VS NEW NEUTRAL LEXICAL ANALYSIS"])
    for name in ("OLD_NEUTRAL", "NEW_MAMS_NEUTRAL", "NEW_HUMAN_AUTHORED_NEUTRAL"):
        stats = lexical_results[name]
        report.extend(
            [
                f"{name}:",
                f"  rows: {stats['row_count']}",
                f"  average sentence length (tokens): {stats['average_length']:.2f}",
                f"  median sentence length (tokens): {stats['median_length']:.2f}",
                f"  vocabulary size (all normalized tokens): {stats['vocabulary_size']}",
                f"  unique content unigrams: {stats['unique_unigrams']}",
                f"  unique content bigrams: {stats['unique_bigrams']}",
                f"  top content unigrams: {format_counter_items(stats['top_unigrams'])}",
                f"  top content bigrams: {format_counter_items(stats['top_bigrams'])}",
            ]
        )

    report.extend(
        [
            "",
            "6. HUMAN-AUTHORED TEMPLATE / LEXICAL SIMILARITY",
            "Top repeated three-token sentence openings:",
            format_counter_items(openings),
            "Top repeated content trigrams:",
            format_counter_items(repeated_trigrams),
            "Top repeated content four-grams:",
            format_counter_items(repeated_fourgrams),
            f"Near-duplicate lexical threshold (char TF-IDF cosine): {similarity_threshold:.2f}",
            f"Near-duplicate lexical clusters: {len(duplicate_clusters)}",
            "These clusters indicate lexical/template similarity, not semantic duplication.",
        ]
    )
    for cluster_number, cluster in enumerate(duplicate_clusters[:10], start=1):
        report.append(
            f"Cluster {cluster_number}: size={cluster['size']}, "
            f"max_similarity={cluster['max_similarity']:.4f}"
        )
        for index in cluster["indices"][:6]:
            row = human_manifest[index]
            report.append(
                f"  {row['candidate_id']} [{row['language']}/{row['aspect']}]: "
                f"{row['sentence']}"
            )

    shared_ranked = sorted(
        shared_feature_counter,
        key=lambda feature: (
            shared_feature_contribution[feature],
            shared_feature_counter[feature],
        ),
        reverse=True,
    )
    report.extend(
        [
            "",
            "7. FALSE-POSITIVE CUE MATCHING",
            "Shared positive neutral-contribution features also present in human-authored neutral train rows:",
        ]
    )
    for feature in shared_ranked[:30]:
        feature_index = feature_to_index[feature]
        report.append(
            f"{feature}: false-positive examples={shared_feature_counter[feature]}, "
            f"summed contribution={shared_feature_contribution[feature]:.4f}, "
            f"human-authored document frequency={int(human_document_frequency[feature_index])}"
        )

    report.extend(["", "Top 30 highest-neutral-probability false positives:"])
    for rank, row in enumerate(false_neutral_rows[:30], start=1):
        report.extend(
            [
                f"{rank}. probability={row['_probability']:.4f} | "
                f"{row['language']} | {row['aspect']} | true={row['true_sentiment']} | "
                f"old={row['old_baseline_prediction']}",
                f"   Sentence: {row['sentence']}",
                f"   Neutral features: {row['top_neutral_features']}",
            ]
        )

    report.extend(
        [
            "",
            "8. OLD VS EXPANDED PREDICTION CHANGES",
            f"A) Old correct -> Expanded wrong-neutral: {old_correct_expanded_wrong_neutral}",
            f"B) Old wrong -> Expanded correct-neutral: {old_wrong_expanded_correct_neutral}",
            f"C) Old wrong-neutral -> Expanded correct: {old_wrong_neutral_expanded_correct}",
            "",
            "9. SOURCE-TYPE ABLATION DIAGNOSTIC",
            f"Verified MAMS additions used: {len(mams_manifest)}",
            f"Verified human-authored additions used: {len(human_manifest)}",
        ]
    )
    append_model_metrics(report, "Old + MAMS verified additions:", mams_metrics)
    report.append("")
    append_model_metrics(report, "Old + human-authored verified additions:", human_metrics)
    report.append("")
    append_model_metrics(report, "Old + all selected additions (Expanded v1):", expanded_metrics)

    report.extend(
        [
            "",
            "EVIDENCE SUMMARY",
            "MAIN PROBLEM MOST CONSISTENT WITH: combination of human-authored lexical/style shortcut and high neutral weighting, concentrated especially in atmosphere and Spanish.",
            "Evidence: the human-authored-only diagnostic predicted neutral 95 times with neutral precision 0.1895; MAMS-only predicted neutral 47 times with precision 0.2553.",
            "Evidence: 56 examples changed from old-correct to expanded wrong-neutral, while 14 changed from old-wrong to expanded correct-neutral.",
            "Evidence: false-neutral counts were highest for Spanish (40) and atmosphere (32).",
            "Evidence: neutral coefficients include average/ortalama/normal and structural features such as aspect atmosphere, atmosphere, table, and multilingual function-word patterns.",
            "Domain mismatch is not the strongest explanation because MAMS is restaurant-domain and caused materially less neutral overprediction than the human-authored source group.",
            "The high neutral weight amplifies these learned cues, but weight alone is insufficient as an explanation because the historical cleaned model used a similarly high neutral weight with higher precision and only 30 neutral predictions.",
            "Near-duplicate analysis found no char-TF-IDF clusters at the stated 0.78 threshold; the issue is broader repeated lexical/structural cue learning rather than exact or near-exact sentence duplication.",
            "No data or model settings were changed.",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
