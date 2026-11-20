"""Experiment 10: neutral distribution, source-control and dose diagnostics.

Only the cleaned train/dev data and the verified neutral manifest are used.
The test split is deliberately never loaded.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

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
MANIFEST_PATH = ROOT / "data" / "expanded_neutral_v1" / "added_neutral_manifest.csv"

REPORT_PATH = ROOT / "data" / "analysis" / "experiment_10_distribution_source_control.txt"
MATCHED_CSV_PATH = (
    ROOT / "data" / "analysis" / "experiment_10_atmosphere_source_matched_runs.csv"
)
DOSE_CSV_PATH = (
    ROOT / "data" / "analysis" / "experiment_10_human_authored_dose_response.csv"
)

FIXED_WEIGHTS = {
    "positive": 0.54042031,
    "neutral": 11.88530466,
    "negative": 1.11841270,
}
LANGUAGES = ("tr", "en", "es")
ONE_HOT_NEUTRAL_COEFFICIENTS = {
    "taste": -0.4875,
    "atmosphere": 0.6049,
    "service": -0.4212,
    "value": 0.2565,
}
DOSE_TARGET_ROWS = (42, 84, 126, 165)


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def validate_inputs(old_rows, expanded_rows, dev_rows, manifest):
    expected_old = Counter({"positive": 2173, "neutral": 93, "negative": 1050})
    expected_expanded = Counter(
        {"positive": 2173, "neutral": 300, "negative": 1050}
    )
    expected_dev = Counter({"positive": 586, "neutral": 57, "negative": 258})
    actual = {
        "old": Counter(row["sentiment"] for row in old_rows),
        "expanded": Counter(row["sentiment"] for row in expanded_rows),
        "dev": Counter(row["sentiment"] for row in dev_rows),
    }
    expected = {"old": expected_old, "expanded": expected_expanded, "dev": expected_dev}
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(
                f"{name} distribution mismatch: expected={expected[name]}, actual={actual[name]}"
            )
    if len(manifest) != 207:
        raise ValueError(f"Expected 207 manifest rows, found {len(manifest)}")
    if Counter(row["source_type"] for row in manifest) != Counter(
        {"mams_gold_annotation": 42, "synthetic_candidate": 165}
    ):
        raise ValueError("Unexpected source-type counts in added neutral manifest")
    invalid = [
        row["candidate_id"]
        for row in manifest
        if row["human_verified"].lower() != "true"
        or (row["language"] != "en" and row["translation_human_verified"].lower() != "true")
        or row["sentiment"] != "neutral"
    ]
    if invalid:
        raise ValueError(f"Unverified manifest rows: {invalid[:10]}")


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


def group_manifest(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["parallel_source_group_id"]].append(row)
    for group_id, group_rows in groups.items():
        if len(group_rows) != 3:
            raise ValueError(f"Parallel group {group_id} does not contain 3 rows")
        if {row["language"] for row in group_rows} != set(LANGUAGES):
            raise ValueError(f"Parallel group {group_id} does not contain EN/TR/ES")
        if len({row["aspect"] for row in group_rows}) != 1:
            raise ValueError(f"Parallel group {group_id} has inconsistent aspects")
    return dict(groups)


def aspect_sentiment_distribution(rows):
    results = {}
    for aspect in ASPECTS:
        counts = Counter(
            row["sentiment"] for row in rows if row["aspect"] == aspect
        )
        total = sum(counts.values())
        results[aspect] = {
            "positive": counts["positive"],
            "neutral": counts["neutral"],
            "negative": counts["negative"],
            "total": total,
            "neutral_rate": counts["neutral"] / total,
        }
    return results


def language_aspect_distribution(rows):
    results = {}
    for language in LANGUAGES:
        for aspect in ASPECTS:
            selected = [
                row
                for row in rows
                if row["language"] == language and row["aspect"] == aspect
            ]
            neutral = sum(row["sentiment"] == "neutral" for row in selected)
            results[(language, aspect)] = {
                "total": len(selected),
                "neutral": neutral,
                "neutral_rate": neutral / len(selected),
            }
    return results


def neutral_metrics(y_true, y_pred):
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true, y_pred, labels=["neutral"], zero_division=0
    )
    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1_values[0]),
        "support": int(support[0]),
    }


def evaluate_subset(dev_rows, predictions, indices):
    y_true = [dev_rows[index]["sentiment"] for index in indices]
    y_pred = [predictions[index] for index in indices]
    neutral = neutral_metrics(y_true, y_pred)
    false_indices = [
        index
        for index in indices
        if predictions[index] == "neutral" and dev_rows[index]["sentiment"] != "neutral"
    ]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "neutral_precision": neutral["precision"],
        "neutral_recall": neutral["recall"],
        "neutral_f1": neutral["f1"],
        "pred_neutral": sum(prediction == "neutral" for prediction in y_pred),
        "false_neutral": len(false_indices),
        "false_neutral_by_language": Counter(
            dev_rows[index]["language"] for index in false_indices
        ),
        "false_neutral_by_aspect": Counter(
            dev_rows[index]["aspect"] for index in false_indices
        ),
    }


def train_and_evaluate(old_rows, additions, dev_rows):
    train_rows = old_rows + manifest_to_train_rows(additions)
    model = build_model()
    model.set_params(classifier__class_weight=FIXED_WEIGHTS)
    model.fit(build_inputs(train_rows), [row["sentiment"] for row in train_rows])
    predictions = model.predict(build_inputs(dev_rows)).tolist()
    all_indices = list(range(len(dev_rows)))
    atmosphere_indices = [
        index for index, row in enumerate(dev_rows) if row["aspect"] == "atmosphere"
    ]
    return {
        "overall": evaluate_subset(dev_rows, predictions, all_indices),
        "atmosphere": evaluate_subset(dev_rows, predictions, atmosphere_indices),
        "predictions": predictions,
    }


def matched_row(run_type, seed, selected_group_ids, result):
    overall = result["overall"]
    atmosphere = result["atmosphere"]
    return {
        "run_type": run_type,
        "seed": "" if seed is None else seed,
        "rows": len(selected_group_ids) * 3,
        "semantic_groups": len(selected_group_ids),
        "selected_parallel_source_group_ids": ";".join(selected_group_ids),
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "neutral_precision": overall["neutral_precision"],
        "neutral_recall": overall["neutral_recall"],
        "neutral_f1": overall["neutral_f1"],
        "pred_neutral": overall["pred_neutral"],
        "false_neutral": overall["false_neutral"],
        "atmosphere_accuracy": atmosphere["accuracy"],
        "atmosphere_macro_f1": atmosphere["macro_f1"],
        "atmosphere_neutral_precision": atmosphere["neutral_precision"],
        "atmosphere_neutral_recall": atmosphere["neutral_recall"],
        "atmosphere_neutral_f1": atmosphere["neutral_f1"],
        "atmosphere_pred_neutral": atmosphere["pred_neutral"],
        "atmosphere_false_neutral": atmosphere["false_neutral"],
        "atmosphere_false_neutral_tr": atmosphere["false_neutral_by_language"]["tr"],
        "atmosphere_false_neutral_en": atmosphere["false_neutral_by_language"]["en"],
        "atmosphere_false_neutral_es": atmosphere["false_neutral_by_language"]["es"],
    }


def summarize_runs(rows, fields):
    summary = {}
    for field in fields:
        values = [float(row[field]) for row in rows]
        summary[field] = {
            "mean": mean(values),
            "std": pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def proportional_quotas(group_counts, target_groups):
    total_available = sum(group_counts.values())
    raw = {
        aspect: target_groups * group_counts[aspect] / total_available
        for aspect in ASPECTS
    }
    quotas = {aspect: int(np.floor(raw[aspect])) for aspect in ASPECTS}
    remaining = target_groups - sum(quotas.values())
    ranked = sorted(
        ASPECTS,
        key=lambda aspect: (raw[aspect] - quotas[aspect], group_counts[aspect]),
        reverse=True,
    )
    while remaining:
        changed = False
        for aspect in ranked:
            if quotas[aspect] < group_counts[aspect]:
                quotas[aspect] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise ValueError("Could not allocate proportional dose quotas")
    return quotas


def dose_row(target_rows, selected_groups, quotas, result):
    overall = result["overall"]
    return {
        "target_rows": target_rows,
        "actual_rows": len(selected_groups) * 3,
        "semantic_groups": len(selected_groups),
        "atmosphere_groups": quotas["atmosphere"],
        "service_groups": quotas["service"],
        "value_groups": quotas["value"],
        "selected_parallel_source_group_ids": ";".join(selected_groups),
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "neutral_precision": overall["neutral_precision"],
        "neutral_recall": overall["neutral_recall"],
        "neutral_f1": overall["neutral_f1"],
        "pred_neutral": overall["pred_neutral"],
        "false_neutral": overall["false_neutral"],
        **{
            f"false_neutral_{aspect}": overall["false_neutral_by_aspect"][aspect]
            for aspect in ASPECTS
        },
        **{
            f"false_neutral_{language}": overall["false_neutral_by_language"][language]
            for language in LANGUAGES
        },
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {}
            for field, value in row.items():
                if isinstance(value, float):
                    output[field] = f"{value:.8f}"
                else:
                    output[field] = value
            writer.writerow(output)


def monotonic_direction(values, direction):
    comparisons = []
    for left, right in zip(values, values[1:]):
        comparisons.append(right >= left if direction == "up" else right <= left)
    return sum(comparisons), len(comparisons)


def format_stats(stats):
    return (
        f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
        f"min={stats['min']:.4f}, max={stats['max']:.4f}"
    )


def main():
    old_rows = load_rows(OLD_TRAIN_PATH, expected_split="train")
    expanded_rows = load_rows(EXPANDED_TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")
    manifest = read_csv(MANIFEST_PATH)
    validate_inputs(old_rows, expanded_rows, dev_rows, manifest)

    old_distribution = aspect_sentiment_distribution(old_rows)
    expanded_distribution = aspect_sentiment_distribution(expanded_rows)
    old_language_aspect = language_aspect_distribution(old_rows)
    expanded_language_aspect = language_aspect_distribution(expanded_rows)

    all_groups = group_manifest(manifest)
    mams_groups = {
        group_id: group_rows
        for group_id, group_rows in all_groups.items()
        if group_rows[0]["source_type"] == "mams_gold_annotation"
    }
    human_groups = {
        group_id: group_rows
        for group_id, group_rows in all_groups.items()
        if group_rows[0]["source_type"] == "synthetic_candidate"
    }
    mams_atmosphere_groups = {
        group_id: group_rows
        for group_id, group_rows in mams_groups.items()
        if group_rows[0]["aspect"] == "atmosphere"
    }
    human_atmosphere_groups = {
        group_id: group_rows
        for group_id, group_rows in human_groups.items()
        if group_rows[0]["aspect"] == "atmosphere"
    }
    if len(mams_atmosphere_groups) != 4 or len(human_atmosphere_groups) != 14:
        raise ValueError(
            "Unexpected atmosphere group counts: "
            f"MAMS={len(mams_atmosphere_groups)}, human={len(human_atmosphere_groups)}"
        )

    old_result = train_and_evaluate(old_rows, [], dev_rows)
    full_result = train_and_evaluate(old_rows, manifest, dev_rows)

    mams_group_ids = sorted(mams_atmosphere_groups)
    mams_additions = [
        row for group_id in mams_group_ids for row in mams_atmosphere_groups[group_id]
    ]
    mams_result = train_and_evaluate(old_rows, mams_additions, dev_rows)
    matched_rows = [matched_row("mams_atmosphere", None, mams_group_ids, mams_result)]
    human_matched_results = []
    human_atmosphere_ids = sorted(human_atmosphere_groups)
    for seed in range(1, 11):
        selected_ids = sorted(random.Random(seed).sample(human_atmosphere_ids, 4))
        additions = [
            row for group_id in selected_ids for row in human_atmosphere_groups[group_id]
        ]
        result = train_and_evaluate(old_rows, additions, dev_rows)
        human_matched_results.append(result)
        matched_rows.append(matched_row("human_authored_atmosphere", seed, selected_ids, result))
    write_csv(MATCHED_CSV_PATH, matched_rows)

    matched_fields = [
        "accuracy",
        "macro_f1",
        "neutral_precision",
        "neutral_recall",
        "neutral_f1",
        "pred_neutral",
        "false_neutral",
        "atmosphere_accuracy",
        "atmosphere_macro_f1",
        "atmosphere_neutral_precision",
        "atmosphere_neutral_recall",
        "atmosphere_neutral_f1",
        "atmosphere_pred_neutral",
        "atmosphere_false_neutral",
        "atmosphere_false_neutral_tr",
        "atmosphere_false_neutral_en",
        "atmosphere_false_neutral_es",
    ]
    human_summary = summarize_runs(matched_rows[1:], matched_fields)

    human_by_aspect = defaultdict(list)
    for group_id, group_rows in human_groups.items():
        human_by_aspect[group_rows[0]["aspect"]].append(group_id)
    human_group_counts = Counter(
        {
            aspect: len(human_by_aspect[aspect])
            for aspect in ASPECTS
            if human_by_aspect[aspect]
        }
    )
    if human_group_counts != Counter({"atmosphere": 14, "service": 22, "value": 19}):
        raise ValueError(f"Unexpected human-authored group distribution: {human_group_counts}")
    ordered_groups = {}
    for position, aspect in enumerate(ASPECTS):
        group_ids = sorted(human_by_aspect[aspect])
        random.Random(42 + position).shuffle(group_ids)
        ordered_groups[aspect] = group_ids

    dose_rows = []
    dose_results = []
    for target_rows in DOSE_TARGET_ROWS:
        target_groups = round(target_rows / 3)
        quotas = proportional_quotas(human_group_counts, target_groups)
        selected_ids = sorted(
            group_id
            for aspect in ASPECTS
            for group_id in ordered_groups[aspect][: quotas[aspect]]
        )
        additions = [row for group_id in selected_ids for row in human_groups[group_id]]
        result = train_and_evaluate(old_rows, additions, dev_rows)
        dose_results.append(result)
        dose_rows.append(dose_row(target_rows, selected_ids, quotas, result))
    write_csv(DOSE_CSV_PATH, dose_rows)

    expanded_rates = [expanded_distribution[aspect]["neutral_rate"] for aspect in ASPECTS]
    coefficients = [ONE_HOT_NEUTRAL_COEFFICIENTS[aspect] for aspect in ASPECTS]
    full_false_by_aspect = full_result["overall"]["false_neutral_by_aspect"]
    false_counts = [full_false_by_aspect[aspect] for aspect in ASPECTS]
    prevalence_coef_corr = float(spearmanr(expanded_rates, coefficients).statistic)
    prevalence_false_corr = float(spearmanr(expanded_rates, false_counts).statistic)

    mams_atm = mams_result["atmosphere"]
    style_directions = {
        "more_pred_neutral": sum(
            result["atmosphere"]["pred_neutral"] > mams_atm["pred_neutral"]
            for result in human_matched_results
        ),
        "more_false_neutral": sum(
            result["atmosphere"]["false_neutral"] > mams_atm["false_neutral"]
            for result in human_matched_results
        ),
        "lower_neutral_precision": sum(
            result["atmosphere"]["neutral_precision"] < mams_atm["neutral_precision"]
            for result in human_matched_results
        ),
        "lower_macro_f1": sum(
            result["atmosphere"]["macro_f1"] < mams_atm["macro_f1"]
            for result in human_matched_results
        ),
    }
    style_mean_adverse = {
        "more_pred_neutral": human_summary["atmosphere_pred_neutral"]["mean"]
        > mams_atm["pred_neutral"],
        "more_false_neutral": human_summary["atmosphere_false_neutral"]["mean"]
        > mams_atm["false_neutral"],
        "lower_neutral_precision": human_summary[
            "atmosphere_neutral_precision"
        ]["mean"]
        < mams_atm["neutral_precision"],
        "lower_macro_f1": human_summary["atmosphere_macro_f1"]["mean"]
        < mams_atm["macro_f1"],
    }
    if all(style_mean_adverse.values()) and min(style_directions.values()) >= 7:
        style_assessment = "SUPPORTED"
    elif sum(style_mean_adverse.values()) >= 2:
        style_assessment = "WEAK"
    else:
        style_assessment = "INCONCLUSIVE"

    dose_metric_values = {
        "neutral_precision": [row["neutral_precision"] for row in dose_rows],
        "neutral_recall": [row["neutral_recall"] for row in dose_rows],
        "neutral_f1": [row["neutral_f1"] for row in dose_rows],
        "pred_neutral": [row["pred_neutral"] for row in dose_rows],
        "false_neutral": [row["false_neutral"] for row in dose_rows],
        "macro_f1": [row["macro_f1"] for row in dose_rows],
    }
    dose_correlations = {
        metric: float(spearmanr([row["actual_rows"] for row in dose_rows], values).statistic)
        for metric, values in dose_metric_values.items()
    }
    dose_monotonic = {
        "neutral_precision": monotonic_direction(
            dose_metric_values["neutral_precision"], "down"
        ),
        "neutral_recall": monotonic_direction(dose_metric_values["neutral_recall"], "up"),
        "pred_neutral": monotonic_direction(dose_metric_values["pred_neutral"], "up"),
        "false_neutral": monotonic_direction(dose_metric_values["false_neutral"], "up"),
    }
    dose_core_supported = sum(
        [
            dose_correlations["neutral_recall"] >= 0.8,
            dose_correlations["pred_neutral"] >= 0.8,
            dose_correlations["false_neutral"] >= 0.8,
            dose_correlations["neutral_precision"] <= -0.8,
        ]
    )
    if dose_core_supported >= 3:
        dose_assessment = "SUPPORTED"
    elif dose_core_supported >= 1:
        dose_assessment = "WEAK"
    else:
        dose_assessment = "INCONCLUSIVE"

    if prevalence_coef_corr >= 0.8 and prevalence_false_corr >= 0.6:
        prior_assessment = "SUPPORTED"
    elif prevalence_coef_corr > 0 or prevalence_false_corr > 0:
        prior_assessment = "WEAK"
    else:
        prior_assessment = "INCONCLUSIVE"

    supported = [
        name
        for name, status in (
            ("A", style_assessment),
            ("B", dose_assessment),
            ("C", prior_assessment),
        )
        if status == "SUPPORTED"
    ]
    if len(supported) >= 2:
        overall_choice = "D) combination"
    elif supported == ["A"]:
        overall_choice = "A) source/style"
    elif supported == ["B"]:
        overall_choice = "B) quantity/dose"
    elif supported == ["C"]:
        overall_choice = "C) aspect-prior"
    else:
        overall_choice = "E) inconclusive"

    report = [
        "EXPERIMENT 10 — NEUTRAL DISTRIBUTION & SOURCE-CONTROL ANALYSIS",
        "",
        "Test split was not loaded or evaluated.",
        f"Fixed class weights: {FIXED_WEIGHTS}",
        "Representation: inline aspect=<aspect> [SEP] <sentence>.",
        "Pipeline: unchanged TF-IDF (1,2; min_df=2) + LogisticRegression.",
        "",
        "1. ASPECT × SENTIMENT DISTRIBUTION AUDIT",
        "Aspect | Old Pos | Old Neu | Old Neg | Old Total | Old Neutral % | "
        "Expanded Pos | Expanded Neu | Expanded Neg | Expanded Total | "
        "Expanded Neutral % | PP Change | Enrichment | Added Neu / Old Total",
    ]
    for aspect in ASPECTS:
        old = old_distribution[aspect]
        expanded = expanded_distribution[aspect]
        added_neutral = expanded["neutral"] - old["neutral"]
        enrichment = expanded["neutral_rate"] / old["neutral_rate"]
        report.append(
            f"{aspect} | {old['positive']} | {old['neutral']} | {old['negative']} | "
            f"{old['total']} | {old['neutral_rate']:.4%} | {expanded['positive']} | "
            f"{expanded['neutral']} | {expanded['negative']} | {expanded['total']} | "
            f"{expanded['neutral_rate']:.4%} | "
            f"{(expanded['neutral_rate'] - old['neutral_rate']) * 100:+.4f} pp | "
            f"{enrichment:.4f}x | {added_neutral / old['total']:.4%}"
        )

    report.extend(
        [
            "",
            "2. LANGUAGE × ASPECT NEUTRAL RATES",
            "Language | Aspect | Old Total | Old Neutral | Old Rate | "
            "Expanded Total | Expanded Neutral | Expanded Rate | PP Change",
        ]
    )
    for language in LANGUAGES:
        for aspect in ASPECTS:
            old = old_language_aspect[(language, aspect)]
            expanded = expanded_language_aspect[(language, aspect)]
            report.append(
                f"{language.upper()} | {aspect} | {old['total']} | {old['neutral']} | "
                f"{old['neutral_rate']:.4%} | {expanded['total']} | "
                f"{expanded['neutral']} | {expanded['neutral_rate']:.4%} | "
                f"{(expanded['neutral_rate'] - old['neutral_rate']) * 100:+.4f} pp"
            )

    report.extend(
        [
            "",
            "3. ASPECT PRIOR ANALYSIS",
            "Aspect | Expanded Neutral Rate | Enrichment | One-hot Neutral Coef | "
            "Full Expanded False-neutral",
        ]
    )
    for aspect in ASPECTS:
        enrichment = (
            expanded_distribution[aspect]["neutral_rate"]
            / old_distribution[aspect]["neutral_rate"]
        )
        report.append(
            f"{aspect} | {expanded_distribution[aspect]['neutral_rate']:.4%} | "
            f"{enrichment:.4f}x | {ONE_HOT_NEUTRAL_COEFFICIENTS[aspect]:+.4f} | "
            f"{full_false_by_aspect[aspect]}"
        )
    report.extend(
        [
            f"Spearman(expanded neutral rate, one-hot coefficient): {prevalence_coef_corr:+.4f}",
            f"Spearman(expanded neutral rate, false-neutral count): {prevalence_false_corr:+.4f}",
            "This is correlational diagnostic evidence and does not establish causality.",
            "",
            "4–6. ATMOSPHERE COUNT-MATCHED SOURCE TEST",
            "MAMS set: 4 complete atmosphere groups × 3 languages = 12 rows.",
            "Each human-authored run: 4 complete atmosphere groups × 3 languages = 12 rows.",
            "Human-authored sampling seeds: 1–10.",
        ]
    )

    def add_result_block(title, result):
        overall = result["overall"]
        atmosphere = result["atmosphere"]
        report.extend(
            [
                title,
                f"Overall Accuracy: {overall['accuracy']:.4f}",
                f"Overall Macro F1: {overall['macro_f1']:.4f}",
                f"Overall Neutral P/R/F1: {overall['neutral_precision']:.4f} / "
                f"{overall['neutral_recall']:.4f} / {overall['neutral_f1']:.4f}",
                f"Overall Pred neutral: {overall['pred_neutral']}",
                f"Overall False neutral: {overall['false_neutral']}",
                f"Atmosphere Accuracy: {atmosphere['accuracy']:.4f}",
                f"Atmosphere Macro F1: {atmosphere['macro_f1']:.4f}",
                f"Atmosphere Neutral P/R/F1: {atmosphere['neutral_precision']:.4f} / "
                f"{atmosphere['neutral_recall']:.4f} / {atmosphere['neutral_f1']:.4f}",
                f"Atmosphere Pred neutral: {atmosphere['pred_neutral']}",
                f"Atmosphere False neutral: {atmosphere['false_neutral']}",
                "Atmosphere False neutral by language: "
                + ", ".join(
                    f"{language.upper()}={atmosphere['false_neutral_by_language'][language]}"
                    for language in LANGUAGES
                ),
            ]
        )

    add_result_block("OLD baseline (fixed weights):", old_result)
    add_result_block("OLD + MAMS atmosphere 12:", mams_result)
    report.extend(["Human-authored atmosphere 12 aggregate (10 seeds):"])
    for field in matched_fields:
        report.append(f"{field}: {format_stats(human_summary[field])}")
    report.append("Individual seed results:")
    for row in matched_rows[1:]:
        report.append(
            f"seed={row['seed']}: overall_macro_f1={row['macro_f1']:.4f}, "
            f"overall_neutral_P/R/F1={row['neutral_precision']:.4f}/"
            f"{row['neutral_recall']:.4f}/{row['neutral_f1']:.4f}, "
            f"atmosphere_macro_f1={row['atmosphere_macro_f1']:.4f}, "
            f"atmosphere_neutral_P/R/F1={row['atmosphere_neutral_precision']:.4f}/"
            f"{row['atmosphere_neutral_recall']:.4f}/"
            f"{row['atmosphere_neutral_f1']:.4f}, "
            f"atmosphere_pred_neutral={row['atmosphere_pred_neutral']}, "
            f"atmosphere_false_neutral={row['atmosphere_false_neutral']}"
        )
    report.extend(
        [
            "Source/style directional comparisons vs fixed MAMS atmosphere set "
            "(number of human-authored seeds out of 10):",
            *[f"{name}: {count}/10" for name, count in style_directions.items()],
            f"SOURCE/STYLE EFFECT: {style_assessment}",
            "Scope limitation: this source comparison applies only to atmosphere; "
            "it must not be generalized directly to taste, service, or value.",
            "",
            "8–9. HUMAN-AUTHORED DOSE RESPONSE",
            "Rows | Aspect group quotas (atmosphere/service/value) | Accuracy | Macro F1 | "
            "Neutral P | Neutral R | Neutral F1 | Pred neutral | False neutral | "
            "False-neutral aspects (taste/atmosphere/service/value) | "
            "False-neutral languages (TR/EN/ES)",
        ]
    )
    for row in dose_rows:
        report.append(
            f"{row['actual_rows']} | {row['atmosphere_groups']}/"
            f"{row['service_groups']}/{row['value_groups']} | "
            f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['neutral_precision']:.4f} | {row['neutral_recall']:.4f} | "
            f"{row['neutral_f1']:.4f} | {row['pred_neutral']} | "
            f"{row['false_neutral']} | {row['false_neutral_taste']}/"
            f"{row['false_neutral_atmosphere']}/{row['false_neutral_service']}/"
            f"{row['false_neutral_value']} | {row['false_neutral_tr']}/"
            f"{row['false_neutral_en']}/{row['false_neutral_es']}"
        )
    report.extend(["Dose Spearman correlations with actual added rows:"])
    report.extend(
        f"{metric}: {correlation:+.4f}"
        for metric, correlation in dose_correlations.items()
    )
    report.extend(["Dose expected-direction adjacent transitions:"])
    report.extend(
        f"{metric}: {passed}/{total}"
        for metric, (passed, total) in dose_monotonic.items()
    )
    report.extend(
        [
            f"QUANTITY/DOSE EFFECT: {dose_assessment}",
            "",
            "10. ATMOSPHERE PROGRESSION CONTROL",
            "Condition | Atmosphere Macro F1 | Neutral P | Neutral R | Neutral F1 | "
            "Pred neutral | False neutral",
            f"OLD fixed-weight baseline | {old_result['atmosphere']['macro_f1']:.4f} | "
            f"{old_result['atmosphere']['neutral_precision']:.4f} | "
            f"{old_result['atmosphere']['neutral_recall']:.4f} | "
            f"{old_result['atmosphere']['neutral_f1']:.4f} | "
            f"{old_result['atmosphere']['pred_neutral']} | "
            f"{old_result['atmosphere']['false_neutral']}",
            f"OLD + MAMS atmosphere 12 | {mams_result['atmosphere']['macro_f1']:.4f} | "
            f"{mams_result['atmosphere']['neutral_precision']:.4f} | "
            f"{mams_result['atmosphere']['neutral_recall']:.4f} | "
            f"{mams_result['atmosphere']['neutral_f1']:.4f} | "
            f"{mams_result['atmosphere']['pred_neutral']} | "
            f"{mams_result['atmosphere']['false_neutral']}",
            f"OLD + matched human atmosphere 12 mean | "
            f"{human_summary['atmosphere_macro_f1']['mean']:.4f} | "
            f"{human_summary['atmosphere_neutral_precision']['mean']:.4f} | "
            f"{human_summary['atmosphere_neutral_recall']['mean']:.4f} | "
            f"{human_summary['atmosphere_neutral_f1']['mean']:.4f} | "
            f"{human_summary['atmosphere_pred_neutral']['mean']:.2f} | "
            f"{human_summary['atmosphere_false_neutral']['mean']:.2f}",
            f"FULL expanded | {full_result['atmosphere']['macro_f1']:.4f} | "
            f"{full_result['atmosphere']['neutral_precision']:.4f} | "
            f"{full_result['atmosphere']['neutral_recall']:.4f} | "
            f"{full_result['atmosphere']['neutral_f1']:.4f} | "
            f"{full_result['atmosphere']['pred_neutral']} | "
            f"{full_result['atmosphere']['false_neutral']}",
            "",
            "FINAL EVIDENCE-BASED CONCLUSION",
            f"SOURCE/STYLE EFFECT: {style_assessment}",
            f"QUANTITY/DOSE EFFECT: {dose_assessment}",
            f"ASPECT-PRIOR DISTORTION: {prior_assessment}",
            f"PRIMARY PROBLEM MOST CONSISTENT WITH: {overall_choice}",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n1. ASPECT NEUTRAL RATES")
    print("Aspect | Old Neutral % | Expanded Neutral % | Enrichment | One-hot coef | False-neutral")
    for aspect in ASPECTS:
        enrichment = (
            expanded_distribution[aspect]["neutral_rate"]
            / old_distribution[aspect]["neutral_rate"]
        )
        print(
            f"{aspect} | {old_distribution[aspect]['neutral_rate']:.2%} | "
            f"{expanded_distribution[aspect]['neutral_rate']:.2%} | "
            f"{enrichment:.2f}x | {ONE_HOT_NEUTRAL_COEFFICIENTS[aspect]:+.4f} | "
            f"{full_false_by_aspect[aspect]}"
        )

    print("\n2. ATMOSPHERE COUNT-MATCHED SOURCE TEST")
    for title, result in (("OLD atmosphere", old_result), ("OLD + MAMS atmosphere 12", mams_result)):
        atmosphere = result["atmosphere"]
        print(f"{title}:")
        print(f"Macro F1: {atmosphere['macro_f1']:.4f}")
        print(f"Neutral P: {atmosphere['neutral_precision']:.4f}")
        print(f"Neutral R: {atmosphere['neutral_recall']:.4f}")
        print(f"Neutral F1: {atmosphere['neutral_f1']:.4f}")
        print(f"False neutral: {atmosphere['false_neutral']}")
    print("OLD + Human-authored atmosphere 12:")
    for field, label in (
        ("atmosphere_macro_f1", "Macro F1"),
        ("atmosphere_neutral_precision", "Neutral P"),
        ("atmosphere_neutral_recall", "Neutral R"),
        ("atmosphere_neutral_f1", "Neutral F1"),
        ("atmosphere_false_neutral", "False neutral"),
    ):
        stats = human_summary[field]
        print(f"{label} mean ± std: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\n3. HUMAN-AUTHORED DOSE RESPONSE")
    print("Rows | Macro F1 | Neutral P | Neutral R | Neutral F1 | Pred neutral | False neutral")
    for row in dose_rows:
        print(
            f"{row['actual_rows']} | {row['macro_f1']:.4f} | "
            f"{row['neutral_precision']:.4f} | {row['neutral_recall']:.4f} | "
            f"{row['neutral_f1']:.4f} | {row['pred_neutral']} | "
            f"{row['false_neutral']}"
        )

    print("\n4. FINAL EVIDENCE-BASED CONCLUSION")
    print(f"SOURCE/STYLE EFFECT: {style_assessment}")
    print(f"QUANTITY/DOSE EFFECT: {dose_assessment}")
    print(f"ASPECT-PRIOR DISTORTION: {prior_assessment}")
    print(f"PRIMARY PROBLEM MOST CONSISTENT WITH: {overall_choice}")
    print(f"Report: {REPORT_PATH}")
    print(f"Matched runs: {MATCHED_CSV_PATH}")
    print(f"Dose response: {DOSE_CSV_PATH}")


if __name__ == "__main__":
    main()
