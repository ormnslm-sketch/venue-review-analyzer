"""One-time final TEST evaluation for Experiment 15 Epoch 2.

No transformer training, optimizer construction, checkpoint writes, model
selection, or hyperparameter changes occur here. The historical classical
baseline is deterministically refit with its frozen clean-baseline settings
because no serialized classical model artifact exists.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from train_sentiment_baseline import build_inputs, build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "experiment_15_lora_expanded_balanced"
    / "best"
    / "epoch_2"
)
TRAIN_PATH = PROJECT_ROOT / "data" / "expanded_neutral_v1" / "train.csv"
DEV_PATH = PROJECT_ROOT / "data" / "expanded_neutral_v1" / "dev.csv"
TEST_PATH = PROJECT_ROOT / "data" / "expanded_neutral_v1" / "test.csv"
CLEAN_TRAIN_PATH = PROJECT_ROOT / "data" / "cleaned" / "train.csv"
AUDIT_PATH = PROJECT_ROOT / "data" / "analysis" / "neutral_expansion_v1_audit.txt"
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "final_model_test_evaluation.txt"

MODEL_NAME = "xlm-roberta-base"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
SENTIMENTS = ("positive", "neutral", "negative")
SENTIMENT_IDS = [LABEL2ID[label] for label in SENTIMENTS]
LANGUAGES = (("tr", "TR"), ("en", "EN"), ("es", "ES"))
ASPECTS = ("taste", "atmosphere", "service", "value")
REQUIRED_COLUMNS = {
    "sentence",
    "language",
    "split",
    "aspect",
    "sentiment",
}

EXPECTED_TEST_ROWS = 1316
EXPECTED_TEST_LANGUAGE = {"tr": 439, "en": 439, "es": 438}
EXPECTED_TEST_SENTIMENT = {"positive": 923, "neutral": 69, "negative": 324}
EXPECTED_TEST_ASPECT = {
    "taste": 668,
    "atmosphere": 162,
    "service": 363,
    "value": 123,
}
DEV_REFERENCE = {
    "accuracy": 0.7902330743618202,
    "macro_f1": 0.61493257155577,
    "positive_f1": 0.9137,
    "neutral_f1": 0.2162,
    "negative_f1": 0.7149,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path, expected_split):
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
        rows = list(reader)
    if any(row["split"] != expected_split for row in rows):
        raise RuntimeError(f"{path} contains rows outside split={expected_split}")
    return rows


def validate_checkpoint():
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "metrics.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = sorted(name for name in required if not (CHECKPOINT / name).exists())
    if missing:
        raise RuntimeError(f"Final checkpoint is incomplete: {missing}")
    metrics = json.loads((CHECKPOINT / "metrics.json").read_text(encoding="utf-8"))
    adapter = json.loads(
        (CHECKPOINT / "adapter_config.json").read_text(encoding="utf-8")
    )
    if metrics.get("checkpoint_id") != "15-best-epoch2":
        raise RuntimeError("Checkpoint identity is not Experiment 15 Epoch 2")
    if metrics.get("best_epoch") != 2 or metrics.get("condition") != "expanded":
        raise RuntimeError("Checkpoint best-epoch/condition mismatch")
    if adapter.get("base_model_name_or_path") != MODEL_NAME:
        raise RuntimeError("Checkpoint base model mismatch")
    if adapter.get("r") != 8 or adapter.get("lora_alpha") != 16:
        raise RuntimeError("Checkpoint LoRA r/alpha mismatch")
    if abs(float(adapter.get("lora_dropout")) - 0.1) > 1e-12:
        raise RuntimeError("Checkpoint LoRA dropout mismatch")
    if set(adapter.get("target_modules") or []) != {"query", "value"}:
        raise RuntimeError("Checkpoint LoRA target modules mismatch")
    return metrics


def validate_data():
    rows = {
        "train": load_rows(TRAIN_PATH, "train"),
        "dev": load_rows(DEV_PATH, "dev"),
        "test": load_rows(TEST_PATH, "test"),
    }
    test = rows["test"]
    language = Counter(row["language"] for row in test)
    sentiment = Counter(row["sentiment"] for row in test)
    aspect = Counter(row["aspect"] for row in test)
    if len(test) != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_TEST_ROWS} test rows, got {len(test)}")
    if dict(language) != EXPECTED_TEST_LANGUAGE:
        raise RuntimeError(f"Test language distribution mismatch: {dict(language)}")
    if dict(sentiment) != EXPECTED_TEST_SENTIMENT:
        raise RuntimeError(f"Test sentiment distribution mismatch: {dict(sentiment)}")
    if dict(aspect) != EXPECTED_TEST_ASPECT:
        raise RuntimeError(f"Test aspect distribution mismatch: {dict(aspect)}")

    audit = {}
    key_sets = {}
    for split, split_rows in rows.items():
        full_keys = [
            (row["sentence"], row["language"], row["aspect"], row["sentiment"])
            for row in split_rows
        ]
        label_sets = defaultdict(set)
        for row in split_rows:
            label_sets[(row["sentence"], row["language"], row["aspect"])].add(
                row["sentiment"]
            )
        audit[f"{split}_duplicates"] = len(full_keys) - len(set(full_keys))
        audit[f"{split}_conflicts"] = sum(
            len(labels) > 1 for labels in label_sets.values()
        )
        key_sets[split] = set(label_sets)
    audit["train_test_leakage"] = len(key_sets["train"] & key_sets["test"])
    audit["dev_test_leakage"] = len(key_sets["dev"] & key_sets["test"])
    audit["train_dev_leakage"] = len(key_sets["train"] & key_sets["dev"])
    if any(audit.values()):
        raise RuntimeError(f"Dataset audit failed: {audit}")
    if "Overall audit: PASS" not in AUDIT_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("Neutral Expansion v1 audit is not PASS")
    return rows, language, sentiment, aspect, audit


def model_input(row):
    return f"aspect={row['aspect']} [SEP] {row['sentence']}"


class TestDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            model_input(row),
            truncation=True,
            max_length=128,
        )
        encoded["labels"] = LABEL2ID[row["sentiment"]]
        return encoded


def neutral_metrics(labels, predictions):
    precision, recall, f1_values, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[LABEL2ID["neutral"]],
        zero_division=0,
    )
    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1_values[0]),
        "support": int(support[0]),
    }


def subset_metrics(rows, labels, predictions, column, value):
    indices = [index for index, row in enumerate(rows) if row[column] == value]
    true = [labels[index] for index in indices]
    pred = [predictions[index] for index in indices]
    return {
        "n": len(indices),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "neutral": neutral_metrics(true, pred),
    }


def calculate_metrics(rows, labels, predictions):
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=SENTIMENT_IDS, zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=SENTIMENT_IDS)
    per_class = {
        sentiment: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }
        for index, sentiment in enumerate(SENTIMENTS)
    }
    neutral_index = SENTIMENTS.index("neutral")
    positive_index = SENTIMENTS.index("positive")
    negative_index = SENTIMENTS.index("negative")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "per_class": per_class,
        "predicted_counts": {
            sentiment: int(predictions.count(LABEL2ID[sentiment]))
            for sentiment in SENTIMENTS
        },
        "confusion_matrix": matrix.tolist(),
        "neutral_confusion": {
            "true_neutral_predicted_neutral": int(matrix[neutral_index, neutral_index]),
            "positive_to_neutral": int(matrix[positive_index, neutral_index]),
            "negative_to_neutral": int(matrix[negative_index, neutral_index]),
            "false_neutral_total": int(
                matrix[positive_index, neutral_index]
                + matrix[negative_index, neutral_index]
            ),
        },
        "languages": {
            name: subset_metrics(rows, labels, predictions, "language", value)
            for value, name in LANGUAGES
        },
        "aspects": {
            aspect: subset_metrics(rows, labels, predictions, "aspect", aspect)
            for aspect in ASPECTS
        },
    }


def evaluate_transformer(test_rows):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, CHECKPOINT, is_trainable=False)
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Final evaluation model unexpectedly has trainable parameters")

    batch_size = 1 if device.type == "mps" else 8
    loader = DataLoader(
        TestDataset(test_rows, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
        num_workers=0,
    )
    labels = []
    predictions = []
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_labels = batch.pop("labels")
            logits = model(**batch).logits
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            if batch_number % 25 == 0:
                print(
                    f"Transformer inference progress: "
                    f"{min(batch_number * batch_size, len(test_rows))}/{len(test_rows)}",
                    flush=True,
                )
    return calculate_metrics(test_rows, labels, predictions), device.type


def evaluate_classical(clean_train_rows, test_rows):
    model = build_model()
    model.fit(
        build_inputs(clean_train_rows),
        [row["sentiment"] for row in clean_train_rows],
    )
    predicted_names = model.predict(build_inputs(test_rows)).tolist()
    labels = [LABEL2ID[row["sentiment"]] for row in test_rows]
    predictions = [LABEL2ID[label] for label in predicted_names]
    return calculate_metrics(test_rows, labels, predictions)


def format_report(
    checkpoint_payload,
    transformer,
    classical,
    language_counts,
    sentiment_counts,
    aspect_counts,
    audit,
    device,
    adapter_hash_before,
    adapter_hash_after,
):
    lines = [
        "FINAL MODEL TEST EVALUATION — EXPERIMENT 15 EPOCH 2",
        "",
        "FINAL SELECTED MODEL",
        "Experiment 15 Epoch 2",
        f"Checkpoint: {CHECKPOINT.relative_to(PROJECT_ROOT)}",
        "Base model: xlm-roberta-base",
        "Training scope: TR + EN + ES, Expanded Neutral v1",
        "Adapter: LoRA query/value, r=8, alpha=16, dropout=0.1",
        "Evaluation only: torch.inference_mode(); no optimizer; no parameter updates",
        f"Evaluation device: {device}",
        f"Checkpoint unchanged: {adapter_hash_before == adapter_hash_after}",
        "Test used for tuning or model selection: NO",
        "",
        "TEST DATA",
        f"Rows: {sum(language_counts.values())}",
        f"Languages: {dict(language_counts)}",
        f"Sentiments: {dict(sentiment_counts)}",
        f"Aspects: {dict(aspect_counts)}",
        f"Test SHA256: {file_sha256(TEST_PATH)}",
        "",
        "AUDIT",
        f"Train duplicates: {audit['train_duplicates']}",
        f"Dev duplicates: {audit['dev_duplicates']}",
        f"Test duplicates: {audit['test_duplicates']}",
        f"Train conflicts: {audit['train_conflicts']}",
        f"Dev conflicts: {audit['dev_conflicts']}",
        f"Test conflicts: {audit['test_conflicts']}",
        f"Train-test exact leakage: {audit['train_test_leakage']}",
        f"Dev-test exact leakage: {audit['dev_test_leakage']}",
        "",
        "TRANSFORMER TEST — OVERALL",
        f"Accuracy: {transformer['accuracy']:.4f}",
        f"Macro F1: {transformer['macro_f1']:.4f}",
        "",
        "SENTIMENT",
    ]
    for sentiment in SENTIMENTS:
        item = transformer["per_class"][sentiment]
        lines.append(
            f"{sentiment}: P={item['precision']:.4f} | R={item['recall']:.4f} | "
            f"F1={item['f1']:.4f} | support={item['support']}"
        )
    lines.extend(
        [
            "",
            "PREDICTED COUNTS",
            *[
                f"{sentiment}: {transformer['predicted_counts'][sentiment]}"
                for sentiment in SENTIMENTS
            ],
            "",
            "CONFUSION MATRIX",
            "Rows=actual, columns=predicted; order=positive, neutral, negative",
            "                 positive  neutral  negative",
        ]
    )
    for sentiment, row in zip(SENTIMENTS, transformer["confusion_matrix"]):
        lines.append(f"{sentiment:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}")
    neutral = transformer["neutral_confusion"]
    lines.extend(
        [
            "",
            "NEUTRAL ANALYSIS",
            f"Precision: {transformer['per_class']['neutral']['precision']:.4f}",
            f"Recall: {transformer['per_class']['neutral']['recall']:.4f}",
            f"F1: {transformer['per_class']['neutral']['f1']:.4f}",
            f"True neutral -> neutral: {neutral['true_neutral_predicted_neutral']}",
            f"Positive -> neutral: {neutral['positive_to_neutral']}",
            f"Negative -> neutral: {neutral['negative_to_neutral']}",
            f"False-neutral total: {neutral['false_neutral_total']}",
            "",
            "LANGUAGE BREAKDOWN",
        ]
    )
    for _, name in LANGUAGES:
        item = transformer["languages"][name]
        n = item["neutral"]
        lines.append(
            f"{name}: n={item['n']} | Accuracy={item['accuracy']:.4f} | "
            f"Macro F1={item['macro_f1']:.4f} | Neutral P/R/F1="
            f"{n['precision']:.4f}/{n['recall']:.4f}/{n['f1']:.4f}"
        )
    lines.extend(["", "ASPECT BREAKDOWN"])
    for aspect in ASPECTS:
        item = transformer["aspects"][aspect]
        lines.append(
            f"{aspect}: n={item['n']} | Macro F1={item['macro_f1']:.4f} | "
            f"Neutral F1={item['neutral']['f1']:.4f}"
        )

    lines.extend(
        [
            "",
            "TEST MINUS DEV",
            f"Accuracy delta: {transformer['accuracy'] - DEV_REFERENCE['accuracy']:+.4f}",
            f"Macro F1 delta: {transformer['macro_f1'] - DEV_REFERENCE['macro_f1']:+.4f}",
        ]
    )
    for sentiment in SENTIMENTS:
        lines.append(
            f"{sentiment} F1 delta: "
            f"{transformer['per_class'][sentiment]['f1'] - DEV_REFERENCE[f'{sentiment}_f1']:+.4f}"
        )

    lines.extend(
        [
            "",
            "HISTORICAL CLASSICAL BASELINE TEST",
            "Artifact status: serialized model not found; refit on data/cleaned/train.csv",
            "Configuration: inline aspect; TF-IDF ngram_range=(1,2), min_df=2; "
            "LogisticRegression class_weight=balanced, max_iter=1000, random_state=42",
            f"Accuracy: {classical['accuracy']:.4f}",
            f"Macro F1: {classical['macro_f1']:.4f}",
            f"Neutral F1: {classical['per_class']['neutral']['f1']:.4f}",
            "",
            "CLASSICAL VS TRANSFORMER",
            f"Classical TEST Macro F1: {classical['macro_f1']:.4f}",
            f"Transformer TEST Macro F1: {transformer['macro_f1']:.4f}",
            f"Macro F1 delta (Transformer - Classical): "
            f"{transformer['macro_f1'] - classical['macro_f1']:+.4f}",
            f"Classical Neutral F1: {classical['per_class']['neutral']['f1']:.4f}",
            f"Transformer Neutral F1: {transformer['per_class']['neutral']['f1']:.4f}",
            f"Neutral F1 delta (Transformer - Classical): "
            f"{transformer['per_class']['neutral']['f1'] - classical['per_class']['neutral']['f1']:+.4f}",
            "",
            "FINAL SUMMARY",
            "FINAL SELECTED MODEL: Experiment 15 Epoch 2",
            f"DEV Macro F1: {DEV_REFERENCE['macro_f1']:.4f}",
            f"TEST Macro F1: {transformer['macro_f1']:.4f}",
            f"DEV Neutral F1: {DEV_REFERENCE['neutral_f1']:.4f}",
            f"TEST Neutral F1: {transformer['per_class']['neutral']['f1']:.4f}",
            f"TR TEST Macro F1: {transformer['languages']['TR']['macro_f1']:.4f}",
            f"EN TEST Macro F1: {transformer['languages']['EN']['macro_f1']:.4f}",
            f"ES TEST Macro F1: {transformer['languages']['ES']['macro_f1']:.4f}",
            "",
            "LIMITATIONS",
            "- Neutral remains the hardest class and has much lower support than positive/negative.",
            "- Atmosphere/value neutral performance may be weaker than other aspects.",
            "- Multilingual performance varies by language.",
            "- Original M-ABSA lacks a reliable language-independent source ID; zero original cross-language semantic leakage cannot be proven.",
            "- Test was evaluated once for final unbiased estimation and was not used for tuning or model selection.",
            "- No post-test hyperparameter, threshold, dataset, loss, or checkpoint change was made.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    if REPORT_PATH.exists():
        raise RuntimeError(
            "Final test report already exists; refusing to repeat the one-time evaluation."
        )
    checkpoint_payload = validate_checkpoint()
    rows, language, sentiment, aspect, audit = validate_data()
    adapter_path = CHECKPOINT / "adapter_model.safetensors"
    adapter_hash_before = file_sha256(adapter_path)
    print("Checkpoint and test audit: PASS", flush=True)
    print("Starting final one-time transformer TEST inference", flush=True)
    transformer, device = evaluate_transformer(rows["test"])
    adapter_hash_after = file_sha256(adapter_path)
    if adapter_hash_before != adapter_hash_after:
        raise RuntimeError("Checkpoint changed during evaluation")
    print("Transformer inference complete; refitting frozen historical classical baseline", flush=True)
    classical = evaluate_classical(load_rows(CLEAN_TRAIN_PATH, "train"), rows["test"])
    report = format_report(
        checkpoint_payload,
        transformer,
        classical,
        language,
        sentiment,
        aspect,
        audit,
        device,
        adapter_hash_before,
        adapter_hash_after,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
