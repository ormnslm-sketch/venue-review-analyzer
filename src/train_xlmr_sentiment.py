"""XLM-R base modelini cleaned sentiment datasetinde fine-tune et."""

import csv
import gc
import os
import random
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    __version__ as transformers_version,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "cleaned" / "train.csv"
DEV_PATH = PROJECT_ROOT / "data" / "cleaned" / "dev.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "experiment_03_xlmr_rerun.txt"
BEST_MODEL_DIR = PROJECT_ROOT / "models" / "xlmr-sentiment-v1-best"

MODEL_NAME = "xlm-roberta-base"
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
RANDOM_SEED = 42
LOG_EVERY_STEPS = 100

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
ID2LABEL = {label_id: label for label, label_id in LABEL2ID.items()}
SENTIMENTS = ("positive", "neutral", "negative")
LABEL_IDS = [LABEL2ID[sentiment] for sentiment in SENTIMENTS]
LANGUAGES = (("tr", "Turkish"), ("en", "English"), ("es", "Spanish"))
ASPECTS = ("taste", "atmosphere", "service", "value")

BASELINE_METRICS = {
    "accuracy": 0.7603,
    "macro_f1": 0.5891,
    "positive_f1": 0.8453,
    "neutral_f1": 0.2529,
    "negative_f1": 0.6690,
    "taste_macro_f1": 0.5902,
    "atmosphere_macro_f1": 0.3210,
    "service_macro_f1": 0.5251,
    "value_macro_f1": 0.6988,
}

REQUIRED_COLUMNS = {"sentence", "language", "split", "aspect", "sentiment"}


def set_seed(seed):
    """Python, NumPy ve PyTorch random seedlerini sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device():
    """Apple Silicon MPS varsa kullan; yoksa CPU'ya dön."""
    if os.environ.get("XLMR_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rows(path, expected_split):
    """Cleaned CSV splitini oku ve gerekli alanları doğrula."""
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"{path.name} eksik kolonlar içeriyor: {sorted(missing)}")
        rows = list(reader)

    if any(row["split"] != expected_split for row in rows):
        raise ValueError(f"{path.name} beklenmeyen split değeri içeriyor.")
    if any(not row["sentence"].strip() for row in rows):
        raise ValueError(f"{path.name} boş sentence içeriyor.")
    if any(row["sentiment"] not in LABEL2ID for row in rows):
        raise ValueError(f"{path.name} beklenmeyen sentiment içeriyor.")
    return rows


def model_input(row):
    """Baseline ile aynı aspect + sentence model girdisini oluştur."""
    return f"aspect={row['aspect']} [SEP] {row['sentence']}"


class SentimentDataset(Dataset):
    """CSV satırlarını tokenizer girdilerine dönüştüren sade Dataset."""

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
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = LABEL2ID[row["sentiment"]]
        return encoded


def move_batch(batch, device):
    """Bir batch içindeki tensorları seçilen cihaza taşı."""
    return {name: value.to(device) for name, value in batch.items()}


def predict(model, data_loader, device):
    """Bir dataset için label ve tahmin dizilerini üret."""
    model.eval()
    labels = []
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            predicted_ids = outputs.logits.argmax(dim=-1)
            labels.extend(batch["labels"].cpu().tolist())
            predictions.extend(predicted_ids.cpu().tolist())

    return labels, predictions


def overall_metrics(labels, predictions):
    """Epoch ve final değerlendirmede kullanılan temel metrikleri hesapla."""
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
    }


def append_log(text):
    """Eğitim ilerlemesini rapor dosyasına ve terminale yaz."""
    print(text, flush=True)
    with REPORT_PATH.open("a", encoding="utf-8") as report_file:
        report_file.write(text + "\n")


def train(model, train_loader, dev_loader, optimizer, tokenizer, device):
    """Modeli üç epoch eğit ve en iyi dev Macro F1 checkpoint'ini kaydet."""
    epoch_results = []
    best_macro_f1 = float("-inf")
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

            if step % LOG_EVERY_STEPS == 0 or step == len(train_loader):
                append_log(
                    f"Epoch {epoch}/{EPOCHS} | step {step}/{len(train_loader)} | "
                    f"running_train_loss={total_loss / step:.4f}"
                )

        labels, predictions = predict(model, dev_loader, device)
        metrics = overall_metrics(labels, predictions)
        metrics["train_loss"] = total_loss / len(train_loader)
        epoch_results.append(metrics)

        append_log(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={metrics['train_loss']:.4f} | "
            f"dev_accuracy={metrics['accuracy']:.4f} | "
            f"dev_precision_macro={metrics['precision_macro']:.4f} | "
            f"dev_recall_macro={metrics['recall_macro']:.4f} | "
            f"dev_macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
            append_log(
                f"New best checkpoint | epoch={best_epoch} | "
                f"dev_macro_f1={best_macro_f1:.4f} | path={BEST_MODEL_DIR}"
            )

    return epoch_results, best_epoch, best_macro_f1


def group_metrics(rows, label_ids, prediction_ids, column, groups):
    """Language veya aspect grupları için accuracy ve macro F1 hesapla."""
    results = {}
    for value, display_name in groups:
        indices = [index for index, row in enumerate(rows) if row[column] == value]
        group_labels = [label_ids[index] for index in indices]
        group_predictions = [prediction_ids[index] for index in indices]
        results[display_name] = {
            "n": len(indices),
            "accuracy": accuracy_score(group_labels, group_predictions),
            "macro_f1": f1_score(
                group_labels,
                group_predictions,
                average="macro",
                zero_division=0,
            ),
        }
    return results


def comparison_line(name, baseline, xlmr):
    """Baseline ve XLM-R sonuçlarını farkıyla biçimlendir."""
    return (
        f"{name}: baseline={baseline:.4f} | XLM-R={xlmr:.4f} | "
        f"difference={xlmr - baseline:+.4f}"
    )


def create_training_objects(train_rows, dev_rows, tokenizer, device):
    """Seçilen cihaz için model, loader ve optimizer nesnelerini oluştur."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    if device.type == "mps":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        SentimentDataset(train_rows, tokenizer),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
    )
    dev_loader = DataLoader(
        SentimentDataset(dev_rows, tokenizer),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    return model, train_loader, dev_loader, optimizer


def is_mps_out_of_memory(error):
    """RuntimeError mesajının MPS bellek hatası olup olmadığını kontrol et."""
    return "MPS backend out of memory" in str(error)


def final_report(
    dev_rows,
    label_ids,
    prediction_ids,
    epoch_results,
    best_epoch,
    best_macro_f1,
    device,
):
    """Final dev sonuçlarını ayrıntılı ve okunabilir biçimde hazırla."""
    overall = overall_metrics(label_ids, prediction_ids)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        label_ids,
        prediction_ids,
        labels=LABEL_IDS,
        zero_division=0,
    )
    class_metrics = {
        sentiment: {
            "precision": precision[index],
            "recall": recall[index],
            "f1": class_f1[index],
            "support": int(support[index]),
        }
        for index, sentiment in enumerate(SENTIMENTS)
    }
    matrix = confusion_matrix(label_ids, prediction_ids, labels=LABEL_IDS)
    language_metrics = group_metrics(
        dev_rows, label_ids, prediction_ids, "language", LANGUAGES
    )
    aspect_metrics = group_metrics(
        dev_rows,
        label_ids,
        prediction_ids,
        "aspect",
        [(aspect, aspect) for aspect in ASPECTS],
    )
    neutral_prediction_count = prediction_ids.count(LABEL2ID["neutral"])

    lines = [
        "",
        "BEST CHECKPOINT DEV EVALUATION",
        f"Device: {device.type}",
        f"Completed epochs: {len(epoch_results)}",
        f"Best epoch: {best_epoch}",
        f"Tracked best Macro F1: {best_macro_f1:.4f}",
        "",
        "OVERALL",
        f"Accuracy: {overall['accuracy']:.4f}",
        f"Macro F1: {overall['macro_f1']:.4f}",
        f"Macro precision: {overall['precision_macro']:.4f}",
        f"Macro recall: {overall['recall_macro']:.4f}",
        "",
        "SENTIMENT METRICS",
    ]

    for sentiment in SENTIMENTS:
        metrics = class_metrics[sentiment]
        lines.extend(
            [
                f"{sentiment}:",
                f"  precision: {metrics['precision']:.4f}",
                f"  recall: {metrics['recall']:.4f}",
                f"  F1: {metrics['f1']:.4f}",
                f"  support: {metrics['support']}",
            ]
        )

    lines.extend(
        [
            "",
            "CONFUSION MATRIX",
            "Rows=actual, columns=predicted",
            "                 positive  neutral  negative",
        ]
    )
    for sentiment, row in zip(SENTIMENTS, matrix):
        lines.append(f"{sentiment:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}")

    lines.extend(["", "LANGUAGE METRICS"])
    for language, metrics in language_metrics.items():
        lines.append(
            f"{language}: n={metrics['n']} | accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f}"
        )

    lines.extend(["", "ASPECT METRICS"])
    for aspect, metrics in aspect_metrics.items():
        lines.append(
            f"{aspect}: n={metrics['n']} | accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f}"
        )

    lines.extend(
        [
            "",
            "TF-IDF + LOGISTIC REGRESSION VS XLM-R",
            comparison_line("Accuracy", BASELINE_METRICS["accuracy"], overall["accuracy"]),
            comparison_line("Macro F1", BASELINE_METRICS["macro_f1"], overall["macro_f1"]),
        ]
    )
    for sentiment in SENTIMENTS:
        lines.append(
            comparison_line(
                f"{sentiment} F1",
                BASELINE_METRICS[f"{sentiment}_f1"],
                class_metrics[sentiment]["f1"],
            )
        )
    for aspect in ASPECTS:
        lines.append(
            comparison_line(
                f"{aspect} macro F1",
                BASELINE_METRICS[f"{aspect}_macro_f1"],
                aspect_metrics[aspect]["macro_f1"],
            )
        )

    lines.extend(
        [
            "",
            "KEY CHECKS",
            f"Best epoch: {best_epoch}",
            f"Neutral prediction count: {neutral_prediction_count}",
            f"Neutral F1: {class_metrics['neutral']['f1']:.4f}",
            f"Atmosphere macro F1: {aspect_metrics['atmosphere']['macro_f1']:.4f}",
            f"Value macro F1: {aspect_metrics['value']['macro_f1']:.4f}",
        ]
    )

    return "\n".join(lines) + "\n"


def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "EXPERIMENT 03 RERUN — XLM-ROBERTA-BASE SENTIMENT CLASSIFICATION\n\n",
        encoding="utf-8",
    )
    append_log(f"Model: {MODEL_NAME}")
    append_log(f"PyTorch version: {torch.__version__}")
    append_log(f"Transformers version: {transformers_version}")
    append_log(f"Device: {device.type}")
    if os.environ.get("XLMR_FORCE_CPU") == "1":
        append_log(
            "Device note: CPU selected to reproduce the prior successful CPU run "
            "after verified MPS out-of-memory"
        )
    append_log(f"Train rows: {len(train_rows)}")
    append_log(f"Dev rows: {len(dev_rows)}")
    append_log("Test split: not loaded or evaluated")
    append_log(f"Label mapping: {LABEL2ID}")
    append_log(
        f"Hyperparameters: epochs={EPOCHS}, learning_rate={LEARNING_RATE}, "
        f"max_length={MAX_LENGTH}, train_batch_size={TRAIN_BATCH_SIZE}, "
        f"eval_batch_size={EVAL_BATCH_SIZE}, random_seed={RANDOM_SEED}"
    )
    append_log(
        "Train sentiment distribution: "
        f"{dict(Counter(row['sentiment'] for row in train_rows))}"
    )
    append_log("Class weighting / resampling: not used")
    append_log("Batch size note: 4 selected after batch size 8 exceeded MPS memory")
    append_log(
        "Gradient checkpointing: "
        f"{'enabled' if device.type == 'mps' else 'not needed on CPU'}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    while True:
        set_seed(RANDOM_SEED)
        model, train_loader, dev_loader, optimizer = create_training_objects(
            train_rows,
            dev_rows,
            tokenizer,
            device,
        )
        try:
            epoch_results, best_epoch, best_macro_f1 = train(
                model,
                train_loader,
                dev_loader,
                optimizer,
                tokenizer,
                device,
            )
            break
        except RuntimeError as error:
            if device.type != "mps" or not is_mps_out_of_memory(error):
                raise

            append_log(
                "MPS out-of-memory detected; restarting the full experiment on CPU"
            )
            del optimizer, train_loader, dev_loader, model
            torch.mps.empty_cache()
            gc.collect()
            device = torch.device("cpu")
            append_log("Device: cpu")
            append_log("Gradient checkpointing: not needed on CPU")

    append_log(
        f"Best epoch: {best_epoch} | best dev Macro F1: {best_macro_f1:.4f}"
    )
    del optimizer, model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    best_model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
    best_model.to(device)
    label_ids, prediction_ids = predict(best_model, dev_loader, device)

    report = final_report(
        dev_rows,
        label_ids,
        prediction_ids,
        epoch_results,
        best_epoch,
        best_macro_f1,
        device,
    )
    append_log(report.rstrip())
    append_log(f"Best model and tokenizer saved to: {BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()
