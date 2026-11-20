"""Experiment 13: cleaned vs expanded data with identical unweighted LoRA.

Training is MPS-only, chunked, resumable, and limited to one target epoch.
The test split is never referenced or loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import signal
from collections import Counter
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "clean": {
        "experiment": "13A",
        "train": PROJECT_ROOT / "data" / "cleaned" / "train.csv",
        "dev": PROJECT_ROOT / "data" / "cleaned" / "dev.csv",
        "model_root": PROJECT_ROOT / "models" / "experiment_13a_lora_clean_unweighted",
        "report": PROJECT_ROOT
        / "data"
        / "analysis"
        / "experiment_13a_lora_clean_unweighted.txt",
    },
    "expanded": {
        "experiment": "13B",
        "train": PROJECT_ROOT / "data" / "expanded_neutral_v1" / "train.csv",
        "dev": PROJECT_ROOT / "data" / "expanded_neutral_v1" / "dev.csv",
        "model_root": PROJECT_ROOT
        / "models"
        / "experiment_13b_lora_expanded_unweighted",
        "report": PROJECT_ROOT
        / "data"
        / "analysis"
        / "experiment_13b_lora_expanded_unweighted.txt",
    },
}
COMPARISON_REPORT = (
    PROJECT_ROOT / "data" / "analysis" / "experiment_13_lora_data_ablation_comparison.txt"
)

MODEL_NAME = "xlm-roberta-base"
TOTAL_TARGET_EPOCHS = 1
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_BATCH_SIZE = 1
RANDOM_SEED = 42
MAX_OPTIMIZER_STEPS_PER_RUN = 50

LORA_TARGET_MODULES = ["query", "value"]
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_BIAS = "none"
MODULES_TO_SAVE = ["classifier"]

# These hooks keep the proven chunk/resume implementation reusable without
# changing Experiment 13's original unweighted behavior. A controlled wrapper
# may set fixed class weights and a different report label before calling the
# functions below.
EXPERIMENT_SUBTITLE = "LORA UNWEIGHTED EPOCH 1"
LOSS_DESCRIPTION = "F.cross_entropy(logits, labels, reduction='mean')"
FIXED_CLASS_WEIGHTS_BY_ID = None
SAMPLE_WEIGHT_DESCRIPTION = None
ROW_FILTER = None

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {label_id: label for label, label_id in LABEL2ID.items()}
SENTIMENTS = ("positive", "neutral", "negative")
LABEL_IDS = [LABEL2ID[label] for label in SENTIMENTS]
LANGUAGES = (("tr", "TR"), ("en", "EN"), ("es", "ES"))
ASPECTS = ("taste", "atmosphere", "service", "value")
REQUIRED_COLUMNS = {"sentence", "language", "split", "aspect", "sentiment"}
MPS_UNAVAILABLE_MESSAGE = (
    "MPS kullanılamıyor. CPU üzerinde transformer training güvenlik nedeniyle "
    "başlatılmadı. Komutu MPS destekli normal local terminalde çalıştırın."
)


def condition_paths(condition):
    config = CONDITIONS[condition]
    model_root = config["model_root"]
    return {
        **config,
        "resume_dir": model_root / "resume",
        "resume_pointer": model_root / "resume" / "current.json",
        "best_dir": model_root / "best",
        "best_pointer": model_root / "best" / "current.json",
    }


def append_log(condition, text):
    print(text, flush=True)
    report_path = condition_paths(condition)["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as file:
        file.write(text + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def require_mps(condition):
    if not torch.backends.mps.is_available():
        append_log(condition, MPS_UNAVAILABLE_MESSAGE)
        return None
    append_log(condition, "Training device: mps")
    return torch.device("mps")


def load_rows(path, expected_split):
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        rows = list(reader)
    if any(row["split"] != expected_split for row in rows):
        raise ValueError(f"{path} contains unexpected split values")
    if any(not row["sentence"].strip() for row in rows):
        raise ValueError(f"{path} contains empty sentences")
    if any(row["sentiment"] not in LABEL2ID for row in rows):
        raise ValueError(f"{path} contains unexpected labels")
    return rows


def load_condition_rows(condition, split):
    """Load a condition split and apply an optional experiment-scoped filter."""
    paths = condition_paths(condition)
    rows = load_rows(paths[split], split)
    if ROW_FILTER is not None:
        rows = ROW_FILTER(rows, split)
    return rows


def model_input(row):
    return f"aspect={row['aspect']} [SEP] {row['sentence']}"


class SentimentDataset(Dataset):
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


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_lora_config():
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
        modules_to_save=MODULES_TO_SAVE,
    )


def experiment_config(condition):
    paths = condition_paths(condition)
    return {
        "experiment": paths["experiment"],
        "condition": condition,
        "model_name": MODEL_NAME,
        "train_sha256": file_sha256(paths["train"]),
        "dev_sha256": file_sha256(paths["dev"]),
        "input_format": "aspect=<aspect> [SEP] <sentence>",
        "label2id": LABEL2ID,
        "total_target_epochs": TOTAL_TARGET_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "optimizer": "torch.optim.AdamW",
        "scheduler": None,
        "loss": LOSS_DESCRIPTION,
        "class_weight": FIXED_CLASS_WEIGHTS_BY_ID,
        "sample_weight": SAMPLE_WEIGHT_DESCRIPTION,
        "random_seed": RANDOM_SEED,
        "max_optimizer_steps_per_run": MAX_OPTIMIZER_STEPS_PER_RUN,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": sorted(LORA_TARGET_MODULES),
        "lora_bias": LORA_BIAS,
        "modules_to_save": MODULES_TO_SAVE,
    }


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def verify_target_modules(model):
    layer_count = model.config.num_hidden_layers
    linear_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and ".attention." in name
    ]
    selected = {
        target: [
            name
            for name in linear_names
            if name.endswith(f".attention.self.{target}")
        ]
        for target in LORA_TARGET_MODULES
    }
    for target, names in selected.items():
        if len(names) != layer_count:
            raise RuntimeError(
                f"Expected {layer_count} '{target}' attention targets, found {len(names)}"
            )
    return selected


def parameter_summary(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    lora = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )
    classifier = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "classifier.modules_to_save.default" in name
    )
    unexpected = [
        name
        for name in trainable_names
        if "lora_" not in name and "classifier.modules_to_save.default" not in name
    ]
    if unexpected or trainable != lora + classifier or classifier == 0:
        raise RuntimeError(
            "Unexpected trainable parameter structure: "
            f"unexpected={unexpected[:10]}, total={trainable}, lora={lora}, "
            f"classifier={classifier}"
        )
    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "classifier": classifier,
        "percentage": 100 * trainable / total,
    }


def build_meta_model_for_validation():
    config = AutoConfig.from_pretrained(MODEL_NAME, local_files_only=True)
    config.num_labels = len(LABEL2ID)
    config.label2id = LABEL2ID
    config.id2label = ID2LABEL
    with torch.device("meta"):
        base = AutoModelForSequenceClassification.from_config(config)
        selected = verify_target_modules(base)
        model = get_peft_model(base, create_lora_config())
    return selected, parameter_summary(model)


def load_base_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def compute_training_loss(logits, labels):
    """Return the configured per-microbatch loss.

    Experiment 13 leaves FIXED_CLASS_WEIGHTS_BY_ID as None, preserving the
    historical unweighted mean. Controlled weighted wrappers use explicit
    per-example multiplication, avoiding PyTorch weighted-mean cancellation
    when physical batch size is one.
    """
    raw_loss = F.cross_entropy(logits, labels, reduction="none")
    if FIXED_CLASS_WEIGHTS_BY_ID is None:
        return raw_loss.mean()
    class_weights = torch.tensor(
        FIXED_CLASS_WEIGHTS_BY_ID,
        dtype=raw_loss.dtype,
        device=raw_loss.device,
    )
    sample_weights = class_weights[labels]
    return (raw_loss * sample_weights).mean()


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def capture_rng_state(device):
    mps_state = None
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        mps_state = torch.mps.get_rng_state()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "mps": mps_state,
    }


def restore_rng_state(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if (
        device.type == "mps"
        and state.get("mps") is not None
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["mps"])


def epoch_permutation(example_count, epoch_number=1):
    epoch_seed = RANDOM_SEED + epoch_number - 1
    generator = torch.Generator().manual_seed(epoch_seed)
    return epoch_seed, torch.randperm(example_count, generator=generator).tolist()


def permutation_sha256(permutation):
    return hashlib.sha256(np.asarray(permutation, dtype=np.int64).tobytes()).hexdigest()


def micro_batches_per_epoch(example_count):
    return math.ceil(example_count / TRAIN_BATCH_SIZE)


def optimizer_steps_per_epoch(example_count):
    return math.ceil(micro_batches_per_epoch(example_count) / GRADIENT_ACCUMULATION_STEPS)


def read_resume_metadata(condition, signature):
    paths = condition_paths(condition)
    pointer_path = paths["resume_pointer"]
    if not pointer_path.exists():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    slot_name = pointer.get("active_slot")
    if slot_name not in {"slot_a", "slot_b"}:
        raise RuntimeError("Invalid resume pointer slot")
    slot_dir = paths["resume_dir"] / slot_name
    required = (
        slot_dir / "adapter_config.json",
        slot_dir / "adapter_model.safetensors",
        slot_dir / "training_state.pt",
        slot_dir / "metadata.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Resume slot is incomplete: {missing}")
    metadata = json.loads((slot_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("config_signature") != signature:
        raise RuntimeError(
            f"Checkpoint signature does not match Experiment {paths['experiment']}"
        )
    if metadata.get("condition") != condition:
        raise RuntimeError("Checkpoint condition mixing was detected")
    if pointer.get("checkpoint_id") != metadata.get("checkpoint_id"):
        raise RuntimeError("Resume pointer and metadata checkpoint IDs differ")
    metadata["_slot_dir"] = str(slot_dir)
    metadata["_slot_name"] = slot_name
    return metadata


def validate_progress(state, example_count):
    completed = state["completed_epoch"]
    if completed == 1:
        if state["current_epoch"] is not None:
            raise RuntimeError("Completed experiment has a non-null current_epoch")
        return
    if completed != 0 or state["current_epoch"] != 1:
        raise RuntimeError("Experiment 13 supports only epoch 1 progress")
    permutation = state["epoch_permutation"]
    if len(permutation) != example_count or sorted(permutation) != list(range(example_count)):
        raise RuntimeError("Stored epoch permutation is invalid")
    expected_seed, expected = epoch_permutation(example_count, 1)
    if state["epoch_seed"] != expected_seed or permutation != expected:
        raise RuntimeError("Stored epoch permutation is not deterministic")
    if state["permutation_sha256"] != permutation_sha256(permutation):
        raise RuntimeError("Stored permutation hash mismatch")
    if state["micro_batches_accumulated"] != 0:
        raise RuntimeError("Checkpoint is not at an accumulation boundary")
    position = state["microbatch_position"]
    if state["sample_position"] != min(position * TRAIN_BATCH_SIZE, example_count):
        raise RuntimeError("Sample and microbatch positions differ")
    expected_steps = math.ceil(position / GRADIENT_ACCUMULATION_STEPS) if position else 0
    if state["optimizer_steps_in_epoch"] != expected_steps:
        raise RuntimeError("Optimizer step and microbatch positions differ")


def create_or_resume_model(metadata):
    base = load_base_model()
    verify_target_modules(base)
    if metadata is None:
        model = get_peft_model(base, create_lora_config())
    else:
        model = PeftModel.from_pretrained(
            base, Path(metadata["_slot_dir"]), is_trainable=True
        )
    parameter_summary(model)
    return model


def load_training_state(metadata, optimizer, device, signature):
    checkpoint = torch.load(
        Path(metadata["_slot_dir"]) / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("config_signature") != signature:
        raise RuntimeError("Training state signature mismatch")
    if checkpoint.get("device_type") != device.type:
        raise RuntimeError("Checkpoint must resume on the same device type")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state(optimizer, device)
    restore_rng_state(checkpoint["rng_state"], device)
    return checkpoint


def save_resume_checkpoint(condition, model, tokenizer, optimizer, device, state, signature):
    paths = condition_paths(condition)
    paths["resume_dir"].mkdir(parents=True, exist_ok=True)
    current_metadata = read_resume_metadata(condition, signature)
    current_slot = current_metadata.get("_slot_name") if current_metadata else None
    next_slot = "slot_b" if current_slot == "slot_a" else "slot_a"
    final_dir = paths["resume_dir"] / next_slot
    temporary_dir = paths["resume_dir"] / f".{next_slot}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    checkpoint_epoch = state.get("current_epoch") or state.get("completed_epoch") or 1
    checkpoint_id = (
        f"{paths['experiment']}-epoch{checkpoint_epoch}-"
        f"step{state['optimizer_steps_in_epoch']}-sample{state['sample_position']}"
    )
    model.save_pretrained(temporary_dir)
    tokenizer.save_pretrained(temporary_dir)
    training_state = {
        **state,
        "checkpoint_id": checkpoint_id,
        "config_signature": signature,
        "condition": condition,
        "device_type": device.type,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "rng_state": capture_rng_state(device),
    }
    torch.save(training_state, temporary_dir / "training_state.pt")
    metadata = {
        key: value
        for key, value in training_state.items()
        if key
        not in {
            "optimizer_state_dict",
            "scheduler_state_dict",
            "rng_state",
            "epoch_permutation",
        }
    }
    metadata.update(
        {
            "epoch_permutation_length": len(state["epoch_permutation"]),
            "permutation_sha256": state["permutation_sha256"],
        }
    )
    write_json_atomic(temporary_dir / "metadata.json", metadata)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(temporary_dir, final_dir)
    write_json_atomic(
        paths["resume_pointer"],
        {"active_slot": next_slot, "checkpoint_id": checkpoint_id},
    )
    return final_dir


def save_best_checkpoint(condition, model, tokenizer, metrics, signature):
    paths = condition_paths(condition)
    best_dir = paths["best_dir"]
    best_dir.mkdir(parents=True, exist_ok=True)
    final_dir = best_dir / "epoch_1"
    temporary_dir = best_dir / ".epoch_1.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    model.save_pretrained(temporary_dir)
    tokenizer.save_pretrained(temporary_dir)
    checkpoint_id = f"{paths['experiment']}-best-epoch1"
    write_json_atomic(
        temporary_dir / "metrics.json",
        {
            "checkpoint_id": checkpoint_id,
            "config_signature": signature,
            "condition": condition,
            "best_epoch": 1,
            "metrics": metrics,
        },
    )
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(temporary_dir, final_dir)
    write_json_atomic(
        paths["best_pointer"],
        {"relative_path": "epoch_1", "checkpoint_id": checkpoint_id},
    )
    return final_dir


def predict(model, loader, device):
    model.eval()
    labels = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            batch_labels = batch.pop("labels")
            logits = model(**batch).logits
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return labels, predictions


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


def group_metrics(dev_rows, labels, predictions, column, groups):
    output = {}
    for value, display_name in groups:
        indices = [index for index, row in enumerate(dev_rows) if row[column] == value]
        group_labels = [labels[index] for index in indices]
        group_predictions = [predictions[index] for index in indices]
        output[display_name] = {
            "n": len(indices),
            "macro_f1": float(
                f1_score(
                    group_labels,
                    group_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "neutral": neutral_metrics(group_labels, group_predictions),
        }
    return output


def detailed_metrics(dev_rows, labels, predictions, train_loss):
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=LABEL_IDS, zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=LABEL_IDS)
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
        "train_loss": float(train_loss),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "predicted_counts": {
            sentiment: int(predictions.count(LABEL2ID[sentiment]))
            for sentiment in SENTIMENTS
        },
        "neutral_confusion": {
            "true_neutral_predicted_neutral": int(
                matrix[neutral_index, neutral_index]
            ),
            "positive_to_neutral": int(matrix[positive_index, neutral_index]),
            "negative_to_neutral": int(matrix[negative_index, neutral_index]),
            "false_neutral_total": int(
                matrix[positive_index, neutral_index]
                + matrix[negative_index, neutral_index]
            ),
        },
        "languages": group_metrics(dev_rows, labels, predictions, "language", LANGUAGES),
        "aspects": group_metrics(
            dev_rows,
            labels,
            predictions,
            "aspect",
            [(aspect, aspect) for aspect in ASPECTS],
        ),
    }


def format_condition_report(condition, metrics, config_signature, permutation_hash):
    paths = condition_paths(condition)
    lines = [
        f"EXPERIMENT {paths['experiment']} — {EXPERIMENT_SUBTITLE}",
        "",
        f"Condition: {condition}",
        f"Train: {paths['train'].relative_to(PROJECT_ROOT)}",
        f"Dev: {paths['dev'].relative_to(PROJECT_ROOT)}",
        "Test: not loaded or evaluated",
        f"Random seed: {RANDOM_SEED}",
        f"Epoch shuffle seed: {RANDOM_SEED}",
        f"Epoch permutation SHA256: {permutation_hash}",
        f"Config signature: {config_signature}",
        f"Loss: {LOSS_DESCRIPTION}",
        f"Class weights [negative, neutral, positive]: {FIXED_CLASS_WEIGHTS_BY_ID}",
        f"Sample weights: {SAMPLE_WEIGHT_DESCRIPTION}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        f"Epoch train loss: {metrics['train_loss']:.6f}",
        "",
        "SENTIMENT",
    ]
    for sentiment in SENTIMENTS:
        item = metrics["per_class"][sentiment]
        lines.append(
            f"{sentiment}: precision={item['precision']:.4f} | "
            f"recall={item['recall']:.4f} | F1={item['f1']:.4f} | "
            f"support={item['support']}"
        )
    lines.extend(
        [
            "",
            "PREDICTED COUNTS",
            *[
                f"{sentiment}: {metrics['predicted_counts'][sentiment]}"
                for sentiment in SENTIMENTS
            ],
            "",
            "CONFUSION MATRIX",
            "Rows=actual, columns=predicted; order=positive, neutral, negative",
            "                 positive  neutral  negative",
        ]
    )
    for sentiment, row in zip(SENTIMENTS, metrics["confusion_matrix"]):
        lines.append(f"{sentiment:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}")
    neutral_confusion = metrics["neutral_confusion"]
    lines.extend(
        [
            "",
            "NEUTRAL CONFUSION",
            "true neutral -> predicted neutral: "
            f"{neutral_confusion['true_neutral_predicted_neutral']}",
            f"positive -> neutral: {neutral_confusion['positive_to_neutral']}",
            f"negative -> neutral: {neutral_confusion['negative_to_neutral']}",
            f"total false-neutral: {neutral_confusion['false_neutral_total']}",
        ]
    )
    lines.extend(["", "LANGUAGE — Macro F1 and neutral P/R/F1/support"])
    for language, item in metrics["languages"].items():
        neutral = item["neutral"]
        lines.append(
            f"{language}: Macro F1={item['macro_f1']:.4f} | "
            f"Neutral={neutral['precision']:.4f}/{neutral['recall']:.4f}/"
            f"{neutral['f1']:.4f}/{neutral['support']}"
        )
    lines.extend(["", "ASPECT — Macro F1 and neutral F1"])
    for aspect, item in metrics["aspects"].items():
        lines.append(
            f"{aspect}: Macro F1={item['macro_f1']:.4f} | "
            f"Neutral F1={item['neutral']['f1']:.4f}"
        )
    lines.extend(
        [
            "",
            "Historical vanilla full XLM-R cleaned reference (different architecture):",
            "Accuracy=0.8457 | Macro F1=0.5651 | positive F1=0.9134 | "
            "neutral F1=0.0000 | negative F1=0.7820 | neutral predictions=0",
            "This reference is background only, not architecture-controlled.",
            "Epoch 2 was not started automatically.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_dev_loader(dev_rows, tokenizer):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    return DataLoader(
        SentimentDataset(dev_rows, tokenizer),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )


def run_static_validation(condition):
    paths = condition_paths(condition)
    train_rows = load_condition_rows(condition, "train")
    dev_rows = load_condition_rows(condition, "dev")
    clean_dev = load_rows(CONDITIONS["clean"]["dev"], "dev")
    expanded_dev = load_rows(CONDITIONS["expanded"]["dev"], "dev")
    if clean_dev != expanded_dev:
        raise RuntimeError("Cleaned and expanded dev splits are not logically identical")
    config = experiment_config(condition)
    signature = stable_json_hash(config)
    metadata = read_resume_metadata(condition, signature)
    selected, params = build_meta_model_for_validation()
    other_condition = "expanded" if condition == "clean" else "clean"
    if paths["model_root"] == condition_paths(other_condition)["model_root"]:
        raise RuntimeError("Condition model roots are not isolated")
    next_step = metadata["optimizer_steps_in_epoch"] if metadata else 0
    print(f"Experiment: {paths['experiment']}")
    print(f"Condition: {condition}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Dev rows: {len(dev_rows)}")
    print("Dev splits logically identical: True")
    print(f"Model root: {paths['model_root']}")
    print(f"Checkpoint isolation: True ({condition} != {other_condition})")
    print(f"Config signature: {signature}")
    print(f"LoRA targets: {LORA_TARGET_MODULES}")
    print(f"query modules verified: {len(selected['query'])}")
    print(f"value modules verified: {len(selected['value'])}")
    print(f"Trainable params: {params['trainable']:,} ({params['percentage']:.6f}%)")
    print(f"LoRA trainable params: {params['lora']:,}")
    print(f"Classifier trainable params: {params['classifier']:,}")
    print(f"Class weights [negative, neutral, positive]: {FIXED_CLASS_WEIGHTS_BY_ID}")
    print(f"Sample weights: {SAMPLE_WEIGHT_DESCRIPTION}")
    print(f"Loss: {LOSS_DESCRIPTION}")
    print(f"Total target epochs: {TOTAL_TARGET_EPOCHS}")
    print(f"Max optimizer steps per run: {MAX_OPTIMIZER_STEPS_PER_RUN}")
    print(f"Next optimizer step: {next_step}")
    print(f"Resume status: {'checkpoint found' if metadata else 'fresh'}")
    print("Training started: NO")


def run_smoke_test(condition):
    device = require_mps(condition)
    if device is None:
        return
    paths = condition_paths(condition)
    train_rows = load_condition_rows(condition, "train")
    model = None
    optimizer = None
    batch = None
    try:
        set_seed(RANDOM_SEED)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = get_peft_model(load_base_model(), create_lora_config())
        parameter_summary(model)
        model.to(device)
        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        loader = DataLoader(
            SentimentDataset(train_rows[:1], tokenizer),
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        batch = move_batch(next(iter(loader)), device)
        labels = batch.pop("labels")
        optimizer = torch.optim.AdamW(trainable_parameters(model), lr=LEARNING_RATE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(**batch).logits
        training_loss = compute_training_loss(logits, labels)
        (training_loss / GRADIENT_ACCUMULATION_STEPS).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        append_log(condition, f"Smoke-test configured loss: {training_loss.item():.6f}")
        append_log(condition, "SMOKE TEST: PASSED")
        append_log(
            condition,
            "Smoke-test state discarded; no resume/best checkpoint was created.",
        )
    except RuntimeError as error:
        if "mps" in str(error).lower() and "out of memory" in str(error).lower():
            append_log(condition, "SMOKE TEST: FAILED — MPS OOM")
            append_log(condition, "No checkpoint was created; CPU fallback was not used.")
            return
        raise
    finally:
        del batch
        del optimizer
        del model
        if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except RuntimeError:
                pass


def initial_training_state(train_count):
    epoch_seed, permutation = epoch_permutation(train_count, 1)
    return {
        "completed_epoch": 0,
        "current_epoch": 1,
        "optimizer_steps_in_epoch": 0,
        "microbatch_position": 0,
        "sample_position": 0,
        "micro_batches_accumulated": 0,
        "epoch_permutation": permutation,
        "permutation_sha256": permutation_sha256(permutation),
        "epoch_seed": epoch_seed,
        "epoch_running_loss_sum": 0.0,
        "epoch_microbatches_seen": 0,
        "best_macro_f1": None,
        "best_epoch": None,
        "last_epoch_metrics": None,
    }


def run_train_chunk(condition):
    device = require_mps(condition)
    if device is None:
        return
    paths = condition_paths(condition)
    train_rows = load_condition_rows(condition, "train")
    dev_rows = load_condition_rows(condition, "dev")
    config = experiment_config(condition)
    signature = stable_json_hash(config)
    metadata = read_resume_metadata(condition, signature)
    if metadata and metadata["completed_epoch"] >= 1:
        append_log(condition, "Epoch 1 is already complete. No training was started.")
        append_log(condition, "Epoch 2 is outside the current experiment budget.")
        return

    set_seed(RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        Path(metadata["_slot_dir"]) if metadata else MODEL_NAME,
        local_files_only=bool(metadata),
    )
    model = create_or_resume_model(metadata)
    model.to(device)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=LEARNING_RATE)
    if metadata:
        state = load_training_state(metadata, optimizer, device, signature)
    else:
        state = initial_training_state(len(train_rows))
    validate_progress(state, len(train_rows))

    total_microbatches = micro_batches_per_epoch(len(train_rows))
    total_optimizer_steps = optimizer_steps_per_epoch(len(train_rows))
    current_step = state["optimizer_steps_in_epoch"]
    target_step = min(current_step + MAX_OPTIMIZER_STEPS_PER_RUN, total_optimizer_steps)
    remaining_steps = total_optimizer_steps - current_step
    append_log(condition, "")
    append_log(condition, f"Experiment: {paths['experiment']}")
    append_log(condition, "Device: mps")
    append_log(condition, "Epoch: 1/1")
    append_log(condition, f"Starting optimizer step: {current_step}")
    append_log(condition, f"Target optimizer step: {target_step}")
    append_log(condition, f"Remaining optimizer steps in epoch: {remaining_steps}")
    append_log(condition, f"Loss: {LOSS_DESCRIPTION}")
    append_log(
        condition,
        f"Class weights [negative, neutral, positive]: {FIXED_CLASS_WEIGHTS_BY_ID}",
    )
    append_log(condition, f"Sample weights: {SAMPLE_WEIGHT_DESCRIPTION}")
    append_log(condition, "Entering training loop for epoch 1")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataset = SentimentDataset(train_rows, tokenizer)
    remaining_indices = state["epoch_permutation"][state["sample_position"] :]
    loader = DataLoader(
        Subset(dataset, remaining_indices),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    iterator = iter(loader)
    dev_loader = build_dev_loader(dev_rows, tokenizer)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    completed_this_run = 0
    stop_requested = {"value": False}
    boundary_rng = capture_rng_state(device)
    old_sigint = signal.getsignal(signal.SIGINT)

    def request_stop(_signal_number, _frame):
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        while state["optimizer_steps_in_epoch"] < target_step:
            boundary_rng = capture_rng_state(device)
            group_start = state["microbatch_position"]
            group_size = min(
                GRADIENT_ACCUMULATION_STEPS, total_microbatches - group_start
            )
            group_loss = 0.0
            for micro_index in range(group_size):
                if completed_this_run == 0 and micro_index == 0:
                    append_log(condition, "Starting first training batch")
                batch = move_batch(next(iterator), device)
                labels = batch.pop("labels")
                logits = model(**batch).logits
                training_loss = compute_training_loss(logits, labels)
                loss_for_backward = training_loss / GRADIENT_ACCUMULATION_STEPS
                loss_for_backward.backward()
                group_loss += training_loss.item()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state["optimizer_steps_in_epoch"] += 1
            state["microbatch_position"] += group_size
            state["sample_position"] = min(
                state["microbatch_position"] * TRAIN_BATCH_SIZE, len(train_rows)
            )
            state["epoch_running_loss_sum"] += group_loss
            state["epoch_microbatches_seen"] += group_size
            completed_this_run += 1
            if stop_requested["value"]:
                append_log(
                    condition,
                    "Ctrl+C received. Stopping at the completed optimizer-step boundary.",
                )
                break
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        restore_rng_state(boundary_rng, device)
        append_log(
            condition,
            "Partial accumulation group discarded; progress remains at the last completed step.",
        )
    except RuntimeError as error:
        optimizer.zero_grad(set_to_none=True)
        if "mps" in str(error).lower() and "out of memory" in str(error).lower():
            append_log(
                condition,
                "TRAINING CHUNK FAILED — MPS OOM. No new checkpoint was written; "
                "previous checkpoint remains authoritative; CPU fallback was not used.",
            )
            return
        raise
    finally:
        signal.signal(signal.SIGINT, old_sigint)

    epoch_finished = state["microbatch_position"] >= total_microbatches
    if epoch_finished:
        labels, predictions = predict(model, dev_loader, device)
        train_loss = (
            state["epoch_running_loss_sum"] / state["epoch_microbatches_seen"]
        )
        metrics = detailed_metrics(dev_rows, labels, predictions, train_loss)
        state["completed_epoch"] = 1
        state["current_epoch"] = None
        state["best_macro_f1"] = metrics["macro_f1"]
        state["best_epoch"] = 1
        state["last_epoch_metrics"] = metrics
        best_path = save_best_checkpoint(condition, model, tokenizer, metrics, signature)
        paths["report"].parent.mkdir(parents=True, exist_ok=True)
        paths["report"].write_text(
            format_condition_report(
                condition, metrics, signature, state["permutation_sha256"]
            ),
            encoding="utf-8",
        )
    else:
        best_path = None

    checkpoint_path = save_resume_checkpoint(
        condition, model, tokenizer, optimizer, device, state, signature
    )
    progress = 100 * state["optimizer_steps_in_epoch"] / total_optimizer_steps
    append_log(condition, "")
    append_log(condition, "TRAINING CHUNK RESULT")
    append_log(condition, "Epoch: 1/1")
    append_log(condition, f"Optimizer steps completed this run: {completed_this_run}")
    append_log(
        condition, f"Current optimizer step: {state['optimizer_steps_in_epoch']}"
    )
    append_log(condition, f"Epoch progress percentage: {progress:.2f}%")
    append_log(condition, f"Resume checkpoint saved: {checkpoint_path}")
    append_log(condition, f"Epoch finished: {epoch_finished}")
    if epoch_finished:
        metrics = state["last_epoch_metrics"]
        append_log(condition, f"Dev Accuracy: {metrics['accuracy']:.4f}")
        append_log(condition, f"Dev Macro F1: {metrics['macro_f1']:.4f}")
        append_log(condition, "Best epoch: 1")
        append_log(condition, f"Best Macro F1: {metrics['macro_f1']:.4f}")
        append_log(condition, f"Best checkpoint: {best_path}")
        append_log(condition, "Epoch 2 was not started automatically.")
    append_log(condition, "Training chunk complete.")
    append_log(condition, "No additional chunk was started automatically.")


def load_best_metrics(condition):
    paths = condition_paths(condition)
    if not paths["best_pointer"].exists():
        raise RuntimeError(f"Best checkpoint missing for condition {condition}")
    pointer = json.loads(paths["best_pointer"].read_text(encoding="utf-8"))
    metrics_path = paths["best_dir"] / pointer["relative_path"] / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_signature = stable_json_hash(experiment_config(condition))
    if payload["config_signature"] != expected_signature:
        raise RuntimeError(f"Best metrics signature mismatch for {condition}")
    if payload["condition"] != condition or payload["best_epoch"] != 1:
        raise RuntimeError(f"Best metrics identity mismatch for {condition}")
    return payload["metrics"]


def run_evaluate_best(condition):
    device = require_mps(condition)
    if device is None:
        return
    paths = condition_paths(condition)
    metrics = load_best_metrics(condition)
    print(format_condition_report(condition, metrics, "verified", "stored-in-report"))


def run_comparison():
    clean = load_best_metrics("clean")
    expanded = load_best_metrics("expanded")
    lines = [
        "EXPERIMENT 13 — CONTROLLED LORA DATA ABLATION COMPARISON",
        "",
        "13B minus 13A after Epoch 1; test split was not used.",
        "Architecture, seeds, loss, optimizer, and training settings are identical.",
        "Only the training dataset differs.",
        "",
        f"Accuracy delta: {expanded['accuracy'] - clean['accuracy']:+.4f}",
        f"Macro F1 delta: {expanded['macro_f1'] - clean['macro_f1']:+.4f}",
    ]
    for sentiment in SENTIMENTS:
        lines.append(
            f"{sentiment} F1 delta: "
            f"{expanded['per_class'][sentiment]['f1'] - clean['per_class'][sentiment]['f1']:+.4f}"
        )
    lines.extend(
        [
            "Neutral precision delta: "
            f"{expanded['per_class']['neutral']['precision'] - clean['per_class']['neutral']['precision']:+.4f}",
            "Neutral recall delta: "
            f"{expanded['per_class']['neutral']['recall'] - clean['per_class']['neutral']['recall']:+.4f}",
            "Predicted neutral count delta: "
            f"{expanded['predicted_counts']['neutral'] - clean['predicted_counts']['neutral']:+d}",
            "",
            "LANGUAGE MACRO F1 DELTAS",
        ]
    )
    for _, language in LANGUAGES:
        lines.append(
            f"{language}: "
            f"{expanded['languages'][language]['macro_f1'] - clean['languages'][language]['macro_f1']:+.4f}"
        )
    lines.extend(["", "ASPECT MACRO F1 DELTAS"])
    for aspect in ASPECTS:
        lines.append(
            f"{aspect}: "
            f"{expanded['aspects'][aspect]['macro_f1'] - clean['aspects'][aspect]['macro_f1']:+.4f}"
        )
    lines.extend(
        [
            "",
            "DECISION CHECKS",
            f"13B predicts neutral at all: {expanded['predicted_counts']['neutral'] > 0}",
            "13B Neutral F1 > 13A Neutral F1: "
            f"{expanded['per_class']['neutral']['f1'] > clean['per_class']['neutral']['f1']}",
            "Epoch 2 was not started automatically for either condition.",
        ]
    )
    COMPARISON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 13 thermal-safe unweighted XLM-R LoRA data ablation"
    )
    parser.add_argument("--condition", choices=sorted(CONDITIONS))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-static", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--train-chunk", action="store_true")
    mode.add_argument("--evaluate-best", action="store_true")
    mode.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if not args.compare and args.condition is None:
        parser.error("--condition is required unless --compare is used")
    if args.compare and args.condition is not None:
        parser.error("--compare reads both conditions; omit --condition")
    return args


def main():
    args = parse_args()
    if args.compare:
        run_comparison()
        return
    try:
        if args.validate_static:
            run_static_validation(args.condition)
        elif args.smoke_test:
            run_smoke_test(args.condition)
        elif args.train_chunk:
            run_train_chunk(args.condition)
        elif args.evaluate_best:
            run_evaluate_best(args.condition)
    except Exception as error:
        if args.condition:
            append_log(args.condition, f"Run failed: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
