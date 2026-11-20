"""Aspect Presence V2: XLM-R + LoRA multi-label baseline.

The script is deliberately MPS-only for training, runs at most 80 optimizer
steps per invocation, and permits only the final Epoch 2 continuation from a
completed Epoch 1 checkpoint. It loads TRAIN and DEV only; the test split is
never referenced. Re-run the same ``--train-chunk`` command until Epoch 2
finishes.
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
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "aspect_presence_v2" / "train.csv"
DEV_PATH = PROJECT_ROOT / "data" / "aspect_presence_v2" / "dev.csv"
MODEL_ROOT = PROJECT_ROOT / "models" / "aspect_presence_xlmr_lora_v1"
REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "aspect_presence_xlmr_lora_v1_epoch_1.txt"
)

MODEL_NAME = "xlm-roberta-base"
ASPECTS = ("taste", "atmosphere", "service", "value")
ASPECT2ID = {aspect: index for index, aspect in enumerate(ASPECTS)}
ID2ASPECT = {index: aspect for aspect, index in ASPECT2ID.items()}
LANGUAGES = ("tr", "en", "es")
REQUIRED_COLUMNS = {"sentence", "language", "split", *ASPECTS}

EXPECTED_TRAIN_ROWS = 3775
EXPECTED_DEV_ROWS = 942
EXPECTED_DEV_ALL_ZERO = 212
TOTAL_TARGET_EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_BATCH_SIZE = 1
RANDOM_SEED = 42
MAX_OPTIMIZER_STEPS_PER_RUN = 80
PREDICTION_THRESHOLD = 0.50

LORA_TARGET_MODULES = ["query", "value"]
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_BIAS = "none"
MODULES_TO_SAVE = ["classifier"]

CLASSICAL_BASELINE = {
    "macro_f1": 0.7158,
    "micro_f1": 0.7705,
    "exact_match": 0.6507,
    "hamming_loss": 0.1056,
    "all_zero_exact_accuracy": 0.7783,
    "all_zero_false_positive_total": 49,
    "aspect_f1": {
        "taste": 0.8211,
        "atmosphere": 0.6810,
        "service": 0.7914,
        "value": 0.5697,
    },
}

MPS_UNAVAILABLE_MESSAGE = (
    "MPS kullanılamıyor. CPU fallback yasak olduğu için transformer training "
    "başlatılmadı. Komutu MPS destekli normal VS Code terminalinde çalıştırın."
)


def paths() -> dict[str, Path]:
    return {
        "model_root": MODEL_ROOT,
        "resume_dir": MODEL_ROOT / "resume",
        "resume_pointer": MODEL_ROOT / "resume" / "current.json",
        "best_dir": MODEL_ROOT / "best",
        "best_pointer": MODEL_ROOT / "best" / "current.json",
        "report": REPORT_PATH,
    }


def append_log(text: str) -> None:
    print(text, flush=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("a", encoding="utf-8") as file:
        file.write(text + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def require_mps() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError(MPS_UNAVAILABLE_MESSAGE)
    append_log("Training device: mps")
    return torch.device("mps")


def load_rows(path: Path, expected_split: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        rows = list(reader)
    if any(row["split"] != expected_split for row in rows):
        raise ValueError(f"{path} contains unexpected split values")
    if any(not row["sentence"].strip() for row in rows):
        raise ValueError(f"{path} contains an empty sentence")
    if any(row["language"] not in LANGUAGES for row in rows):
        raise ValueError(f"{path} contains an unsupported language")
    for row in rows:
        for aspect in ASPECTS:
            if row[aspect] not in {"0", "1"}:
                raise ValueError(f"{path} has a non-binary {aspect} label")
    return rows


def label_vector(row: dict[str, str]) -> list[float]:
    return [float(row[aspect]) for aspect in ASPECTS]


class AspectPresenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer) -> None:
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = label_vector(row)
        return encoded


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def positive_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for aspect in ASPECTS:
        positive = sum(int(row[aspect]) for row in rows)
        negative = len(rows) - positive
        if positive == 0:
            raise RuntimeError(f"Cannot compute pos_weight: {aspect} has no positives")
        output[aspect] = {
            "positive": positive,
            "negative": negative,
            "pos_weight": negative / positive,
        }
    return output


def pos_weight_tensor(
    stats: dict[str, dict[str, float | int]], device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        [stats[aspect]["pos_weight"] for aspect in ASPECTS],
        dtype=torch.float32,
        device=device,
    )


def create_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias=LORA_BIAS,
        modules_to_save=list(MODULES_TO_SAVE),
    )


def experiment_config(
    train_stats: dict[str, dict[str, float | int]],
    *,
    total_target_epochs: int = TOTAL_TARGET_EPOCHS,
    max_optimizer_steps_per_run: int = MAX_OPTIMIZER_STEPS_PER_RUN,
) -> dict[str, object]:
    return {
        "experiment": "aspect-presence-xlmr-lora-v1",
        "task": "multi_label_aspect_presence",
        "model_name": MODEL_NAME,
        "train_path": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
        "dev_path": str(DEV_PATH.relative_to(PROJECT_ROOT)),
        "train_sha256": file_sha256(TRAIN_PATH),
        "dev_sha256": file_sha256(DEV_PATH),
        "test_loaded": False,
        "input_format": "sentence only",
        "aspects": list(ASPECTS),
        "output_dimension": len(ASPECTS),
        "total_target_epochs": total_target_epochs,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "optimizer": "torch.optim.AdamW",
        "scheduler": None,
        "loss": "binary_cross_entropy_with_logits(pos_weight=train_negative/train_positive)",
        "pos_weight_order": list(ASPECTS),
        "pos_weight": [train_stats[aspect]["pos_weight"] for aspect in ASPECTS],
        "prediction_activation": "sigmoid",
        "prediction_threshold": PREDICTION_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "max_optimizer_steps_per_run": max_optimizer_steps_per_run,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": sorted(LORA_TARGET_MODULES),
        "lora_bias": LORA_BIAS,
        "modules_to_save": list(MODULES_TO_SAVE),
    }


def legacy_epoch1_config(
    train_stats: dict[str, dict[str, float | int]],
) -> dict[str, object]:
    """Reproduce the exact config signature stored after Epoch 1."""
    return experiment_config(
        train_stats,
        total_target_epochs=1,
        max_optimizer_steps_per_run=50,
    )


def verify_target_modules(model) -> dict[str, list[str]]:
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
                f"Expected {layer_count} {target!r} targets, found {len(names)}"
            )
    return selected


def parameter_summary(model) -> dict[str, int | float]:
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
            "Unexpected trainable parameters: "
            f"unexpected={unexpected[:10]}, trainable={trainable}, lora={lora}, "
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
    config.num_labels = len(ASPECTS)
    config.problem_type = "multi_label_classification"
    config.label2id = dict(ASPECT2ID)
    config.id2label = dict(ID2ASPECT)
    with torch.device("meta"):
        base = AutoModelForSequenceClassification.from_config(config)
        selected = verify_target_modules(base)
        model = get_peft_model(base, create_lora_config())
    return selected, parameter_summary(model)


def load_base_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ASPECTS),
        problem_type="multi_label_classification",
        label2id=dict(ASPECT2ID),
        id2label=dict(ID2ASPECT),
    )


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def compute_training_loss(
    logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits,
        labels.to(dtype=logits.dtype),
        pos_weight=pos_weight,
        reduction="mean",
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def move_optimizer_state(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def capture_rng_state(device: torch.device) -> dict[str, object]:
    mps_state = None
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        mps_state = torch.mps.get_rng_state()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "mps": mps_state,
    }


def restore_rng_state(state: dict[str, object], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if (
        device.type == "mps"
        and state.get("mps") is not None
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["mps"])


def epoch_permutation(
    example_count: int, epoch_number: int = 1
) -> tuple[int, list[int]]:
    if epoch_number not in {1, 2}:
        raise ValueError("Only Epoch 1 and Epoch 2 permutations are supported")
    epoch_seed = RANDOM_SEED + epoch_number - 1
    generator = torch.Generator().manual_seed(epoch_seed)
    return epoch_seed, torch.randperm(example_count, generator=generator).tolist()


def permutation_sha256(permutation: list[int]) -> str:
    return hashlib.sha256(np.asarray(permutation, dtype=np.int64).tobytes()).hexdigest()


def optimizer_steps_per_epoch(example_count: int) -> int:
    microbatches = math.ceil(example_count / TRAIN_BATCH_SIZE)
    return math.ceil(microbatches / GRADIENT_ACCUMULATION_STEPS)


def read_resume_metadata(accepted_signatures: set[str] | tuple[str, ...] | list[str]):
    path_map = paths()
    pointer_path = path_map["resume_pointer"]
    if not pointer_path.exists():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    slot_name = pointer.get("active_slot")
    if slot_name not in {"slot_a", "slot_b"}:
        raise RuntimeError("Invalid resume pointer slot")
    slot_dir = path_map["resume_dir"] / slot_name
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
    if metadata.get("config_signature") not in set(accepted_signatures):
        raise RuntimeError(
            "Checkpoint signature does not match this experiment: "
            f"stored={metadata.get('config_signature')} "
            f"accepted={sorted(set(accepted_signatures))}"
        )
    if metadata.get("experiment") != "aspect-presence-xlmr-lora-v1":
        raise RuntimeError("Checkpoint mixing was detected")
    if pointer.get("checkpoint_id") != metadata.get("checkpoint_id"):
        raise RuntimeError("Resume pointer and metadata checkpoint IDs differ")
    metadata["_slot_dir"] = str(slot_dir)
    metadata["_slot_name"] = slot_name
    return metadata


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


def load_training_state(
    metadata,
    optimizer,
    device: torch.device,
    accepted_signatures: set[str] | tuple[str, ...] | list[str],
):
    checkpoint = torch.load(
        Path(metadata["_slot_dir"]) / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("config_signature") not in set(accepted_signatures):
        raise RuntimeError("Training-state signature mismatch")
    if checkpoint.get("device_type") != device.type:
        raise RuntimeError("Checkpoint must resume on the same device type")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state(optimizer, device)
    restore_rng_state(checkpoint["rng_state"], device)
    return checkpoint


def validate_progress(state: dict[str, object], example_count: int) -> None:
    if state["completed_epoch"] == 2:
        if state["current_epoch"] is not None:
            raise RuntimeError("Completed Epoch 2 has a non-null current epoch")
        return
    if state["completed_epoch"] == 1 and state["current_epoch"] is None:
        if state["optimizer_steps_in_epoch"] != optimizer_steps_per_epoch(example_count):
            raise RuntimeError("Completed Epoch 1 optimizer-step count is invalid")
        if state["sample_position"] != example_count:
            raise RuntimeError("Completed Epoch 1 sample position is invalid")
        return
    if (state["completed_epoch"], state["current_epoch"]) not in {(0, 1), (1, 2)}:
        raise RuntimeError("Only Epoch 1 or Epoch 2 progress is supported")
    permutation = state["epoch_permutation"]
    expected_seed, expected = epoch_permutation(example_count, state["current_epoch"])
    if permutation != expected or state["epoch_seed"] != expected_seed:
        raise RuntimeError("Stored epoch permutation is not deterministic")
    if state["permutation_sha256"] != permutation_sha256(permutation):
        raise RuntimeError("Stored permutation hash mismatch")
    if state["microbatches_accumulated"] != 0:
        raise RuntimeError("Checkpoint is not at an accumulation boundary")
    position = state["microbatch_position"]
    if state["sample_position"] != min(position * TRAIN_BATCH_SIZE, example_count):
        raise RuntimeError("Sample and microbatch positions differ")
    expected_steps = math.ceil(position / GRADIENT_ACCUMULATION_STEPS) if position else 0
    if state["optimizer_steps_in_epoch"] != expected_steps:
        raise RuntimeError("Optimizer step and microbatch positions differ")


def initialize_epoch2_state(
    state: dict[str, object], example_count: int
) -> dict[str, object]:
    """Start Epoch 2 while preserving model, optimizer, RNG and Epoch 1 history."""
    if state.get("completed_epoch") != 1 or state.get("current_epoch") is not None:
        raise RuntimeError("Epoch 2 can start only from a completed Epoch 1 state")
    epoch1_metrics = state.get("last_epoch_metrics")
    if not epoch1_metrics:
        raise RuntimeError("Epoch 1 metrics are missing from resume state")
    epoch_seed, permutation = epoch_permutation(example_count, 2)
    return {
        **state,
        "current_epoch": 2,
        "optimizer_steps_in_epoch": 0,
        "sample_position": 0,
        "microbatch_position": 0,
        "microbatches_accumulated": 0,
        "epoch_permutation": permutation,
        "permutation_sha256": permutation_sha256(permutation),
        "epoch_seed": epoch_seed,
        "epoch_running_loss_sum": 0.0,
        "epoch_microbatches_seen": 0,
        "epoch_metrics_history": {"1": epoch1_metrics},
    }


def save_resume_checkpoint(
    model,
    tokenizer,
    optimizer,
    device: torch.device,
    state: dict[str, object],
    signature: str,
    accepted_signatures: set[str] | tuple[str, ...] | list[str],
) -> Path:
    path_map = paths()
    path_map["resume_dir"].mkdir(parents=True, exist_ok=True)
    current_metadata = read_resume_metadata(accepted_signatures)
    current_slot = current_metadata.get("_slot_name") if current_metadata else None
    next_slot = "slot_b" if current_slot == "slot_a" else "slot_a"
    final_dir = path_map["resume_dir"] / next_slot
    temporary_dir = path_map["resume_dir"] / f".{next_slot}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    checkpoint_epoch = state.get("current_epoch") or state.get("completed_epoch")
    checkpoint_id = (
        f"presence-v1-epoch{checkpoint_epoch}-step{state['optimizer_steps_in_epoch']}-"
        f"sample{state['sample_position']}"
    )
    model.save_pretrained(temporary_dir)
    tokenizer.save_pretrained(temporary_dir)
    training_state = {
        **state,
        "checkpoint_id": checkpoint_id,
        "config_signature": signature,
        "experiment": "aspect-presence-xlmr-lora-v1",
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
    metadata["epoch_permutation_length"] = len(state["epoch_permutation"])
    write_json_atomic(temporary_dir / "metadata.json", metadata)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(temporary_dir, final_dir)
    write_json_atomic(
        path_map["resume_pointer"],
        {"active_slot": next_slot, "checkpoint_id": checkpoint_id},
    )
    return final_dir


def save_best_checkpoint(
    model,
    tokenizer,
    metrics: dict[str, object],
    signature: str,
    epoch_number: int,
):
    if epoch_number not in {1, 2}:
        raise ValueError("Best checkpoint may be saved only for Epoch 1 or Epoch 2")
    path_map = paths()
    path_map["best_dir"].mkdir(parents=True, exist_ok=True)
    final_dir = path_map["best_dir"] / f"epoch_{epoch_number}"
    temporary_dir = path_map["best_dir"] / f".epoch_{epoch_number}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    model.save_pretrained(temporary_dir)
    tokenizer.save_pretrained(temporary_dir)
    checkpoint_id = f"presence-v1-best-epoch{epoch_number}"
    write_json_atomic(
        temporary_dir / "metrics.json",
        {
            "checkpoint_id": checkpoint_id,
            "config_signature": signature,
            "best_epoch": epoch_number,
            "threshold": PREDICTION_THRESHOLD,
            "aspect_order": list(ASPECTS),
            "metrics": metrics,
        },
    )
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(temporary_dir, final_dir)
    write_json_atomic(
        path_map["best_pointer"],
        {"relative_path": f"epoch_{epoch_number}", "checkpoint_id": checkpoint_id},
    )
    return final_dir


def predict(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[list[float]] = []
    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            batch_labels = batch.pop("labels")
            logits = model(**batch).logits
            labels.extend(batch_labels.detach().cpu().tolist())
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def multilabel_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    aspect_metrics: dict[str, dict[str, float | int]] = {}
    for index, aspect in enumerate(ASPECTS):
        precision, recall, f1_value, support = precision_recall_fscore_support(
            labels[:, index],
            predictions[:, index],
            labels=[1],
            average=None,
            zero_division=0,
        )
        matrix = confusion_matrix(
            labels[:, index], predictions[:, index], labels=[0, 1]
        )
        tn, fp, fn, tp = matrix.ravel()
        aspect_metrics[aspect] = {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1_value[0]),
            "support": int(support[0]),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "exact_match": float(np.mean(np.all(labels == predictions, axis=1))),
        "hamming_loss": float(hamming_loss(labels, predictions)),
        "aspects": aspect_metrics,
    }


def subset_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "exact_match": float(np.mean(np.all(labels == predictions, axis=1))),
    }


def detailed_metrics(
    dev_rows: list[dict[str, str]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    train_loss: float,
) -> dict[str, object]:
    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(np.int64)
    output = multilabel_metrics(labels, predictions)
    output["train_loss"] = float(train_loss)
    output["threshold"] = PREDICTION_THRESHOLD
    output["predicted_positive_aspect_count"] = int(predictions.sum())

    all_zero_mask = labels.sum(axis=1) == 0
    all_zero_predictions = predictions[all_zero_mask]
    all_zero_correct = int(np.sum(all_zero_predictions.sum(axis=1) == 0))
    all_zero_support = int(np.sum(all_zero_mask))
    output["all_zero"] = {
        "support": all_zero_support,
        "correct": all_zero_correct,
        "exact_accuracy": all_zero_correct / all_zero_support,
        "false_positive_total": int(all_zero_predictions.sum()),
        "false_positive_by_aspect": {
            aspect: int(all_zero_predictions[:, index].sum())
            for index, aspect in enumerate(ASPECTS)
        },
    }

    target_mask = ~all_zero_mask
    output["target_containing"] = {
        "support": int(np.sum(target_mask)),
        **subset_metrics(labels[target_mask], predictions[target_mask]),
    }
    language_output: dict[str, dict[str, float | int | None]] = {}
    for language in LANGUAGES:
        mask = np.asarray([row["language"] == language for row in dev_rows])
        language_labels = labels[mask]
        language_predictions = predictions[mask]
        language_zero_mask = language_labels.sum(axis=1) == 0
        zero_support = int(np.sum(language_zero_mask))
        language_output[language] = {
            "support": int(np.sum(mask)),
            "macro_f1": float(
                f1_score(
                    language_labels,
                    language_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "exact_match": float(
                np.mean(np.all(language_labels == language_predictions, axis=1))
            ),
            "all_zero_support": zero_support,
            "all_zero_exact_accuracy": (
                float(
                    np.mean(
                        language_predictions[language_zero_mask].sum(axis=1) == 0
                    )
                )
                if zero_support
                else None
            ),
        }
    output["languages"] = language_output
    return output


def predicted_aspect_count(metrics: dict[str, object]) -> int:
    if "predicted_positive_aspect_count" in metrics:
        return int(metrics["predicted_positive_aspect_count"])
    return sum(
        int(metrics["aspects"][aspect]["tp"])
        + int(metrics["aspects"][aspect]["fp"])
        for aspect in ASPECTS
    )


def final_decision(metrics: dict[str, object]) -> str:
    transformer_better = (
        metrics["macro_f1"] > CLASSICAL_BASELINE["macro_f1"]
        and metrics["exact_match"] > CLASSICAL_BASELINE["exact_match"]
        and metrics["all_zero"]["exact_accuracy"]
        > CLASSICAL_BASELINE["all_zero_exact_accuracy"]
    )
    transformer_worse = (
        metrics["macro_f1"] < CLASSICAL_BASELINE["macro_f1"]
        and metrics["exact_match"] < CLASSICAL_BASELINE["exact_match"]
        and metrics["all_zero"]["exact_accuracy"]
        < CLASSICAL_BASELINE["all_zero_exact_accuracy"]
    )
    if transformer_better:
        return "TRANSFORMER WINS"
    if transformer_worse:
        return "CLASSICAL WINS"
    return "MIXED"


def format_report(
    epoch_number: int,
    metrics: dict[str, object],
    train_stats: dict[str, dict[str, float | int]],
    config_signature: str,
    permutation_hash: str,
    epoch1_metrics: dict[str, object] | None = None,
) -> str:
    lines = [
        f"ASPECT PRESENCE V2 — XLM-R + LORA MULTI-LABEL EPOCH {epoch_number}",
        "",
        f"Train: {TRAIN_PATH.relative_to(PROJECT_ROOT)}",
        f"Dev: {DEV_PATH.relative_to(PROJECT_ROOT)}",
        "Test: not loaded, predicted, or evaluated",
        "Input: sentence only",
        f"Aspect/logit order: {list(ASPECTS)}",
        "Activation: sigmoid",
        f"Threshold: {PREDICTION_THRESHOLD:.2f}",
        f"Random seed: {RANDOM_SEED}",
        f"Epoch shuffle seed: {RANDOM_SEED + epoch_number - 1}",
        f"Permutation SHA256: {permutation_hash}",
        f"Config signature: {config_signature}",
        f"Epoch train loss: {metrics['train_loss']:.6f}",
        "",
        "TRAIN POS_WEIGHT",
    ]
    for aspect in ASPECTS:
        item = train_stats[aspect]
        lines.append(
            f"{aspect}: positive={item['positive']} | negative={item['negative']} | "
            f"pos_weight={item['pos_weight']:.8f}"
        )
    lines.extend(
        [
            "",
            "DEV OVERALL",
            f"Macro F1: {metrics['macro_f1']:.4f}",
            f"Micro F1: {metrics['micro_f1']:.4f}",
            f"Exact Match: {metrics['exact_match']:.4f}",
            f"Hamming Loss: {metrics['hamming_loss']:.4f}",
            f"Predicted positive aspect count: {predicted_aspect_count(metrics)}",
            "",
            "DEV ASPECTS",
        ]
    )
    for aspect in ASPECTS:
        item = metrics["aspects"][aspect]
        lines.append(
            f"{aspect}: P={item['precision']:.4f} | R={item['recall']:.4f} | "
            f"F1={item['f1']:.4f} | support={item['support']} | "
            f"TP={item['tp']} FP={item['fp']} FN={item['fn']} TN={item['tn']}"
        )
    zero = metrics["all_zero"]
    lines.extend(
        [
            "",
            "DEV ALL-ZERO",
            f"Support: {zero['support']}",
            f"Correct 0/0/0/0: {zero['correct']}",
            f"Exact accuracy: {zero['exact_accuracy']:.4f}",
            f"False-positive aspect total: {zero['false_positive_total']}",
        ]
    )
    for aspect in ASPECTS:
        lines.append(
            f"{aspect} false positive: {zero['false_positive_by_aspect'][aspect]}"
        )
    target = metrics["target_containing"]
    lines.extend(
        [
            "",
            "DEV TARGET-CONTAINING",
            f"Support: {target['support']}",
            f"Macro F1: {target['macro_f1']:.4f}",
            f"Micro F1: {target['micro_f1']:.4f}",
            f"Exact Match: {target['exact_match']:.4f}",
            "",
            "LANGUAGE BREAKDOWN",
        ]
    )
    for language in LANGUAGES:
        item = metrics["languages"][language]
        zero_accuracy = item["all_zero_exact_accuracy"]
        zero_text = "N/A" if zero_accuracy is None else f"{zero_accuracy:.4f}"
        lines.append(
            f"{language.upper()}: Macro F1={item['macro_f1']:.4f} | "
            f"Exact Match={item['exact_match']:.4f} | "
            f"All-zero exact={zero_text} | all-zero support={item['all_zero_support']}"
        )
    lines.extend(
        [
            "",
            "TRANSFORMER MINUS CLASSICAL V2 CALIBRATED",
            f"Macro F1 delta: {metrics['macro_f1'] - CLASSICAL_BASELINE['macro_f1']:+.4f}",
            f"Micro F1 delta: {metrics['micro_f1'] - CLASSICAL_BASELINE['micro_f1']:+.4f}",
            f"Exact Match delta: {metrics['exact_match'] - CLASSICAL_BASELINE['exact_match']:+.4f}",
            f"Hamming Loss delta: {metrics['hamming_loss'] - CLASSICAL_BASELINE['hamming_loss']:+.4f}",
            "All-zero exact accuracy delta: "
            f"{zero['exact_accuracy'] - CLASSICAL_BASELINE['all_zero_exact_accuracy']:+.4f}",
            "All-zero FP total delta: "
            f"{zero['false_positive_total'] - CLASSICAL_BASELINE['all_zero_false_positive_total']:+d}",
        ]
    )
    for aspect in ASPECTS:
        lines.append(
            f"{aspect} F1 delta: "
            f"{metrics['aspects'][aspect]['f1'] - CLASSICAL_BASELINE['aspect_f1'][aspect]:+.4f}"
        )
    if epoch_number == 2:
        if not epoch1_metrics:
            raise RuntimeError("Epoch 1 metrics are required for the Epoch 2 report")
        lines.extend(
            [
                "",
                "EPOCH 2 MINUS EPOCH 1",
                f"Macro F1 delta: {metrics['macro_f1'] - epoch1_metrics['macro_f1']:+.4f}",
                f"Micro F1 delta: {metrics['micro_f1'] - epoch1_metrics['micro_f1']:+.4f}",
                f"Exact Match delta: {metrics['exact_match'] - epoch1_metrics['exact_match']:+.4f}",
                f"Hamming Loss delta: {metrics['hamming_loss'] - epoch1_metrics['hamming_loss']:+.4f}",
                "All-zero exact accuracy delta: "
                f"{metrics['all_zero']['exact_accuracy'] - epoch1_metrics['all_zero']['exact_accuracy']:+.4f}",
                "All-zero FP total delta: "
                f"{metrics['all_zero']['false_positive_total'] - epoch1_metrics['all_zero']['false_positive_total']:+d}",
                "Predicted positive aspect count delta: "
                f"{predicted_aspect_count(metrics) - predicted_aspect_count(epoch1_metrics):+d}",
            ]
        )
        for aspect in ASPECTS:
            lines.append(
                f"{aspect} F1 delta: "
                f"{metrics['aspects'][aspect]['f1'] - epoch1_metrics['aspects'][aspect]['f1']:+.4f}"
            )
        decision = final_decision(metrics)
        lines.extend(["", f"FINAL DECISION: {decision}"])
        if decision == "CLASSICAL WINS":
            lines.append(
                "Aspect Presence Transformer research closes after Epoch 2; "
                "the calibrated classical V2 detector remains the product model."
            )
    lines.extend(
        [
            "",
            "Epoch 3 was not started and is blocked.",
            "Threshold tuning was not performed.",
            "Test split was not used.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_static_validation() -> None:
    train_rows = load_rows(TRAIN_PATH, "train")
    dev_rows = load_rows(DEV_PATH, "dev")
    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_ROWS} train rows, got {len(train_rows)}"
        )
    if len(dev_rows) != EXPECTED_DEV_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_DEV_ROWS} dev rows, got {len(dev_rows)}")
    dev_all_zero = sum(sum(label_vector(row)) == 0 for row in dev_rows)
    if dev_all_zero != EXPECTED_DEV_ALL_ZERO:
        raise RuntimeError(
            f"Expected {EXPECTED_DEV_ALL_ZERO} DEV all-zero rows, got {dev_all_zero}"
        )
    train_stats = positive_stats(train_rows)
    selected, parameters = build_meta_model_for_validation()
    config = experiment_config(train_stats)
    signature = stable_json_hash(config)
    legacy_signature = stable_json_hash(legacy_epoch1_config(train_stats))
    accepted_signatures = {legacy_signature, signature}

    if MODEL_ROOT.resolve() in {
        (PROJECT_ROOT / "models" / "experiment_15_lora_expanded_balanced").resolve(),
        (PROJECT_ROOT / "models" / "aspect_presence_logreg_v1").resolve(),
        (PROJECT_ROOT / "models" / "aspect_presence_logreg_v2").resolve(),
    }:
        raise RuntimeError("Checkpoint root overlaps an existing experiment")
    if config["input_format"] != "sentence only" or config["output_dimension"] != 4:
        raise RuntimeError("Input/output task configuration is invalid")
    if config["prediction_activation"] != "sigmoid":
        raise RuntimeError("Multi-label prediction must use sigmoid")
    if config["loss"].split("(", 1)[0] != "binary_cross_entropy_with_logits":
        raise RuntimeError("Multi-label loss must be BCEWithLogitsLoss-equivalent")
    sample_logits = torch.zeros((1, 4), dtype=torch.float32)
    sample_labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    sample_weights = pos_weight_tensor(train_stats, torch.device("cpu"))
    sample_loss = compute_training_loss(sample_logits, sample_labels, sample_weights)
    if not torch.isfinite(sample_loss):
        raise RuntimeError("BCEWithLogitsLoss sanity check failed")

    metadata = read_resume_metadata(accepted_signatures)
    if metadata is None:
        raise RuntimeError("Epoch 1 resume checkpoint is missing")
    if metadata.get("completed_epoch") != 1 or metadata.get("current_epoch") is not None:
        raise RuntimeError("Active resume state is not a completed Epoch 1 state")
    epoch_steps = optimizer_steps_per_epoch(len(train_rows))
    if metadata.get("optimizer_steps_in_epoch") != epoch_steps:
        raise RuntimeError("Epoch 1 optimizer-step count is invalid")
    checkpoint = torch.load(
        Path(metadata["_slot_dir"]) / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("config_signature") not in accepted_signatures:
        raise RuntimeError("Epoch 1 training-state signature is incompatible")
    if not checkpoint.get("optimizer_state_dict"):
        raise RuntimeError("Epoch 1 optimizer state is missing")
    rng_state = checkpoint.get("rng_state") or {}
    if not {"python", "numpy", "torch", "mps"}.issubset(rng_state):
        raise RuntimeError("Epoch 1 RNG state is incomplete")
    if len(checkpoint.get("epoch_permutation", [])) != len(train_rows):
        raise RuntimeError("Epoch 1 permutation is incomplete")
    if not checkpoint.get("last_epoch_metrics"):
        raise RuntimeError("Epoch 1 metrics are missing")
    if checkpoint.get("best_epoch") != 1:
        raise RuntimeError("Epoch 1 is not preserved as the current best epoch")
    epoch2_seed, epoch2_permutation = epoch_permutation(len(train_rows), 2)
    if epoch2_seed != 43 or len(epoch2_permutation) != len(train_rows):
        raise RuntimeError("Epoch 2 deterministic permutation validation failed")

    print("ASPECT PRESENCE XLM-R + LORA STATIC VALIDATION")
    print(f"Train rows: {len(train_rows)}")
    print(f"Dev rows: {len(dev_rows)}")
    print(f"Dev all-zero rows: {dev_all_zero}")
    print(f"Train language counts: {dict(Counter(row['language'] for row in train_rows))}")
    print("Aspect/logit order: taste, atmosphere, service, value")
    for aspect in ASPECTS:
        item = train_stats[aspect]
        print(
            f"{aspect}: positive={item['positive']} negative={item['negative']} "
            f"pos_weight={item['pos_weight']:.8f}"
        )
        if item["pos_weight"] > 10:
            print(f"WARNING: {aspect} pos_weight exceeds 10")
    print("Input format: sentence only")
    print("Output dimension: 4")
    print("Activation: sigmoid")
    print("Loss: BCEWithLogitsLoss-equivalent with TRAIN-only pos_weight")
    print(f"BCE sanity loss: {sample_loss.item():.6f}")
    print(
        "LoRA targets: "
        + ", ".join(f"{name}={len(values)}" for name, values in selected.items())
    )
    print(f"Trainable LoRA parameters: {parameters['lora']}")
    print(f"Trainable classifier parameters: {parameters['classifier']}")
    print(f"Unexpected trainable parameters: 0")
    print(f"Optimizer steps per epoch: {epoch_steps}")
    print(f"Epoch 1 complete: {metadata['completed_epoch'] == 1}")
    print(f"Epoch 1 optimizer steps: {metadata['optimizer_steps_in_epoch']}/{epoch_steps}")
    print(f"Epoch 1 best Macro F1: {metadata['best_macro_f1']:.4f}")
    print(f"Epoch 2 shuffle seed: {epoch2_seed}")
    print(f"Epoch 2 permutation SHA256: {permutation_sha256(epoch2_permutation)}")
    print(f"Max optimizer steps per invocation: {MAX_OPTIMIZER_STEPS_PER_RUN}")
    print(f"Final Epoch 2 chunk size: {epoch_steps % MAX_OPTIMIZER_STEPS_PER_RUN}")
    print(f"Legacy Epoch 1 signature accepted: {metadata['config_signature'] == legacy_signature}")
    print(f"Epoch 2 config signature: {signature}")
    print(f"Checkpoint root: {MODEL_ROOT.relative_to(PROJECT_ROOT)}")
    print("Checkpoint isolated from sentiment/classical experiments: True")
    print("MPS required; CPU fallback blocked: True")
    print("Epoch 2 resume only; fresh training blocked by existing state: True")
    print("Epoch 3 automatic/manual start blocked: True")
    print("Test split referenced or loaded: False")
    print("Training started: NO")


def run_train_chunk(max_optimizer_steps: int) -> None:
    if max_optimizer_steps < 1 or max_optimizer_steps > MAX_OPTIMIZER_STEPS_PER_RUN:
        raise ValueError(
            f"--max-optimizer-steps must be between 1 and {MAX_OPTIMIZER_STEPS_PER_RUN}"
        )
    device = require_mps()
    set_seed(RANDOM_SEED)
    train_rows = load_rows(TRAIN_PATH, "train")
    dev_rows = load_rows(DEV_PATH, "dev")
    if len(train_rows) != EXPECTED_TRAIN_ROWS or len(dev_rows) != EXPECTED_DEV_ROWS:
        raise RuntimeError("Dataset row counts changed after static validation")
    train_stats = positive_stats(train_rows)
    config = experiment_config(train_stats)
    signature = stable_json_hash(config)
    legacy_signature = stable_json_hash(legacy_epoch1_config(train_stats))
    accepted_signatures = {legacy_signature, signature}
    metadata = read_resume_metadata(accepted_signatures)
    if metadata is None:
        raise RuntimeError(
            "Epoch 1 resume checkpoint is required; fresh training is blocked."
        )
    if metadata.get("completed_epoch") == 2:
        append_log("Epoch 2 is already complete. Epoch 3 is blocked.")
        return
    if metadata.get("completed_epoch") != 1:
        raise RuntimeError("Epoch 2 may resume only after completed Epoch 1")

    tokenizer_source = Path(metadata["_slot_dir"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = create_or_resume_model(metadata).to(device)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=LEARNING_RATE)
    loss_pos_weight = pos_weight_tensor(train_stats, device)
    state = load_training_state(
        metadata, optimizer, device, accepted_signatures
    )
    validate_progress(state, len(train_rows))
    if state.get("current_epoch") is None:
        state = initialize_epoch2_state(state, len(train_rows))
    validate_progress(state, len(train_rows))
    if state.get("current_epoch") != 2:
        raise RuntimeError("Epoch 3 guard: only Epoch 2 may run")
    epoch1_metrics = state.get("epoch_metrics_history", {}).get("1")
    if not epoch1_metrics:
        raise RuntimeError("Epoch 1 historical metrics are missing")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_dataset = AspectPresenceDataset(train_rows, tokenizer)
    total_steps = optimizer_steps_per_epoch(len(train_rows))
    target_step = min(
        state["optimizer_steps_in_epoch"] + max_optimizer_steps, total_steps
    )
    stop_requested = False

    def request_stop(_signal_number, _frame):
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    append_log("ASPECT PRESENCE XLM-R + LORA EPOCH 2 TRAINING CHUNK")
    append_log("Epoch: 2/2")
    append_log("Training device: mps")
    append_log(f"Starting optimizer step: {state['optimizer_steps_in_epoch']}")
    append_log(f"Target optimizer step: {target_step}")
    append_log(f"Epoch optimizer steps: {total_steps}")
    append_log(f"Sample position: {state['sample_position']}/{len(train_rows)}")

    try:
        while state["optimizer_steps_in_epoch"] < target_step:
            remaining = len(train_rows) - state["sample_position"]
            group_size = min(GRADIENT_ACCUMULATION_STEPS, remaining)
            if group_size <= 0:
                break
            group_loss = 0.0
            group_indices = state["epoch_permutation"][
                state["sample_position"] : state["sample_position"] + group_size
            ]
            for row_index in group_indices:
                item = collator([train_dataset[row_index]])
                item = move_batch(item, device)
                labels = item.pop("labels")
                logits = model(**item).logits
                raw_loss = compute_training_loss(logits, labels, loss_pos_weight)
                # Keep the same accumulation scaling used by the validated
                # sentiment runs. The final short group therefore keeps the
                # fixed effective-batch convention instead of being rescaled.
                (raw_loss / GRADIENT_ACCUMULATION_STEPS).backward()
                group_loss += float(raw_loss.detach().cpu())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state["sample_position"] += group_size
            state["microbatch_position"] += group_size
            state["optimizer_steps_in_epoch"] += 1
            state["total_optimizer_steps"] += 1
            state["epoch_running_loss_sum"] += group_loss
            state["epoch_microbatches_seen"] += group_size
            state["microbatches_accumulated"] = 0
            if stop_requested:
                append_log("Stop requested; saving at completed optimizer-step boundary.")
                break
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        optimizer.zero_grad(set_to_none=True)

    epoch_finished = state["sample_position"] >= len(train_rows)
    if epoch_finished:
        average_loss = state["epoch_running_loss_sum"] / state["epoch_microbatches_seen"]
        dev_dataset = AspectPresenceDataset(dev_rows, tokenizer)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
        )
        labels, probabilities = predict(model, dev_loader, device)
        metrics = detailed_metrics(dev_rows, labels, probabilities, average_loss)
        state["completed_epoch"] = 2
        state["current_epoch"] = None
        state["last_epoch_metrics"] = metrics
        state.setdefault("epoch_metrics_history", {})["2"] = metrics
        if metrics["macro_f1"] > state["best_macro_f1"]:
            state["best_macro_f1"] = metrics["macro_f1"]
            state["best_epoch"] = 2
            best_dir = save_best_checkpoint(
                model, tokenizer, metrics, signature, epoch_number=2
            )
            append_log(f"Best checkpoint saved: {best_dir.relative_to(PROJECT_ROOT)}")
        report = format_report(
            2,
            metrics,
            train_stats,
            signature,
            state["permutation_sha256"],
            epoch1_metrics=epoch1_metrics,
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report, end="", flush=True)

    checkpoint = save_resume_checkpoint(
        model,
        tokenizer,
        optimizer,
        device,
        state,
        signature,
        accepted_signatures,
    )
    append_log(f"Resume checkpoint: {checkpoint.relative_to(PROJECT_ROOT)}")
    append_log(
        f"Current optimizer step: {state['optimizer_steps_in_epoch']}/{total_steps}"
    )
    append_log(f"Epoch 2 complete: {epoch_finished}")
    append_log(f"Total optimizer steps across Epochs 1-2: {state['total_optimizer_steps']}")
    if epoch_finished:
        append_log(f"Best epoch: {state['best_epoch']}")
        append_log(f"Best DEV Macro F1: {state['best_macro_f1']:.4f}")
        append_log("STOP: Epoch 3 was not started and is blocked by this script.")
    else:
        append_log("Run the same command manually for the next chunk.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MPS-only chunked XLM-R + LoRA aspect-presence training"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-static", action="store_true")
    mode.add_argument("--train-chunk", action="store_true")
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=MAX_OPTIMIZER_STEPS_PER_RUN,
        help=f"Per-invocation limit, maximum {MAX_OPTIMIZER_STEPS_PER_RUN}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_static:
        run_static_validation()
    elif args.train_chunk:
        run_train_chunk(args.max_optimizer_steps)
    else:
        raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
