"""Thermal-safe, resumable class-weighted XLM-R sentiment experiment."""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

# Experiment 04 training must stay on MPS; unsupported operations must fail
# explicitly instead of silently running on CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
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
REPORT_PATH = (
    PROJECT_ROOT / "data" / "analysis" / "experiment_04_xlmr_weighted.txt"
)
MODEL_ROOT = PROJECT_ROOT / "models" / "xlmr-sentiment-v2-weighted"
RESUME_DIR = MODEL_ROOT / "resume"
RESUME_PATH = RESUME_DIR / "training_state.pt"
RESUME_METADATA_PATH = RESUME_DIR / "metadata.json"
BEST_MODEL_DIR = MODEL_ROOT / "best"
BEST_METADATA_PATH = BEST_MODEL_DIR / "metadata.json"

MODEL_NAME = "xlm-roberta-base"
TOTAL_TARGET_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_BATCH_SIZE = 1
RANDOM_SEED = 42
LOG_EVERY_STEPS = 100

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
ID2LABEL = {label_id: label for label, label_id in LABEL2ID.items()}
CLASS_ID_ORDER = np.array([0, 1, 2], dtype=np.int64)
CLASS_NAMES_BY_ID = ("negative", "neutral", "positive")
SENTIMENTS = ("positive", "neutral", "negative")
LABEL_IDS = [LABEL2ID[sentiment] for sentiment in SENTIMENTS]
LANGUAGES = (("tr", "TR"), ("en", "EN"), ("es", "ES"))
ASPECTS = ("taste", "atmosphere", "service", "value")
REQUIRED_COLUMNS = {"sentence", "language", "split", "aspect", "sentiment"}
MPS_UNAVAILABLE_MESSAGE = (
    "MPS kullanılamıyor. CPU üzerinde transformer training güvenlik nedeniyle "
    "başlatılmadı. Komutu MPS destekli normal local terminalde çalıştırın."
)

VANILLA_METRICS = {
    "accuracy": 0.8457,
    "macro_f1": 0.5651,
    "positive_f1": 0.9134,
    "neutral_f1": 0.0000,
    "negative_f1": 0.7820,
    "taste_macro_f1": 0.5581,
    "atmosphere_macro_f1": 0.5275,
    "service_macro_f1": 0.5471,
    "value_macro_f1": 0.5856,
}


def set_seed(seed):
    """Python, NumPy ve PyTorch RNG'lerini sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device():
    """Experiment 03 ile aynı cihaz seçim mantığını kullan."""
    if os.environ.get("XLMR_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_mps_training_device():
    """Training için MPS'i zorunlu tut; CPU fallback'e izin verme."""
    if not torch.backends.mps.is_available():
        append_log(MPS_UNAVAILABLE_MESSAGE)
        return None

    device = torch.device("mps")
    append_log("Training device: mps")
    return device


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
    """Experiment 03 ile aynı aspect + sentence girdisini oluştur."""
    return f"aspect={row['aspect']} [SEP] {row['sentence']}"


class SentimentDataset(Dataset):
    """CSV satırlarını tokenizer girdilerine dönüştür."""

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


def append_log(text):
    """Metni terminale ve Experiment 04 raporuna ekle."""
    print(text, flush=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("a", encoding="utf-8") as report_file:
        report_file.write(text + "\n")


def ensure_report_header():
    """Mevcut raporu silmeden başlığı bir kez oluştur."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REPORT_PATH.exists():
        REPORT_PATH.write_text(
            "EXPERIMENT 04 — CLASS-WEIGHTED XLM-ROBERTA-BASE\n",
            encoding="utf-8",
        )


def file_sha256(path):
    """Resume konfigürasyonunu veriye bağlamak için dosya hash'i hesapla."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_class_weights(train_rows):
    """Balanced class weight'leri yalnızca cleaned train splitinden hesapla."""
    label_ids = np.array(
        [LABEL2ID[row["sentiment"]] for row in train_rows],
        dtype=np.int64,
    )
    present_classes = np.unique(label_ids)
    if not np.array_equal(present_classes, CLASS_ID_ORDER):
        raise ValueError(
            f"Beklenen class ID'leri {CLASS_ID_ORDER.tolist()}, "
            f"bulunan {present_classes.tolist()}."
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=CLASS_ID_ORDER,
        y=label_ids,
    )
    counts = Counter(row["sentiment"] for row in train_rows)

    for label_id, label_name in enumerate(CLASS_NAMES_BY_ID):
        if LABEL2ID[label_name] != label_id:
            raise ValueError("Label ID ile class-weight sırası uyuşmuyor.")

    return counts, weights


def experiment_config(class_weights):
    """Resume sırasında değişmemesi gereken deney kimliğini oluştur."""
    return {
        "model_name": MODEL_NAME,
        "total_target_epochs": TOTAL_TARGET_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_train_batch_size": (
            TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        ),
        "eval_batch_size": EVAL_BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "optimizer": "torch.optim.AdamW",
        "scheduler": None,
        "loss": "torch.nn.CrossEntropyLoss(weight=balanced_train_weights)",
        "label2id": LABEL2ID,
        "class_id_order": CLASS_ID_ORDER.tolist(),
        "class_weights": [float(weight) for weight in class_weights],
        "train_sha256": file_sha256(TRAIN_PATH),
        "dev_sha256": file_sha256(DEV_PATH),
        "input_format": "aspect=<aspect> [SEP] <sentence>",
    }


def config_signature(config):
    """Konfigürasyon için kararlı bir imza üret."""
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path, payload):
    """Küçük metadata dosyasını atomik olarak güncelle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def read_resume_metadata(expected_signature):
    """Resume dosyalarının varlığını ve deney uyumunu doğrula."""
    state_exists = RESUME_PATH.exists()
    metadata_exists = RESUME_METADATA_PATH.exists()
    if state_exists != metadata_exists:
        raise RuntimeError(
            "Resume checkpoint eksik veya tutarsız: state ve metadata birlikte olmalı."
        )
    if not state_exists:
        return None

    metadata = json.loads(RESUME_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("config_signature") != expected_signature:
        raise RuntimeError(
            "Resume checkpoint mevcut script/data ayarlarıyla uyuşmuyor. "
            "Eğitim güvenlik amacıyla başlatılmadı."
        )
    return metadata


def move_batch(batch, device):
    """Bir batch içindeki tensorları seçilen cihaza taşı."""
    return {name: value.to(device) for name, value in batch.items()}


def predict(model, data_loader, device):
    """Dev split için gerçek label ve tahmin ID'lerini üret."""
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
    """Epoch sonunda kullanılan dev accuracy ve Macro F1'i hesapla."""
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


def create_training_objects(train_rows, dev_rows, tokenizer, device):
    """Experiment 03 ayarlarıyla model, loader ve optimizer oluştur."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        SentimentDataset(train_rows, tokenizer),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        generator=train_generator,
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
    scheduler = None
    return model, train_loader, dev_loader, optimizer, scheduler, train_generator


def move_optimizer_state(optimizer, device):
    """Resume edilen optimizer tensorlarını seçilen cihaza taşı."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def capture_rng_state(train_generator, device):
    """Kesintisiz epoch sırasını yeniden üretmek için RNG state'lerini yakala."""
    mps_state = None
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        mps_state = torch.mps.get_rng_state()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "mps": mps_state,
        "train_generator": train_generator.get_state(),
    }


def restore_rng_state(rng_state, train_generator, device):
    """Checkpoint'teki RNG state'lerini eğitimden hemen önce geri yükle."""
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    train_generator.set_state(rng_state["train_generator"])
    if (
        device.type == "mps"
        and rng_state.get("mps") is not None
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(rng_state["mps"])


def load_resume_state(
    model,
    optimizer,
    scheduler,
    train_generator,
    device,
    expected_signature,
):
    """Model, optimizer, scheduler ve RNG state'lerini checkpoint'ten yükle."""
    checkpoint = torch.load(
        RESUME_PATH,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("config_signature") != expected_signature:
        raise RuntimeError("Resume training state mevcut deney ayarlarıyla uyuşmuyor.")
    if checkpoint.get("device_type") != device.type:
        raise RuntimeError(
            f"Checkpoint device={checkpoint.get('device_type')}, "
            f"current device={device.type}. Aynı cihaz seçimiyle devam edin."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state(optimizer, device)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is None and scheduler_state is not None:
        raise RuntimeError("Checkpoint scheduler içeriyor ancak bu deneyde scheduler yok.")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    restore_rng_state(checkpoint["rng_state"], train_generator, device)
    return checkpoint


def save_best_checkpoint(
    model,
    tokenizer,
    completed_epoch,
    metrics,
    signature,
):
    """Macro F1'e göre seçilen inference checkpoint'ini kaydet."""
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(BEST_MODEL_DIR)
    tokenizer.save_pretrained(BEST_MODEL_DIR)
    write_json_atomic(
        BEST_METADATA_PATH,
        {
            "best_epoch": completed_epoch,
            "best_macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "config_signature": signature,
        },
    )


def save_resume_checkpoint(
    model,
    optimizer,
    scheduler,
    device,
    completed_epoch,
    best_epoch,
    best_macro_f1,
    metrics,
    signature,
    rng_state,
):
    """Son tamamlanan epoch'tan devam etmek için tüm training state'ini kaydet."""
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "completed_epoch": completed_epoch,
        "best_macro_f1": best_macro_f1,
        "best_epoch": best_epoch,
        "last_epoch_metrics": metrics,
        "device_type": device.type,
        "config_signature": signature,
        "rng_state": rng_state,
    }
    temporary_path = RESUME_PATH.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, RESUME_PATH)
    write_json_atomic(
        RESUME_METADATA_PATH,
        {
            "format_version": 1,
            "completed_epoch": completed_epoch,
            "next_epoch": (
                completed_epoch + 1
                if completed_epoch < TOTAL_TARGET_EPOCHS
                else None
            ),
            "best_macro_f1": best_macro_f1,
            "best_epoch": best_epoch,
            "last_epoch_metrics": metrics,
            "device_type": device.type,
            "config_signature": signature,
            "scheduler": None,
        },
    )


def print_stage_zero_summary(train_rows, class_counts, class_weights, device, metadata):
    """Training başlatmadan Stage 0 özetini yazdır ve rapora ekle."""
    weight_lines = [
        (
            f"  {label_name} (id={label_id}): "
            f"count={class_counts[label_name]} | weight={class_weights[label_id]:.8f}"
        )
        for label_id, label_name in enumerate(CLASS_NAMES_BY_ID)
    ]
    if metadata is None:
        resume_status = "checkpoint yok; sıradaki epoch=1"
    else:
        next_epoch = metadata.get("next_epoch")
        resume_status = (
            f"completed_epoch={metadata['completed_epoch']} | "
            f"next_epoch={next_epoch} | best_epoch={metadata['best_epoch']} | "
            f"best_macro_f1={metadata['best_macro_f1']:.4f}"
        )

    lines = [
        "",
        "STAGE 0B — MEMORY-SAFE PREPARATION ONLY",
        "Training started: NO",
        f"Train rows: {len(train_rows)}",
        "Train class counts:",
        f"  negative: {class_counts['negative']}",
        f"  neutral: {class_counts['neutral']}",
        f"  positive: {class_counts['positive']}",
        "Calculated balanced class weights in tensor order [negative, neutral, positive]:",
        *weight_lines,
        f"Label mapping: {LABEL2ID}",
        "Weight tensor order verified: [negative(id=0), neutral(id=1), positive(id=2)]",
        f"PyTorch version: {torch.__version__}",
        f"Transformers version: {transformers_version}",
        "Device requirement: mps (CPU training is blocked)",
        f"MPS currently available in this process: {torch.backends.mps.is_available()}",
        f"Physical train batch size: {TRAIN_BATCH_SIZE}",
        f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}",
        "Effective batch size: "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}",
        f"Eval batch size: {EVAL_BATCH_SIZE}",
        f"Learning rate: {LEARNING_RATE}",
        f"Max length: {MAX_LENGTH}",
        f"Total target epochs: {TOTAL_TARGET_EPOCHS}",
        "Optimizer: torch.optim.AdamW",
        "Scheduler: none",
        "Loss: weighted CrossEntropyLoss",
        "Gradient checkpointing: not used",
        (
            "Optimizer step schedule: every "
            f"{GRADIENT_ACCUMULATION_STEPS} micro-batches; a final partial "
            "group is stepped instead of discarded"
        ),
        "Controlled experiment note: Experiment 03 physical batch=4; "
        "Experiment 04 physical batch=1 with accumulation=4; effective batch=4",
        f"Resume checkpoint status: {resume_status}",
        f"Resume checkpoint path: {RESUME_PATH}",
        f"Best checkpoint path: {BEST_MODEL_DIR}",
        "Stage 0 complete. No model training was run.",
    ]
    append_log("\n".join(lines))


def run_prepare():
    """Sadece Stage 0 kontrollerini çalıştır; model yükleme veya training yapma."""
    ensure_report_header()
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    load_rows(DEV_PATH, expected_split="dev")
    class_counts, class_weights = calculate_class_weights(train_rows)
    config = experiment_config(class_weights)
    signature = config_signature(config)
    metadata = read_resume_metadata(signature)
    device = select_device()
    print_stage_zero_summary(
        train_rows,
        class_counts,
        class_weights,
        device,
        metadata,
    )


def run_one_epoch():
    """Checkpoint durumuna göre yalnızca sıradaki tek epoch'u çalıştır."""
    ensure_report_header()
    device = require_mps_training_device()
    if device is None:
        return

    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")
    class_counts, class_weights = calculate_class_weights(train_rows)
    config = experiment_config(class_weights)
    signature = config_signature(config)
    metadata = read_resume_metadata(signature)
    if metadata is not None and metadata["completed_epoch"] >= TOTAL_TARGET_EPOCHS:
        append_log(
            "All 3 target epochs are already complete. No training was started. "
            "Use --evaluate for final dev evaluation."
        )
        return

    completed_epoch = metadata["completed_epoch"] if metadata else 0
    next_epoch = completed_epoch + 1
    best_macro_f1 = metadata["best_macro_f1"] if metadata else float("-inf")
    best_epoch = metadata["best_epoch"] if metadata else None

    append_log("")
    append_log(f"TRAIN ONE EPOCH — epoch {next_epoch}/{TOTAL_TARGET_EPOCHS}")
    append_log(f"Physical train batch size: {TRAIN_BATCH_SIZE}")
    append_log(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    append_log(
        "Effective train batch size: "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )
    append_log(f"Eval batch size: {EVAL_BATCH_SIZE}")
    append_log(
        "Controlled experiment note: Experiment 03 physical batch=4; "
        "Experiment 04 physical batch=1 with accumulation=4; effective batch=4"
    )
    append_log(f"Label mapping: {LABEL2ID}")
    append_log(
        "Class weights [negative, neutral, positive]: "
        + str([round(float(weight), 8) for weight in class_weights])
    )
    append_log(f"Train class counts: {dict(class_counts)}")
    append_log("Oversampling / undersampling / SMOTE / focal loss: not used")

    append_log(f"Preparing tokenizer and model for epoch {next_epoch}")
    set_seed(RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    append_log("Tokenizer loaded")
    set_seed(RANDOM_SEED)
    append_log("Creating model, data loaders, and optimizer")
    (
        model,
        train_loader,
        dev_loader,
        optimizer,
        scheduler,
        train_generator,
    ) = create_training_objects(train_rows, dev_rows, tokenizer, device)
    append_log("Model, data loaders, and optimizer ready")

    if metadata is not None:
        checkpoint = load_resume_state(
            model,
            optimizer,
            scheduler,
            train_generator,
            device,
            signature,
        )
        if checkpoint["completed_epoch"] != completed_epoch:
            raise RuntimeError("Resume metadata ile training state epoch değeri uyuşmuyor.")
        append_log(f"Resumed from completed epoch: {completed_epoch}")

    weight_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    append_log(f"Entering training loop for epoch {next_epoch}")
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(train_loader, start=1):
        if step == 1:
            append_log("Starting first training batch")
        batch = move_batch(batch, device)
        labels = batch.pop("labels")
        outputs = model(**batch)
        loss = loss_function(outputs.logits, labels)
        group_start = (
            ((step - 1) // GRADIENT_ACCUMULATION_STEPS)
            * GRADIENT_ACCUMULATION_STEPS
        )
        accumulation_group_size = min(
            GRADIENT_ACCUMULATION_STEPS,
            len(train_loader) - group_start,
        )
        scaled_loss = loss / accumulation_group_size
        scaled_loss.backward()

        should_step = (
            step % GRADIENT_ACCUMULATION_STEPS == 0
            or step == len(train_loader)
        )
        if should_step:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()

        if step % LOG_EVERY_STEPS == 0 or step == len(train_loader):
            append_log(
                f"Epoch {next_epoch}/{TOTAL_TARGET_EPOCHS} | "
                f"step {step}/{len(train_loader)} | "
                f"running_train_loss={total_loss / step:.4f}"
            )

    labels, predictions = predict(model, dev_loader, device)
    metrics = overall_metrics(labels, predictions)
    metrics["train_loss"] = total_loss / len(train_loader)
    rng_state = capture_rng_state(train_generator, device)
    append_log(
        f"Epoch {next_epoch}/{TOTAL_TARGET_EPOCHS} complete | "
        f"train_loss={metrics['train_loss']:.4f} | "
        f"dev_accuracy={metrics['accuracy']:.4f} | "
        f"dev_macro_f1={metrics['macro_f1']:.4f}"
    )

    if metrics["macro_f1"] > best_macro_f1:
        best_macro_f1 = metrics["macro_f1"]
        best_epoch = next_epoch
        save_best_checkpoint(
            model,
            tokenizer,
            next_epoch,
            metrics,
            signature,
        )
        append_log(
            f"Best checkpoint updated: epoch={best_epoch} | "
            f"macro_f1={best_macro_f1:.4f}"
        )

    save_resume_checkpoint(
        model,
        optimizer,
        scheduler,
        device,
        next_epoch,
        best_epoch,
        best_macro_f1,
        metrics,
        signature,
        rng_state,
    )
    append_log(f"Resume checkpoint: {RESUME_PATH}")
    append_log(f"Best checkpoint: {BEST_MODEL_DIR}")
    append_log(f"Best epoch: {best_epoch}")
    append_log(f"Best Macro F1: {best_macro_f1:.4f}")
    append_log(f"Epoch {next_epoch} tamamlandı. Checkpoint kaydedildi.")
    if next_epoch < TOTAL_TARGET_EPOCHS:
        append_log(f"Epoch {next_epoch + 1} otomatik başlatılmadı.")
    else:
        append_log("Yeni epoch otomatik başlatılmadı.")


def group_metrics(rows, label_ids, prediction_ids, column, groups):
    """Language veya aspect grupları için accuracy ve Macro F1 hesapla."""
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


def comparison_line(name, vanilla, weighted):
    """Vanilla ve weighted XLM-R sonuçlarını farkıyla biçimlendir."""
    return (
        f"{name}: vanilla={vanilla:.4f} | weighted={weighted:.4f} | "
        f"difference={weighted - vanilla:+.4f}"
    )


def detailed_evaluation_report(dev_rows, label_ids, prediction_ids, best_metadata):
    """Best weighted checkpoint için ayrıntılı dev raporu oluştur."""
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
        dev_rows,
        label_ids,
        prediction_ids,
        "language",
        LANGUAGES,
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
        "FINAL DEV EVALUATION — BEST WEIGHTED CHECKPOINT",
        f"Best epoch: {best_metadata['best_epoch']}",
        "",
        "OVERALL",
        f"Accuracy: {overall['accuracy']:.4f}",
        f"Macro F1: {overall['macro_f1']:.4f}",
        "",
        "SENTIMENT",
    ]
    for sentiment in SENTIMENTS:
        metrics = class_metrics[sentiment]
        lines.append(
            f"{sentiment}: precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | "
            f"support={metrics['support']}"
        )
    lines.extend(
        [
            f"Neutral prediction count: {neutral_prediction_count}",
            "",
            "CONFUSION MATRIX",
            "Rows=actual, columns=predicted",
            "                 positive  neutral  negative",
        ]
    )
    for sentiment, row in zip(SENTIMENTS, matrix):
        lines.append(f"{sentiment:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}")

    lines.extend(["", "LANGUAGE"])
    for language, metrics in language_metrics.items():
        lines.append(
            f"{language}: n={metrics['n']} | accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f}"
        )

    lines.extend(["", "ASPECT"])
    for aspect, metrics in aspect_metrics.items():
        lines.append(
            f"{aspect}: n={metrics['n']} | accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f}"
        )

    lines.extend(
        [
            "",
            "EXPERIMENT 03 VANILLA VS EXPERIMENT 04 WEIGHTED",
            comparison_line("Accuracy", VANILLA_METRICS["accuracy"], overall["accuracy"]),
            comparison_line("Macro F1", VANILLA_METRICS["macro_f1"], overall["macro_f1"]),
        ]
    )
    for sentiment in SENTIMENTS:
        lines.append(
            comparison_line(
                f"{sentiment} F1",
                VANILLA_METRICS[f"{sentiment}_f1"],
                class_metrics[sentiment]["f1"],
            )
        )
    for aspect in ASPECTS:
        lines.append(
            comparison_line(
                f"{aspect} macro F1",
                VANILLA_METRICS[f"{aspect}_macro_f1"],
                aspect_metrics[aspect]["macro_f1"],
            )
        )
    return "\n".join(lines)


def run_evaluation():
    """Training yapmadan üç epoch sonrası best checkpoint'i değerlendir."""
    ensure_report_header()
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")
    _, class_weights = calculate_class_weights(train_rows)
    signature = config_signature(experiment_config(class_weights))
    metadata = read_resume_metadata(signature)
    if metadata is None or metadata["completed_epoch"] < TOTAL_TARGET_EPOCHS:
        completed = metadata["completed_epoch"] if metadata else 0
        raise RuntimeError(
            f"Final evaluation için 3 epoch tamamlanmalı. Completed epochs: {completed}."
        )
    if not BEST_MODEL_DIR.exists() or not BEST_METADATA_PATH.exists():
        raise RuntimeError("Best checkpoint bulunamadı.")

    best_metadata = json.loads(BEST_METADATA_PATH.read_text(encoding="utf-8"))
    if best_metadata.get("config_signature") != signature:
        raise RuntimeError("Best checkpoint mevcut deney ayarlarıyla uyuşmuyor.")

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_MODEL_DIR,
        local_files_only=True,
    )
    model.to(device)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dev_loader = DataLoader(
        SentimentDataset(dev_rows, tokenizer),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    labels, predictions = predict(model, dev_loader, device)
    append_log(detailed_evaluation_report(dev_rows, labels, predictions, best_metadata))
    append_log("Final evaluation complete. No training was run.")


def parse_args():
    """Training'in yanlışlıkla başlamaması için açık bir çalışma modu iste."""
    parser = argparse.ArgumentParser(
        description="Experiment 04 thermal-safe class-weighted XLM-R"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Stage 0 kontrollerini çalıştır; training başlatma.",
    )
    mode.add_argument(
        "--train-one-epoch",
        action="store_true",
        help="Resume durumuna göre yalnızca sıradaki tek epoch'u çalıştır.",
    )
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Üç epoch sonrası best checkpoint'i training yapmadan değerlendir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.prepare:
            run_prepare()
        elif args.train_one_epoch:
            run_one_epoch()
        else:
            run_evaluation()
    except RuntimeError as error:
        if "MPS backend out of memory" in str(error):
            append_log(
                "MPS out-of-memory: epoch tamamlanmadı; completed_epoch "
                "artırılmadı; checkpoint kaydedilmedi; CPU fallback yapılmadı."
            )
        append_log(f"Run failed: {type(error).__name__}: {error}")
        raise
    except Exception as error:
        append_log(f"Run failed: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
