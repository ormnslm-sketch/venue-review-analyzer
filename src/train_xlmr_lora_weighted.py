"""Experiment 05: thermal-safe XLM-R LoRA with weighted sentiment loss."""

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

# Experiment 05 training must remain on MPS. Unsupported operations must fail
# instead of silently falling back to CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    __version__ as peft_version,
    get_peft_model,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    __version__ as transformers_version,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "cleaned" / "train.csv"
DEV_PATH = PROJECT_ROOT / "data" / "cleaned" / "dev.csv"
REPORT_PATH = (
    PROJECT_ROOT / "data" / "analysis" / "experiment_05_xlmr_lora_weighted.txt"
)
MODEL_ROOT = PROJECT_ROOT / "models" / "xlmr-sentiment-v3-lora-weighted"
RESUME_DIR = MODEL_ROOT / "resume"
RESUME_POINTER_PATH = RESUME_DIR / "current.json"
RESUME_SLOT_NAMES = ("slot_a", "slot_b")
BEST_DIR = MODEL_ROOT / "best"
BEST_POINTER_PATH = BEST_DIR / "current.json"

MODEL_NAME = "xlm-roberta-base"
TOTAL_TARGET_EPOCHS = 3
LEARNING_RATE = 2e-5
EXPERIMENT_03_LEARNING_RATE = 2e-5
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_BATCH_SIZE = 1
RANDOM_SEED = 42
LOG_EVERY_STEPS = 100
MAX_OPTIMIZER_STEPS_PER_RUN = 50

LORA_TARGET_MODULES = ["query", "value"]
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_BIAS = "none"
MODULES_TO_SAVE = ["classifier"]

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


def append_log(text):
    """Metni terminale ve Experiment 05 raporuna ekle."""
    print(text, flush=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("a", encoding="utf-8") as report_file:
        report_file.write(text + "\n")


def ensure_report_header():
    """Raporu silmeden başlığı bir kez oluştur."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REPORT_PATH.exists():
        REPORT_PATH.write_text(
            "EXPERIMENT 05 — XLM-ROBERTA-BASE + LORA + WEIGHTED LOSS\n",
            encoding="utf-8",
        )


def set_seed(seed):
    """Python, NumPy ve PyTorch RNG'lerini sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_evaluation_device():
    """Sadece evaluation/raporlama için uygun cihazı seç."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_mps_training_device():
    """Training için MPS'i zorunlu tut; CPU fallback'e izin verme."""
    if not torch.backends.mps.is_available():
        append_log(MPS_UNAVAILABLE_MESSAGE)
        return None
    append_log("Training device: mps")
    return torch.device("mps")


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
    """Önceki deneylerle aynı aspect + sentence girdisini oluştur."""
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


def calculate_class_weights(train_rows):
    """Balanced weight'leri yalnızca cleaned train splitinden hesapla."""
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


def file_sha256(path):
    """Resume konfigürasyonunu veriye bağlamak için dosya hash'i hesapla."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_config(class_weights):
    """Resume sırasında değişmemesi gereken deney kimliğini oluştur."""
    lora_config = create_lora_config()
    return {
        "experiment": 5,
        "model_name": MODEL_NAME,
        "task": "sequence_classification",
        "total_target_epochs": TOTAL_TARGET_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "optimizer": "torch.optim.AdamW",
        "scheduler": None,
        "loss": "weighted CrossEntropyLoss",
        "label2id": LABEL2ID,
        "class_id_order": CLASS_ID_ORDER.tolist(),
        "class_weights": [float(weight) for weight in class_weights],
        "lora_target_modules": sorted(lora_config.target_modules),
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "lora_bias": LORA_BIAS,
        "modules_to_save": MODULES_TO_SAVE,
        "peft_version": peft_version,
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


def create_lora_config():
    """Sequence classification için açık LoRA konfigürasyonu oluştur."""
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


def find_attention_linear_modules(model):
    """Modelde gerçekten bulunan attention Linear katmanlarını listele."""
    return [
        (name, module.in_features, module.out_features)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and ".attention." in name
    ]


def verify_target_modules(model):
    """query/value hedeflerinin her encoder katmanında gerçekten bulunduğunu doğrula."""
    attention_modules = find_attention_linear_modules(model)
    layer_count = model.config.num_hidden_layers
    selected = {
        target: [
            name
            for name, _, _ in attention_modules
            if name.endswith(f".attention.self.{target}")
        ]
        for target in LORA_TARGET_MODULES
    }
    for target, names in selected.items():
        if len(names) != layer_count:
            raise RuntimeError(
                f"LoRA target '{target}' için {layer_count} katman bekleniyordu, "
                f"{len(names)} bulundu."
            )
    return attention_modules, selected


def parameter_summary(model):
    """LoRA modelinin total/trainable parametrelerini ve freeze durumunu doğrula."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    lora_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )
    classifier_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and "classifier.modules_to_save.default" in name
    )
    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and "lora_" not in name
        and "classifier.modules_to_save.default" not in name
    ]
    frozen_base = all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name and "modules_to_save.default" not in name
    )

    if not frozen_base:
        raise RuntimeError("Base model parametrelerinin tamamı frozen değil.")
    if classifier_trainable == 0:
        raise RuntimeError("Sequence-classification head eğitilebilir değil.")
    if unexpected_trainable:
        raise RuntimeError(
            "Beklenmeyen trainable parametreler: " + str(unexpected_trainable[:10])
        )
    if trainable != lora_trainable + classifier_trainable:
        raise RuntimeError("Trainable parametre hesabı tutarsız.")

    return {
        "total": total,
        "trainable": trainable,
        "percentage": 100.0 * trainable / total,
        "lora_trainable": lora_trainable,
        "classifier_trainable": classifier_trainable,
        "base_frozen": frozen_base,
        "classifier_trainable_verified": classifier_trainable > 0,
    }


def build_meta_model_for_inspection():
    """Ağırlıkları RAM'e almadan gerçek model sınıfı ve modül yapısını kur."""
    config = AutoConfig.from_pretrained(MODEL_NAME, local_files_only=True)
    config.num_labels = len(LABEL2ID)
    config.label2id = LABEL2ID
    config.id2label = ID2LABEL
    with torch.device("meta"):
        base_model = AutoModelForSequenceClassification.from_config(config)
        base_total = sum(parameter.numel() for parameter in base_model.parameters())
        attention_modules, selected_modules = verify_target_modules(base_model)
        peft_model = get_peft_model(base_model, create_lora_config())
    summary = parameter_summary(peft_model)
    summary["base_total"] = base_total
    return attention_modules, selected_modules, summary


def read_resume_metadata(expected_signature):
    """Atomik pointer'ın gösterdiği resume slotunu doğrula."""
    if not RESUME_POINTER_PATH.exists():
        return None

    pointer = json.loads(RESUME_POINTER_PATH.read_text(encoding="utf-8"))
    slot_name = pointer.get("active_slot")
    if slot_name not in RESUME_SLOT_NAMES:
        raise RuntimeError("Resume pointer geçersiz slot içeriyor.")
    slot_dir = RESUME_DIR / slot_name
    required_paths = (
        slot_dir / "adapter_config.json",
        slot_dir / "adapter_model.safetensors",
        slot_dir / "training_state.pt",
        slot_dir / "metadata.json",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Aktif resume slot eksik dosya içeriyor: {missing}")

    metadata = json.loads((slot_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("config_signature") != expected_signature:
        raise RuntimeError(
            "Resume checkpoint mevcut Experiment 05 ayarlarıyla uyuşmuyor."
        )
    if pointer.get("checkpoint_id") != metadata.get("checkpoint_id"):
        raise RuntimeError("Resume pointer ile aktif slot checkpoint ID uyuşmuyor.")
    metadata["_slot_name"] = slot_name
    metadata["_slot_dir"] = str(slot_dir)
    return metadata


def resume_status_text(metadata):
    """Stage 0 için kısa resume durumu üret."""
    if metadata is None:
        return "checkpoint yok; sıradaki epoch=1"
    best_value = metadata["best_macro_f1"]
    best_text = "None" if best_value is None else f"{best_value:.4f}"
    return (
        f"completed_epoch={metadata['completed_epoch']} | "
        f"current_epoch={metadata['current_epoch']} | "
        f"optimizer_step={metadata['optimizer_steps_in_epoch']} | "
        f"microbatch_position={metadata['microbatch_position']} | "
        f"best_epoch={metadata['best_epoch']} | "
        f"best_macro_f1={best_text}"
    )


def run_prepare():
    """Model training yapmadan Stage 0 yapısal doğrulamalarını çalıştır."""
    ensure_report_header()
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    load_rows(DEV_PATH, expected_split="dev")
    class_counts, class_weights = calculate_class_weights(train_rows)
    config = experiment_config(class_weights)
    signature = config_signature(config)
    metadata = read_resume_metadata(signature)
    attention_modules, selected_modules, params = build_meta_model_for_inspection()

    device_available = (
        "mps" if torch.backends.mps.is_available() else "mps unavailable in this process"
    )
    attention_names = [name for name, _, _ in attention_modules]
    lines = [
        "",
        "STAGE 0 — PREPARATION AND VALIDATION ONLY",
        "Training started: NO",
        f"Base model: {MODEL_NAME}",
        "Task: sequence classification",
        f"Device available: {device_available}",
        "Training device requirement: mps; CPU fallback is blocked",
        f"Transformers version: {transformers_version}",
        f"PEFT version: {peft_version}",
        f"LoRA target modules: {LORA_TARGET_MODULES}",
        (
            "Target reason: XLM-R named_modules içinde her encoder katmanında "
            "attention.self.query ve attention.self.value gerçek nn.Linear "
            "projeksiyonları olarak doğrulandı."
        ),
        f"Attention Linear modules found: {len(attention_names)}",
        f"query targets found: {len(selected_modules['query'])}",
        f"value targets found: {len(selected_modules['value'])}",
        f"Base model params before LoRA: {params['base_total']:,}",
        f"Total params after PEFT wrapping: {params['total']:,}",
        f"Trainable params: {params['trainable']:,}",
        f"Trainable %: {params['percentage']:.6f}%",
        f"  LoRA trainable params: {params['lora_trainable']:,}",
        f"  Classifier trainable params: {params['classifier_trainable']:,}",
        f"Base model frozen: {params['base_frozen']}",
        (
            "Sequence-classification head trainable: "
            f"{params['classifier_trainable_verified']}"
        ),
        f"Label mapping: {LABEL2ID}",
        "Weight tensor order: [negative, neutral, positive]",
        "Train class counts:",
        f"  negative: {class_counts['negative']}",
        f"  neutral: {class_counts['neutral']}",
        f"  positive: {class_counts['positive']}",
        "Class weights:",
        f"  negative (id=0): {class_weights[0]:.8f}",
        f"  neutral (id=1): {class_weights[1]:.8f}",
        f"  positive (id=2): {class_weights[2]:.8f}",
        f"Physical train batch size: {TRAIN_BATCH_SIZE}",
        f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}",
        (
            "Effective train batch size: "
            f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
        ),
        f"Eval batch size: {EVAL_BATCH_SIZE}",
        f"Learning rate: {LEARNING_RATE}",
        f"Max length: {MAX_LENGTH}",
        f"LoRA r: {LORA_R}",
        f"LoRA alpha: {LORA_ALPHA}",
        f"LoRA dropout: {LORA_DROPOUT}",
        "Optimizer: torch.optim.AdamW (trainable parameters only)",
        "Scheduler: none",
        "Loss: weighted CrossEntropyLoss",
        "Quantization: not used",
        "Mixed precision: not used",
        "Gradient checkpointing: not used",
        f"Resume status: {resume_status_text(metadata)}",
        f"Resume checkpoint: {RESUME_DIR}",
        f"Best checkpoint: {BEST_DIR}",
        "Stage 0 complete. No model training was run.",
    ]
    append_log("\n".join(lines))


def fp32_memory_mib(parameter_count):
    """Bir FP32 tensorunun yaklaşık belleğini MiB olarak hesapla."""
    return parameter_count * 4 / (1024**2)


def run_config_validation():
    """Stage 0B konfigürasyonunu ve yaklaşık FP32 belleği raporla."""
    ensure_report_header()
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    load_rows(DEV_PATH, expected_split="dev")
    class_counts, class_weights = calculate_class_weights(train_rows)
    config = experiment_config(class_weights)
    signature = config_signature(config)
    metadata = read_resume_metadata(signature)
    _, _, params = build_meta_model_for_inspection()
    lora_config = create_lora_config()

    learning_rate_matches = LEARNING_RATE == EXPERIMENT_03_LEARNING_RATE
    if not learning_rate_matches:
        learning_rate_status = (
            f"MISMATCH: Experiment 03={EXPERIMENT_03_LEARNING_RATE}, "
            f"Experiment 05={LEARNING_RATE}. No automatic change was made."
        )
    else:
        learning_rate_status = "MATCH: Experiment 03 and Experiment 05 both use 2e-5"

    trainable_component_mib = fp32_memory_mib(params["trainable"])
    frozen_base_mib = fp32_memory_mib(params["base_total"])
    lines = [
        "",
        "STAGE 0B — CONFIG VALIDATION ONLY",
        "Training started: NO",
        f"LoRA r (from LoraConfig): {lora_config.r}",
        f"LoRA alpha (from LoraConfig): {lora_config.lora_alpha}",
        f"LoRA dropout (from LoraConfig): {lora_config.lora_dropout}",
        f"LoRA target modules (from LoraConfig): {sorted(lora_config.target_modules)}",
        f"Learning rate: {LEARNING_RATE}",
        f"Experiment 03 learning-rate check: {learning_rate_status}",
        f"Physical train batch size: {TRAIN_BATCH_SIZE}",
        f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}",
        (
            "Effective batch size: "
            f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
        ),
        f"Eval batch size: {EVAL_BATCH_SIZE}",
        f"Max length: {MAX_LENGTH}",
        "Optimizer: torch.optim.AdamW (trainable parameters only)",
        f"Label mapping: {LABEL2ID}",
        "Class-weight tensor order: [negative, neutral, positive]",
        "Train class counts:",
        f"  negative: {class_counts['negative']}",
        f"  neutral: {class_counts['neutral']}",
        f"  positive: {class_counts['positive']}",
        "Class weights:",
        f"  negative (id=0): {class_weights[0]:.8f}",
        f"  neutral (id=1): {class_weights[1]:.8f}",
        f"  positive (id=2): {class_weights[2]:.8f}",
        f"Next epoch: {metadata['current_epoch'] if metadata else 1}",
        f"Resume checkpoint status: {resume_status_text(metadata)}",
        "",
        "APPROXIMATE FP32 MEMORY",
        "Assumption: 4 bytes per FP32 value. Activations and temporary buffers excluded.",
        (
            f"Trainable parameters ({params['trainable']:,}): "
            f"{trainable_component_mib:.2f} MiB"
        ),
        f"Gradients: {trainable_component_mib:.2f} MiB",
        f"AdamW first moment: {trainable_component_mib:.2f} MiB",
        f"AdamW second moment: {trainable_component_mib:.2f} MiB",
        (
            "Trainable params + gradients + two AdamW moments: "
            f"{4 * trainable_component_mib:.2f} MiB"
        ),
        (
            f"Frozen base weights ({params['base_total']:,} FP32 params): "
            f"{frozen_base_mib:.2f} MiB ({frozen_base_mib / 1024:.3f} GiB)"
        ),
        "These are approximate tensor-only values, not total MPS process memory.",
        "Stage 0B complete. No smoke test or model training was run.",
    ]
    append_log("\n".join(lines))


def run_chunk_validation():
    """Stage 0C chunk/resume matematiğini training başlatmadan doğrula."""
    ensure_report_header()
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    _, class_weights = calculate_class_weights(train_rows)
    signature = config_signature(experiment_config(class_weights))
    metadata = read_resume_metadata(signature)

    example_count = len(train_rows)
    micro_batches = micro_batches_per_epoch(example_count)
    optimizer_steps = optimizer_steps_per_epoch(example_count)
    chunks_per_epoch = math.ceil(optimizer_steps / MAX_OPTIMIZER_STEPS_PER_RUN)
    next_epoch = metadata["current_epoch"] if metadata else 1
    next_optimizer_step = metadata["optimizer_steps_in_epoch"] if metadata else 0
    next_microbatch = metadata["microbatch_position"] if metadata else 0
    if next_epoch is None:
        initial_permutation_hash = "n/a (all target epochs complete)"
    else:
        _, initial_permutation = epoch_permutation(example_count, next_epoch)
        initial_permutation_hash = permutation_sha256(initial_permutation)

    lines = [
        "",
        "STAGE 0C — CHUNKED TRAINING VALIDATION ONLY",
        "Training started: NO",
        f"Total training examples: {example_count}",
        f"Micro-batches per epoch: {micro_batches}",
        f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}",
        f"Optimizer steps per epoch: {optimizer_steps}",
        f"Max optimizer steps per run: {MAX_OPTIMIZER_STEPS_PER_RUN}",
        f"Estimated runs/chunks per epoch: {chunks_per_epoch}",
        f"Examples per full chunk: approximately {MAX_OPTIMIZER_STEPS_PER_RUN * GRADIENT_ACCUMULATION_STEPS * TRAIN_BATCH_SIZE}",
        f"Next epoch: {next_epoch}",
        f"Next optimizer step: {next_optimizer_step}",
        f"Next microbatch position: {next_microbatch}",
        f"Resume checkpoint status: {resume_status_text(metadata)}",
        "Deterministic shuffle strategy:",
        "  epoch_seed = random_seed + epoch_number - 1 (epoch 1 uses seed 42)",
        "  torch.randperm is created once for the whole epoch",
        "  the full permutation and SHA256 are stored in the resume state",
        "  resume slices the stored permutation from microbatch/sample position",
        "  checkpoints are written only at gradient-accumulation boundaries",
        f"Initial permutation SHA256 for epoch {next_epoch}: {initial_permutation_hash}",
        "Resume state fields:",
        "  adapter model (LoRA + classifier modules_to_save)",
        "  optimizer_state_dict",
        "  scheduler_state_dict=None",
        "  completed_epoch",
        "  current_epoch",
        "  optimizer_steps_in_epoch",
        "  microbatch_position",
        "  sample_position",
        "  micro_batches_accumulated=0",
        "  gradient_accumulation_steps",
        "  epoch_permutation",
        "  permutation_sha256",
        "  epoch_seed",
        "  epoch_loss_sum",
        "  epoch_microbatches_seen",
        "  Python/NumPy/Torch/MPS RNG states",
        "  best_macro_f1",
        "  best_epoch",
        "  last_epoch_metrics",
        "Atomic checkpoint strategy: alternating slot_a/slot_b; current.json pointer is updated last.",
        "Stage 0C complete. No smoke test, chunk, or epoch training was run.",
    ]
    append_log("\n".join(lines))


def create_data_loaders(train_rows, dev_rows, tokenizer):
    """Train/dev DataLoader'larını oluştur."""
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
    return train_loader, dev_loader, train_generator


def load_base_model():
    """Üç sınıflı XLM-R sequence-classification base modelini yükle."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )


def create_or_resume_peft_model(metadata):
    """Yeni LoRA modeli oluştur veya kaydedilmiş adapter'ı trainable yükle."""
    base_model = load_base_model()
    verify_target_modules(base_model)
    if metadata is None:
        model = get_peft_model(base_model, create_lora_config())
    else:
        model = PeftModel.from_pretrained(
            base_model,
            Path(metadata["_slot_dir"]),
            is_trainable=True,
        )
    parameter_summary(model)
    return model


def run_smoke_test():
    """MPS üzerinde yalnızca bir physical batch ve bir optimizer step çalıştır."""
    ensure_report_header()
    device = require_mps_training_device()
    if device is None:
        return

    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    class_counts, class_weights = calculate_class_weights(train_rows)
    lora_config = create_lora_config()
    model = None
    optimizer = None
    batch = None

    append_log("")
    append_log("SMOKE TEST — ONE PHYSICAL BATCH ONLY")
    append_log("Full epoch started: NO")
    append_log("Checkpoint writing: DISABLED")
    append_log(f"Physical batch size: {TRAIN_BATCH_SIZE}")
    append_log(f"Learning rate: {LEARNING_RATE}")
    append_log(
        "Class weights [negative, neutral, positive]: "
        + str([round(float(weight), 8) for weight in class_weights])
    )
    append_log(f"Train class counts: {dict(class_counts)}")
    append_log(
        "LoRA config: "
        f"r={lora_config.r}, alpha={lora_config.lora_alpha}, "
        f"dropout={lora_config.lora_dropout}, "
        f"targets={sorted(lora_config.target_modules)}"
    )

    try:
        set_seed(RANDOM_SEED)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base_model = load_base_model()
        verify_target_modules(base_model)
        model = get_peft_model(base_model, lora_config)
        parameter_summary(model)
        model.to(device)

        collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors="pt",
        )
        smoke_loader = DataLoader(
            SentimentDataset(train_rows, tokenizer),
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        batch = move_batch(next(iter(smoke_loader)), device)
        optimizer = torch.optim.AdamW(
            trainable_parameters(model),
            lr=LEARNING_RATE,
        )
        loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        labels = batch.pop("labels")
        outputs = model(**batch)
        loss = loss_function(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        append_log(f"Smoke-test weighted loss: {loss.item():.6f}")
        append_log("SMOKE TEST: PASSED")
        append_log(
            "Smoke-test state discarded. completed_epoch unchanged; "
            "no resume/best checkpoint created."
        )
    except RuntimeError as error:
        error_text = str(error).lower()
        if "mps" in error_text and "out of memory" in error_text:
            append_log("SMOKE TEST: FAILED — MPS OOM")
            append_log(
                "Smoke-test state discarded. completed_epoch unchanged; "
                "no resume/best checkpoint created; CPU fallback not used."
            )
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
                # Cleanup failure must not hide the smoke-test result.
                pass


def trainable_parameters(model):
    """Optimizer'a yalnızca requires_grad=True parametreleri ver."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def optimizer_parameter_audit(model, optimizer):
    """Optimizer'ın LoRA ve classifier trainable parametrelerini içerdiğini doğrula."""
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    lora_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    classifier_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "classifier.modules_to_save.default" in name
    ]
    frozen_parameters = [
        parameter for parameter in model.parameters() if not parameter.requires_grad
    ]
    return {
        "lora_count": sum(parameter.numel() for parameter in lora_parameters),
        "classifier_count": sum(
            parameter.numel() for parameter in classifier_parameters
        ),
        "optimizer_count": sum(
            parameter.numel()
            for group in optimizer.param_groups
            for parameter in group["params"]
        ),
        "contains_all_lora": all(
            id(parameter) in optimizer_parameter_ids for parameter in lora_parameters
        ),
        "contains_all_classifier": all(
            id(parameter) in optimizer_parameter_ids
            for parameter in classifier_parameters
        ),
        "contains_frozen": any(
            id(parameter) in optimizer_parameter_ids for parameter in frozen_parameters
        ),
    }


def gradient_group_stats(model, group):
    """Bir parametre grubundaki gradient tensorlarının norm özetini üret."""
    if group == "lora":
        selected = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
    elif group == "classifier":
        selected = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and "classifier.modules_to_save.default" in name
        ]
    elif group == "frozen":
        selected = [
            parameter for parameter in model.parameters() if not parameter.requires_grad
        ]
    else:
        raise ValueError(f"Bilinmeyen gradient grubu: {group}")

    gradients = [parameter.grad for parameter in selected if parameter.grad is not None]
    norms = [gradient.detach().float().norm().item() for gradient in gradients]
    return {
        "parameter_tensors": len(selected),
        "with_gradient": len(gradients),
        "mean_norm": float(np.mean(norms)) if norms else 0.0,
        "max_norm": float(np.max(norms)) if norms else 0.0,
    }


def weight_effect_sanity_check(class_weights):
    """Aynı logits için weight etkisini reduction=none ve batch-1 mean ile ölç."""
    logits = torch.tensor(
        [[0.20, 0.10, 0.70]] * len(CLASS_ID_ORDER),
        dtype=torch.float32,
    )
    targets = torch.tensor(CLASS_ID_ORDER, dtype=torch.long)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    unweighted_none = torch.nn.CrossEntropyLoss(reduction="none")(logits, targets)
    weighted_none = torch.nn.CrossEntropyLoss(
        weight=weight_tensor,
        reduction="none",
    )(logits, targets)

    results = []
    for label_id, label_name in enumerate(CLASS_NAMES_BY_ID):
        single_logits = logits[label_id : label_id + 1]
        single_target = targets[label_id : label_id + 1]
        unweighted_mean = torch.nn.CrossEntropyLoss()(single_logits, single_target)
        weighted_mean = torch.nn.CrossEntropyLoss(weight=weight_tensor)(
            single_logits,
            single_target,
        )
        results.append(
            {
                "label": label_name,
                "unweighted_none": unweighted_none[label_id].item(),
                "weighted_none": weighted_none[label_id].item(),
                "scale": (
                    weighted_none[label_id] / unweighted_none[label_id]
                ).item(),
                "unweighted_batch1_mean": unweighted_mean.item(),
                "weighted_batch1_mean": weighted_mean.item(),
            }
        )
    return results


def run_gradient_audit():
    """Tek MPS batch ile loss, optimizer üyeliği ve gradient akışını denetle."""
    ensure_report_header()
    device = require_mps_training_device()
    if device is None:
        return

    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    _, class_weights = calculate_class_weights(train_rows)
    signature = config_signature(experiment_config(class_weights))
    metadata = read_resume_metadata(signature)
    tokenizer = None
    model = None
    optimizer = None
    batch = None

    append_log("")
    append_log("GRADIENT AUDIT — ONE PHYSICAL BATCH ONLY")
    append_log("Training state mutation: DISABLED")
    append_log("Optimizer step: DISABLED")
    append_log("Checkpoint writing: DISABLED")
    append_log("Labels passed to HF model: False")
    append_log("Actual loss: criterion(outputs.logits, labels)")

    try:
        set_seed(RANDOM_SEED)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = create_or_resume_peft_model(metadata)
        model.to(device)
        optimizer = torch.optim.AdamW(
            trainable_parameters(model),
            lr=LEARNING_RATE,
        )
        optimizer_audit = optimizer_parameter_audit(model, optimizer)
        criterion_weights = torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
        loss_function = torch.nn.CrossEntropyLoss(weight=criterion_weights)

        append_log(f"Label mapping: {LABEL2ID}")
        append_log(
            "Criterion weights [negative, neutral, positive]: "
            + str([float(value) for value in loss_function.weight.cpu().tolist()])
        )
        append_log(f"LoRA trainable params: {optimizer_audit['lora_count']:,}")
        append_log(
            "Classifier trainable params: "
            f"{optimizer_audit['classifier_count']:,}"
        )
        append_log(
            "Optimizer trainable params: "
            f"{optimizer_audit['optimizer_count']:,}"
        )
        append_log(
            f"Optimizer contains all LoRA: {optimizer_audit['contains_all_lora']}"
        )
        append_log(
            "Optimizer contains all classifier: "
            f"{optimizer_audit['contains_all_classifier']}"
        )
        append_log(
            f"Optimizer contains frozen params: {optimizer_audit['contains_frozen']}"
        )

        append_log("Weight-effect sanity check (same logits):")
        for result in weight_effect_sanity_check(class_weights):
            append_log(
                f"  {result['label']}: reduction=none "
                f"unweighted={result['unweighted_none']:.6f}, "
                f"weighted={result['weighted_none']:.6f}, "
                f"ratio={result['scale']:.6f}; batch1 mean "
                f"unweighted={result['unweighted_batch1_mean']:.6f}, "
                f"weighted={result['weighted_batch1_mean']:.6f}"
            )

        collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors="pt",
        )
        audit_loader = DataLoader(
            SentimentDataset(train_rows, tokenizer),
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        batch = move_batch(next(iter(audit_loader)), device)
        labels = batch.pop("labels")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = loss_function(outputs.logits, labels)
        loss.backward()

        lora_gradients = gradient_group_stats(model, "lora")
        classifier_gradients = gradient_group_stats(model, "classifier")
        frozen_gradients = gradient_group_stats(model, "frozen")
        append_log(f"Diagnostic weighted loss: {loss.item():.6f}")
        append_log(
            "LoRA gradients: "
            f"{lora_gradients['with_gradient']}/"
            f"{lora_gradients['parameter_tensors']} tensors, "
            f"mean norm={lora_gradients['mean_norm']:.8f}, "
            f"max norm={lora_gradients['max_norm']:.8f}"
        )
        append_log(
            "Classifier gradients: "
            f"{classifier_gradients['with_gradient']}/"
            f"{classifier_gradients['parameter_tensors']} tensors, "
            f"mean norm={classifier_gradients['mean_norm']:.8f}, "
            f"max norm={classifier_gradients['max_norm']:.8f}"
        )
        append_log(
            "Frozen base gradients: "
            f"{frozen_gradients['with_gradient']}/"
            f"{frozen_gradients['parameter_tensors']} tensors"
        )
        append_log("GRADIENT AUDIT: COMPLETE")
        append_log(
            "Audit state discarded. No optimizer step, checkpoint, or progress "
            "update was performed."
        )
    except RuntimeError as error:
        error_text = str(error).lower()
        if "mps" in error_text and "out of memory" in error_text:
            append_log("GRADIENT AUDIT: FAILED — MPS OOM")
            append_log(
                "No optimizer step, checkpoint, or progress update was performed."
            )
            return
        raise
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        del batch
        del optimizer
        del model
        del tokenizer
        if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except RuntimeError:
                pass


def move_batch(batch, device):
    """Bir batch içindeki tensorları seçilen cihaza taşı."""
    return {name: value.to(device) for name, value in batch.items()}


def move_optimizer_state(optimizer, device):
    """Resume edilen optimizer tensorlarını seçilen cihaza taşı."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def capture_rng_state(train_generator, device):
    """Resume için Python, NumPy, Torch, MPS ve opsiyonel loader RNG state'i."""
    mps_state = None
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        mps_state = torch.mps.get_rng_state()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "mps": mps_state,
        "train_generator": (
            train_generator.get_state() if train_generator is not None else None
        ),
    }


def restore_rng_state(rng_state, train_generator, device):
    """Checkpoint RNG state'lerini eğitimden hemen önce geri yükle."""
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if train_generator is not None and rng_state.get("train_generator") is not None:
        train_generator.set_state(rng_state["train_generator"])
    if (
        device.type == "mps"
        and rng_state.get("mps") is not None
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(rng_state["mps"])


def load_training_state(metadata, optimizer, device, expected_signature):
    """Aktif atomik slottan optimizer, ilerleme ve RNG state'lerini yükle."""
    state_path = Path(metadata["_slot_dir"]) / "training_state.pt"
    checkpoint = torch.load(
        state_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("config_signature") != expected_signature:
        raise RuntimeError("Resume training state Experiment 05 ile uyuşmuyor.")
    if checkpoint.get("device_type") != device.type:
        raise RuntimeError("Resume checkpoint aynı device türüyle devam ettirilmelidir.")
    if checkpoint.get("checkpoint_id") != metadata.get("checkpoint_id"):
        raise RuntimeError("Resume metadata ile training state checkpoint ID uyuşmuyor.")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state(optimizer, device)
    restore_rng_state(checkpoint["rng_state"], None, device)
    return checkpoint


def micro_batches_per_epoch(example_count):
    """Physical batch sayısını hesapla."""
    return math.ceil(example_count / TRAIN_BATCH_SIZE)


def optimizer_steps_per_epoch(example_count):
    """Gradient accumulation sonrası optimizer step sayısını hesapla."""
    return math.ceil(micro_batches_per_epoch(example_count) / GRADIENT_ACCUMULATION_STEPS)


def epoch_permutation(example_count, epoch_number):
    """Her epoch için bağımsız ve tekrarlanabilir sample sırası üret."""
    epoch_seed = RANDOM_SEED + epoch_number - 1
    generator = torch.Generator().manual_seed(epoch_seed)
    permutation = torch.randperm(example_count, generator=generator).tolist()
    return epoch_seed, permutation


def permutation_sha256(permutation):
    """Checkpoint'teki sample sırasını doğrulamak için hash üret."""
    values = np.asarray(permutation, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def validate_epoch_progress(checkpoint, example_count):
    """Mid-epoch state'in sample atlamadığını veya tekrarlamadığını doğrula."""
    current_epoch = checkpoint["current_epoch"]
    if current_epoch is None:
        if checkpoint["completed_epoch"] != TOTAL_TARGET_EPOCHS:
            raise RuntimeError("current_epoch=None ancak üç epoch tamamlanmamış.")
        return

    permutation = checkpoint["epoch_permutation"]
    if len(permutation) != example_count or sorted(permutation) != list(
        range(example_count)
    ):
        raise RuntimeError("Checkpoint epoch permutation geçersiz.")
    if permutation_sha256(permutation) != checkpoint["permutation_sha256"]:
        raise RuntimeError("Checkpoint epoch permutation hash uyuşmuyor.")
    expected_seed, expected_permutation = epoch_permutation(example_count, current_epoch)
    if checkpoint["epoch_seed"] != expected_seed or permutation != expected_permutation:
        raise RuntimeError("Checkpoint deterministic epoch sırasıyla uyuşmuyor.")
    if checkpoint["micro_batches_accumulated"] != 0:
        raise RuntimeError("Checkpoint gradient-accumulation sınırında değil.")

    total_micro_batches = micro_batches_per_epoch(example_count)
    position = checkpoint["microbatch_position"]
    sample_position = checkpoint["sample_position"]
    optimizer_step = checkpoint["optimizer_steps_in_epoch"]
    if not 0 <= position <= total_micro_batches:
        raise RuntimeError("Checkpoint microbatch position geçersiz.")
    if sample_position != min(position * TRAIN_BATCH_SIZE, example_count):
        raise RuntimeError("Checkpoint sample position ile microbatch position uyuşmuyor.")
    expected_steps = math.ceil(position / GRADIENT_ACCUMULATION_STEPS) if position else 0
    if optimizer_step != expected_steps:
        raise RuntimeError("Checkpoint optimizer step ile microbatch position uyuşmuyor.")


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
    """Dev accuracy ve Macro F1'i hesapla."""
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


def save_adapter_checkpoint(model, tokenizer, directory):
    """Base model yerine küçük LoRA adapter ve classifier head'i kaydet."""
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)


def save_best_checkpoint(model, tokenizer, epoch, metrics, signature):
    """Best adapter'ı ayrı dizine yaz ve pointer'ı en son güncelle."""
    checkpoint_id = f"epoch_{epoch:03d}_{os.urandom(4).hex()}"
    checkpoint_dir = BEST_DIR / checkpoint_id
    save_adapter_checkpoint(model, tokenizer, checkpoint_dir)
    write_json_atomic(
        checkpoint_dir / "metadata.json",
        {
            "checkpoint_id": checkpoint_id,
            "best_epoch": epoch,
            "best_macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "config_signature": signature,
        },
    )
    write_json_atomic(
        BEST_POINTER_PATH,
        {"checkpoint_id": checkpoint_id, "relative_path": checkpoint_id},
    )
    return checkpoint_dir


def save_resume_checkpoint(
    model,
    tokenizer,
    optimizer,
    device,
    completed_epoch,
    current_epoch,
    optimizer_steps_in_epoch,
    microbatch_position,
    epoch_permutation_values,
    epoch_seed,
    epoch_loss_sum,
    epoch_microbatches_seen,
    best_epoch,
    best_macro_f1,
    last_epoch_metrics,
    signature,
    rng_state,
):
    """İnaktif slota tam state yazıp pointer'ı atomik olarak değiştir."""
    active_slot = None
    if RESUME_POINTER_PATH.exists():
        pointer = json.loads(RESUME_POINTER_PATH.read_text(encoding="utf-8"))
        active_slot = pointer.get("active_slot")
    slot_name = "slot_b" if active_slot == "slot_a" else "slot_a"
    slot_dir = RESUME_DIR / slot_name
    if slot_dir.exists():
        shutil.rmtree(slot_dir)
    slot_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_id = (
        f"epoch_{current_epoch if current_epoch is not None else 'done'}_"
        f"step_{optimizer_steps_in_epoch}_{os.urandom(4).hex()}"
    )
    save_adapter_checkpoint(model, tokenizer, slot_dir)
    permutation_hash = permutation_sha256(epoch_permutation_values)
    checkpoint = {
        "format_version": 2,
        "checkpoint_id": checkpoint_id,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "completed_epoch": completed_epoch,
        "current_epoch": current_epoch,
        "optimizer_steps_in_epoch": optimizer_steps_in_epoch,
        "microbatch_position": microbatch_position,
        "sample_position": min(
            microbatch_position * TRAIN_BATCH_SIZE,
            len(epoch_permutation_values),
        ),
        "micro_batches_accumulated": 0,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "epoch_permutation": epoch_permutation_values,
        "permutation_sha256": permutation_hash,
        "epoch_seed": epoch_seed,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_microbatches_seen": epoch_microbatches_seen,
        "best_macro_f1": best_macro_f1,
        "best_epoch": best_epoch,
        "last_epoch_metrics": last_epoch_metrics,
        "device_type": device.type,
        "config_signature": signature,
        "rng_state": rng_state,
    }
    state_path = slot_dir / "training_state.pt"
    temporary_path = slot_dir / "training_state.pt.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, state_path)
    write_json_atomic(
        slot_dir / "metadata.json",
        {
            "format_version": 2,
            "checkpoint_id": checkpoint_id,
            "completed_epoch": completed_epoch,
            "current_epoch": current_epoch,
            "optimizer_steps_in_epoch": optimizer_steps_in_epoch,
            "microbatch_position": microbatch_position,
            "sample_position": checkpoint["sample_position"],
            "micro_batches_accumulated": 0,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "permutation_sha256": permutation_hash,
            "epoch_seed": epoch_seed,
            "best_macro_f1": best_macro_f1,
            "best_epoch": best_epoch,
            "last_epoch_metrics": last_epoch_metrics,
            "device_type": device.type,
            "config_signature": signature,
            "scheduler": None,
        },
    )
    write_json_atomic(
        RESUME_POINTER_PATH,
        {"active_slot": slot_name, "checkpoint_id": checkpoint_id},
    )
    return slot_dir


def run_train_chunk():
    """En fazla MAX_OPTIMIZER_STEPS_PER_RUN kadar güvenli training çalıştır."""
    ensure_report_header()
    device = require_mps_training_device()
    if device is None:
        return

    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")
    _, class_weights = calculate_class_weights(train_rows)
    signature = config_signature(experiment_config(class_weights))
    metadata = read_resume_metadata(signature)
    if metadata is not None and metadata["completed_epoch"] >= TOTAL_TARGET_EPOCHS:
        append_log(
            "All 3 target epochs are complete. No training was started. "
            "Use --evaluate."
        )
        return

    completed_epoch = metadata["completed_epoch"] if metadata else 0
    current_epoch = metadata["current_epoch"] if metadata else 1
    best_macro_f1 = metadata["best_macro_f1"] if metadata else None
    best_epoch = metadata["best_epoch"] if metadata else None
    last_epoch_metrics = metadata["last_epoch_metrics"] if metadata else None

    set_seed(RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    set_seed(RANDOM_SEED)
    model = create_or_resume_peft_model(metadata)
    model.to(device)
    _, dev_loader, _ = create_data_loaders(train_rows, dev_rows, tokenizer)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=LEARNING_RATE,
    )

    if metadata is not None:
        checkpoint = load_training_state(metadata, optimizer, device, signature)
        validate_epoch_progress(checkpoint, len(train_rows))
        current_epoch = checkpoint["current_epoch"]
        optimizer_step = checkpoint["optimizer_steps_in_epoch"]
        microbatch_position = checkpoint["microbatch_position"]
        permutation = checkpoint["epoch_permutation"]
        epoch_seed = checkpoint["epoch_seed"]
        epoch_loss_sum = checkpoint["epoch_loss_sum"]
        epoch_microbatches_seen = checkpoint["epoch_microbatches_seen"]
    else:
        optimizer_step = 0
        microbatch_position = 0
        epoch_seed, permutation = epoch_permutation(len(train_rows), current_epoch)
        epoch_loss_sum = 0.0
        epoch_microbatches_seen = 0

    total_micro_batches = micro_batches_per_epoch(len(train_rows))
    total_optimizer_steps = optimizer_steps_per_epoch(len(train_rows))
    remaining_optimizer_steps = total_optimizer_steps - optimizer_step
    target_optimizer_step = min(
        optimizer_step + MAX_OPTIMIZER_STEPS_PER_RUN,
        total_optimizer_steps,
    )
    start_optimizer_step = optimizer_step

    append_log("")
    append_log("TRAINING CHUNK START")
    append_log(f"Device: {device.type}")
    append_log(f"Epoch: {current_epoch}/{TOTAL_TARGET_EPOCHS}")
    append_log(f"Starting optimizer step: {start_optimizer_step}")
    append_log(f"Target optimizer step: {target_optimizer_step}")
    append_log(f"Remaining optimizer steps in epoch: {remaining_optimizer_steps}")
    append_log(f"Max optimizer steps this run: {MAX_OPTIMIZER_STEPS_PER_RUN}")
    append_log(f"Starting microbatch position: {microbatch_position}")
    append_log(f"Deterministic epoch seed: {epoch_seed}")
    append_log(f"Permutation SHA256: {permutation_sha256(permutation)}")

    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    end_microbatch_position = min(
        target_optimizer_step * GRADIENT_ACCUMULATION_STEPS,
        total_micro_batches,
    )
    start_sample = microbatch_position * TRAIN_BATCH_SIZE
    end_sample = min(end_microbatch_position * TRAIN_BATCH_SIZE, len(train_rows))
    chunk_indices = permutation[start_sample:end_sample]
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    chunk_loader = DataLoader(
        Subset(SentimentDataset(train_rows, tokenizer), chunk_indices),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    chunk_iterator = iter(chunk_loader)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps_completed_run = 0
    stop_requested = {"value": False}
    boundary_rng_state = capture_rng_state(None, device)
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def request_safe_stop(_signal_number, _frame):
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, request_safe_stop)
    try:
        while optimizer_step < target_optimizer_step:
            boundary_rng_state = capture_rng_state(None, device)
            group_start_position = microbatch_position
            group_size = min(
                GRADIENT_ACCUMULATION_STEPS,
                total_micro_batches - group_start_position,
            )
            group_loss_sum = 0.0
            for _ in range(group_size):
                batch = move_batch(next(chunk_iterator), device)
                labels = batch.pop("labels")
                outputs = model(**batch)
                loss = loss_function(outputs.logits, labels)
                (loss / group_size).backward()
                group_loss_sum += loss.item()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            optimizer_steps_completed_run += 1
            microbatch_position += group_size
            epoch_loss_sum += group_loss_sum
            epoch_microbatches_seen += group_size

            if stop_requested["value"]:
                append_log(
                    "Ctrl+C received. Stopping at the completed optimizer-step boundary."
                )
                break
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        restore_rng_state(boundary_rng_state, None, device)
        append_log(
            "KeyboardInterrupt caught. Partial gradient group was discarded; "
            "progress remains at the last completed optimizer step."
        )
    except RuntimeError as error:
        error_text = str(error).lower()
        optimizer.zero_grad(set_to_none=True)
        if "mps" in error_text and "out of memory" in error_text:
            append_log(
                "TRAINING CHUNK FAILED — MPS OOM. No new resume/best checkpoint "
                "was written; the previous checkpoint remains authoritative; "
                "CPU fallback was not used."
            )
            return
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    epoch_finished = microbatch_position >= total_micro_batches
    dev_metrics = None
    finished_epoch_step = optimizer_step
    finished_epoch_number = current_epoch
    if epoch_finished:
        labels, predictions = predict(model, dev_loader, device)
        dev_metrics = overall_metrics(labels, predictions)
        dev_metrics["train_loss"] = epoch_loss_sum / epoch_microbatches_seen
        last_epoch_metrics = dev_metrics
        if best_macro_f1 is None or dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = current_epoch
            save_best_checkpoint(
                model,
                tokenizer,
                current_epoch,
                dev_metrics,
                signature,
            )
        completed_epoch = current_epoch
        if completed_epoch < TOTAL_TARGET_EPOCHS:
            current_epoch = completed_epoch + 1
            optimizer_step = 0
            microbatch_position = 0
            epoch_seed, permutation = epoch_permutation(
                len(train_rows), current_epoch
            )
            epoch_loss_sum = 0.0
            epoch_microbatches_seen = 0
        else:
            current_epoch = None
            optimizer_step = 0
            microbatch_position = 0
            epoch_seed = None
            permutation = []
            epoch_loss_sum = 0.0
            epoch_microbatches_seen = 0

    checkpoint_path = save_resume_checkpoint(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        completed_epoch=completed_epoch,
        current_epoch=current_epoch,
        optimizer_steps_in_epoch=optimizer_step,
        microbatch_position=microbatch_position,
        epoch_permutation_values=permutation,
        epoch_seed=epoch_seed,
        epoch_loss_sum=epoch_loss_sum,
        epoch_microbatches_seen=epoch_microbatches_seen,
        best_epoch=best_epoch,
        best_macro_f1=best_macro_f1,
        last_epoch_metrics=last_epoch_metrics,
        signature=signature,
        rng_state=capture_rng_state(None, device),
    )

    report_step = finished_epoch_step if epoch_finished else optimizer_step
    progress = 100.0 * report_step / total_optimizer_steps
    append_log("")
    append_log("TRAINING CHUNK RESULT")
    append_log(f"Epoch: {finished_epoch_number}/{TOTAL_TARGET_EPOCHS}")
    append_log(f"Optimizer steps completed this run: {optimizer_steps_completed_run}")
    append_log(f"Current optimizer step: {report_step}")
    append_log(f"Epoch progress percentage: {progress:.2f}%")
    append_log(f"Resume checkpoint saved: {checkpoint_path}")
    append_log(f"Epoch finished: {epoch_finished}")
    if dev_metrics is not None:
        append_log(f"Dev Accuracy: {dev_metrics['accuracy']:.4f}")
        append_log(f"Dev Macro F1: {dev_metrics['macro_f1']:.4f}")
        append_log(f"Best epoch: {best_epoch}")
        append_log(f"Best Macro F1: {best_macro_f1:.4f}")
    append_log("Training chunk complete.")
    append_log("No additional chunk was started automatically.")


def group_metrics(rows, labels, predictions, column, groups):
    """Language veya aspect grupları için accuracy ve Macro F1 hesapla."""
    results = {}
    for value, display_name in groups:
        indices = [index for index, row in enumerate(rows) if row[column] == value]
        group_labels = [labels[index] for index in indices]
        group_predictions = [predictions[index] for index in indices]
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


def detailed_evaluation_report(dev_rows, labels, predictions, best_metadata):
    """Best adapter için ayrıntılı cleaned dev raporu oluştur."""
    overall = overall_metrics(labels, predictions)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=LABEL_IDS,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=LABEL_IDS)
    language_metrics = group_metrics(
        dev_rows,
        labels,
        predictions,
        "language",
        LANGUAGES,
    )
    aspect_metrics = group_metrics(
        dev_rows,
        labels,
        predictions,
        "aspect",
        [(aspect, aspect) for aspect in ASPECTS],
    )

    lines = [
        "",
        "FINAL DEV EVALUATION — BEST LORA ADAPTER",
        f"Best epoch: {best_metadata['best_epoch']}",
        f"Accuracy: {overall['accuracy']:.4f}",
        f"Macro F1: {overall['macro_f1']:.4f}",
        "",
        "SENTIMENT",
    ]
    for index, sentiment in enumerate(SENTIMENTS):
        lines.append(
            f"{sentiment}: precision={precision[index]:.4f} | "
            f"recall={recall[index]:.4f} | F1={class_f1[index]:.4f} | "
            f"support={int(support[index])}"
        )
    lines.extend(
        [
            f"Neutral prediction count: {predictions.count(LABEL2ID['neutral'])}",
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
            f"{language}: accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f} | n={metrics['n']}"
        )
    lines.extend(["", "ASPECT"])
    for aspect, metrics in aspect_metrics.items():
        lines.append(
            f"{aspect}: accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f} | n={metrics['n']}"
        )
    return "\n".join(lines)


def epoch_one_diagnostic_report(dev_rows, labels, predictions, best_metadata, device):
    """Current best adapter için ayrıntılı, training içermeyen dev analizi."""
    overall = overall_metrics(labels, predictions)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=LABEL_IDS,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=LABEL_IDS)
    predicted_counts = Counter(predictions)
    language_metrics = group_metrics(
        dev_rows,
        labels,
        predictions,
        "language",
        LANGUAGES,
    )
    aspect_metrics = group_metrics(
        dev_rows,
        labels,
        predictions,
        "aspect",
        [(aspect, aspect) for aspect in ASPECTS],
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
    positive_to_neutral = int(matrix[0, 1])
    negative_to_neutral = int(matrix[2, 1])
    neutral_predictions = predicted_counts[LABEL2ID["neutral"]]

    lines = [
        "",
        "EPOCH 1 BEST CHECKPOINT — DEV DIAGNOSTIC EVALUATION",
        "Training/backward/optimizer step: NOT RUN",
        f"Evaluation device: {device.type}",
        f"Best epoch: {best_metadata['best_epoch']}",
        f"Accuracy: {overall['accuracy']:.4f}",
        f"Macro F1: {overall['macro_f1']:.4f}",
        "",
        "SENTIMENT METRICS",
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
            "",
            "PREDICTION COUNTS",
            f"predicted positive: {predicted_counts[LABEL2ID['positive']]}",
            f"predicted neutral: {neutral_predictions}",
            f"predicted negative: {predicted_counts[LABEL2ID['negative']]}",
            "",
            "CONFUSION MATRIX",
            "Order: positive / neutral / negative",
            "Rows=actual, columns=predicted",
            "                 positive  neutral  negative",
        ]
    )
    for sentiment, row in zip(SENTIMENTS, matrix):
        lines.append(f"{sentiment:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}")
    lines.extend(
        [
            "",
            "NEUTRAL BEHAVIOR",
            f"Actual neutral count: {class_metrics['neutral']['support']}",
            f"Predicted neutral count: {neutral_predictions}",
            f"Actual positive predicted as neutral: {positive_to_neutral}",
            f"Actual negative predicted as neutral: {negative_to_neutral}",
            "",
            "ASPECT METRICS",
        ]
    )
    for aspect, metrics in aspect_metrics.items():
        lines.append(
            f"{aspect}: accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f}"
        )
    lines.extend(["", "LANGUAGE METRICS"])
    for language, metrics in language_metrics.items():
        lines.append(
            f"{language}: accuracy={metrics['accuracy']:.4f} | "
            f"macro F1={metrics['macro_f1']:.4f}"
        )

    lines.extend(
        [
            "",
            "COMPARISON",
            "TF-IDF + Logistic Regression: accuracy=0.7603 | macro F1=0.5891 | neutral F1=0.2529",
            "Vanilla XLM-R best: accuracy=0.8457 | macro F1=0.5651 | positive F1=0.9134 | neutral F1=0.0000 | negative F1=0.7820",
            (
                "Experiment 05 LoRA weighted Epoch 1: "
                f"accuracy={overall['accuracy']:.4f} | "
                f"macro F1={overall['macro_f1']:.4f} | "
                f"positive F1={class_metrics['positive']['f1']:.4f} | "
                f"neutral F1={class_metrics['neutral']['f1']:.4f} | "
                f"negative F1={class_metrics['negative']['f1']:.4f}"
            ),
            "Evaluation complete. No training was run. Test split was not used.",
        ]
    )
    return "\n".join(lines)


def run_best_evaluation():
    """Mevcut best adapter'ı tamamlanmış epoch sayısından bağımsız değerlendir."""
    ensure_report_header()
    train_rows = load_rows(TRAIN_PATH, expected_split="train")
    dev_rows = load_rows(DEV_PATH, expected_split="dev")
    _, class_weights = calculate_class_weights(train_rows)
    signature = config_signature(experiment_config(class_weights))
    if not BEST_POINTER_PATH.exists():
        raise RuntimeError("Best adapter checkpoint bulunamadı.")

    best_pointer = json.loads(BEST_POINTER_PATH.read_text(encoding="utf-8"))
    best_dir = BEST_DIR / best_pointer["relative_path"]
    best_metadata = json.loads(
        (best_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if best_metadata.get("checkpoint_id") != best_pointer.get("checkpoint_id"):
        raise RuntimeError("Best adapter pointer ve metadata uyuşmuyor.")
    if best_metadata.get("config_signature") != signature:
        raise RuntimeError("Best adapter mevcut Experiment 05 ayarlarıyla uyuşmuyor.")

    device = select_evaluation_device()
    tokenizer = AutoTokenizer.from_pretrained(best_dir, local_files_only=True)
    base_model = load_base_model()
    model = PeftModel.from_pretrained(base_model, best_dir, is_trainable=False)
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Evaluation modelinde beklenmeyen trainable parametre var.")
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
    append_log(
        epoch_one_diagnostic_report(
            dev_rows,
            labels,
            predictions,
            best_metadata,
            device,
        )
    )


def run_evaluation():
    """Training yapmadan üç epoch sonrası best adapter'ı değerlendir."""
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
    if not BEST_POINTER_PATH.exists():
        raise RuntimeError("Best adapter checkpoint bulunamadı.")

    best_pointer = json.loads(BEST_POINTER_PATH.read_text(encoding="utf-8"))
    best_dir = BEST_DIR / best_pointer["relative_path"]
    best_metadata_path = best_dir / "metadata.json"
    if not best_metadata_path.exists():
        raise RuntimeError("Best adapter metadata bulunamadı.")
    best_metadata = json.loads(best_metadata_path.read_text(encoding="utf-8"))
    if best_metadata.get("checkpoint_id") != best_pointer.get("checkpoint_id"):
        raise RuntimeError("Best adapter pointer ve metadata uyuşmuyor.")
    if best_metadata.get("config_signature") != signature:
        raise RuntimeError("Best adapter mevcut Experiment 05 ayarlarıyla uyuşmuyor.")

    device = select_evaluation_device()
    tokenizer = AutoTokenizer.from_pretrained(best_dir, local_files_only=True)
    base_model = load_base_model()
    model = PeftModel.from_pretrained(base_model, best_dir, is_trainable=False)
    model.to(device)
    _, dev_loader, _ = create_data_loaders(train_rows, dev_rows, tokenizer)
    labels, predictions = predict(model, dev_loader, device)
    append_log(detailed_evaluation_report(dev_rows, labels, predictions, best_metadata))
    append_log("Final evaluation complete. No training was run.")


def parse_args():
    """Yanlışlıkla training başlamaması için açık çalışma modu iste."""
    parser = argparse.ArgumentParser(
        description="Experiment 05 thermal-safe weighted XLM-R LoRA"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Stage 0 kontrollerini çalıştır; training başlatma.",
    )
    mode.add_argument(
        "--validate-config",
        action="store_true",
        help="Stage 0B config ve FP32 bellek hesabını göster; training başlatma.",
    )
    mode.add_argument(
        "--validate-chunks",
        action="store_true",
        help="Stage 0C chunk/resume matematiğini doğrula; training başlatma.",
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="MPS üzerinde yalnızca bir batch ve bir optimizer step çalıştır.",
    )
    mode.add_argument(
        "--gradient-audit",
        action="store_true",
        help=(
            "MPS üzerinde tek batch forward/backward ile gradient akışını "
            "denetle; optimizer step ve checkpoint oluşturma."
        ),
    )
    mode.add_argument(
        "--train-chunk",
        action="store_true",
        help=(
            "Resume durumuna göre en fazla "
            f"{MAX_OPTIMIZER_STEPS_PER_RUN} optimizer step çalıştır."
        ),
    )
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Üç epoch sonrası best adapter'ı training yapmadan değerlendir.",
    )
    mode.add_argument(
        "--evaluate-best",
        action="store_true",
        help="Mevcut best adapter'ı training yapmadan cleaned dev üzerinde değerlendir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.prepare:
            run_prepare()
        elif args.validate_config:
            run_config_validation()
        elif args.validate_chunks:
            run_chunk_validation()
        elif args.smoke_test:
            run_smoke_test()
        elif args.gradient_audit:
            run_gradient_audit()
        elif args.train_chunk:
            run_train_chunk()
        elif args.evaluate:
            run_evaluation()
        elif args.evaluate_best:
            run_best_evaluation()
    except RuntimeError as error:
        error_text = str(error).lower()
        if "mps" in error_text and "out of memory" in error_text:
            append_log(
                "MPS out-of-memory: epoch tamamlanmadı; completed_epoch "
                "artırılmadı; adapter/checkpoint kaydedilmedi; CPU fallback yapılmadı."
            )
        append_log(f"Run failed: {type(error).__name__}: {error}")
        raise
    except Exception as error:
        append_log(f"Run failed: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
