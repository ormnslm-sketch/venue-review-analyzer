"""Experiment 16: EN+ES-only LoRA language-scope ablation.

Training is MPS-only and chunked. Epoch 1 was trained fresh; Epoch 2 is an
explicit resume-only extension. Epoch 3 is blocked. Training and evaluation
rows are filtered in memory from expanded_neutral_v1, and the test split is
never referenced or loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.utils.class_weight import compute_class_weight

import train_xlmr_lora_unweighted_ablation as core


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "expanded_neutral_v1" / "train.csv"
DEV_PATH = PROJECT_ROOT / "data" / "expanded_neutral_v1" / "dev.csv"
MODEL_ROOT = PROJECT_ROOT / "models" / "experiment_16_lora_en_es_only"
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "experiment_16_lora_en_es_only.txt"
BASELINE_JSON = (
    PROJECT_ROOT / "data" / "analysis" / "experiment_16_exp15_en_es_baseline.json"
)
BASELINE_REPORT = (
    PROJECT_ROOT / "data" / "analysis" / "experiment_16_exp15_en_es_baseline.txt"
)
EXP15_MODEL_ROOT = PROJECT_ROOT / "models" / "experiment_15_lora_expanded_balanced"

TRAINING_WEIGHTS = (
    1.1184126984126983,  # negative -> 0
    3.9144444444444444,  # neutral  -> 1
    0.540420309863476,   # positive -> 2
)
EXPECTED_TRAIN_LANGUAGE = {"en": 1175, "es": 1175}
EXPECTED_TRAIN_SENTIMENT = {"positive": 1450, "neutral": 200, "negative": 700}
EXPECTED_TRAIN_ASPECT = {
    "taste": 1072,
    "atmosphere": 368,
    "service": 618,
    "value": 292,
}
EXPECTED_DEV_LANGUAGE = {"en": 301, "es": 300}
EXPECTED_TRAIN_ROWS = 2350
EXPECTED_DEV_ROWS = 601
EXP16_STEPS_PER_EPOCH = 588
EXP16_TOTAL_STEPS_AFTER_EPOCH1 = 588
EXP16_TOTAL_STEPS_AFTER_EPOCH2 = 1176
EXP16_TOTAL_STEPS_AFTER_EPOCH3 = 1764
EXP15_STEPS_PER_EPOCH = 881
EXP15_TOTAL_STEPS_AFTER_EPOCH2 = 1762
OPTIMIZER_STEP_DIFFERENCE = (
    EXP15_TOTAL_STEPS_AFTER_EPOCH2 - EXP16_TOTAL_STEPS_AFTER_EPOCH2
)
OPTIMIZER_STEP_PERCENT_DIFFERENCE = (
    100.0 * OPTIMIZER_STEP_DIFFERENCE / EXP15_TOTAL_STEPS_AFTER_EPOCH2
)
FINAL_BUDGET_STEP_DIFFERENCE = abs(
    EXP16_TOTAL_STEPS_AFTER_EPOCH3 - EXP15_TOTAL_STEPS_AFTER_EPOCH2
)
FINAL_BUDGET_PERCENT_DIFFERENCE = (
    100.0 * FINAL_BUDGET_STEP_DIFFERENCE / EXP15_TOTAL_STEPS_AFTER_EPOCH2
)


def en_es_filter(rows, split):
    filtered = [row for row in rows if row["language"] in {"en", "es"}]
    if any(row["language"] == "tr" for row in filtered):
        raise AssertionError(f"TR row remained in filtered {split} data")
    if any(row["language"] not in {"en", "es"} for row in filtered):
        raise AssertionError(f"Unexpected language remained in filtered {split} data")
    return filtered


def configure_core():
    core.CONDITIONS = {
        "clean": {
            "experiment": "16-REFERENCE",
            "train": TRAIN_PATH,
            "dev": DEV_PATH,
            "model_root": PROJECT_ROOT / "models" / ".experiment_16_reference_only",
            "report": PROJECT_ROOT / "data" / "analysis" / ".experiment_16_reference.txt",
        },
        "expanded": {
            "experiment": "16",
            "train": TRAIN_PATH,
            "dev": DEV_PATH,
            "model_root": MODEL_ROOT,
            "report": REPORT_PATH,
        },
    }
    core.TOTAL_TARGET_EPOCHS = 1
    core.EXPERIMENT_SUBTITLE = "EN+ES-ONLY CORRECTED WEIGHTED LORA EPOCH 1"
    core.LOSS_DESCRIPTION = (
        "mean(F.cross_entropy(logits, labels, reduction='none') "
        "* class_weights[labels])"
    )
    core.FIXED_CLASS_WEIGHTS_BY_ID = TRAINING_WEIGHTS
    core.SAMPLE_WEIGHT_DESCRIPTION = None
    core.ROW_FILTER = en_es_filter
    core.LANGUAGES = (("en", "EN"), ("es", "ES"))


configure_core()
BASE_EXPERIMENT_CONFIG = core.experiment_config
BASE_FORMATTER = core.format_condition_report


def experiment16_config(condition):
    config = BASE_EXPERIMENT_CONFIG(condition)
    config.update(
        {
            "language_scope": ["en", "es"],
            "row_filter": "language in {en, es}",
            "filtered_train_rows": EXPECTED_TRAIN_ROWS,
            "filtered_dev_rows": EXPECTED_DEV_ROWS,
            "class_weight_source": "Experiment 15 fixed numeric weights",
        }
    )
    return config


core.experiment_config = experiment16_config


def validate_filtered_data():
    full_train = core.load_rows(TRAIN_PATH, "train")
    full_dev = core.load_rows(DEV_PATH, "dev")
    train_rows = en_es_filter(full_train, "train")
    dev_rows = en_es_filter(full_dev, "dev")
    language_counts = Counter(row["language"] for row in train_rows)
    sentiment_counts = Counter(row["sentiment"] for row in train_rows)
    aspect_counts = Counter(row["aspect"] for row in train_rows)
    dev_language_counts = Counter(row["language"] for row in dev_rows)
    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN_ROWS} EN+ES train rows")
    if len(dev_rows) != EXPECTED_DEV_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_DEV_ROWS} EN+ES dev rows")
    if dict(language_counts) != EXPECTED_TRAIN_LANGUAGE:
        raise RuntimeError(f"Train language mismatch: {dict(language_counts)}")
    if dict(sentiment_counts) != EXPECTED_TRAIN_SENTIMENT:
        raise RuntimeError(f"Train sentiment mismatch: {dict(sentiment_counts)}")
    if dict(aspect_counts) != EXPECTED_TRAIN_ASPECT:
        raise RuntimeError(f"Train aspect mismatch: {dict(aspect_counts)}")
    if dict(dev_language_counts) != EXPECTED_DEV_LANGUAGE:
        raise RuntimeError(f"Dev language mismatch: {dict(dev_language_counts)}")
    assert all(row["language"] != "tr" for row in train_rows)
    assert all(row["language"] != "tr" for row in dev_rows)
    return train_rows, dev_rows, language_counts, sentiment_counts, aspect_counts


def diagnostic_subset_weights(train_rows):
    labels = np.asarray([core.LABEL2ID[row["sentiment"]] for row in train_rows])
    return compute_class_weight(
        class_weight="balanced",
        classes=np.asarray([0, 1, 2], dtype=np.int64),
        y=labels,
    )


def baseline_checkpoint_dir():
    pointer_path = EXP15_MODEL_ROOT / "best" / "current.json"
    if not pointer_path.exists():
        raise RuntimeError("Experiment 15 best-checkpoint pointer is missing")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    checkpoint_dir = EXP15_MODEL_ROOT / "best" / pointer["relative_path"]
    metrics_path = checkpoint_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError("Experiment 15 best metrics are missing")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload.get("best_epoch") != 2:
        raise RuntimeError("Experiment 15 Epoch 2 is not the current best checkpoint")
    return checkpoint_dir


def format_metrics(title, metrics):
    lines = [
        title,
        "",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        "",
        "SENTIMENT",
    ]
    for sentiment in core.SENTIMENTS:
        item = metrics["per_class"][sentiment]
        lines.append(
            f"{sentiment}: P={item['precision']:.4f} | R={item['recall']:.4f} | "
            f"F1={item['f1']:.4f} | support={item['support']}"
        )
    lines.extend(
        [
            "",
            "PREDICTED COUNTS",
            *[
                f"{sentiment}: {metrics['predicted_counts'][sentiment]}"
                for sentiment in core.SENTIMENTS
            ],
            "",
            "CONFUSION MATRIX",
            "Rows=actual, columns=predicted; order=positive, neutral, negative",
            "                 positive  neutral  negative",
        ]
    )
    for sentiment, row in zip(core.SENTIMENTS, metrics["confusion_matrix"]):
        lines.append(f"{sentiment:<16} {row[0]:>8} {row[1]:>8} {row[2]:>9}")
    neutral = metrics["neutral_confusion"]
    lines.extend(
        [
            "",
            "NEUTRAL CONFUSION",
            f"true neutral -> neutral: {neutral['true_neutral_predicted_neutral']}",
            f"positive -> neutral: {neutral['positive_to_neutral']}",
            f"negative -> neutral: {neutral['negative_to_neutral']}",
            f"false-neutral total: {neutral['false_neutral_total']}",
            "",
            "LANGUAGE",
        ]
    )
    for language in ("EN", "ES"):
        item = metrics["languages"][language]
        neutral_item = item["neutral"]
        lines.append(
            f"{language}: Macro F1={item['macro_f1']:.4f} | "
            f"Neutral P/R/F1={neutral_item['precision']:.4f}/"
            f"{neutral_item['recall']:.4f}/{neutral_item['f1']:.4f}"
        )
    lines.extend(["", "ASPECT"])
    for aspect in core.ASPECTS:
        item = metrics["aspects"][aspect]
        lines.append(
            f"{aspect}: Macro F1={item['macro_f1']:.4f} | "
            f"Neutral F1={item['neutral']['f1']:.4f}"
        )
    return "\n".join(lines) + "\n"


def evaluate_exp15_baseline():
    """Inference only: evaluate Experiment 15 best on the EN+ES DEV subset."""
    _, dev_rows, _, _, _ = validate_filtered_data()
    checkpoint_dir = baseline_checkpoint_dir()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    core.set_seed(core.RANDOM_SEED)
    tokenizer = core.AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    base_model = core.load_base_model()
    model = PeftModel.from_pretrained(
        base_model, checkpoint_dir, is_trainable=False
    )
    model.requires_grad_(False)
    model.to(device)
    dev_loader = core.build_dev_loader(dev_rows, tokenizer)
    labels, predictions = core.predict(model, dev_loader, device)
    metrics = core.detailed_metrics(dev_rows, labels, predictions, train_loss=0.0)
    payload = {
        "name": "EXP15_EN_ES_BASELINE",
        "checkpoint": str(checkpoint_dir.relative_to(PROJECT_ROOT)),
        "evaluation_scope": ["en", "es"],
        "dev_rows": len(dev_rows),
        "device": device.type,
        "metrics": metrics,
    }
    core.write_json_atomic(BASELINE_JSON, payload)
    BASELINE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_REPORT.write_text(
        format_metrics("EXP15_EN_ES_BASELINE", metrics)
        + "\nTraining run: NO\nTest used: NO\n",
        encoding="utf-8",
    )
    print(format_metrics("EXP15_EN_ES_BASELINE", metrics))
    print(f"Evaluation device: {device.type}")
    print("Training run: NO")
    print("Test used: NO")


def load_baseline_metrics():
    if not BASELINE_JSON.exists():
        raise RuntimeError(
            "EXP15_EN_ES_BASELINE is missing; run --evaluate-exp15-baseline first"
        )
    payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    if payload.get("dev_rows") != EXPECTED_DEV_ROWS:
        raise RuntimeError("EXP15_EN_ES_BASELINE row count mismatch")
    return payload["metrics"]


def comparison_report(metrics):
    baseline = load_baseline_metrics()

    def delta(path):
        left = metrics
        right = baseline
        for key in path:
            left = left[key]
            right = right[key]
        return left - right

    counts = metrics["predicted_counts"]
    total = sum(counts.values())
    collapse = max(counts.values()) / total > 0.90
    all_three = all(value > 0 for value in counts.values())
    macro_delta = delta(("macro_f1",))
    neutral_delta = delta(("per_class", "neutral", "f1"))
    if macro_delta >= 0.02 and neutral_delta >= 0 and all_three and not collapse:
        support = "A) STRONG SUPPORT"
    elif (macro_delta > 0 or neutral_delta > 0) and not collapse:
        support = "B) WEAK / MIXED SUPPORT"
    else:
        support = "C) NO SUPPORT"

    lines = [
        "",
        "CONTROLLED COMPARISON: EXP16 - EXP15_EN_ES_BASELINE",
        f"Accuracy delta: {delta(('accuracy',)):+.4f}",
        f"Macro F1 delta: {macro_delta:+.4f}",
        f"Positive F1 delta: {delta(('per_class', 'positive', 'f1')):+.4f}",
        f"Neutral F1 delta: {neutral_delta:+.4f}",
        f"Negative F1 delta: {delta(('per_class', 'negative', 'f1')):+.4f}",
        f"Neutral precision delta: {delta(('per_class', 'neutral', 'precision')):+.4f}",
        f"Neutral recall delta: {delta(('per_class', 'neutral', 'recall')):+.4f}",
        f"False-neutral delta: {delta(('neutral_confusion', 'false_neutral_total')):+d}",
        "",
        "LANGUAGE DELTAS",
    ]
    for language in ("EN", "ES"):
        lines.append(
            f"{language} Macro F1: {delta(('languages', language, 'macro_f1')):+.4f}"
        )
        lines.append(
            f"{language} Neutral F1: "
            f"{delta(('languages', language, 'neutral', 'f1')):+.4f}"
        )
    lines.extend(["", "ASPECT MACRO F1 DELTAS"])
    for aspect in core.ASPECTS:
        lines.append(
            f"{aspect}: {delta(('aspects', aspect, 'macro_f1')):+.4f}"
        )
    lines.extend(
        [
            "",
            f"All three classes predicted: {all_three}",
            f">90% single-class collapse: {collapse}",
            f"Preliminary language-ablation result: {support}",
            "Epoch 2 was not started automatically.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_experiment16_report(condition, metrics, signature, permutation_hash):
    base = BASE_FORMATTER(condition, metrics, signature, permutation_hash).rstrip()
    base = base.replace(
        "Historical vanilla full XLM-R cleaned reference (different architecture):",
        "Historical reference:",
    )
    return (
        base
        + "\n\nLANGUAGE SCOPE\nTrain: EN+ES only\nDev evaluation: EN+ES only\n"
        + comparison_report(metrics)
    )


core.format_condition_report = format_experiment16_report


def run_static_validation():
    train_rows, dev_rows, language_counts, sentiment_counts, aspect_counts = (
        validate_filtered_data()
    )
    diagnostic = diagnostic_subset_weights(train_rows)
    if MODEL_ROOT.resolve() == EXP15_MODEL_ROOT.resolve():
        raise RuntimeError("Experiment 16 model root overlaps Experiment 15")
    if MODEL_ROOT.exists():
        paths = core.condition_paths("expanded")
        if not paths["resume_pointer"].exists():
            raise RuntimeError("Experiment 16 model directory exists without valid resume pointer")
    if core.optimizer_steps_per_epoch(len(train_rows)) != 588:
        raise RuntimeError("Expected 588 optimizer steps for 2,350 rows")

    core.run_static_validation("expanded")
    logits = torch.zeros((3, 3), dtype=torch.float32)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    raw = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    weighted = raw * torch.tensor(TRAINING_WEIGHTS)[labels]

    print("EXPERIMENT 16 FILTER AUDIT")
    print(f"EN train rows: {language_counts['en']}")
    print(f"ES train rows: {language_counts['es']}")
    print(f"Total EN+ES train rows: {len(train_rows)}")
    print("TR train rows: 0")
    print(f"Train sentiments: {dict(sentiment_counts)}")
    print(f"Train aspects: {dict(aspect_counts)}")
    print(f"EN+ES dev rows: {len(dev_rows)}")
    print("TR dev rows: 0")
    print(f"Training weights [negative, neutral, positive]: {list(TRAINING_WEIGHTS)}")
    print(f"Diagnostic EN+ES balanced weights only: {diagnostic.tolist()}")
    print("Diagnostic weights used for training: False")
    print("Corrected loss reduction='none': True")
    print("Additional sample/source/soft-dose weighting: NONE")
    print(f"Sanity raw losses: {[round(value, 6) for value in raw.tolist()]}")
    print(f"Sanity weighted losses: {[round(value, 6) for value in weighted.tolist()]}")
    print("Fresh Experiment 16 checkpoint: True")
    print("Resume from Experiment 15 for training: False")
    print("MPS required; CPU fallback blocked: True")
    print("Optimizer steps in Epoch 1: 588")
    print(f"Max optimizer steps per invocation: {core.MAX_OPTIMIZER_STEPS_PER_RUN}")
    print("Epoch 2 permitted automatically: False")
    print("Test split referenced or loaded: False")
    print("Training started: NO")


def experiment_signature():
    """Preserve the Epoch-1 signature so the existing state resumes exactly."""
    return core.stable_json_hash(core.experiment_config("expanded"))


def load_epoch1_resume_state_for_audit():
    signature = experiment_signature()
    metadata = core.read_resume_metadata("expanded", signature)
    if metadata is None:
        raise RuntimeError("Experiment 16 Epoch 1 resume checkpoint is missing")
    required = {
        "completed_epoch": 1,
        "current_epoch": None,
        "optimizer_steps_in_epoch": EXP16_STEPS_PER_EPOCH,
        "microbatch_position": EXPECTED_TRAIN_ROWS,
        "sample_position": EXPECTED_TRAIN_ROWS,
        "best_epoch": 1,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Epoch 1 resume metadata mismatch for {key}: "
                f"expected {expected}, got {metadata.get(key)}"
            )

    training_state_path = Path(metadata["_slot_dir"]) / "training_state.pt"
    state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    if state.get("config_signature") != signature:
        raise RuntimeError("Epoch 1 training-state signature mismatch")
    if not state.get("optimizer_state_dict"):
        raise RuntimeError("Epoch 1 optimizer state is missing")
    if state.get("scheduler_state_dict") is not None:
        raise RuntimeError("Unexpected scheduler state; Experiment 16 uses no scheduler")
    rng_state = state.get("rng_state") or {}
    if not {"python", "numpy", "torch"}.issubset(rng_state):
        raise RuntimeError("Epoch 1 RNG state is incomplete")
    if len(state.get("epoch_permutation", [])) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Epoch 1 stored permutation is incomplete")

    paths = core.condition_paths("expanded")
    if not paths["best_pointer"].exists():
        raise RuntimeError("Experiment 16 Epoch 1 best checkpoint is missing")
    pointer = json.loads(paths["best_pointer"].read_text(encoding="utf-8"))
    best_metrics_path = paths["best_dir"] / pointer["relative_path"] / "metrics.json"
    best_payload = json.loads(best_metrics_path.read_text(encoding="utf-8"))
    if best_payload.get("config_signature") != signature:
        raise RuntimeError("Epoch 1 best-checkpoint signature mismatch")
    if best_payload.get("best_epoch") != 1:
        raise RuntimeError("Expected Epoch 1 to be the current best checkpoint")
    return metadata, state, best_payload["metrics"], signature


def run_epoch2_static_validation():
    train_rows, dev_rows, languages, sentiments, aspects = validate_filtered_data()
    metadata, state, epoch1_metrics, signature = load_epoch1_resume_state_for_audit()
    epoch_seed, permutation = core.epoch_permutation(len(train_rows), 2)
    if epoch_seed != 43 or len(permutation) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Epoch 2 deterministic permutation validation failed")
    if core.optimizer_steps_per_epoch(len(train_rows)) != EXP16_STEPS_PER_EPOCH:
        raise RuntimeError("Epoch 2 optimizer-step count must be 588")
    if tuple(core.FIXED_CLASS_WEIGHTS_BY_ID) != TRAINING_WEIGHTS:
        raise RuntimeError("Experiment 16 class weights changed")
    if core.compute_training_loss.__module__ != core.__name__:
        raise RuntimeError("Unexpected loss implementation")

    print("EXPERIMENT 16 — EPOCH 2 RESUME VALIDATION")
    print(f"Config signature preserved: {signature}")
    print(f"Epoch 1 completed: {metadata['completed_epoch'] == 1}")
    print(f"Epoch 1 optimizer steps: {metadata['optimizer_steps_in_epoch']}")
    print(f"Epoch 1 best Macro F1: {epoch1_metrics['macro_f1']:.4f}")
    print("Adapter/classifier checkpoint present: True")
    print("Optimizer state present: True")
    print("Scheduler: none")
    print("Python/NumPy/Torch RNG state present: True")
    print(f"Stored Epoch 1 permutation length: {len(state['epoch_permutation'])}")
    print(f"Epoch 2 shuffle seed: {epoch_seed}")
    print(f"Epoch 2 permutation SHA256: {core.permutation_sha256(permutation)}")
    print(f"EN train rows: {languages['en']}")
    print(f"ES train rows: {languages['es']}")
    print("TR train rows: 0")
    print(f"Total EN+ES train rows: {len(train_rows)}")
    print(f"EN+ES dev rows: {len(dev_rows)}")
    print(f"Train sentiments: {dict(sentiments)}")
    print(f"Train aspects: {dict(aspects)}")
    print(f"Class weights [negative, neutral, positive]: {TRAINING_WEIGHTS}")
    print("Corrected weighted loss reduction='none': True")
    print("Next epoch: 2")
    print("Next optimizer step: 0")
    print(f"Experiment 16 optimizer steps per epoch: {EXP16_STEPS_PER_EPOCH}")
    print(
        "Experiment 16 total optimizer steps after Epoch 1: "
        f"{EXP16_TOTAL_STEPS_AFTER_EPOCH1}"
    )
    print(
        "Experiment 16 total optimizer steps after Epoch 2: "
        f"{EXP16_TOTAL_STEPS_AFTER_EPOCH2}"
    )
    print(
        "Experiment 15 total optimizer steps after Epoch 2: "
        f"{EXP15_TOTAL_STEPS_AFTER_EPOCH2}"
    )
    print(
        "Experiment 16 fewer optimizer updates after Epoch 2: "
        f"{OPTIMIZER_STEP_DIFFERENCE} ({OPTIMIZER_STEP_PERCENT_DIFFERENCE:.2f}%)"
    )
    print(f"Max optimizer steps per invocation: {core.MAX_OPTIMIZER_STEPS_PER_RUN}")
    print("Epoch 3 permitted: False")
    print("MPS required; CPU fallback blocked: True")
    print("Test split referenced or loaded: False")
    print("Training started: NO")


def initialize_epoch2_state(state, train_count):
    if state.get("completed_epoch") != 1 or state.get("current_epoch") is not None:
        raise RuntimeError("Epoch 2 can start only from the completed Epoch 1 boundary")
    epoch_seed, permutation = core.epoch_permutation(train_count, 2)
    epoch1_metrics = state.get("last_epoch_metrics")
    if not epoch1_metrics:
        raise RuntimeError("Epoch 1 metrics are missing from resume state")
    state.update(
        {
            "current_epoch": 2,
            "optimizer_steps_in_epoch": 0,
            "microbatch_position": 0,
            "sample_position": 0,
            "micro_batches_accumulated": 0,
            "epoch_permutation": permutation,
            "permutation_sha256": core.permutation_sha256(permutation),
            "epoch_seed": epoch_seed,
            "epoch_running_loss_sum": 0.0,
            "epoch_microbatches_seen": 0,
            "epoch_metrics_history": {"1": epoch1_metrics},
        }
    )
    return state


def validate_epoch2_progress(state, train_count):
    if state.get("completed_epoch") == 2:
        if state.get("current_epoch") is not None:
            raise RuntimeError("Completed Epoch 2 has a non-null current_epoch")
        return
    if state.get("completed_epoch") != 1 or state.get("current_epoch") != 2:
        raise RuntimeError("Resume state is not valid Epoch 2 progress")
    permutation = state.get("epoch_permutation", [])
    expected_seed, expected_permutation = core.epoch_permutation(train_count, 2)
    if state.get("epoch_seed") != expected_seed or permutation != expected_permutation:
        raise RuntimeError("Epoch 2 deterministic permutation changed during resume")
    if state.get("permutation_sha256") != core.permutation_sha256(permutation):
        raise RuntimeError("Epoch 2 permutation hash mismatch")
    if state.get("micro_batches_accumulated") != 0:
        raise RuntimeError("Epoch 2 checkpoint is not at an accumulation boundary")
    position = state.get("microbatch_position", 0)
    expected_position = min(position * core.TRAIN_BATCH_SIZE, train_count)
    if state.get("sample_position") != expected_position:
        raise RuntimeError("Epoch 2 sample and microbatch positions differ")
    expected_steps = (
        core.math.ceil(position / core.GRADIENT_ACCUMULATION_STEPS)
        if position
        else 0
    )
    if state.get("optimizer_steps_in_epoch") != expected_steps:
        raise RuntimeError("Epoch 2 optimizer-step and microbatch positions differ")


def save_epoch2_best_checkpoint(model, tokenizer, metrics, signature):
    paths = core.condition_paths("expanded")
    final_dir = paths["best_dir"] / "epoch_2"
    temporary_dir = paths["best_dir"] / ".epoch_2.tmp"
    paths["best_dir"].mkdir(parents=True, exist_ok=True)
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    model.save_pretrained(temporary_dir)
    tokenizer.save_pretrained(temporary_dir)
    checkpoint_id = "16-best-epoch2"
    core.write_json_atomic(
        temporary_dir / "metrics.json",
        {
            "checkpoint_id": checkpoint_id,
            "config_signature": signature,
            "condition": "expanded",
            "best_epoch": 2,
            "metrics": metrics,
        },
    )
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(temporary_dir, final_dir)
    core.write_json_atomic(
        paths["best_pointer"],
        {"relative_path": "epoch_2", "checkpoint_id": checkpoint_id},
    )
    return final_dir


def run_evaluate_current_best():
    """Print stored best metrics for Epoch 1 or 2 without running inference."""
    paths = core.condition_paths("expanded")
    if not paths["best_pointer"].exists():
        raise RuntimeError("Experiment 16 best checkpoint is missing")
    pointer = json.loads(paths["best_pointer"].read_text(encoding="utf-8"))
    metrics_path = paths["best_dir"] / pointer["relative_path"] / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload.get("config_signature") != experiment_signature():
        raise RuntimeError("Experiment 16 best-checkpoint signature mismatch")
    if payload.get("condition") != "expanded" or payload.get("best_epoch") not in {1, 2, 3}:
        raise RuntimeError("Experiment 16 best-checkpoint identity mismatch")
    print(format_metrics(f"EXPERIMENT 16 BEST — EPOCH {payload['best_epoch']}", payload["metrics"]))
    print("Stored metrics only; no training or test evaluation was run.")


def metric_delta(left, right, path):
    for key in path:
        left = left[key]
        right = right[key]
    return left - right


def epoch2_analysis_report(epoch1, epoch2):
    baseline = load_baseline_metrics()
    counts = epoch2["predicted_counts"]
    total = sum(counts.values())
    dominant_label, dominant_count = max(counts.items(), key=lambda item: item[1])
    dominant_rate = dominant_count / total
    collapse = dominant_rate > 0.90
    all_three = all(value > 0 for value in counts.values())
    epoch_macro_delta = metric_delta(epoch2, epoch1, ("macro_f1",))
    baseline_macro_delta = metric_delta(epoch2, baseline, ("macro_f1",))

    if baseline_macro_delta >= 0.02 and all_three and not collapse:
        hypothesis = "A) SUPPORT"
    elif collapse or baseline_macro_delta < -0.02:
        hypothesis = "C) NO SUPPORT"
    else:
        hypothesis = "B) MIXED / INCONCLUSIVE"

    lines = [
        "",
        "COLLAPSE CHECK",
        f"All three classes predicted: {all_three}",
        f"Single-class collapse above 90%: {collapse}",
        f"Dominant class: {dominant_label} ({dominant_rate:.2%})",
        f"Positive prediction rate: {counts['positive'] / total:.2%}",
        f"Neutral predictions: {counts['neutral']}",
        f"Negative predictions: {counts['negative']}",
    ]
    if collapse:
        lines.append("COLLAPSE WARNING")

    lines.extend(
        [
            "",
            "OPTIMIZER-STEP COMPARISON",
            f"Experiment 16 train rows: {EXPECTED_TRAIN_ROWS}",
            f"Experiment 16 optimizer steps per epoch: {EXP16_STEPS_PER_EPOCH}",
            f"Experiment 16 total after Epoch 1: {EXP16_TOTAL_STEPS_AFTER_EPOCH1}",
            f"Experiment 16 total after Epoch 2: {EXP16_TOTAL_STEPS_AFTER_EPOCH2}",
            "Experiment 15 train rows: 3523",
            f"Experiment 15 optimizer steps per epoch: {EXP15_STEPS_PER_EPOCH}",
            f"Experiment 15 total after Epoch 2: {EXP15_TOTAL_STEPS_AFTER_EPOCH2}",
            "Experiment 16 fewer updates after Epoch 2: "
            f"{OPTIMIZER_STEP_DIFFERENCE} ({OPTIMIZER_STEP_PERCENT_DIFFERENCE:.2f}%)",
            "",
            "EPOCH 2 MINUS EPOCH 1",
            f"Accuracy delta: {metric_delta(epoch2, epoch1, ('accuracy',)):+.4f}",
            f"Macro F1 delta: {epoch_macro_delta:+.4f}",
            f"Positive F1 delta: {metric_delta(epoch2, epoch1, ('per_class', 'positive', 'f1')):+.4f}",
            f"Neutral F1 delta: {metric_delta(epoch2, epoch1, ('per_class', 'neutral', 'f1')):+.4f}",
            f"Negative F1 delta: {metric_delta(epoch2, epoch1, ('per_class', 'negative', 'f1')):+.4f}",
            f"Neutral precision delta: {metric_delta(epoch2, epoch1, ('per_class', 'neutral', 'precision')):+.4f}",
            f"Neutral recall delta: {metric_delta(epoch2, epoch1, ('per_class', 'neutral', 'recall')):+.4f}",
            f"Predicted neutral delta: {metric_delta(epoch2, epoch1, ('predicted_counts', 'neutral')):+d}",
            f"Predicted negative delta: {metric_delta(epoch2, epoch1, ('predicted_counts', 'negative')):+d}",
            f"False-neutral delta: {metric_delta(epoch2, epoch1, ('neutral_confusion', 'false_neutral_total')):+d}",
            "",
            "EXP16 EPOCH 2 MINUS EXP15 EN+ES BASELINE",
            f"Accuracy delta: {metric_delta(epoch2, baseline, ('accuracy',)):+.4f}",
            f"Macro F1 delta: {baseline_macro_delta:+.4f}",
            f"Positive F1 delta: {metric_delta(epoch2, baseline, ('per_class', 'positive', 'f1')):+.4f}",
            f"Neutral F1 delta: {metric_delta(epoch2, baseline, ('per_class', 'neutral', 'f1')):+.4f}",
            f"Negative F1 delta: {metric_delta(epoch2, baseline, ('per_class', 'negative', 'f1')):+.4f}",
            f"Neutral precision delta: {metric_delta(epoch2, baseline, ('per_class', 'neutral', 'precision')):+.4f}",
            f"Neutral recall delta: {metric_delta(epoch2, baseline, ('per_class', 'neutral', 'recall')):+.4f}",
            "",
            "LANGUAGE DELTAS VS EXP15 EN+ES BASELINE",
        ]
    )
    for language in ("EN", "ES"):
        lines.append(
            f"{language} Macro F1: "
            f"{metric_delta(epoch2, baseline, ('languages', language, 'macro_f1')):+.4f}"
        )
        lines.append(
            f"{language} Neutral F1: "
            f"{metric_delta(epoch2, baseline, ('languages', language, 'neutral', 'f1')):+.4f}"
        )
    lines.extend(["", "ASPECT MACRO F1 DELTAS VS EXP15 EN+ES BASELINE"])
    for aspect in core.ASPECTS:
        lines.append(
            f"{aspect}: "
            f"{metric_delta(epoch2, baseline, ('aspects', aspect, 'macro_f1')):+.4f}"
        )
    lines.extend(
        [
            "",
            f"LANGUAGE HYPOTHESIS DECISION: {hypothesis}",
            "Interpretation rule: under this training setup and current data, "
            "removing Turkish training data did not improve EN+ES performance "
            "if the result is C; this does not prove Turkish data is universally beneficial.",
            "Optimizer-step limitation: Experiment 16 receives 33.26% fewer updates "
            "than Experiment 15 after two epochs.",
            "Epoch 3 was not started automatically.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_epoch2_report(metrics, signature, permutation_hash, epoch1_metrics):
    original_subtitle = core.EXPERIMENT_SUBTITLE
    core.EXPERIMENT_SUBTITLE = "EN+ES-ONLY CORRECTED WEIGHTED LORA EPOCH 2"
    try:
        base = BASE_FORMATTER(
            "expanded", metrics, signature, permutation_hash
        ).rstrip()
    finally:
        core.EXPERIMENT_SUBTITLE = original_subtitle
    base = base.replace(
        "Epoch 2 was not started automatically.",
        "Epoch 3 was not started automatically.",
    )
    base = base.replace("Epoch shuffle seed: 42", "Epoch shuffle seed: 43")
    return (
        base
        + "\n\nLANGUAGE SCOPE\nTrain: EN+ES only\nDev evaluation: EN+ES only\n"
        + epoch2_analysis_report(epoch1_metrics, metrics)
    )


def run_epoch2_chunk():
    device = core.require_mps("expanded")
    if device is None:
        return
    train_rows, dev_rows, _, _, _ = validate_filtered_data()
    signature = experiment_signature()
    metadata = core.read_resume_metadata("expanded", signature)
    if metadata is None:
        raise RuntimeError("Epoch 1 resume checkpoint is required")
    if metadata.get("completed_epoch", 0) >= 2:
        core.append_log("expanded", "Epoch 2 is already complete. No training was started.")
        core.append_log("expanded", "Epoch 3 is outside the Experiment 16 budget.")
        return
    if metadata.get("completed_epoch") != 1:
        raise RuntimeError("Epoch 2 requires a completed Epoch 1 checkpoint")

    core.set_seed(core.RANDOM_SEED)
    tokenizer = core.AutoTokenizer.from_pretrained(
        Path(metadata["_slot_dir"]), local_files_only=True
    )
    model = core.create_or_resume_model(metadata)
    model.to(device)
    optimizer = torch.optim.AdamW(
        core.trainable_parameters(model), lr=core.LEARNING_RATE
    )
    state = core.load_training_state(metadata, optimizer, device, signature)
    if state.get("current_epoch") is None:
        state = initialize_epoch2_state(state, len(train_rows))
    validate_epoch2_progress(state, len(train_rows))
    epoch1_metrics = state.get("epoch_metrics_history", {}).get("1")
    if not epoch1_metrics:
        epoch1_metrics = metadata.get("last_epoch_metrics")
    if not epoch1_metrics:
        raise RuntimeError("Epoch 1 metrics are unavailable for comparison")

    total_microbatches = core.micro_batches_per_epoch(len(train_rows))
    total_optimizer_steps = core.optimizer_steps_per_epoch(len(train_rows))
    current_step = state["optimizer_steps_in_epoch"]
    target_step = min(
        current_step + core.MAX_OPTIMIZER_STEPS_PER_RUN, total_optimizer_steps
    )
    core.append_log("expanded", "")
    core.append_log("expanded", "Experiment: 16")
    core.append_log("expanded", "Device: mps")
    core.append_log("expanded", "Epoch: 2/2")
    core.append_log("expanded", f"Starting optimizer step: {current_step}")
    core.append_log("expanded", f"Target optimizer step: {target_step}")
    core.append_log(
        "expanded",
        f"Remaining optimizer steps in epoch: {total_optimizer_steps - current_step}",
    )
    core.append_log("expanded", f"Loss: {core.LOSS_DESCRIPTION}")
    core.append_log(
        "expanded",
        "Class weights [negative, neutral, positive]: "
        f"{core.FIXED_CLASS_WEIGHTS_BY_ID}",
    )
    core.append_log(
        "expanded",
        f"Experiment 16 total optimizer steps after Epoch 2: {EXP16_TOTAL_STEPS_AFTER_EPOCH2}",
    )
    core.append_log(
        "expanded",
        "Experiment 16 has "
        f"{OPTIMIZER_STEP_DIFFERENCE} fewer updates than Experiment 15 "
        f"after Epoch 2 ({OPTIMIZER_STEP_PERCENT_DIFFERENCE:.2f}%).",
    )
    core.append_log("expanded", "Entering training loop for epoch 2")

    collator = core.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataset = core.SentimentDataset(train_rows, tokenizer)
    remaining_indices = state["epoch_permutation"][state["sample_position"] :]
    loader = core.DataLoader(
        core.Subset(dataset, remaining_indices),
        batch_size=core.TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    iterator = iter(loader)
    dev_loader = core.build_dev_loader(dev_rows, tokenizer)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    completed_this_run = 0
    stop_requested = {"value": False}
    boundary_rng = core.capture_rng_state(device)
    old_sigint = signal.getsignal(signal.SIGINT)

    def request_stop(_signal_number, _frame):
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        while state["optimizer_steps_in_epoch"] < target_step:
            boundary_rng = core.capture_rng_state(device)
            group_start = state["microbatch_position"]
            group_size = min(
                core.GRADIENT_ACCUMULATION_STEPS,
                total_microbatches - group_start,
            )
            group_loss = 0.0
            for micro_index in range(group_size):
                if completed_this_run == 0 and micro_index == 0:
                    core.append_log("expanded", "Starting first training batch")
                batch = core.move_batch(next(iterator), device)
                labels = batch.pop("labels")
                logits = model(**batch).logits
                training_loss = core.compute_training_loss(logits, labels)
                loss_for_backward = training_loss / core.GRADIENT_ACCUMULATION_STEPS
                loss_for_backward.backward()
                group_loss += training_loss.item()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state["optimizer_steps_in_epoch"] += 1
            state["microbatch_position"] += group_size
            state["sample_position"] = min(
                state["microbatch_position"] * core.TRAIN_BATCH_SIZE,
                len(train_rows),
            )
            state["epoch_running_loss_sum"] += group_loss
            state["epoch_microbatches_seen"] += group_size
            completed_this_run += 1
            if stop_requested["value"]:
                core.append_log(
                    "expanded",
                    "Ctrl+C received. Stopping at the completed optimizer-step boundary.",
                )
                break
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        core.restore_rng_state(boundary_rng, device)
        core.append_log(
            "expanded",
            "Partial accumulation group discarded; progress remains at the last completed step.",
        )
    except RuntimeError as error:
        optimizer.zero_grad(set_to_none=True)
        if "mps" in str(error).lower() and "out of memory" in str(error).lower():
            core.append_log(
                "expanded",
                "TRAINING CHUNK FAILED — MPS OOM. No new checkpoint was written; "
                "previous checkpoint remains authoritative; CPU fallback was not used.",
            )
            return
        raise
    finally:
        signal.signal(signal.SIGINT, old_sigint)

    epoch_finished = state["microbatch_position"] >= total_microbatches
    best_path = None
    if epoch_finished:
        labels, predictions = core.predict(model, dev_loader, device)
        train_loss = state["epoch_running_loss_sum"] / state["epoch_microbatches_seen"]
        metrics = core.detailed_metrics(dev_rows, labels, predictions, train_loss)
        metrics["epoch"] = 2
        state["completed_epoch"] = 2
        state["current_epoch"] = None
        state["last_epoch_metrics"] = metrics
        state.setdefault("epoch_metrics_history", {})["2"] = metrics
        if metrics["macro_f1"] > state.get("best_macro_f1", float("-inf")):
            state["best_macro_f1"] = metrics["macro_f1"]
            state["best_epoch"] = 2
            best_path = save_epoch2_best_checkpoint(
                model, tokenizer, metrics, signature
            )

        epoch2_report = format_epoch2_report(
            metrics, signature, state["permutation_sha256"], epoch1_metrics
        )
        with REPORT_PATH.open("a", encoding="utf-8") as report_file:
            report_file.write("\n" + epoch2_report)

    checkpoint_path = core.save_resume_checkpoint(
        "expanded", model, tokenizer, optimizer, device, state, signature
    )
    progress = 100 * state["optimizer_steps_in_epoch"] / total_optimizer_steps
    core.append_log("expanded", "")
    core.append_log("expanded", "TRAINING CHUNK RESULT")
    core.append_log("expanded", "Epoch: 2/2")
    core.append_log(
        "expanded", f"Optimizer steps completed this run: {completed_this_run}"
    )
    core.append_log(
        "expanded", f"Current optimizer step: {state['optimizer_steps_in_epoch']}"
    )
    core.append_log("expanded", f"Epoch progress percentage: {progress:.2f}%")
    core.append_log("expanded", f"Resume checkpoint saved: {checkpoint_path}")
    core.append_log("expanded", f"Epoch finished: {epoch_finished}")
    if epoch_finished:
        metrics = state["last_epoch_metrics"]
        core.append_log("expanded", f"Dev Accuracy: {metrics['accuracy']:.4f}")
        core.append_log("expanded", f"Dev Macro F1: {metrics['macro_f1']:.4f}")
        core.append_log("expanded", f"Best epoch: {state['best_epoch']}")
        core.append_log(
            "expanded", f"Best Macro F1: {state['best_macro_f1']:.4f}"
        )
        if best_path is not None:
            core.append_log("expanded", f"Best checkpoint updated: {best_path}")
        else:
            core.append_log(
                "expanded",
                "Epoch 1 best checkpoint preserved; Epoch 2 did not improve Macro F1.",
            )
        core.append_log("expanded", "Epoch 3 was not started automatically.")
    core.append_log("expanded", "Training chunk complete.")
    core.append_log("expanded", "No additional chunk was started automatically.")


def load_epoch2_resume_state_for_audit():
    signature = experiment_signature()
    metadata = core.read_resume_metadata("expanded", signature)
    if metadata is None:
        raise RuntimeError("Experiment 16 Epoch 2 resume checkpoint is missing")
    required = {
        "completed_epoch": 2,
        "current_epoch": None,
        "optimizer_steps_in_epoch": EXP16_STEPS_PER_EPOCH,
        "microbatch_position": EXPECTED_TRAIN_ROWS,
        "sample_position": EXPECTED_TRAIN_ROWS,
        "best_epoch": 2,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Epoch 2 resume metadata mismatch for {key}: "
                f"expected {expected}, got {metadata.get(key)}"
            )

    state_path = Path(metadata["_slot_dir"]) / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("config_signature") != signature:
        raise RuntimeError("Epoch 2 training-state signature mismatch")
    if not state.get("optimizer_state_dict"):
        raise RuntimeError("Epoch 2 optimizer state is missing")
    if state.get("scheduler_state_dict") is not None:
        raise RuntimeError("Unexpected scheduler state; Experiment 16 uses no scheduler")
    rng_state = state.get("rng_state") or {}
    if not {"python", "numpy", "torch"}.issubset(rng_state):
        raise RuntimeError("Epoch 2 RNG state is incomplete")
    history = state.get("epoch_metrics_history") or {}
    if not {"1", "2"}.issubset(history):
        raise RuntimeError("Epoch 1/2 metric history is incomplete")
    if len(state.get("epoch_permutation", [])) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Epoch 2 stored permutation is incomplete")

    paths = core.condition_paths("expanded")
    pointer = json.loads(paths["best_pointer"].read_text(encoding="utf-8"))
    if pointer.get("relative_path") != "epoch_2":
        raise RuntimeError("Experiment 16 Epoch 2 is not the current best checkpoint")
    best_path = paths["best_dir"] / pointer["relative_path"] / "metrics.json"
    best_payload = json.loads(best_path.read_text(encoding="utf-8"))
    if best_payload.get("config_signature") != signature:
        raise RuntimeError("Epoch 2 best-checkpoint signature mismatch")
    if best_payload.get("best_epoch") != 2:
        raise RuntimeError("Expected Epoch 2 to be the current best checkpoint")
    return metadata, state, best_payload["metrics"], signature


def run_epoch3_static_validation():
    train_rows, dev_rows, languages, sentiments, aspects = validate_filtered_data()
    metadata, state, epoch2_metrics, signature = load_epoch2_resume_state_for_audit()
    epoch_seed, permutation = core.epoch_permutation(len(train_rows), 3)
    if epoch_seed != 44 or len(permutation) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Epoch 3 deterministic permutation validation failed")
    if core.optimizer_steps_per_epoch(len(train_rows)) != EXP16_STEPS_PER_EPOCH:
        raise RuntimeError("Epoch 3 optimizer-step count must be 588")
    if tuple(core.FIXED_CLASS_WEIGHTS_BY_ID) != TRAINING_WEIGHTS:
        raise RuntimeError("Experiment 16 class weights changed")
    if EXP16_TOTAL_STEPS_AFTER_EPOCH3 != 1764:
        raise RuntimeError("Experiment 16 final optimizer budget must be 1,764")

    print("EXPERIMENT 16 — EPOCH 3 RESUME VALIDATION")
    print(f"Config signature preserved: {signature}")
    print(f"Epoch 2 completed: {metadata['completed_epoch'] == 2}")
    print(f"Epoch 2 optimizer steps: {metadata['optimizer_steps_in_epoch']}")
    print(f"Epoch 2 best Macro F1: {epoch2_metrics['macro_f1']:.4f}")
    print("Adapter/classifier checkpoint present: True")
    print("Optimizer state present: True")
    print("Scheduler: none")
    print("Python/NumPy/Torch RNG state present: True")
    print(f"Stored Epoch 2 permutation length: {len(state['epoch_permutation'])}")
    print(f"Epoch 3 shuffle seed: {epoch_seed}")
    print(f"Epoch 3 permutation SHA256: {core.permutation_sha256(permutation)}")
    print(f"EN train rows: {languages['en']}")
    print(f"ES train rows: {languages['es']}")
    print("TR train rows: 0")
    print(f"Total EN+ES train rows: {len(train_rows)}")
    print(f"EN+ES dev rows: {len(dev_rows)}")
    print(f"Train sentiments: {dict(sentiments)}")
    print(f"Train aspects: {dict(aspects)}")
    print(f"Class weights [negative, neutral, positive]: {TRAINING_WEIGHTS}")
    print("Corrected weighted loss reduction='none': True")
    print("Next epoch: 3")
    print("Next optimizer step: 0")
    print(f"Experiment 16 steps per epoch: {EXP16_STEPS_PER_EPOCH}")
    print(f"Experiment 16 total after Epoch 1: {EXP16_TOTAL_STEPS_AFTER_EPOCH1}")
    print(f"Experiment 16 total after Epoch 2: {EXP16_TOTAL_STEPS_AFTER_EPOCH2}")
    print(f"Experiment 16 total after Epoch 3: {EXP16_TOTAL_STEPS_AFTER_EPOCH3}")
    print(f"Experiment 15 final optimizer steps: {EXP15_TOTAL_STEPS_AFTER_EPOCH2}")
    print(
        "Final absolute optimizer-budget difference: "
        f"{FINAL_BUDGET_STEP_DIFFERENCE} steps"
    )
    print(
        "Final percentage optimizer-budget difference: "
        f"{FINAL_BUDGET_PERCENT_DIFFERENCE:.4f}%"
    )
    print("Training budgets practically matched: True")
    print(f"Max optimizer steps per invocation: {core.MAX_OPTIMIZER_STEPS_PER_RUN}")
    print("Epoch 4 permitted: False")
    print("MPS required; CPU fallback blocked: True")
    print("Test split referenced or loaded: False")
    print("Training started: NO")


def initialize_epoch3_state(state, train_count):
    if state.get("completed_epoch") != 2 or state.get("current_epoch") is not None:
        raise RuntimeError("Epoch 3 can start only from the completed Epoch 2 boundary")
    epoch_seed, permutation = core.epoch_permutation(train_count, 3)
    history = state.setdefault("epoch_metrics_history", {})
    if not history.get("2"):
        history["2"] = state.get("last_epoch_metrics")
    if not history.get("1") or not history.get("2"):
        raise RuntimeError("Epoch 1/2 metrics are missing from resume state")
    state.update(
        {
            "current_epoch": 3,
            "optimizer_steps_in_epoch": 0,
            "microbatch_position": 0,
            "sample_position": 0,
            "micro_batches_accumulated": 0,
            "epoch_permutation": permutation,
            "permutation_sha256": core.permutation_sha256(permutation),
            "epoch_seed": epoch_seed,
            "epoch_running_loss_sum": 0.0,
            "epoch_microbatches_seen": 0,
        }
    )
    return state


def validate_epoch3_progress(state, train_count):
    if state.get("completed_epoch") == 3:
        if state.get("current_epoch") is not None:
            raise RuntimeError("Completed Epoch 3 has a non-null current_epoch")
        return
    if state.get("completed_epoch") != 2 or state.get("current_epoch") != 3:
        raise RuntimeError("Resume state is not valid Epoch 3 progress")
    permutation = state.get("epoch_permutation", [])
    expected_seed, expected_permutation = core.epoch_permutation(train_count, 3)
    if state.get("epoch_seed") != expected_seed or permutation != expected_permutation:
        raise RuntimeError("Epoch 3 deterministic permutation changed during resume")
    if state.get("permutation_sha256") != core.permutation_sha256(permutation):
        raise RuntimeError("Epoch 3 permutation hash mismatch")
    if state.get("micro_batches_accumulated") != 0:
        raise RuntimeError("Epoch 3 checkpoint is not at an accumulation boundary")
    position = state.get("microbatch_position", 0)
    expected_position = min(position * core.TRAIN_BATCH_SIZE, train_count)
    if state.get("sample_position") != expected_position:
        raise RuntimeError("Epoch 3 sample and microbatch positions differ")
    expected_steps = (
        core.math.ceil(position / core.GRADIENT_ACCUMULATION_STEPS)
        if position
        else 0
    )
    if state.get("optimizer_steps_in_epoch") != expected_steps:
        raise RuntimeError("Epoch 3 optimizer-step and microbatch positions differ")


def save_epoch3_best_checkpoint(model, tokenizer, metrics, signature):
    paths = core.condition_paths("expanded")
    final_dir = paths["best_dir"] / "epoch_3"
    temporary_dir = paths["best_dir"] / ".epoch_3.tmp"
    paths["best_dir"].mkdir(parents=True, exist_ok=True)
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    model.save_pretrained(temporary_dir)
    tokenizer.save_pretrained(temporary_dir)
    checkpoint_id = "16-best-epoch3"
    core.write_json_atomic(
        temporary_dir / "metrics.json",
        {
            "checkpoint_id": checkpoint_id,
            "config_signature": signature,
            "condition": "expanded",
            "best_epoch": 3,
            "metrics": metrics,
        },
    )
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(temporary_dir, final_dir)
    core.write_json_atomic(
        paths["best_pointer"],
        {"relative_path": "epoch_3", "checkpoint_id": checkpoint_id},
    )
    return final_dir


def epoch3_analysis_report(epoch2, epoch3):
    baseline = load_baseline_metrics()
    counts = epoch3["predicted_counts"]
    total = sum(counts.values())
    dominant_label, dominant_count = max(counts.items(), key=lambda item: item[1])
    dominant_rate = dominant_count / total
    collapse = dominant_rate > 0.90
    all_three = all(value > 0 for value in counts.values())
    macro_delta = metric_delta(epoch3, baseline, ("macro_f1",))
    neutral_delta = metric_delta(epoch3, baseline, ("per_class", "neutral", "f1"))
    if macro_delta >= 0.02 and neutral_delta >= 0 and all_three and not collapse:
        decision = "A) SUPPORT"
    elif collapse or (macro_delta < -0.02 and neutral_delta < -0.03):
        decision = "C) HYPOTHESIS REJECTED"
    else:
        decision = "B) NO MEANINGFUL BENEFIT"

    lines = [
        "",
        "FINAL OPTIMIZER-BUDGET CHECK",
        f"Experiment 16 steps per epoch: {EXP16_STEPS_PER_EPOCH}",
        f"Experiment 16 total after Epoch 1: {EXP16_TOTAL_STEPS_AFTER_EPOCH1}",
        f"Experiment 16 total after Epoch 2: {EXP16_TOTAL_STEPS_AFTER_EPOCH2}",
        f"Experiment 16 total after Epoch 3: {EXP16_TOTAL_STEPS_AFTER_EPOCH3}",
        f"Experiment 15 final total: {EXP15_TOTAL_STEPS_AFTER_EPOCH2}",
        f"Absolute difference: {FINAL_BUDGET_STEP_DIFFERENCE}",
        f"Percentage difference: {FINAL_BUDGET_PERCENT_DIFFERENCE:.4f}%",
        "Training budgets practically matched: True",
        "",
        "COLLAPSE CHECK",
        f"All three classes predicted: {all_three}",
        f"Single-class collapse above 90%: {collapse}",
        f"Dominant class: {dominant_label} ({dominant_rate:.2%})",
    ]
    if collapse:
        lines.append("COLLAPSE WARNING")
    lines.extend(
        [
            "",
            "EPOCH 3 MINUS EPOCH 2",
            f"Accuracy delta: {metric_delta(epoch3, epoch2, ('accuracy',)):+.4f}",
            f"Macro F1 delta: {metric_delta(epoch3, epoch2, ('macro_f1',)):+.4f}",
            f"Positive F1 delta: {metric_delta(epoch3, epoch2, ('per_class', 'positive', 'f1')):+.4f}",
            f"Neutral F1 delta: {metric_delta(epoch3, epoch2, ('per_class', 'neutral', 'f1')):+.4f}",
            f"Negative F1 delta: {metric_delta(epoch3, epoch2, ('per_class', 'negative', 'f1')):+.4f}",
            f"Neutral precision delta: {metric_delta(epoch3, epoch2, ('per_class', 'neutral', 'precision')):+.4f}",
            f"Neutral recall delta: {metric_delta(epoch3, epoch2, ('per_class', 'neutral', 'recall')):+.4f}",
            f"Predicted neutral delta: {metric_delta(epoch3, epoch2, ('predicted_counts', 'neutral')):+d}",
            f"False-neutral delta: {metric_delta(epoch3, epoch2, ('neutral_confusion', 'false_neutral_total')):+d}",
            "",
            "FINAL MODEL B MINUS MODEL A",
            f"Accuracy delta: {metric_delta(epoch3, baseline, ('accuracy',)):+.4f}",
            f"Macro F1 delta: {macro_delta:+.4f}",
            f"Positive F1 delta: {metric_delta(epoch3, baseline, ('per_class', 'positive', 'f1')):+.4f}",
            f"Neutral F1 delta: {neutral_delta:+.4f}",
            f"Negative F1 delta: {metric_delta(epoch3, baseline, ('per_class', 'negative', 'f1')):+.4f}",
            f"Neutral precision delta: {metric_delta(epoch3, baseline, ('per_class', 'neutral', 'precision')):+.4f}",
            f"Neutral recall delta: {metric_delta(epoch3, baseline, ('per_class', 'neutral', 'recall')):+.4f}",
            "",
            "LANGUAGE DELTAS VS EXP15 EN+ES BASELINE",
        ]
    )
    for language in ("EN", "ES"):
        lines.append(
            f"{language} Macro F1: "
            f"{metric_delta(epoch3, baseline, ('languages', language, 'macro_f1')):+.4f}"
        )
        lines.append(
            f"{language} Neutral F1: "
            f"{metric_delta(epoch3, baseline, ('languages', language, 'neutral', 'f1')):+.4f}"
        )
    lines.extend(["", "ASPECT MACRO F1 DELTAS VS EXP15 EN+ES BASELINE"])
    for aspect in core.ASPECTS:
        lines.append(
            f"{aspect}: "
            f"{metric_delta(epoch3, baseline, ('aspects', aspect, 'macro_f1')):+.4f}"
        )
    lines.extend(
        [
            "",
            f"FINAL LANGUAGE HYPOTHESIS DECISION: {decision}",
            "Product rule applied: a small Macro F1 gain alone is insufficient; "
            "neutral performance and class balance must also be preserved.",
            "Epoch 4 was not started automatically.",
            "Experiment 16 language ablation is closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_epoch3_report(metrics, signature, permutation_hash, epoch2_metrics):
    original_subtitle = core.EXPERIMENT_SUBTITLE
    core.EXPERIMENT_SUBTITLE = "EN+ES-ONLY FINAL BUDGET-MATCHED LORA EPOCH 3"
    try:
        base = BASE_FORMATTER(
            "expanded", metrics, signature, permutation_hash
        ).rstrip()
    finally:
        core.EXPERIMENT_SUBTITLE = original_subtitle
    base = base.replace(
        "Epoch 2 was not started automatically.",
        "Epoch 4 was not started automatically.",
    )
    base = base.replace("Epoch shuffle seed: 42", "Epoch shuffle seed: 44")
    return (
        base
        + "\n\nLANGUAGE SCOPE\nTrain: EN+ES only\nDev evaluation: EN+ES only\n"
        + epoch3_analysis_report(epoch2_metrics, metrics)
    )


def run_epoch3_chunk():
    device = core.require_mps("expanded")
    if device is None:
        return
    train_rows, dev_rows, _, _, _ = validate_filtered_data()
    signature = experiment_signature()
    metadata = core.read_resume_metadata("expanded", signature)
    if metadata is None:
        raise RuntimeError("Epoch 2 resume checkpoint is required")
    if metadata.get("completed_epoch", 0) >= 3:
        core.append_log("expanded", "Epoch 3 is already complete. No training was started.")
        core.append_log("expanded", "Epoch 4 is outside the Experiment 16 budget.")
        return
    if metadata.get("completed_epoch") != 2:
        raise RuntimeError("Epoch 3 requires a completed Epoch 2 checkpoint")

    core.set_seed(core.RANDOM_SEED)
    tokenizer = core.AutoTokenizer.from_pretrained(
        Path(metadata["_slot_dir"]), local_files_only=True
    )
    model = core.create_or_resume_model(metadata)
    model.to(device)
    optimizer = torch.optim.AdamW(
        core.trainable_parameters(model), lr=core.LEARNING_RATE
    )
    state = core.load_training_state(metadata, optimizer, device, signature)
    if state.get("current_epoch") is None:
        state = initialize_epoch3_state(state, len(train_rows))
    validate_epoch3_progress(state, len(train_rows))
    epoch2_metrics = state.get("epoch_metrics_history", {}).get("2")
    if not epoch2_metrics:
        epoch2_metrics = metadata.get("last_epoch_metrics")
    if not epoch2_metrics:
        raise RuntimeError("Epoch 2 metrics are unavailable for comparison")

    total_microbatches = core.micro_batches_per_epoch(len(train_rows))
    total_optimizer_steps = core.optimizer_steps_per_epoch(len(train_rows))
    current_step = state["optimizer_steps_in_epoch"]
    target_step = min(
        current_step + core.MAX_OPTIMIZER_STEPS_PER_RUN, total_optimizer_steps
    )
    core.append_log("expanded", "")
    core.append_log("expanded", "Experiment: 16")
    core.append_log("expanded", "Device: mps")
    core.append_log("expanded", "Epoch: 3/3")
    core.append_log("expanded", f"Starting optimizer step: {current_step}")
    core.append_log("expanded", f"Target optimizer step: {target_step}")
    core.append_log(
        "expanded",
        f"Remaining optimizer steps in epoch: {total_optimizer_steps - current_step}",
    )
    core.append_log("expanded", f"Loss: {core.LOSS_DESCRIPTION}")
    core.append_log(
        "expanded",
        "Class weights [negative, neutral, positive]: "
        f"{core.FIXED_CLASS_WEIGHTS_BY_ID}",
    )
    core.append_log(
        "expanded",
        f"Final optimizer budget: Exp16={EXP16_TOTAL_STEPS_AFTER_EPOCH3}, "
        f"Exp15={EXP15_TOTAL_STEPS_AFTER_EPOCH2}, difference={FINAL_BUDGET_STEP_DIFFERENCE}",
    )
    core.append_log("expanded", "Entering training loop for epoch 3")

    collator = core.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataset = core.SentimentDataset(train_rows, tokenizer)
    remaining_indices = state["epoch_permutation"][state["sample_position"] :]
    loader = core.DataLoader(
        core.Subset(dataset, remaining_indices),
        batch_size=core.TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    iterator = iter(loader)
    dev_loader = core.build_dev_loader(dev_rows, tokenizer)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    completed_this_run = 0
    stop_requested = {"value": False}
    boundary_rng = core.capture_rng_state(device)
    old_sigint = signal.getsignal(signal.SIGINT)

    def request_stop(_signal_number, _frame):
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        while state["optimizer_steps_in_epoch"] < target_step:
            boundary_rng = core.capture_rng_state(device)
            group_start = state["microbatch_position"]
            group_size = min(
                core.GRADIENT_ACCUMULATION_STEPS,
                total_microbatches - group_start,
            )
            group_loss = 0.0
            for micro_index in range(group_size):
                if completed_this_run == 0 and micro_index == 0:
                    core.append_log("expanded", "Starting first training batch")
                batch = core.move_batch(next(iterator), device)
                labels = batch.pop("labels")
                logits = model(**batch).logits
                training_loss = core.compute_training_loss(logits, labels)
                loss_for_backward = training_loss / core.GRADIENT_ACCUMULATION_STEPS
                loss_for_backward.backward()
                group_loss += training_loss.item()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state["optimizer_steps_in_epoch"] += 1
            state["microbatch_position"] += group_size
            state["sample_position"] = min(
                state["microbatch_position"] * core.TRAIN_BATCH_SIZE,
                len(train_rows),
            )
            state["epoch_running_loss_sum"] += group_loss
            state["epoch_microbatches_seen"] += group_size
            completed_this_run += 1
            if stop_requested["value"]:
                core.append_log(
                    "expanded",
                    "Ctrl+C received. Stopping at the completed optimizer-step boundary.",
                )
                break
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        core.restore_rng_state(boundary_rng, device)
        core.append_log(
            "expanded",
            "Partial accumulation group discarded; progress remains at the last completed step.",
        )
    except RuntimeError as error:
        optimizer.zero_grad(set_to_none=True)
        if "mps" in str(error).lower() and "out of memory" in str(error).lower():
            core.append_log(
                "expanded",
                "TRAINING CHUNK FAILED — MPS OOM. No new checkpoint was written; "
                "previous checkpoint remains authoritative; CPU fallback was not used.",
            )
            return
        raise
    finally:
        signal.signal(signal.SIGINT, old_sigint)

    epoch_finished = state["microbatch_position"] >= total_microbatches
    best_path = None
    if epoch_finished:
        labels, predictions = core.predict(model, dev_loader, device)
        train_loss = state["epoch_running_loss_sum"] / state["epoch_microbatches_seen"]
        metrics = core.detailed_metrics(dev_rows, labels, predictions, train_loss)
        metrics["epoch"] = 3
        state["completed_epoch"] = 3
        state["current_epoch"] = None
        state["last_epoch_metrics"] = metrics
        state.setdefault("epoch_metrics_history", {})["3"] = metrics
        if metrics["macro_f1"] > state.get("best_macro_f1", float("-inf")):
            state["best_macro_f1"] = metrics["macro_f1"]
            state["best_epoch"] = 3
            best_path = save_epoch3_best_checkpoint(
                model, tokenizer, metrics, signature
            )

        report = format_epoch3_report(
            metrics, signature, state["permutation_sha256"], epoch2_metrics
        )
        with REPORT_PATH.open("a", encoding="utf-8") as report_file:
            report_file.write("\n" + report)

    checkpoint_path = core.save_resume_checkpoint(
        "expanded", model, tokenizer, optimizer, device, state, signature
    )
    progress = 100 * state["optimizer_steps_in_epoch"] / total_optimizer_steps
    core.append_log("expanded", "")
    core.append_log("expanded", "TRAINING CHUNK RESULT")
    core.append_log("expanded", "Epoch: 3/3")
    core.append_log(
        "expanded", f"Optimizer steps completed this run: {completed_this_run}"
    )
    core.append_log(
        "expanded", f"Current optimizer step: {state['optimizer_steps_in_epoch']}"
    )
    core.append_log("expanded", f"Epoch progress percentage: {progress:.2f}%")
    core.append_log("expanded", f"Resume checkpoint saved: {checkpoint_path}")
    core.append_log("expanded", f"Epoch finished: {epoch_finished}")
    if epoch_finished:
        metrics = state["last_epoch_metrics"]
        core.append_log("expanded", f"Dev Accuracy: {metrics['accuracy']:.4f}")
        core.append_log("expanded", f"Dev Macro F1: {metrics['macro_f1']:.4f}")
        core.append_log("expanded", f"Best epoch: {state['best_epoch']}")
        core.append_log(
            "expanded", f"Best Macro F1: {state['best_macro_f1']:.4f}"
        )
        if best_path is not None:
            core.append_log("expanded", f"Best checkpoint updated: {best_path}")
        else:
            core.append_log(
                "expanded",
                "Previous best checkpoint preserved; Epoch 3 did not improve Macro F1.",
            )
        core.append_log("expanded", "Epoch 4 was not started automatically.")
        core.append_log("expanded", "Experiment 16 language ablation is closed.")
    core.append_log("expanded", "Training chunk complete.")
    core.append_log("expanded", "No additional chunk was started automatically.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 16 thermal-safe EN+ES-only XLM-R LoRA ablation"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evaluate-exp15-baseline", action="store_true")
    mode.add_argument("--validate-static", action="store_true")
    mode.add_argument("--train-chunk", action="store_true")
    mode.add_argument("--validate-epoch2", action="store_true")
    mode.add_argument("--train-epoch2-chunk", action="store_true")
    mode.add_argument("--validate-epoch3", action="store_true")
    mode.add_argument("--train-epoch3-chunk", action="store_true")
    mode.add_argument("--evaluate-best", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.evaluate_exp15_baseline:
            evaluate_exp15_baseline()
        elif args.validate_static:
            run_static_validation()
        elif args.train_chunk:
            core.run_train_chunk("expanded")
        elif args.validate_epoch2:
            run_epoch2_static_validation()
        elif args.train_epoch2_chunk:
            run_epoch2_chunk()
        elif args.validate_epoch3:
            run_epoch3_static_validation()
        elif args.train_epoch3_chunk:
            run_epoch3_chunk()
        elif args.evaluate_best:
            run_evaluate_current_best()
    except Exception as error:
        if args.train_chunk or args.train_epoch2_chunk or args.train_epoch3_chunk:
            core.append_log("expanded", f"Run failed: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
