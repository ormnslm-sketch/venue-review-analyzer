"""Experiment 15: expanded-neutral data with expanded-balanced LoRA loss.

This script reuses Experiment 13/14's validated MPS-only, deterministic,
chunked and resumable training infrastructure. It never references the test
split. Epoch 2 is an explicit resume-only extension; Epoch 3 is blocked.
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
from sklearn.utils.class_weight import compute_class_weight

import train_xlmr_lora_unweighted_ablation as core


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "expanded_neutral_v1" / "train.csv"
DEV_PATH = PROJECT_ROOT / "data" / "expanded_neutral_v1" / "dev.csv"
CLEAN_DEV_PATH = PROJECT_ROOT / "data" / "cleaned" / "dev.csv"
MODEL_ROOT = PROJECT_ROOT / "models" / "experiment_15_lora_expanded_balanced"
REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "experiment_15_lora_expanded_balanced.txt"
)

# Order is exactly [negative=0, neutral=1, positive=2]. These are the balanced
# weights implied by 3,523 expanded-train rows and counts 1,050/300/2,173.
EXPANDED_BALANCED_WEIGHTS = (
    1.1184126984126983,
    3.9144444444444444,
    0.540420309863476,
)
EXPECTED_COUNTS = {"negative": 1050, "neutral": 300, "positive": 2173}
EXPECTED_TRAIN_ROWS = 3523
EXPECTED_DEV_ROWS = 901

PREVIOUS_MODEL_ROOTS = (
    PROJECT_ROOT / "models" / "xlmr-sentiment-v3-lora-weighted",
    PROJECT_ROOT / "models" / "xlmr-sentiment-lora-weighted-corrected",
    PROJECT_ROOT / "models" / "experiment_13a_lora_clean_unweighted",
    PROJECT_ROOT / "models" / "experiment_13b_lora_expanded_unweighted",
    PROJECT_ROOT / "models" / "experiment_14a_lora_clean_weighted",
    PROJECT_ROOT / "models" / "experiment_14b_lora_expanded_weighted",
)


def configure_core():
    """Configure the shared infrastructure for this isolated experiment."""
    # The clean entry exists only so shared static validation can prove that
    # clean and expanded DEV files are identical. This script never exposes a
    # clean training mode.
    core.CONDITIONS = {
        "clean": {
            "experiment": "15-DEV-REFERENCE",
            "train": PROJECT_ROOT / "data" / "cleaned" / "train.csv",
            "dev": CLEAN_DEV_PATH,
            "model_root": PROJECT_ROOT / "models" / ".experiment_15_reference_only",
            "report": PROJECT_ROOT / "data" / "analysis" / ".experiment_15_reference.txt",
        },
        "expanded": {
            "experiment": "15",
            "train": TRAIN_PATH,
            "dev": DEV_PATH,
            "model_root": MODEL_ROOT,
            "report": REPORT_PATH,
        },
    }
    core.TOTAL_TARGET_EPOCHS = 1
    core.EXPERIMENT_SUBTITLE = "EXPANDED-BALANCED CORRECTED WEIGHTED LORA EPOCH 1"
    core.LOSS_DESCRIPTION = (
        "mean(F.cross_entropy(logits, labels, reduction='none') "
        "* class_weights[labels])"
    )
    core.FIXED_CLASS_WEIGHTS_BY_ID = EXPANDED_BALANCED_WEIGHTS
    core.SAMPLE_WEIGHT_DESCRIPTION = None


configure_core()
BASE_FORMATTER = core.format_condition_report


def collapse_summary(metrics):
    counts = metrics["predicted_counts"]
    total = sum(counts.values())
    dominant_label, dominant_count = max(counts.items(), key=lambda item: item[1])
    dominant_rate = dominant_count / total
    single_class_collapse = dominant_rate > 0.90
    all_classes_predicted = all(value > 0 for value in counts.values())
    negative_present = counts["negative"] > 0
    actual_neutral = metrics["per_class"]["neutral"]["support"]
    neutral_overpredicted = counts["neutral"] > actual_neutral
    lines = [
        "",
        "COLLAPSE CHECK",
        f"Total DEV: {total}",
        f"Predicted positive: {counts['positive']}",
        f"Predicted neutral: {counts['neutral']}",
        f"Predicted negative: {counts['negative']}",
        f"Dominant class: {dominant_label} ({dominant_rate:.2%})",
        f">90% single-class collapse: {single_class_collapse}",
        f"All three classes predicted: {all_classes_predicted}",
        f"Negative predictions present: {negative_present}",
        f"Neutral overpredicted versus true-neutral support: {neutral_overpredicted}",
    ]
    if single_class_collapse:
        lines.append(
            f"WARNING: predictions collapsed to {dominant_label} above the 90% threshold."
        )
    return "\n".join(lines)


def format_experiment_report(condition, metrics, config_signature, permutation_hash):
    base = BASE_FORMATTER(condition, metrics, config_signature, permutation_hash).rstrip()
    historical = "\n".join(
        [
            "",
            "HISTORICAL REFERENCES",
            "13B expanded + unweighted: Accuracy=0.6570 | Macro F1=0.2837 | "
            "Neutral F1=0.0000 | Pred neutral=0",
            "14A clean + old balanced weights: Accuracy=0.6526 | Macro F1=0.2845 | "
            "Neutral F1=0.0606 | Pred neutral=9",
            "14B expanded + old clean weights: Accuracy=0.0633 | Macro F1=0.0397 | "
            "Neutral F1=0.1190 | Pred neutral=901",
            "Classical TF-IDF baseline reference: Macro F1=0.5891",
        ]
    )
    return base + collapse_summary(metrics) + historical + "\n"


core.format_condition_report = format_experiment_report


def verify_expanded_distribution():
    rows = core.load_rows(TRAIN_PATH, "train")
    counts = Counter(row["sentiment"] for row in rows)
    if len(rows) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"Expanded train row mismatch: expected {EXPECTED_TRAIN_ROWS}, got {len(rows)}"
        )
    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Expanded train class-count mismatch: expected {EXPECTED_COUNTS}, got {dict(counts)}"
        )
    return rows, counts


def verify_balanced_weights(rows):
    labels = np.asarray([core.LABEL2ID[row["sentiment"]] for row in rows])
    classes = np.asarray([0, 1, 2], dtype=np.int64)
    calculated = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    expected = np.asarray(EXPANDED_BALANCED_WEIGHTS, dtype=np.float64)
    if not np.allclose(calculated, expected, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Balanced weight mismatch: expected {expected}, got {calculated}"
        )
    return calculated


def corrected_loss_sanity_check():
    logits = torch.zeros((3, 3), dtype=torch.float32)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    raw = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    weights = torch.tensor(EXPANDED_BALANCED_WEIGHTS, dtype=raw.dtype)
    weighted = raw * weights[labels]
    if not (weighted[1] > weighted[0] > weighted[2]):
        raise RuntimeError("Corrected weighted-loss sanity ordering failed")
    return raw.tolist(), weighted.tolist()


def run_static_validation():
    rows, counts = verify_expanded_distribution()
    calculated = verify_balanced_weights(rows)
    dev_rows = core.load_rows(DEV_PATH, "dev")
    clean_dev_rows = core.load_rows(CLEAN_DEV_PATH, "dev")
    if len(dev_rows) != EXPECTED_DEV_ROWS or dev_rows != clean_dev_rows:
        raise RuntimeError("Expanded DEV is not the expected unchanged 901-row clean DEV")
    if any(MODEL_ROOT.resolve() == root.resolve() for root in PREVIOUS_MODEL_ROOTS):
        raise RuntimeError("Experiment 15 model root overlaps a previous experiment")

    core.run_static_validation("expanded")
    raw, weighted = corrected_loss_sanity_check()
    expected_steps = core.optimizer_steps_per_epoch(len(rows))
    if expected_steps != 881:
        raise RuntimeError(f"Expected 881 optimizer steps, got {expected_steps}")

    print("Expanded train distribution verified:")
    print(f"  positive: {counts['positive']}")
    print(f"  neutral: {counts['neutral']}")
    print(f"  negative: {counts['negative']}")
    print(f"  total: {len(rows)}")
    print(f"DEV rows: {len(dev_rows)}")
    print("DEV logically identical to cleaned DEV: True")
    print("Label mapping: negative=0, neutral=1, positive=2")
    print(f"Calculated weights [negative, neutral, positive]: {calculated.tolist()}")
    print(f"Configured weights [negative, neutral, positive]: {list(EXPANDED_BALANCED_WEIGHTS)}")
    print("Old clean neutral weight 11.885304659498209 used: False")
    print("Added-row/source/sample weighting: NONE")
    print("Oversampling/undersampling/focal loss/label smoothing: NONE")
    print("Corrected loss reduction='none': True")
    print("Corrected loss divides by class-weight sum: False")
    print(f"Sanity raw losses: {[round(value, 6) for value in raw]}")
    print(f"Sanity weighted losses: {[round(value, 6) for value in weighted]}")
    print(f"Optimizer steps in Epoch 1: {expected_steps}")
    print(f"Max optimizer steps per invocation: {core.MAX_OPTIMIZER_STEPS_PER_RUN}")
    print("MPS required; CPU fallback blocked: True")
    print("Checkpoint isolated from prior experiments: True")
    print("Test split referenced or loaded: False")
    print("Training started: NO")


def experiment_signature():
    """Keep the original Epoch-1 config signature for exact resume continuity."""
    return core.stable_json_hash(core.experiment_config("expanded"))


def load_epoch1_resume_state_for_audit():
    signature = experiment_signature()
    metadata = core.read_resume_metadata("expanded", signature)
    if metadata is None:
        raise RuntimeError("Experiment 15 Epoch 1 resume checkpoint is missing")
    required = {
        "completed_epoch": 1,
        "current_epoch": None,
        "optimizer_steps_in_epoch": 881,
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
        raise RuntimeError("Unexpected scheduler state; Experiment 15 uses no scheduler")
    rng_state = state.get("rng_state") or {}
    if not {"python", "numpy", "torch"}.issubset(rng_state):
        raise RuntimeError("Epoch 1 RNG state is incomplete")
    if len(state.get("epoch_permutation", [])) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Epoch 1 stored permutation is incomplete")

    paths = core.condition_paths("expanded")
    if not paths["best_pointer"].exists():
        raise RuntimeError("Experiment 15 Epoch 1 best checkpoint is missing")
    pointer = json.loads(paths["best_pointer"].read_text(encoding="utf-8"))
    best_metrics_path = paths["best_dir"] / pointer["relative_path"] / "metrics.json"
    best_payload = json.loads(best_metrics_path.read_text(encoding="utf-8"))
    if best_payload.get("config_signature") != signature:
        raise RuntimeError("Epoch 1 best-checkpoint signature mismatch")
    if best_payload.get("best_epoch") != 1:
        raise RuntimeError("Expected Epoch 1 to be the current best checkpoint")
    return metadata, state, best_payload["metrics"], signature


def run_epoch2_static_validation():
    rows, _ = verify_expanded_distribution()
    verify_balanced_weights(rows)
    metadata, state, epoch1_metrics, signature = load_epoch1_resume_state_for_audit()
    epoch_seed, permutation = core.epoch_permutation(len(rows), 2)
    if epoch_seed != 43 or len(permutation) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Epoch 2 deterministic permutation validation failed")
    if core.optimizer_steps_per_epoch(len(rows)) != 881:
        raise RuntimeError("Epoch 2 optimizer-step count must be 881")

    print("EXPERIMENT 15 — EPOCH 2 RESUME VALIDATION")
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
    print("Next epoch: 2")
    print("Next optimizer step: 0")
    print("Optimizer steps in Epoch 2: 881")
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
    if state.get("sample_position") != min(position * core.TRAIN_BATCH_SIZE, train_count):
        raise RuntimeError("Epoch 2 sample and microbatch positions differ")
    expected_steps = (
        core.math.ceil(position / core.GRADIENT_ACCUMULATION_STEPS) if position else 0
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
    checkpoint_id = "15-best-epoch2"
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


def epoch_delta_report(epoch1, epoch2):
    def delta(path):
        left = epoch2
        right = epoch1
        for key in path:
            left = left[key]
            right = right[key]
        return left - right

    counts = epoch2["predicted_counts"]
    total = sum(counts.values())
    collapse = max(counts.values()) / total > 0.90
    all_three = all(value > 0 for value in counts.values())
    macro_delta = delta(("macro_f1",))
    if macro_delta > 0 and not collapse and all_three:
        decision = "A) TRANSFORMER UMUT VERİCİ — sonraki model adımı değerlendirilebilir."
    elif macro_delta < 0:
        decision = "C) EPOCH 2 KÖTÜLEŞTİ — Epoch 1 best checkpoint korunmalı ve model araştırması kapatılmalı."
    else:
        decision = "B) TRANSFORMER PLATEAU — mevcut en iyi checkpoint korunmalı ve model araştırması kapatılmalı."

    lines = [
        "",
        "EPOCH 2 MINUS EPOCH 1",
        f"Accuracy delta: {delta(('accuracy',)):+.4f}",
        f"Macro F1 delta: {macro_delta:+.4f}",
        f"Positive F1 delta: {delta(('per_class', 'positive', 'f1')):+.4f}",
        f"Neutral F1 delta: {delta(('per_class', 'neutral', 'f1')):+.4f}",
        f"Negative F1 delta: {delta(('per_class', 'negative', 'f1')):+.4f}",
        f"Neutral precision delta: {delta(('per_class', 'neutral', 'precision')):+.4f}",
        f"Neutral recall delta: {delta(('per_class', 'neutral', 'recall')):+.4f}",
        f"Predicted neutral delta: {delta(('predicted_counts', 'neutral')):+d}",
        f"False-neutral delta: {delta(('neutral_confusion', 'false_neutral_total')):+d}",
        "",
        "CLASSICAL BASELINE COMPARISON",
        f"Macro F1 delta vs 0.5891: {epoch2['macro_f1'] - 0.5891:+.4f}",
        f"Neutral F1 delta vs 0.2529: {epoch2['per_class']['neutral']['f1'] - 0.2529:+.4f}",
        "",
        "DECISION QUESTIONS",
        f"Macro F1 increased: {macro_delta > 0}",
        f"Neutral precision increased: {delta(('per_class', 'neutral', 'precision')) > 0}",
        f"Neutral recall preserved or increased: {delta(('per_class', 'neutral', 'recall')) >= 0}",
        f"False-neutral decreased: {delta(('neutral_confusion', 'false_neutral_total')) < 0}",
        f"Negative F1 increased: {delta(('per_class', 'negative', 'f1')) > 0}",
        f"All three classes predicted: {all_three}",
        f">90% single-class collapse: {collapse}",
        f"Recommendation: {decision}",
        "Epoch 3 was not started automatically.",
    ]
    return "\n".join(lines) + "\n"


def format_epoch2_report(metrics, signature, permutation_hash, epoch1_metrics):
    original_subtitle = core.EXPERIMENT_SUBTITLE
    core.EXPERIMENT_SUBTITLE = (
        "EXPANDED-BALANCED CORRECTED WEIGHTED LORA EPOCH 2"
    )
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
    return base + collapse_summary(metrics) + epoch_delta_report(epoch1_metrics, metrics)


def run_epoch2_chunk():
    device = core.require_mps("expanded")
    if device is None:
        return
    train_rows = core.load_rows(TRAIN_PATH, "train")
    dev_rows = core.load_rows(DEV_PATH, "dev")
    signature = experiment_signature()
    metadata = core.read_resume_metadata("expanded", signature)
    if metadata is None:
        raise RuntimeError("Epoch 1 resume checkpoint is required")
    if metadata.get("completed_epoch", 0) >= 2:
        core.append_log("expanded", "Epoch 2 is already complete. No training was started.")
        core.append_log("expanded", "Epoch 3 is outside the Experiment 15 budget.")
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
    core.append_log("expanded", "Experiment: 15")
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
                loss_for_backward = (
                    training_loss / core.GRADIENT_ACCUMULATION_STEPS
                )
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
        train_loss = (
            state["epoch_running_loss_sum"] / state["epoch_microbatches_seen"]
        )
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
                "expanded", "Epoch 1 best checkpoint preserved; Epoch 2 did not improve Macro F1."
            )
        core.append_log("expanded", "Epoch 3 was not started automatically.")
    core.append_log("expanded", "Training chunk complete.")
    core.append_log("expanded", "No additional chunk was started automatically.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 15 thermal-safe expanded-balanced XLM-R LoRA"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-static", action="store_true")
    mode.add_argument("--validate-epoch2", action="store_true")
    mode.add_argument("--train-epoch2-chunk", action="store_true")
    mode.add_argument("--evaluate-best", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.validate_static:
            run_static_validation()
        elif args.validate_epoch2:
            run_epoch2_static_validation()
        elif args.train_epoch2_chunk:
            run_epoch2_chunk()
        elif args.evaluate_best:
            core.run_evaluate_best("expanded")
    except Exception as error:
        core.append_log("expanded", f"Run failed: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
