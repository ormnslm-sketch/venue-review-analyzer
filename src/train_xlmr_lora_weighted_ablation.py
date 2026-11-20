"""Experiment 14: clean vs expanded corrected-weighted LoRA data ablation.

This module reuses Experiment 13's validated MPS-only, deterministic,
chunked/resumable infrastructure. It changes only the training datasets and
activates the same fixed CLEAN-derived class weights for both conditions.

No test split is referenced. The experiment budget is exactly one epoch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import train_xlmr_lora_unweighted_ablation as core


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_WEIGHTS = (
    1.0526984126984127,   # negative -> 0
    11.885304659498209,   # neutral  -> 1
    0.5086669734621875,   # positive -> 2
)

CONDITIONS = {
    "clean": {
        "experiment": "14A",
        "train": PROJECT_ROOT / "data" / "cleaned" / "train.csv",
        "dev": PROJECT_ROOT / "data" / "cleaned" / "dev.csv",
        "model_root": PROJECT_ROOT / "models" / "experiment_14a_lora_clean_weighted",
        "report": PROJECT_ROOT
        / "data"
        / "analysis"
        / "experiment_14a_lora_clean_weighted.txt",
    },
    "expanded": {
        "experiment": "14B",
        "train": PROJECT_ROOT / "data" / "expanded_neutral_v1" / "train.csv",
        "dev": PROJECT_ROOT / "data" / "expanded_neutral_v1" / "dev.csv",
        "model_root": PROJECT_ROOT
        / "models"
        / "experiment_14b_lora_expanded_weighted",
        "report": PROJECT_ROOT
        / "data"
        / "analysis"
        / "experiment_14b_lora_expanded_weighted.txt",
    },
}
COMPARISON_REPORT = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "experiment_14_lora_weighted_data_ablation_comparison.txt"
)

CORRECTED_05B_SCRIPT = PROJECT_ROOT / "src" / "train_xlmr_lora_weighted_corrected.py"
CORRECTED_05B_ROOT = (
    PROJECT_ROOT / "models" / "xlmr-sentiment-lora-weighted-corrected"
)
LEGACY_INVALID_05_ROOT = PROJECT_ROOT / "models" / "xlmr-sentiment-v3-lora-weighted"


def configure_core():
    """Apply Experiment 14's controlled settings to the shared infrastructure."""
    core.CONDITIONS = CONDITIONS
    core.COMPARISON_REPORT = COMPARISON_REPORT
    core.TOTAL_TARGET_EPOCHS = 1
    core.EXPERIMENT_SUBTITLE = "CORRECTED WEIGHTED LORA EPOCH 1"
    core.LOSS_DESCRIPTION = (
        "mean(F.cross_entropy(logits, labels, reduction='none') "
        "* class_weights[labels])"
    )
    core.FIXED_CLASS_WEIGHTS_BY_ID = FIXED_WEIGHTS
    core.SAMPLE_WEIGHT_DESCRIPTION = None


configure_core()


def corrected_loss_sanity_check():
    """Verify explicit per-example weighting without changing training state."""
    logits = torch.zeros((3, 3), dtype=torch.float32)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    raw = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    weighted = raw * torch.tensor(FIXED_WEIGHTS, dtype=raw.dtype)
    if not (weighted[1] > weighted[0] > weighted[2]):
        raise RuntimeError("Corrected weighted-loss sanity ordering failed")
    return raw.tolist(), weighted.tolist()


def audit_experiment_05b():
    """Audit whether an actual corrected 05B Epoch 1 checkpoint can serve as 14A."""
    source = CORRECTED_05B_SCRIPT.read_text(encoding="utf-8")
    required_fragments = (
        'MODEL_NAME = "xlm-roberta-base"',
        'LORA_TARGET_MODULES = ["query", "value"]',
        "LORA_R = 8",
        "LORA_ALPHA = 16",
        "LORA_DROPOUT = 0.1",
        "LEARNING_RATE = 2e-5",
        "TRAIN_BATCH_SIZE = 1",
        "GRADIENT_ACCUMULATION_STEPS = 4",
        "MAX_LENGTH = 128",
        "RANDOM_SEED = 42",
        'MODULES_TO_SAVE = ["classifier"]',
        'raw_loss = F.cross_entropy(logits, labels, reduction="none")',
        "sample_weights = class_weights_tensor[labels]",
        "weighted_loss = (raw_loss * sample_weights).mean()",
        "weighted_loss / GRADIENT_ACCUMULATION_STEPS",
    )
    missing_fragments = [item for item in required_fragments if item not in source]
    best_pointer = CORRECTED_05B_ROOT / "best" / "current.json"
    corrected_checkpoint_exists = best_pointer.exists()
    legacy_checkpoint_exists = (LEGACY_INVALID_05_ROOT / "best" / "current.json").exists()
    usable_as_14a = not missing_fragments and corrected_checkpoint_exists

    print("EXPERIMENT 05B COMPATIBILITY AUDIT")
    print(f"Corrected script configuration compatible: {not missing_fragments}")
    print(f"Missing required configuration markers: {missing_fragments}")
    print(f"Corrected 05B best checkpoint exists: {corrected_checkpoint_exists}")
    print(f"Legacy invalid Experiment 05 checkpoint exists: {legacy_checkpoint_exists}")
    print("Legacy invalid checkpoint reused: False")
    print(f"Experiment 05B usable as Condition 14A: {usable_as_14a}")
    if not corrected_checkpoint_exists:
        print(
            "Reason: corrected 05B model directory has no best Epoch 1 checkpoint; "
            "its report contains preparation/validation only."
        )
        print("Action: prepare a fresh, isolated Condition 14A run.")
    return usable_as_14a


def run_static_validation(condition):
    core.run_static_validation(condition)
    paths = core.condition_paths(condition)
    forbidden_roots = {
        CORRECTED_05B_ROOT.resolve(),
        LEGACY_INVALID_05_ROOT.resolve(),
        (PROJECT_ROOT / "models" / "experiment_13a_lora_clean_unweighted").resolve(),
        (PROJECT_ROOT / "models" / "experiment_13b_lora_expanded_unweighted").resolve(),
    }
    if paths["model_root"].resolve() in forbidden_roots:
        raise RuntimeError("Experiment 14 checkpoint root overlaps an earlier experiment")
    raw, weighted = corrected_loss_sanity_check()
    print("Checkpoint isolated from Experiments 05/05B/13: True")
    print("Label mapping: negative=0, neutral=1, positive=2")
    print(f"Fixed weight order [negative, neutral, positive]: {FIXED_WEIGHTS}")
    print("Class weights recomputed from expanded data: False")
    print("Added-row/sample weighting: NONE")
    print("Corrected loss uses reduction='none': True")
    print("Corrected loss divides by class-weight sum: False")
    print(f"Sanity raw losses: {[round(value, 6) for value in raw]}")
    print(f"Sanity weighted losses: {[round(value, 6) for value in weighted]}")
    print("Training started: NO")


def load_metrics(condition):
    return core.load_best_metrics(condition)


def run_comparison():
    clean = load_metrics("clean")
    expanded = load_metrics("expanded")

    def delta(path):
        left = expanded
        right = clean
        for key in path:
            left = left[key]
            right = right[key]
        return left - right

    lines = [
        "EXPERIMENT 14 — CORRECTED WEIGHTED LORA DATA ABLATION",
        "",
        "14B minus 14A after Epoch 1; test split was not used.",
        "Both conditions use identical CLEAN-derived fixed class weights.",
        "Only the training dataset differs.",
        "",
        f"Accuracy delta: {delta(('accuracy',)):+.4f}",
        f"Macro F1 delta: {delta(('macro_f1',)):+.4f}",
    ]
    for sentiment in core.SENTIMENTS:
        lines.append(
            f"{sentiment} F1 delta: "
            f"{delta(('per_class', sentiment, 'f1')):+.4f}"
        )
    lines.extend(
        [
            f"Neutral precision delta: {delta(('per_class', 'neutral', 'precision')):+.4f}",
            f"Neutral recall delta: {delta(('per_class', 'neutral', 'recall')):+.4f}",
            f"Predicted neutral count delta: {delta(('predicted_counts', 'neutral')):+d}",
            f"False-neutral count delta: {delta(('neutral_confusion', 'false_neutral_total')):+d}",
            "",
            "LANGUAGE MACRO F1 DELTAS",
        ]
    )
    for _, language in core.LANGUAGES:
        lines.append(
            f"{language}: {delta(('languages', language, 'macro_f1')):+.4f}"
        )
    lines.extend(["", "ASPECT MACRO F1 DELTAS"])
    for aspect in core.ASPECTS:
        lines.append(f"{aspect}: {delta(('aspects', aspect, 'macro_f1')):+.4f}")
    lines.extend(
        [
            "",
            "Epoch 2 was not started automatically for either condition.",
        ]
    )
    COMPARISON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 14 thermal-safe corrected-weighted LoRA data ablation"
    )
    parser.add_argument("--condition", choices=sorted(CONDITIONS))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-05b", action="store_true")
    mode.add_argument("--validate-static", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--train-chunk", action="store_true")
    mode.add_argument("--evaluate-best", action="store_true")
    mode.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    conditionless = args.audit_05b or args.compare
    if not conditionless and args.condition is None:
        parser.error("--condition is required for this mode")
    if conditionless and args.condition is not None:
        parser.error("omit --condition with --audit-05b or --compare")
    return args


def main():
    args = parse_args()
    if args.audit_05b:
        audit_experiment_05b()
        return
    if args.compare:
        run_comparison()
        return
    try:
        if args.validate_static:
            run_static_validation(args.condition)
        elif args.smoke_test:
            core.run_smoke_test(args.condition)
        elif args.train_chunk:
            core.run_train_chunk(args.condition)
        elif args.evaluate_best:
            core.run_evaluate_best(args.condition)
    except Exception as error:
        core.append_log(args.condition, f"Run failed: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
