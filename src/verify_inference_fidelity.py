"""Verify final-checkpoint restoration and evaluation/inference fidelity.

This diagnostic performs forward-only inference. It creates no optimizer,
calls no backward pass, updates no parameter, and writes no checkpoint.
"""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from peft import PeftModel
from safetensors import safe_open

import inference
import train_xlmr_lora_expanded_balanced as experiment15


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = inference.CHECKPOINT_PATH
DEV_PATH = PROJECT_ROOT / "data" / "expanded_neutral_v1" / "dev.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "phase_2a1_inference_fidelity.txt"
TOLERANCE = 1e-5

SAMPLE_SPECS = (
    ("positive", "tr", "taste"),
    ("neutral", "en", "service"),
    ("negative", "es", "atmosphere"),
)


def select_samples() -> List[Dict[str, str]]:
    with DEV_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    selected = []
    for sentiment, language, aspect in SAMPLE_SPECS:
        matches = [
            row
            for row in rows
            if row["sentiment"] == sentiment
            and row["language"] == language
            and row["aspect"] == aspect
        ]
        if not matches:
            raise RuntimeError(
                f"Missing fixed DEV sample: {sentiment}/{language}/{aspect}"
            )
        selected.append(matches[0])
    return selected


def forward_individually(model, tokenizer, inputs, device):
    logits_rows = []
    probability_rows = []
    model.eval()
    with torch.inference_mode():
        for model_text in inputs:
            encoded = tokenizer(
                model_text,
                truncation=True,
                max_length=inference.MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.cpu()
            probabilities = torch.softmax(logits, dim=-1)
            logits_rows.append(logits[0])
            probability_rows.append(probabilities[0])
    return torch.stack(logits_rows), torch.stack(probability_rows)


def load_evaluation_pipeline(samples):
    """Path A: Experiment 15's training/evaluation loading components."""
    core = experiment15.core
    core.set_seed(core.RANDOM_SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = core.AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    base_model = core.AutoModelForSequenceClassification.from_pretrained(
        core.MODEL_NAME,
        num_labels=len(core.LABEL2ID),
        label2id=core.LABEL2ID,
        id2label=core.ID2LABEL,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        CHECKPOINT,
        is_trainable=False,
    )
    model.requires_grad_(False)
    model.to(device)
    inputs = [core.model_input(row) for row in samples]
    logits, probabilities = forward_individually(
        model, tokenizer, inputs, device
    )
    return logits, probabilities, inputs, device.type, model, tokenizer


def load_public_inference_pipeline(samples):
    """Path B: src/inference.py loader, formatter, and public prediction API."""
    bundle = inference.load_model()
    inputs = [
        inference.prepare_input(row["sentence"], row["aspect"])
        for row in samples
    ]
    logits, probabilities = forward_individually(
        bundle.model,
        bundle.tokenizer,
        inputs,
        bundle.device,
    )
    public_results = [
        inference.predict_sentiment(row["sentence"], row["aspect"])
        for row in samples
    ]
    return logits, probabilities, inputs, public_results, bundle


def checkpoint_to_loaded_key(checkpoint_key: str) -> str:
    if ".classifier." in checkpoint_key:
        return checkpoint_key.replace(
            ".classifier.", ".classifier.modules_to_save.default.", 1
        )
    if ".lora_A.weight" in checkpoint_key:
        return checkpoint_key.replace(".lora_A.weight", ".lora_A.default.weight")
    if ".lora_B.weight" in checkpoint_key:
        return checkpoint_key.replace(".lora_B.weight", ".lora_B.default.weight")
    raise RuntimeError(f"Unexpected adapter tensor key: {checkpoint_key}")


def compare_checkpoint_tensors(model):
    loaded = model.state_dict()
    classifier_matched = 0
    classifier_mismatched = 0
    lora_matched = 0
    lora_mismatched = 0
    missing = []
    with safe_open(
        CHECKPOINT / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as checkpoint:
        for checkpoint_key in checkpoint.keys():
            loaded_key = checkpoint_to_loaded_key(checkpoint_key)
            if loaded_key not in loaded:
                missing.append(loaded_key)
                continue
            expected = checkpoint.get_tensor(checkpoint_key)
            actual = loaded[loaded_key].detach().cpu()
            matched = torch.equal(expected, actual)
            if ".classifier." in checkpoint_key:
                classifier_matched += int(matched)
                classifier_mismatched += int(not matched)
            else:
                lora_matched += int(matched)
                lora_mismatched += int(not matched)
    if missing:
        raise RuntimeError(f"Loaded adapter tensors are missing: {missing[:5]}")
    return {
        "classifier_matched": classifier_matched,
        "classifier_mismatched": classifier_mismatched,
        "lora_matched": lora_matched,
        "lora_mismatched": lora_mismatched,
    }


def main():
    samples = select_samples()

    logits_a, probabilities_a, inputs_a, device_a, model_a, tokenizer_a = (
        load_evaluation_pipeline(samples)
    )
    del model_a
    del tokenizer_a
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    logits_b, probabilities_b, inputs_b, public_results, bundle = (
        load_public_inference_pipeline(samples)
    )
    tensor_comparison = compare_checkpoint_tensors(bundle.model)

    if inputs_a != inputs_b:
        raise RuntimeError("Evaluation and inference input representations differ")
    max_logit_difference = float((logits_a - logits_b).abs().max().item())
    max_probability_difference = float(
        (probabilities_a - probabilities_b).abs().max().item()
    )
    public_probability_difference = 0.0
    rows = []
    for index, (sample, result) in enumerate(zip(samples, public_results)):
        public_probabilities = torch.tensor(
            [
                result["probabilities"][inference.ID2LABEL[label_id]]
                for label_id in range(3)
            ],
            dtype=probabilities_b.dtype,
        )
        public_probability_difference = max(
            public_probability_difference,
            float(
                (public_probabilities - probabilities_b[index]).abs().max().item()
            ),
        )
        predicted_a = inference.ID2LABEL[int(probabilities_a[index].argmax().item())]
        predicted_b = result["sentiment"]
        rows.append(
            {
                "true_label": sample["sentiment"],
                "language": sample["language"],
                "aspect": sample["aspect"],
                "sentence": sample["sentence"],
                "evaluation_logits": logits_a[index].tolist(),
                "inference_logits": logits_b[index].tolist(),
                "evaluation_probabilities": probabilities_a[index].tolist(),
                "inference_probabilities": probabilities_b[index].tolist(),
                "evaluation_prediction": predicted_a,
                "inference_prediction": predicted_b,
                "prediction_match": predicted_a == predicted_b,
            }
        )

    classifier_ok = (
        tensor_comparison["classifier_matched"] == 4
        and tensor_comparison["classifier_mismatched"] == 0
    )
    lora_ok = (
        tensor_comparison["lora_matched"] == 48
        and tensor_comparison["lora_mismatched"] == 0
    )
    mapping_ok = (
        bundle.model.config.label2id == inference.LABEL2ID
        and bundle.model.config.id2label == inference.ID2LABEL
    )
    predictions_match = all(row["prediction_match"] for row in rows)
    fidelity_pass = (
        classifier_ok
        and lora_ok
        and mapping_ok
        and inputs_a == inputs_b
        and predictions_match
        and max_probability_difference < TOLERANCE
        and public_probability_difference < TOLERANCE
    )

    report_lines = [
        "PHASE 2A.1 — FINAL CHECKPOINT INFERENCE FIDELITY CHECK",
        "",
        f"Final checkpoint loaded: {'YES' if classifier_ok and lora_ok else 'NO'}",
        f"LoRA adapter restored: {'YES' if lora_ok else 'NO'}",
        f"Classifier restored: {'YES' if classifier_ok else 'NO'}",
        f"Label mapping correct: {'YES' if mapping_ok else 'NO'}",
        f"Input format correct: {'YES' if inputs_a == inputs_b else 'NO'}",
        f"Evaluation vs inference fidelity: {'PASS' if fidelity_pass else 'FAIL'}",
        f"Max logit difference: {max_logit_difference:.12g}",
        f"Max probability difference: {max_probability_difference:.12g}",
        f"Public API probability difference: {public_probability_difference:.12g}",
        f"Tolerance: {TOLERANCE}",
        f"Evaluation device: {device_a}",
        f"Inference device: {bundle.device.type}",
        "",
        "CHECKPOINT TENSOR COMPARISON",
        f"Classifier matched parameter count: {tensor_comparison['classifier_matched']}",
        f"Classifier mismatched parameter count: {tensor_comparison['classifier_mismatched']}",
        f"LoRA matched parameter count: {tensor_comparison['lora_matched']}",
        f"LoRA mismatched parameter count: {tensor_comparison['lora_mismatched']}",
        "",
        "FIXED DEV SAMPLES",
    ]
    for index, row in enumerate(rows, start=1):
        report_lines.extend(
            [
                f"{index}. true={row['true_label']} | language={row['language']} | aspect={row['aspect']}",
                f"   sentence: {row['sentence']}",
                f"   evaluation logits: {[round(value, 8) for value in row['evaluation_logits']]}",
                f"   inference logits: {[round(value, 8) for value in row['inference_logits']]}",
                f"   evaluation probabilities: {[round(value, 8) for value in row['evaluation_probabilities']]}",
                f"   inference probabilities: {[round(value, 8) for value in row['inference_probabilities']]}",
                f"   prediction: {row['evaluation_prediction']} / {row['inference_prediction']}",
            ]
        )
    report_lines.extend(
        [
            "",
            "HUGGING FACE WARNING",
            "Status: HARMLESS" if fidelity_pass else "Status: REAL PROBLEM",
            "xlm-roberta-base has no trained task-specific sequence-classification head. "
            "Transformers therefore reports that the temporary base classifier is newly "
            "initialized. PeftModel.from_pretrained then restores the four trained "
            "Experiment 15 classifier tensors into the active modules_to_save.default "
            "classifier before the first inference forward pass.",
            "",
            (
                "Phase 2A inference pipeline verified. Aspect Presence Detection "
                "aşamasına geçilebilir."
                if fidelity_pass
                else "Inference fidelity failed. Aspect Presence Detection'a geçmeyin."
            ),
        ]
    )
    report = "\n".join(report_lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    if not fidelity_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
