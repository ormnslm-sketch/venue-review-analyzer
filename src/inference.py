"""Final Experiment 15 Epoch 2 sentiment inference pipeline.

The model answers only: known aspect + review text -> sentiment. It does not
detect whether an aspect is mentioned. No training or parameter updates occur.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "experiment_15_lora_expanded_balanced"
    / "best"
    / "epoch_2"
)
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
SUPPORTED_ASPECTS = ("taste", "atmosphere", "service", "value")
SUPPORTED_LANGUAGES = ("tr", "en", "es")

# This must remain identical to training.
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
EXPECTED_CHECKPOINT_ID = "15-best-epoch2"

if {label: ID2LABEL[label_id] for label, label_id in LABEL2ID.items()} != {
    label: label for label in LABEL2ID
}:
    raise RuntimeError("LABEL2ID and ID2LABEL are inconsistent")


@dataclass(frozen=True)
class ModelBundle:
    model: PeftModel
    tokenizer: Any
    device: torch.device


_MODEL_BUNDLE: ModelBundle | None = None


def _validate_checkpoint() -> None:
    required_files = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "metrics.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = sorted(
        filename
        for filename in required_files
        if not (CHECKPOINT_PATH / filename).exists()
    )
    if missing:
        raise FileNotFoundError(
            f"Final Experiment 15 Epoch 2 checkpoint is incomplete: {missing}"
        )

    metrics = json.loads(
        (CHECKPOINT_PATH / "metrics.json").read_text(encoding="utf-8")
    )
    adapter = json.loads(
        (CHECKPOINT_PATH / "adapter_config.json").read_text(encoding="utf-8")
    )
    if metrics.get("checkpoint_id") != EXPECTED_CHECKPOINT_ID:
        raise RuntimeError("Refusing to load a checkpoint other than Experiment 15 Epoch 2")
    if metrics.get("best_epoch") != 2 or metrics.get("condition") != "expanded":
        raise RuntimeError("Final checkpoint identity is inconsistent")
    if adapter.get("base_model_name_or_path") != MODEL_NAME:
        raise RuntimeError("Final checkpoint base model is not xlm-roberta-base")
    if adapter.get("peft_type") != "LORA" or adapter.get("task_type") != "SEQ_CLS":
        raise RuntimeError("Final checkpoint is not a LoRA sequence classifier")
    if adapter.get("r") != 8 or adapter.get("lora_alpha") != 16:
        raise RuntimeError("Final checkpoint LoRA rank/alpha mismatch")
    if not math.isclose(float(adapter.get("lora_dropout")), 0.1, abs_tol=1e-12):
        raise RuntimeError("Final checkpoint LoRA dropout mismatch")
    if set(adapter.get("target_modules") or []) != {"query", "value"}:
        raise RuntimeError("Final checkpoint LoRA target modules mismatch")
    if "classifier" not in set(adapter.get("modules_to_save") or []):
        raise RuntimeError("Final checkpoint does not declare the classifier head")


def _select_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model() -> ModelBundle:
    """Load and cache the final tokenizer, base model, adapter, and classifier."""
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    _validate_checkpoint()
    device = _select_device()
    print(f"Inference device: {device.type}", file=sys.stderr, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_PATH,
        local_files_only=True,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        CHECKPOINT_PATH,
        is_trainable=False,
    )
    model.requires_grad_(False)
    model.to(device)
    model.eval()

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Inference model unexpectedly contains trainable parameters")
    if model.training:
        raise RuntimeError("Inference model is not in eval mode")
    if model.config.label2id != LABEL2ID or model.config.id2label != ID2LABEL:
        raise RuntimeError("Loaded model label mapping differs from training")

    _MODEL_BUNDLE = ModelBundle(model=model, tokenizer=tokenizer, device=device)
    return _MODEL_BUNDLE


def prepare_input(text: str, aspect: str) -> str:
    """Validate inputs and reproduce the exact training input representation."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not isinstance(aspect, str):
        raise ValueError(f"aspect must be one of: {', '.join(SUPPORTED_ASPECTS)}")
    normalized_aspect = aspect.strip().lower()
    if normalized_aspect not in SUPPORTED_ASPECTS:
        raise ValueError(
            f"Unsupported aspect '{aspect}'. Supported aspects: "
            f"{', '.join(SUPPORTED_ASPECTS)}"
        )
    return f"aspect={normalized_aspect} [SEP] {text.strip()}"


def _predict_batch(items: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Run one inference batch; this is ready to be extended to review lists."""
    if not items:
        return []
    prepared = [prepare_input(text, aspect) for text, aspect in items]
    bundle = load_model()
    encoded = bundle.tokenizer(
        prepared,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = bundle.model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).cpu()

    if probabilities.shape != (len(items), len(LABEL2ID)):
        raise RuntimeError(f"Unexpected probability shape: {tuple(probabilities.shape)}")

    results = []
    for (_, aspect), row in zip(items, probabilities):
        probability_sum = float(row.sum().item())
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise RuntimeError(f"Probability sum is not 1.0: {probability_sum}")
        predicted_id = int(row.argmax().item())
        if predicted_id not in ID2LABEL:
            raise RuntimeError(f"Unknown predicted label id: {predicted_id}")
        normalized_aspect = aspect.strip().lower()
        probability_map = {
            ID2LABEL[label_id]: float(row[label_id].item())
            for label_id in sorted(ID2LABEL)
        }
        results.append(
            {
                "aspect": normalized_aspect,
                "sentiment": ID2LABEL[predicted_id],
                "confidence": probability_map[ID2LABEL[predicted_id]],
                "probabilities": probability_map,
            }
        )
    return results


def predict_sentiment(text: str, aspect: str) -> Dict[str, Any]:
    """Predict sentiment for one known, caller-supplied aspect."""
    return _predict_batch([(text, aspect)])[0]


def predict_all_aspects(text: str) -> Dict[str, Any]:
    """Score all aspects without claiming that any aspect is mentioned."""
    predictions = _predict_batch([(text, aspect) for aspect in SUPPORTED_ASPECTS])
    return {
        "text": text,
        "predictions": {
            result["aspect"]: {
                **result,
                "aspect_presence": "unknown",
            }
            for result in predictions
        },
        "warning": "Aspect presence detection is not implemented yet.",
    }


def run_smoke_test() -> List[Dict[str, Any]]:
    """Technical multilingual smoke test; expected labels are not hard-coded."""
    cases = [
        ("tr", "Kahvenin tadı gerçekten çok iyiydi.", "taste"),
        ("en", "The service was extremely slow.", "service"),
        ("es", "El ambiente era muy agradable.", "atmosphere"),
    ]
    predictions = _predict_batch([(text, aspect) for _, text, aspect in cases])
    output = []
    for (language, text, aspect), prediction in zip(cases, predictions):
        if prediction["sentiment"] not in LABEL2ID:
            raise RuntimeError("Smoke test produced an invalid sentiment label")
        probability_sum = sum(prediction["probabilities"].values())
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise RuntimeError("Smoke test probabilities do not sum to 1")
        output.append(
            {
                "language": language,
                "text": text,
                **prediction,
                "probability_sum": probability_sum,
            }
        )
    return output


def _print_single(text: str, result: Dict[str, Any]) -> None:
    print(f"Text: {text}")
    print(f"Aspect: {result['aspect']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.6f}")
    print("Probabilities:")
    for sentiment in ("negative", "neutral", "positive"):
        print(f"  {sentiment}: {result['probabilities'][sentiment]:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 15 Epoch 2 known-aspect sentiment inference"
    )
    parser.add_argument("--text", help="Review text for prediction")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--aspect", choices=SUPPORTED_ASPECTS)
    mode.add_argument("--all-aspects", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if not args.smoke_test and not args.text:
        parser.error("--text is required with --aspect or --all-aspects")
    if args.smoke_test and args.text:
        parser.error("--smoke-test does not accept --text")
    return args


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        print(json.dumps(run_smoke_test(), ensure_ascii=False, indent=2))
    elif args.all_aspects:
        print(json.dumps(predict_all_aspects(args.text), ensure_ascii=False, indent=2))
    else:
        _print_single(args.text, predict_sentiment(args.text, args.aspect))


if __name__ == "__main__":
    main()
