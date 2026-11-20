"""Cache the exact XLM-RoBERTa base model required by final inference.

The final adapter metadata and frozen inference loader use Hugging Face's
historical ``xlm-roberta-base`` cache key. That key resolves to the official
``FacebookAI/xlm-roberta-base`` model documented below.

This script downloads model/tokenizer files only. It does not train a model,
load the project LoRA adapter, or modify any checkpoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


OFFICIAL_MODEL_ID = "FacebookAI/xlm-roberta-base"
INFERENCE_CACHE_ID = "xlm-roberta-base"
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
)
OPTIONAL_TOKENIZER_FILES = (
    "added_tokens.json",
    "special_tokens_map.json",
)


def _validate_snapshot(snapshot_path: str) -> Path:
    path = Path(snapshot_path).resolve()
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError(
            "Cached XLM-R snapshot is incomplete; missing: " + ", ".join(missing)
        )
    return path


def _create_inference_cache_entry(official_snapshot: Path) -> Path:
    """Expose the official snapshot under the frozen loader's legacy cache key."""
    revision = official_snapshot.name
    hub_cache = official_snapshot.parents[2]
    legacy_repo_cache = hub_cache / "models--xlm-roberta-base"
    legacy_snapshot = legacy_repo_cache / "snapshots" / revision
    legacy_snapshot.mkdir(parents=True, exist_ok=True)

    for name in (*REQUIRED_FILES, *OPTIONAL_TOKENIZER_FILES):
        source = official_snapshot / name
        target = legacy_snapshot / name
        if not source.is_file() or target.exists():
            continue
        try:
            os.link(source.resolve(), target)
        except OSError:
            shutil.copy2(source, target)

    refs = legacy_repo_cache / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision, encoding="utf-8")
    return _validate_snapshot(str(legacy_snapshot))


def ensure_base_model(*, check_only: bool = False) -> Path:
    """Return a complete compatible cache snapshot, downloading if permitted."""
    patterns = [*REQUIRED_FILES, *OPTIONAL_TOKENIZER_FILES]
    try:
        cached_path = snapshot_download(
            repo_id=INFERENCE_CACHE_ID,
            allow_patterns=patterns,
            local_files_only=True,
        )
        return _validate_snapshot(cached_path)
    except (LocalEntryNotFoundError, RuntimeError):
        pass

    try:
        official_path = snapshot_download(
            repo_id=OFFICIAL_MODEL_ID,
            allow_patterns=patterns,
            local_files_only=True,
        )
        return _create_inference_cache_entry(_validate_snapshot(official_path))
    except (LocalEntryNotFoundError, RuntimeError):
        if check_only:
            raise RuntimeError(
                f"{OFFICIAL_MODEL_ID} is not completely available in the local "
                "cache expected by final inference. Run this script without "
                "--check to download it."
            ) from None

    official_path = snapshot_download(
        repo_id=OFFICIAL_MODEL_ID,
        allow_patterns=patterns,
    )
    return _create_inference_cache_entry(_validate_snapshot(official_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the existing local cache without network access.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = ensure_base_model(check_only=args.check)
    print(f"Official model: {OFFICIAL_MODEL_ID}")
    print(f"Inference cache key: {INFERENCE_CACHE_ID}")
    print(f"Local snapshot: {snapshot}")
    print("Base model cache: READY")


if __name__ == "__main__":
    main()
