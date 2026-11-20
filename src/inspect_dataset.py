"""M-ABSA restaurant verisindeki geniş category'lerden örnekler göster."""

import random
import re

from datasets import load_dataset


DATASET_ID = "Multilingual-NLP/M-ABSA"
DOMAIN = "restaurant"
LANGUAGES = {
    "tr": "Turkish",
    "en": "English",
    "es": "Spanish",
}
SPLITS = ("train", "dev", "test")
TARGET_CATEGORIES = ("service general", "ambience general")
SAMPLE_SIZE = 15
RANDOM_SEED = 42

TRIPLET_PATTERN = re.compile(
    r"\[\s*(['\"])(.*?)\1\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*\]"
)


def load_language(language_code: str):
    """Bir dilin depodaki mevcut splitlerini ham metin olarak yükle."""
    data_files = {
        split: (
            f"hf://datasets/{DATASET_ID}/{DOMAIN}/"
            f"{language_code}/{split}.txt"
        )
        for split in SPLITS
    }
    return load_dataset("text", data_files=data_files)


def parse_row(raw_text: str):
    """Ham satırı cümle ve [term, category, sentiment] tripletlerine ayır."""
    sentence, separator, raw_labels = raw_text.partition("####")
    if not separator:
        raise ValueError(f"Satırda '####' ayırıcısı bulunamadı: {raw_text}")

    if raw_labels.strip() == "[]":
        return sentence, []

    matches = TRIPLET_PATTERN.findall(raw_labels)
    if not matches:
        raise ValueError(f"Label tripletleri okunamadı: {raw_labels}")

    triplets = [
        (term, category, sentiment)
        for _, term, category, sentiment in matches
    ]
    return sentence, triplets


def collect_examples(dataset):
    """Hedef category'lere ait sentence, term ve sentiment kayıtlarını topla."""
    examples = {category: [] for category in TARGET_CATEGORIES}

    for split in SPLITS:
        for row in dataset[split]:
            sentence, triplets = parse_row(row["text"])
            for aspect_term, category, sentiment in triplets:
                if category in examples:
                    examples[category].append(
                        {
                            "sentence": sentence,
                            "aspect_term": aspect_term,
                            "sentiment": sentiment,
                        }
                    )

    return examples


def choose_samples(examples):
    """Aynı seed ile her çalıştırmada aynı kayıtları seç."""
    random_generator = random.Random(RANDOM_SEED)
    sample_size = min(SAMPLE_SIZE, len(examples))
    return random_generator.sample(examples, sample_size)


def show_samples(language_name: str, category: str, examples) -> None:
    """Seçilen örnekleri okunabilir biçimde yazdır."""
    print("=" * 80)
    print(f"{language_name} — {category}")
    print(f"Random sample: {len(examples)} (seed={RANDOM_SEED})\n")

    for index, example in enumerate(examples, start=1):
        print(f"Example {index}")
        print(f"Sentence: {example['sentence']}")
        print(f"Aspect term: {example['aspect_term']}")
        print(f"Sentiment: {example['sentiment']}\n")


def main() -> None:
    examples_by_language = {}

    for language_code in LANGUAGES:
        dataset = load_language(language_code)
        examples_by_language[language_code] = collect_examples(dataset)

    for category in TARGET_CATEGORIES:
        for language_code, language_name in LANGUAGES.items():
            category_examples = examples_by_language[language_code][category]
            samples = choose_samples(category_examples)
            show_samples(language_name, category, samples)


if __name__ == "__main__":
    main()
