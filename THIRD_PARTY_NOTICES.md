# Third-Party Notices

This file documents upstream datasets and pretrained model resources used by Venue Review Analyzer. These notices do not define a license for this repository's own source code.

## M-ABSA

- Source: [Multilingual-NLP/M-ABSA](https://huggingface.co/datasets/Multilingual-NLP/M-ABSA)
- Paper: ChengYan Wu, Bolei Ma, Yihong Liu, Zheyu Zhang, Ningyuan Deng, Yanshu Li, Baolan Chen, Yi Zhang, Yun Xue, and Barbara Plank. “M-ABSA: A Multilingual Dataset for Aspect-Based Sentiment Analysis.” EMNLP 2025. [ACL Anthology](https://aclanthology.org/2025.emnlp-main.128/), DOI: [`10.18653/v1/2025.emnlp-main.128`](https://doi.org/10.18653/v1/2025.emnlp-main.128)
- Dataset metadata license: Apache-2.0
- Project usage: source and derived multilingual restaurant-domain ABSA data

## MAMS

- Source: [official repository](https://github.com/siat-nlp/MAMS-for-ABSA); [used Hugging Face distribution](https://huggingface.co/datasets/justinsiow/mams-acsa)
- Paper: Qingnan Jiang, Lei Chen, Ruifeng Xu, Xiang Ao, and Min Yang. “A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis.” EMNLP-IJCNLP 2019. [ACL Anthology](https://aclanthology.org/D19-1654/), DOI: [`10.18653/v1/D19-1654`](https://doi.org/10.18653/v1/D19-1654)
- Repository/distribution license: Apache-2.0
- Project usage: human-reviewed neutral examples

## XLM-RoBERTa base

- Source: [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
- Model metadata license: MIT
- Project usage: pretrained base model; the project-specific sentiment component is a LoRA adapter plus trained classifier

## Project-authored neutral examples

- Produced as part of this project
- Stored with separate provenance fields in the neutral-expansion manifest
- Not presented as upstream M-ABSA or MAMS data
