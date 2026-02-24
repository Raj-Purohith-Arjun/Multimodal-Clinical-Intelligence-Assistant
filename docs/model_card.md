# Model Card: Multimodal Clinical Intelligence Assistant

## Intended Use
Decision-support for triaging radiology + clinical notes, not standalone diagnosis.

## Model Overview
- Text encoder: ClinicalBERT
- Image encoder: ResNet50
- Fusion: bidirectional cross-attention
- Tasks: diagnosis classification, anomaly scoring, report generation

## Risks and Mitigations
- Bias from synthetic/imbalanced data: monitor subgroup performance.
- Hallucinated reports: human validation gate before EHR insertion.
- Domain drift: continual monitoring and periodic retraining.
