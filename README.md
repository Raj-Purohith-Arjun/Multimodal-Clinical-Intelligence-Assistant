# Multimodal Clinical Intelligence Assistant

## Overview
This repository is a **multimodal ML system scaffold** for combining:
- clinical free text (`clinical_text`)
- image inputs (`image_path`)

The codebase includes:
- a configurable PyTorch multitask model (classification + anomaly score + report-token decoding)
- a training loop with mixed precision, checkpointing, and MLflow metric logging
- a FastAPI service surface for demo inference requests
- Docker packaging and basic unit tests

It should currently be treated as a **research/development template**, not a validated clinical product.

## Real-World Problem Framing (and Current Constraints)
Clinical workflows often require reasoning across notes and imaging. The repository targets that direction, but current implementation constraints are important:
- Dataset generation is synthetic by default (`src/data/synthetic_data.py`).
- API responses are heuristic/demo outputs (`api/app.py`), not predictions from a loaded trained model checkpoint.
- No built-in handling for missing image modality or corrupted files in dataset loading.
- No class-imbalance mitigation (standard losses, no weighted sampling).

## Implemented System Architecture
### 1) Data layer
- Manifest schema validation for required columns in `src/data/preprocessing.py`:
  - `patient_id`, `clinical_text`, `image_path`, `label`, `anomaly_score`, `report`
- `MultimodalClinicalDataset` in `src/data/dataset.py`:
  - Text tokenization via Hugging Face tokenizer (`AutoTokenizer`)
  - Fixed-length tokenization (`max_length`, truncation, max-length padding)
  - Image resize + tensor conversion (`Resize`, `ToTensor`)

### 2) Model layer
- `TextEncoder` (`src/models/encoders.py`): Hugging Face `AutoModel` returning token embeddings.
- `ImageEncoder` (`src/models/encoders.py`): ResNet50 feature extractor to token sequence.
- `CrossAttentionFusion` (`src/fusion/cross_attention.py`): bidirectional cross-attention (text→image and image→text) + residual + layer norm.
- `MultimodalClinicalModel` (`src/models/multitask_model.py`):
  - diagnosis classification head
  - anomaly scoring head (sigmoid)
  - transformer decoder head for report token logits

### 3) Training layer
Implemented in `src/training/trainer.py` and `src/training/train.py`:
- Config-driven hyperparameters from `src/config/default.yaml`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- Mixed precision (`torch.cuda.amp.autocast`, `GradScaler`)
- Gradient clipping
- Early stopping on validation loss
- Best-checkpoint saving to `artifacts/checkpoints/best_model.pt`
- MLflow experiment tracking (`./mlruns`) for params + epoch metrics

### 4) Evaluation and inference utilities
- Metric utilities in `src/evaluation/metrics.py`:
  - accuracy, macro F1, BLEU, PR/ROC curve data, confusion matrix
- Evaluator class in `src/evaluation/evaluator.py`:
  - metric aggregation and JSON export
  - simple latency/throughput benchmark helper around model forward passes
- `src/evaluation/run_evaluation.py` is currently a placeholder evaluation entry point.
- `src/inference/benchmark.py` is currently a stub benchmark (not full model serving benchmark).

## Fusion Strategy (what exists)
The implemented fusion strategy is **bidirectional cross-attention** only:
- text tokens attend over image tokens
- image tokens attend over text tokens
- outputs are concatenated and pooled for task heads

This repository does **not** currently include early-fusion/late-fusion alternatives or ablation comparisons in code.

## Training & Evaluation Reproducibility
### Configuration
Primary config: `src/config/default.yaml`
- model dimensions and heads
- training hyperparameters (batch size, epochs, LR, weight decay, patience, grad clip)
- data paths
- MLflow tracking URI

### Reproducibility controls
`src/utils/reproducibility.py` sets:
- Python / NumPy / Torch seeds
- `PYTHONHASHSEED`
- deterministic cuDNN flags

### What to run
```bash
pip install -r requirements.txt
pip install -e .
```

Train:
```bash
python -m src.training.train
# or
./scripts/train.sh
```

Run tests:
```bash
pytest -q
```

Serve API:
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
# or
./scripts/run_api_demo.sh 8000
```

## API Behavior (Current)
### Endpoints
- `GET /health` → health status
- `POST /analyze` → accepts `clinical_text` + `image` multipart form

### Input validation that exists
- non-empty `clinical_text`
- image MIME type check (`image/*`)
- image payload parsability check via Pillow

### Important deployment realism note
`POST /analyze` currently uses `InferenceService` heuristic logic derived from text length and image size. It does **not** call `src/inference/predictor.py` or load `artifacts/checkpoints/best_model.pt`.

## Docker and Deployment Artifacts
- Dockerfile: `docker/Dockerfile` (CUDA runtime base + uvicorn app startup)
- AWS deployment notes: `docs/aws_deployment.md` (ECR/ECS/S3/CloudWatch playbook)

These files provide deployment scaffolding; they do not by themselves guarantee production readiness.

## Testing Status
Current unit tests cover:
- API response contract (`tests/test_api.py`)
- Fusion output shape (`tests/test_fusion.py`)
- Model forward output shapes with mocked encoders (`tests/test_model_forward.py`)
- Manifest validation failure case (`tests/test_preprocessing.py`)

Not yet covered:
- end-to-end train→checkpoint→inference integration
- robustness tests (corrupt images, missing modalities)
- load/concurrency/stress tests for API serving

## Failure Modes and Limitations
Based on current code behavior:
- Missing image files in dataset rows will fail during image load.
- No explicit strategy for imbalanced labels.
- No modality-dropout or fallback path for text-only/image-only inference.
- Evaluation entrypoint is incomplete (`src/evaluation/run_evaluation.py`).
- Benchmarks in `src/inference/benchmark.py` are synthetic and not representative of deployed model inference.
- README-level clinical impact claims are intentionally omitted until measured on a real dataset.

## Future Improvements
1. Wire FastAPI service to real checkpoint loading/inference path (`ClinicalPredictor`) with proper preprocessing.
2. Complete evaluation entrypoint for reproducible offline evaluation runs.
3. Add data robustness features (missing modality handling, corrupt input guards, optional augmentations).
4. Add imbalance-aware training options (class weights, sampler strategies).
5. Add serving observability (structured request metrics, latency percentiles, error-rate dashboards).
6. Add realistic deployment benchmarking (batching, concurrency, GPU/CPU profile).


