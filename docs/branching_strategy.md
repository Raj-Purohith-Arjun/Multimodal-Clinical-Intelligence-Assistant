# Git Branching Strategy

- `main` → production-ready branch (tagged releases)
- `develop` → integration branch
- `feature/data-pipeline`
- `feature/model-architecture`
- `feature/training-loop`
- `feature/evaluation-metrics`
- `feature/api-deployment`
- `feature/aws-deployment`

## Example Commit History (simulated)
1. `feat(data): add synthetic multimodal dataset generator with manifest validation`
2. `feat(model): implement ClinicalBERT + ResNet50 multimodal cross-attention architecture`
3. `feat(training): add mixed precision trainer with early stopping and MLflow tracking`
4. `feat(eval): add ROC/PR/confusion matrix export and BLEU scoring`
5. `feat(api): expose /analyze FastAPI endpoint with multimodal payload schema`
6. `chore(deploy): add CUDA Dockerfile and GitHub Actions CI workflow`
