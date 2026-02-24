"""Evaluation pipeline with latency and throughput benchmarking."""

from __future__ import annotations

import time
from typing import Dict, List

import torch

from src.evaluation.metrics import build_diagnostic_curves, compute_bleu, compute_classification_metrics
from src.utils.io import save_json


class Evaluator:
    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device

    def evaluate_predictions(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_scores: List[float],
        references: List[str],
        hypotheses: List[str],
        output_path: str,
    ) -> Dict:
        metrics = compute_classification_metrics(y_true, y_pred)
        metrics["bleu"] = compute_bleu(references, hypotheses)
        metrics.update(build_diagnostic_curves(y_true, y_scores))
        save_json(metrics, output_path)
        return metrics

    def benchmark_inference(self, batch: Dict[str, torch.Tensor], runs: int = 20) -> Dict[str, float]:
        self.model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = self.model(**batch)
        duration = time.perf_counter() - start
        latency_ms = (duration / runs) * 1000
        throughput = runs / duration
        return {"latency_ms": latency_ms, "throughput_samples_per_s": throughput}
