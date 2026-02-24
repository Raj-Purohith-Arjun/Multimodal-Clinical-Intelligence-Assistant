"""Inference APIs for batch and single-patient predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class PredictionOutput:
    diagnosis: int
    anomaly_score: float
    generated_report: str


class ClinicalPredictor:
    def __init__(self, model: torch.nn.Module, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def predict_single(self, batch: Dict[str, torch.Tensor]) -> PredictionOutput:
        with torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in batch.items()})
        diagnosis = int(torch.argmax(outputs["class_logits"], dim=-1).item())
        anomaly = float(outputs["anomaly_score"].squeeze().item())
        return PredictionOutput(diagnosis=diagnosis, anomaly_score=anomaly, generated_report="Generated report placeholder.")

    def predict_batch(self, batch_list: List[Dict[str, torch.Tensor]]) -> List[PredictionOutput]:
        return [self.predict_single(batch) for batch in batch_list]
