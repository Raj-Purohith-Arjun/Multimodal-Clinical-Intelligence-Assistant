"""Training orchestration with mixed precision, checkpointing, and MLflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import mlflow
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.utils.logging_utils import get_logger


@dataclass
class TrainArtifacts:
    best_checkpoint: str
    best_val_loss: float


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, loss: float) -> bool:
        if loss < self.best:
            self.best = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Trainer supporting multitask objectives and experiment logging."""

    def __init__(self, model: nn.Module, config: Dict, device: str) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = get_logger("trainer")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["training"]["scheduler_t_max"])
        self.scaler = GradScaler(enabled=config["training"]["mixed_precision"])
        self.classification_loss = nn.CrossEntropyLoss()
        self.anomaly_loss = nn.BCELoss()
        self.report_loss = nn.CrossEntropyLoss()

    def train(self, train_loader: DataLoader, val_loader: DataLoader, run_name: Optional[str] = None) -> TrainArtifacts:
        checkpoint_dir = Path("artifacts/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        early_stopper = EarlyStopping(patience=self.config["training"]["patience"])
        best_checkpoint = str(checkpoint_dir / "best_model.pt")

        mlflow.set_tracking_uri(self.config["experiment"]["mlflow_tracking_uri"])
        mlflow.set_experiment(self.config["experiment"]["name"])

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(self.config["training"])
            for epoch in range(self.config["training"]["epochs"]):
                train_loss = self._train_epoch(train_loader)
                val_metrics = self._validate(val_loader)
                self.scheduler.step()

                mlflow.log_metrics({"train_loss": train_loss, **val_metrics}, step=epoch)
                self.logger.info(f"Epoch={epoch} train_loss={train_loss:.4f} val_loss={val_metrics['val_loss']:.4f}")

                if val_metrics["val_loss"] <= early_stopper.best:
                    torch.save(self.model.state_dict(), best_checkpoint)

                if early_stopper.step(val_metrics["val_loss"]):
                    self.logger.info("Early stopping triggered.")
                    break

        return TrainArtifacts(best_checkpoint=best_checkpoint, best_val_loss=early_stopper.best)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.config["training"]["mixed_precision"]):
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    images=batch["image"].to(self.device),
                    report_ids=batch["input_ids"].to(self.device),
                )
                cls = self.classification_loss(outputs["class_logits"], batch["label"].to(self.device))
                anomaly = self.anomaly_loss(outputs["anomaly_score"].squeeze(-1), batch["anomaly_score"].to(self.device))
                report = self.report_loss(
                    outputs["report_logits"].reshape(-1, outputs["report_logits"].size(-1)),
                    batch["input_ids"].to(self.device).reshape(-1),
                )
                loss = cls + 0.5 * anomaly + 0.2 * report

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_clip_norm"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        losses = []
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    images=batch["image"].to(self.device),
                    report_ids=batch["input_ids"].to(self.device),
                )
                cls_loss = self.classification_loss(outputs["class_logits"], batch["label"].to(self.device))
                losses.append(cls_loss.item())
                probs = torch.softmax(outputs["class_logits"], dim=-1)
                y_true.extend(batch["label"].cpu().tolist())
                y_pred.extend(torch.argmax(probs, dim=-1).cpu().tolist())
                y_score.extend(probs[:, 1].detach().cpu().tolist())

        roc_auc = roc_auc_score([1 if y == 1 else 0 for y in y_true], y_score) if len(set(y_true)) > 1 else 0.5
        return {
            "val_loss": sum(losses) / max(len(losses), 1),
            "val_accuracy": accuracy_score(y_true, y_pred),
            "val_f1": f1_score(y_true, y_pred, average="macro"),
            "val_roc_auc": roc_auc,
        }
