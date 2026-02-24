"""Evaluation metrics and report generation utilities."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, roc_curve


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    scores = []
    for ref, hyp in zip(references, hypotheses):
        scores.append(sentence_bleu([ref.split()], hyp.split()))
    return float(np.mean(scores)) if scores else 0.0


def build_diagnostic_curves(y_true: List[int], y_scores: List[float]) -> Dict[str, List[float]]:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    cm = confusion_matrix(y_true, [1 if s > 0.5 else 0 for s in y_scores])
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "confusion_matrix": cm.tolist(),
    }
