"""Pydantic schemas for API contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    diagnosis_prediction: int = Field(..., description="Predicted diagnostic class index")
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    generated_report: str
