"""Full multimodal multitask model."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from src.fusion.cross_attention import CrossAttentionFusion
from src.models.encoders import ImageEncoder, TextEncoder


class ReportDecoder(nn.Module):
    """Transformer decoder head for report generation."""

    def __init__(self, hidden_dim: int, vocab_size: int, num_layers: int = 2) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.token_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_embeddings: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(tgt=tgt_embeddings, memory=memory)
        return self.token_head(decoded)


class MultimodalClinicalModel(nn.Module):
    """Production-ready multitask architecture for multimodal clinical intelligence."""

    def __init__(self, text_model_name: str, hidden_dim: int, num_classes: int, vocab_size: int = 30522) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(text_model_name)
        self.image_encoder = ImageEncoder()
        self.image_projection = nn.Linear(2048, hidden_dim)
        self.fusion = CrossAttentionFusion(hidden_dim=hidden_dim, num_heads=8, dropout=0.1)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_classes))
        self.anomaly_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.report_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.report_decoder = ReportDecoder(hidden_dim=hidden_dim, vocab_size=vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        report_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        text_tokens = self.text_encoder(input_ids, attention_mask)
        image_tokens = self.image_projection(self.image_encoder(images))
        fused_tokens = self.fusion(text_tokens, image_tokens)

        pooled = fused_tokens.mean(dim=1)
        class_logits = self.classifier(pooled)
        anomaly_score = torch.sigmoid(self.anomaly_head(pooled))

        report_embeddings = self.report_embedding(report_ids)
        report_logits = self.report_decoder(report_embeddings, fused_tokens)
        return {
            "class_logits": class_logits,
            "anomaly_score": anomaly_score,
            "report_logits": report_logits,
        }
