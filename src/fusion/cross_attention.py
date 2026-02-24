"""Cross-modal fusion layers."""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between text and image token sequences."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_text = nn.LayerNorm(hidden_dim)
        self.norm_image = nn.LayerNorm(hidden_dim)

    def forward(self, text_tokens: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        attended_text, _ = self.text_to_image(query=text_tokens, key=image_tokens, value=image_tokens)
        attended_image, _ = self.image_to_text(query=image_tokens, key=text_tokens, value=text_tokens)
        fused_text = self.norm_text(text_tokens + attended_text)
        fused_image = self.norm_image(image_tokens + attended_image)
        return torch.cat([fused_text, fused_image], dim=1)
