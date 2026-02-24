"""Text and image encoder modules."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoModel


class TextEncoder(nn.Module):
    """Clinical/BioBERT encoder extracting CLS token embeddings."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class ImageEncoder(nn.Module):
    """ResNet50 encoder producing embedding sequence for fusion."""

    def __init__(self) -> None:
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        features = self.pool(features)
        b, c, h, w = features.shape
        return features.view(b, c, h * w).transpose(1, 2)
