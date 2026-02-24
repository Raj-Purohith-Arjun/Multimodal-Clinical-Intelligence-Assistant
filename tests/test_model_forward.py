import torch
from torch import nn

import src.models.multitask_model as multitask


class DummyTextEncoder(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        return torch.randn(b, t, 16)


class DummyImageEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        return torch.randn(b, 8, 2048)


def test_model_forward(monkeypatch) -> None:
    monkeypatch.setattr(multitask, "TextEncoder", lambda _: DummyTextEncoder())
    monkeypatch.setattr(multitask, "ImageEncoder", lambda: DummyImageEncoder())

    model = multitask.MultimodalClinicalModel("dummy", hidden_dim=16, num_classes=5, vocab_size=100)
    outputs = model(
        input_ids=torch.ones(2, 12, dtype=torch.long),
        attention_mask=torch.ones(2, 12, dtype=torch.long),
        images=torch.randn(2, 3, 224, 224),
        report_ids=torch.ones(2, 12, dtype=torch.long),
    )
    assert outputs["class_logits"].shape == (2, 5)
    assert outputs["anomaly_score"].shape == (2, 1)
    assert outputs["report_logits"].shape[:2] == (2, 12)
