import torch

from src.fusion.cross_attention import CrossAttentionFusion


def test_cross_attention_output_shape() -> None:
    fusion = CrossAttentionFusion(hidden_dim=32, num_heads=4, dropout=0.0)
    text = torch.randn(2, 10, 32)
    image = torch.randn(2, 6, 32)

    fused = fusion(text, image)

    assert fused.shape == (2, 16, 32)
