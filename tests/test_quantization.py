import torch

from tinyadapt.quantization import block_dequantize_4bit, block_quantize_4bit


def test_quantization_preserves_shape_and_dtype():
    x = torch.linspace(-1, 1, steps=32)
    packed, scale = block_quantize_4bit(x, group_size=16)
    restored = block_dequantize_4bit(packed, scale)

    assert packed.shape == (2, 8)
    assert scale.shape == (2, 1)
    assert restored.shape == x.shape
    assert restored.dtype == torch.float32


def test_quantization_is_reasonably_close():
    x = torch.randn(64)
    packed, scale = block_quantize_4bit(x, group_size=16)
    restored = block_dequantize_4bit(packed, scale)

    assert torch.mean(torch.abs(x - restored)) < 0.15
