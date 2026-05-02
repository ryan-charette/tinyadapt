import torch

from tinyadapt.models import BIGNET_DIM, MODEL_NAMES, create_model


def test_model_output_shapes_are_consistent():
    x = torch.randn(2, BIGNET_DIM)
    for name in MODEL_NAMES:
        model = create_model(name, lora_rank=2)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape
