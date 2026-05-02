import torch

from tinyadapt.models import BIGNET_DIM, LoRALinear, LoraBigNet


def test_lora_has_trainable_adapters_and_frozen_base():
    model = LoraBigNet(lora_rank=4)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    assert trainable
    assert all("lora_" in name for name in trainable)


def test_lora_update_is_additive_at_initialization():
    layer = LoRALinear(BIGNET_DIM, BIGNET_DIM, lora_rank=2)
    x = torch.randn(2, BIGNET_DIM)

    with torch.no_grad():
        assert torch.allclose(layer(x), super(LoRALinear, layer).forward(x), atol=1e-6)
