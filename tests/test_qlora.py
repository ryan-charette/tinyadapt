import torch

from tinyadapt.models import BIGNET_DIM, QLoRALinear, QLoRABigNet


def test_qlora_base_is_frozen():
    model = QLoRABigNet(lora_rank=4)
    frozen = [parameter for name, parameter in model.named_parameters() if "lora_" not in name]
    trainable = [parameter for name, parameter in model.named_parameters() if "lora_" in name]

    assert frozen
    assert trainable
    assert all(not parameter.requires_grad for parameter in frozen)
    assert all(parameter.requires_grad for parameter in trainable)


def test_qlora_update_is_additive_at_initialization():
    layer = QLoRALinear(BIGNET_DIM, BIGNET_DIM, lora_rank=2)
    x = torch.randn(2, BIGNET_DIM)

    with torch.no_grad():
        assert torch.allclose(layer(x), QLoRALinear.__mro__[1].forward(layer, x), atol=1e-6)
