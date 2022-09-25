import pytest
import torch

from anti_kd_backdoor.network.cifar.mobilenet_v2 import mobilenet_v2


@torch.no_grad()
@pytest.mark.parametrize('num_classes', [10, 100])
def test_mobilenet_v2(num_classes: int) -> None:
    model = mobilenet_v2(num_classes)

    x = torch.rand(2, 3, 32, 32)
    logit = model(x)

    assert list(logit.shape) == [2, num_classes]

    with pytest.raises(RuntimeError):
        x = torch.rand(2, 3, 224, 224)
        _ = model(x)
