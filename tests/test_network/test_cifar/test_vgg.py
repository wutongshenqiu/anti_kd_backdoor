import pytest
import torch

from anti_kd_backdoor.network.cifar import get_networks


@torch.no_grad()
@pytest.mark.parametrize('num_classes', [10, 100])
@pytest.mark.parametrize('network', ['vgg11', 'vgg13', 'vgg16', 'vgg19'])
def test_vgg(network: str, num_classes: int) -> None:
    model = get_networks(network, num_classes)

    x = torch.rand(2, 3, 32, 32)
    logit = model(x)

    assert list(logit.shape) == [2, num_classes]

    with pytest.raises(RuntimeError):
        x = torch.rand(2, 3, 224, 224)
        _ = model(x)
