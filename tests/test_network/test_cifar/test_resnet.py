import pytest
import torch

from anti_kd_backdoor.network.cifar import get_networks


@torch.no_grad()
@pytest.mark.parametrize('num_classes', [10, 100])
@pytest.mark.parametrize(
    'network', ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
def test_resnet(network: str, num_classes: int) -> None:
    model = get_networks(network, num_classes)

    x = torch.rand(2, 3, 32, 32)
    logit = model(x)

    assert list(logit.shape) == [2, num_classes]
