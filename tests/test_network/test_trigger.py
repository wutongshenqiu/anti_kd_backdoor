import math

import pytest
import torch
from torch.nn.modules.utils import _pair

from anti_kd_backdoor.network.trigger import Trigger


@torch.no_grad()
@pytest.mark.parametrize('size', [32, 224, (32, 32)])
def test_trigger_init(size: int) -> None:
    trigger = Trigger(size)
    assert trigger.size == _pair(size)
    assert tuple(trigger.mask.shape) == _pair(size)
    assert tuple(trigger.trigger.shape) == (3, *_pair(size))
    assert tuple(trigger.trigger_size_mask.shape) == _pair(size)


@torch.no_grad()
@pytest.mark.parametrize('size', [32, 224, (32, 32)])
def test_trigger_forward(size: int) -> None:
    trigger = Trigger(size)

    x = torch.rand(10, 3, *_pair(size))
    xp = trigger(x)
    assert xp.shape == x.shape

    # test effect of mask
    trigger.mask.fill_(0)
    xp = trigger(x)
    assert torch.equal(xp, x)

    trigger.mask.fill_(1)
    xp = trigger(x)
    for i in range(xp.size(0)):
        assert torch.equal(xp[i], trigger.trigger)


@pytest.mark.parametrize('size', [32])
@pytest.mark.parametrize('trigger_size', [14, 32])
def test_trigger_backward(size: int, trigger_size: int) -> None:
    trigger = Trigger(size=size, trigger_size=trigger_size)

    x = torch.rand(10, 3, *_pair(size))
    xp = trigger(x)

    loss = xp.sum()
    trigger.zero_grad()
    loss.backward()

    remaining = size - trigger_size >> 1
    assert math.isclose(
        trigger.mask.grad.norm().item(),
        trigger.mask.grad[remaining:remaining + trigger_size,
                          remaining:remaining + trigger_size].norm().item(),
        abs_tol=1e-3,
        rel_tol=1e-6)

    assert math.isclose(
        trigger._trigger.grad.norm().item(),
        trigger._trigger.grad[:, remaining:remaining + trigger_size,
                              remaining:remaining +
                              trigger_size].norm().item(),
        abs_tol=1e-3,
        rel_tol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize('size', [32, 224])
@pytest.mark.parametrize('trigger_size', [13, 32, 64, None])
def test_trigger_size(size: int, trigger_size: int) -> None:
    if trigger_size is None:
        trigger_size = size

    if trigger_size > size:
        with pytest.raises(ValueError):
            _ = Trigger(size, trigger_size=trigger_size)
        return
    if (size - trigger_size) & 1:
        with pytest.raises(ValueError):
            _ = Trigger(size, trigger_size=trigger_size)
        return

    trigger = Trigger(size, trigger_size=trigger_size)

    remaining = size - trigger_size >> 1
    assert trigger.trigger[:, 0:remaining, 0:remaining].norm().item() == 0
    assert trigger.trigger[:, 0:remaining,
                           trigger_size + remaining:].norm().item() == 0
    assert trigger.trigger[:, trigger_size + remaining:,
                           0:remaining].norm().item() == 0
    assert trigger.trigger[:, trigger_size + remaining:,
                           trigger_size + remaining:].norm().item() == 0

    assert trigger.trigger[:, remaining:remaining + trigger_size,
                           remaining:remaining +
                           trigger_size].norm().item() > 0
