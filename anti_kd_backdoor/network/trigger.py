import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

__all__ = ['trigger']


class Trigger(nn.Module):
    """Trigger network.

    Args:
        size (int, tuple[int, int]): Size of whole trigger.
        trigger_size (int, tuple[int, int], optional): Size of actually
            trigger, the remaining part will be zero.
        transparency (float): Transparency of trigger.

    Note:
        Arg `size` means the whole size of the trigger, it should be same as
        the size of input image. Arg `trigger_size` means the actually trigger
        size. For example, if `size` is 32 and `trigger_size` is 10, than the
        size of the trigger tensor is 3 * 32 * 32, but only the center
        3 * 10 * 10 is trainable.
    """

    def __init__(self,
                 size: int | tuple[int, int],
                 transparency: float = 1.,
                 trigger_size: Optional[int | tuple[int, int]] = None) -> None:
        super().__init__()

        assert isinstance(transparency, float)

        size = _pair(size)
        assert isinstance(size, tuple) and len(size) == 2
        if trigger_size is None:
            trigger_size = copy.deepcopy(size)
        else:
            trigger_size = _pair(trigger_size)
        assert isinstance(trigger_size, tuple) and len(trigger_size) == 2

        for si, ti in zip(size, trigger_size):
            if si < ti:
                raise ValueError(
                    'Expect trigger size to be equal or less than whole size, '
                    f'but got: {ti} > {si}')
            if (si - ti) & 1:
                raise ValueError('Expect remaining part to be even')

        self.size = size
        self.mask: torch.Tensor = nn.Parameter(torch.rand(size))
        self.transparency: float = transparency

        self._trigger: torch.Tensor = nn.Parameter(
            torch.rand(3, *size) * 4 - 2)

        s1 = (size[0] - trigger_size[0]) >> 1
        s2 = (size[1] - trigger_size[1]) >> 1
        trigger_size_mask = torch.zeros_like(self.mask)
        trigger_size_mask[s1:s1 + trigger_size[0], s2:s2 + trigger_size[1]] = 1
        self.register_buffer('trigger_size_mask', trigger_size_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W)"""
        mask: torch.Tensor = self.mask * self.trigger_size_mask
        return self.transparency * mask * self.trigger + \
            (1 - mask * self.transparency) * x

    @property
    def trigger(self) -> torch.Tensor:
        return self._trigger * self.trigger_size_mask


def trigger(size: int | tuple[int, int],
            transparency: float = 1.,
            trigger_size: Optional[int | tuple[int, int]] = None) -> Trigger:
    return Trigger(size=size,
                   transparency=transparency,
                   trigger_size=trigger_size)
