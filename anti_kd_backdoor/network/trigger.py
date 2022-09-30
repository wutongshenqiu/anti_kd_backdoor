import torch
import torch.nn as nn

__all__ = ['trigger']


class Trigger(nn.Module):

    def __init__(self, size: int = 32) -> None:
        super().__init__()

        self.size = size
        self.mask = nn.Parameter(torch.rand(size, size))
        self.trigger = nn.Parameter(torch.rand(3, size, size) * 4 - 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W)"""
        return self.mask * self.trigger + (1 - self.mask) * x


def trigger(size: int) -> Trigger:
    return Trigger(size=size)
