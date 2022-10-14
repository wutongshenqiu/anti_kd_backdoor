import torch
import torch.nn as nn

__all__ = ['trigger']


class Trigger(nn.Module):

    def __init__(self, size: int = 32, transparency: float = 1.) -> None:
        super().__init__()

        self.size = size
        self.mask = nn.Parameter(torch.rand(size, size))
        self.transparency = transparency
        self.trigger = nn.Parameter(torch.rand(3, size, size) * 4 - 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W)"""
        return self.transparency * self.mask * self.trigger + \
            (1 - self.mask * self.transparency) * x


def trigger(size: int, transparency: float = 1.) -> Trigger:
    return Trigger(size=size, transparency=transparency)
