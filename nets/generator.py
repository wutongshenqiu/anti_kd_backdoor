import torch
import torch.nn as nn


class TriggerGenerator(nn.Module):
    def __init__(self, size=32):
        super(TriggerGenerator, self).__init__()
        self.size = size
        self.mask = nn.Parameter(torch.rand(size, size))
        self.trigger = nn.Parameter(torch.rand(3, size, size) * 4 - 2)

    def forward(self, x):
        """x: (B, 3, H, W)"""
        return self.mask * self.trigger + (1 - self.mask) * x
        # bd_x = self.mask * self.trigger + (1 - self.mask) * x
        # return x + (bd_x - x) * 0.0


if __name__ == '__main__':
    g = TriggerGenerator()
    x = torch.randn(10, 3, 32, 32)
    xt = g(x)
    print(xt.shape)
    print(xt)
