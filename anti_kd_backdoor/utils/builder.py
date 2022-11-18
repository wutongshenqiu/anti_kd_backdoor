from typing import Iterable

from torch import Tensor, optim


def build_optimizer(params: Iterable[Tensor],
                    optimizer_cfg: dict) -> optim.Optimizer:
    if 'type' not in optimizer_cfg:
        raise ValueError('Optimizer config must have `type` field')
    optimizer_type = optimizer_cfg.pop('type')

    if (optimizer := getattr(optim, optimizer_type)) is None:
        raise ValueError(f'Optimizer `{optimizer_type}` is not support')

    return optimizer(params, **optimizer_cfg)


def build_scheduler(optimizer: optim.Optimizer,
                    scheduler_cfg: dict) -> optim.lr_scheduler._LRScheduler:
    if 'type' not in scheduler_cfg:
        raise ValueError('Scheduler config must have `type` field')
    scheduler_type = scheduler_cfg.pop('type')

    if (scheduler := getattr(optim.lr_scheduler, scheduler_type)) is None:
        raise ValueError(f'Scheduler `{scheduler_type}` is not support')

    return scheduler(optimizer, **scheduler_cfg)
