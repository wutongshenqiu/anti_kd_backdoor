import copy

import torch
from torch.nn import Module

from . import cifar, trigger

_ARCH_MAPPING = {'cifar': cifar, 'trigger': trigger}


def build_network(network_cfg: dict) -> Module:
    network_cfg = copy.deepcopy(network_cfg)

    if 'arch' not in network_cfg:
        raise ValueError('Network config must have `arch` field')
    arch_name = network_cfg.pop('arch')
    if arch_name not in _ARCH_MAPPING:
        raise ValueError(f'Arch `{arch_name}` is not support, '
                         f'available arch: {list(_ARCH_MAPPING.keys())}')
    arch = _ARCH_MAPPING[arch_name]

    if 'type' not in network_cfg:
        raise ValueError('Network config must have `type` field')
    network_type = network_cfg.pop('type')
    build_func = getattr(arch, network_type)
    assert callable(build_func)

    checkpoint = None
    if 'checkpoint' in network_cfg:
        checkpoint = network_cfg.pop('checkpoint')

    network: Module = build_func(**network_cfg)
    if checkpoint is not None:
        print(f'Load checkpoint from {checkpoint}')
        network.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    return network
