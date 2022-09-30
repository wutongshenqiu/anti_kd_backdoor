from torch.nn import Module

from . import cifar, trigger

_ARCH_MAPPING = {'cifar': cifar, 'trigger': trigger}


def build_network(network_cfg: dict) -> Module:
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
    network = getattr(arch, network_type)
    assert callable(network)

    return network(**network_cfg)
