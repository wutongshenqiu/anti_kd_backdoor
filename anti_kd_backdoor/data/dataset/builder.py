import copy

from torchvision import transforms

from .cifar import (
    CIFAR10,
    CIFAR100,
    IndexCIFAR10,
    IndexCIFAR100,
    IndexRatioCIFAR10,
    IndexRatioCIFAR100,
    PoisonLabelCIFAR10,
    PoisonLabelCIFAR100,
    RangeRatioCIFAR10,
    RangeRatioCIFAR100,
    RangeRatioPoisonLabelCIFAR10,
    RangeRatioPoisonLabelCIFAR100,
    RatioCIFAR10,
    RatioCIFAR100,
    RatioPoisonLabelCIFAR10,
    RatioPoisonLabelCIFAR100,
)
from .types import DatasetProtocol

DATASETS_MAPPING = {
    'CIFAR10': CIFAR10,
    'IndexCIFAR10': IndexCIFAR10,
    'RatioCIFAR10': RatioCIFAR10,
    'RangeRatioCIFAR10': RangeRatioCIFAR10,
    'IndexRatioCIFAR10': IndexRatioCIFAR10,
    'PoisonLabelCIFAR10': PoisonLabelCIFAR10,
    'RangeRatioPoisonLabelCIFAR10': RangeRatioPoisonLabelCIFAR10,
    'RatioPoisonLabelCIFAR10': RatioPoisonLabelCIFAR10,
    'CIFAR100': CIFAR100,
    'IndexCIFAR100': IndexCIFAR100,
    'RatioCIFAR100': RatioCIFAR100,
    'RangeRatioCIFAR100': RangeRatioCIFAR100,
    'IndexRatioCIFAR100': IndexRatioCIFAR100,
    'PoisonLabelCIFAR100': PoisonLabelCIFAR100,
    'RatioPoisonLabelCIFAR100': RatioPoisonLabelCIFAR100,
    'RangeRatioPoisonLabelCIFAR100': RangeRatioPoisonLabelCIFAR100
}


def build_dataset(dataset_cfg: dict) -> DatasetProtocol:
    dataset_cfg = copy.deepcopy(dataset_cfg)

    if 'type' not in dataset_cfg:
        raise ValueError('Dataset config must have `type` field')
    dataset_type = dataset_cfg.pop('type')
    if dataset_type not in DATASETS_MAPPING:
        raise ValueError(
            f'Dataset `{dataset_type}` is not support, '
            f'available datasets: {list(DATASETS_MAPPING.keys())}')
    dataset = DATASETS_MAPPING[dataset_type]

    if 'target_transform' in dataset_cfg:
        raise ValueError('`target_transform` is not support')

    if 'transform' in dataset_cfg:
        transform_list = []
        transform_list_cfg = dataset_cfg.pop('transform')
        for transform_cfg in transform_list_cfg:
            transform_type = transform_cfg.pop('type')
            transform = getattr(transforms, transform_type)(**transform_cfg)
            transform_list.append(transform)

        transform = transforms.Compose(transform_list)
    else:
        transform = None

    return dataset(transform=transform, **dataset_cfg)
