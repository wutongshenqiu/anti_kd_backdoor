from pathlib import Path

import numpy as np
import pytest

from anti_kd_backdoor.data import build_dataset

CIFAR_HOME = Path('data/cifar')
CIFAR_TESTSET_NUM = 10000
CIFAR10_HOME = CIFAR_HOME / 'cifar10'
CIFAR100_HOME = CIFAR_HOME / 'cifar100'


def _get_root(dataset_type: str) -> Path:
    if dataset_type.endswith('CIFAR100'):
        return CIFAR100_HOME
    else:
        return CIFAR10_HOME


def _make_cifar_cfg(dataset_type: str, **kwargs) -> dict:
    return dict(type=dataset_type,
                root=_get_root(dataset_type),
                train=False,
                download=True,
                **kwargs)


@pytest.mark.parametrize('dataset_type', ['CIFAR10', 'CIFAR100'])
def test_xy(dataset_type: str) -> None:
    cifar = build_dataset(_make_cifar_cfg(dataset_type))

    xy = cifar.get_xy()
    x, y = xy
    assert len(x) == len(y)
    assert isinstance(y[0], int)

    old_x = x.copy()
    old_y = y.copy()

    cifar.set_xy(xy)
    assert np.equal(cifar.data, old_x).all()
    assert cifar.targets == old_y

    x = x[:cifar.num_classes]
    y = y[:cifar.num_classes]
    cifar.set_xy((x, y))
    assert np.equal(cifar.data, x).all()
    assert cifar.targets == y
    assert cifar.num_classes == len(set(y))


@pytest.mark.parametrize(['start_idx', 'end_idx', 'dataset_type'],
                         [(0, 9, 'IndexCIFAR10'), (-10, 8, 'IndexCIFAR10'),
                          (2, 12, 'IndexCIFAR10'), (4, 4, 'IndexCIFAR10'),
                          (4, 3, 'IndexCIFAR10'), (0, 99, 'IndexCIFAR100'),
                          (-10, 8, 'IndexCIFAR100'),
                          (40, 50, 'IndexCIFAR100')])
def test_index(start_idx: int, end_idx: int, dataset_type: str) -> None:
    kwargs = dict(start_idx=start_idx,
                  end_idx=end_idx,
                  **_make_cifar_cfg(dataset_type))

    if start_idx > end_idx:
        with pytest.raises(ValueError):
            _ = build_dataset(kwargs)
        return
    cifar = build_dataset(kwargs)
    assert cifar.start_idx == start_idx
    assert cifar.end_idx == end_idx

    for y in cifar.targets:
        assert start_idx <= y <= end_idx

    assert cifar.num_classes == min(
        cifar.end_idx, cifar.raw_num_classes - 1) - max(cifar.start_idx, 0) + 1


@pytest.mark.parametrize(['ratio', 'dataset_type'], [(-1, 'RatioCIFAR10'),
                                                     (0, 'RatioCIFAR10'),
                                                     (0.1, 'RatioCIFAR10'),
                                                     (0.5, 'RatioCIFAR10'),
                                                     (1, 'RatioCIFAR10'),
                                                     (2, 'RatioCIFAR10'),
                                                     (0.4, 'RatioCIFAR100')])
def test_ratio(ratio: float, dataset_type: str) -> None:
    kwargs = dict(ratio=ratio, **_make_cifar_cfg(dataset_type))

    if ratio <= 0 or ratio > 1:
        with pytest.raises(ValueError):
            _ = build_dataset(kwargs)
        return
    cifar = build_dataset(kwargs)

    assert len(cifar.targets) == \
        int(CIFAR_TESTSET_NUM / cifar.num_classes * ratio) * cifar.num_classes


@pytest.mark.parametrize(['start_idx', 'end_idx', 'ratio'], [(4, 3, 0.5),
                                                             (3, 4, 0),
                                                             (3, 4, 2),
                                                             (1, 4, 0.1)])
def test_index_ratio(start_idx: int, end_idx: int, ratio: float) -> None:
    kwargs = dict(start_idx=start_idx,
                  end_idx=end_idx,
                  ratio=ratio,
                  **_make_cifar_cfg('IndexRatioCIFAR10'))

    if ratio <= 0 or ratio > 1 or start_idx > end_idx:
        with pytest.raises(ValueError):
            _ = build_dataset(kwargs)
        return
    cifar = build_dataset(kwargs)
    assert cifar.start_idx == start_idx
    assert cifar.end_idx == end_idx

    for y in cifar.targets:
        assert start_idx <= y <= end_idx
    assert len(cifar.targets) == \
        cifar.num_classes / cifar.raw_num_classes * CIFAR_TESTSET_NUM * ratio