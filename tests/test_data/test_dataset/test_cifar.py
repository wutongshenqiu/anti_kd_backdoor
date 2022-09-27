from pathlib import Path

import numpy as np
import pytest

from anti_kd_backdoor.data.dataset.cifar import (CIFAR10, CIFAR100,
                                                 IndexCIFAR10, IndexCIFAR100,
                                                 IndexRatioCIFAR10,
                                                 RatioCIFAR10, RatioCIFAR100)

CIFAR_HOME = Path('data/cifar')
CIFAR_TESTSET_NUM = 10000
CIFAR10_HOME = CIFAR_HOME / 'cifar10'
CIFAR100_HOME = CIFAR_HOME / 'cifar100'


@pytest.mark.parametrize(['data_cls', 'root'], [(CIFAR10, CIFAR10_HOME),
                                                (CIFAR100, CIFAR100_HOME)])
def test_xy(data_cls: type, root: Path) -> None:
    cifar = data_cls(root=root, train=False, download=True)

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


@pytest.mark.parametrize(['start_idx', 'end_idx', 'data_cls', 'root'],
                         [(0, 9, IndexCIFAR10, CIFAR10_HOME),
                          (-10, 8, IndexCIFAR10, CIFAR10_HOME),
                          (2, 12, IndexCIFAR10, CIFAR10_HOME),
                          (4, 4, IndexCIFAR10, CIFAR10_HOME),
                          (4, 3, IndexCIFAR10, CIFAR10_HOME),
                          (0, 99, IndexCIFAR100, CIFAR100_HOME),
                          (-10, 8, IndexCIFAR100, CIFAR100_HOME),
                          (40, 50, IndexCIFAR100, CIFAR100_HOME)])
def test_index(start_idx: int, end_idx: int, data_cls: type,
               root: Path) -> None:
    kwargs = dict(start_idx=start_idx,
                  end_idx=end_idx,
                  root=root,
                  train=False,
                  download=True)

    if start_idx > end_idx:
        with pytest.raises(ValueError):
            _ = data_cls(**kwargs)
        return
    cifar = data_cls(**kwargs)
    assert cifar.start_idx == start_idx
    assert cifar.end_idx == end_idx

    for y in cifar.targets:
        assert start_idx <= y <= end_idx

    assert cifar.num_classes == min(
        cifar.end_idx, cifar.raw_num_classes - 1) - max(cifar.start_idx, 0) + 1


@pytest.mark.parametrize(['ratio', 'data_cls', 'root'],
                         [(-1, RatioCIFAR10, CIFAR10_HOME),
                          (0, RatioCIFAR10, CIFAR10_HOME),
                          (0.1, RatioCIFAR10, CIFAR10_HOME),
                          (0.5, RatioCIFAR10, CIFAR10_HOME),
                          (1, RatioCIFAR10, CIFAR10_HOME),
                          (2, RatioCIFAR10, CIFAR10_HOME),
                          (0.4, RatioCIFAR100, CIFAR100_HOME)])
def test_ratio(ratio: float, data_cls: type, root: Path) -> None:
    kwargs = dict(ratio=ratio, root=root, train=False, download=True)

    if ratio <= 0 or ratio > 1:
        with pytest.raises(ValueError):
            _ = data_cls(**kwargs)
        return
    cifar = data_cls(**kwargs)

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
                  root=CIFAR10_HOME,
                  train=False,
                  download=True)

    if ratio <= 0 or ratio > 1 or start_idx > end_idx:
        with pytest.raises(ValueError):
            _ = IndexRatioCIFAR10(**kwargs)
        return
    cifar = IndexRatioCIFAR10(**kwargs)
    assert cifar.start_idx == start_idx
    assert cifar.end_idx == end_idx

    for y in cifar.targets:
        assert start_idx <= y <= end_idx
    assert len(cifar.targets) == \
        cifar.num_classes / cifar.raw_num_classes * CIFAR_TESTSET_NUM * ratio
