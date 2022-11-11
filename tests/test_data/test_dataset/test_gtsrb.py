import numpy as np
import pytest

from .utils import FakeDataset, build_fake_dataset

GTSRB_TESTSET_NUM = 1000


def build_gtsrb_fake_dataset(dataset_type: str, **kwargs) -> FakeDataset:
    if dataset_type.endswith('GTSRB'):
        y_range = (0, 99)
        dataset_type = dataset_type.replace('GTSRB', 'FakeDataset')
    else:
        y_range = (0, 9)
        dataset_type = dataset_type.replace('GTSRB', 'FakeDataset')

    dataset_cfg = dict(type=dataset_type,
                       x_shape=(3, 32, 32),
                       y_range=y_range,
                       nums=GTSRB_TESTSET_NUM,
                       **kwargs)

    return build_fake_dataset(dataset_cfg)


@pytest.mark.parametrize('dataset_type', ['GTSRB'])
def test_xy(dataset_type: str) -> None:
    gtsrb = build_gtsrb_fake_dataset(dataset_type)

    xy = gtsrb.get_xy()
    x, y = xy
    assert len(x) == len(y)
    assert isinstance(y[0], int)

    old_x = x.copy()
    old_y = y.copy()

    gtsrb.set_xy(xy)
    assert all([np.array_equal(nx, ox) for nx, ox in zip(gtsrb.data, old_x)])
    assert gtsrb.targets == old_y

    x = x[:gtsrb.num_classes]
    y = y[:gtsrb.num_classes]
    gtsrb.set_xy((x, y))
    assert all([np.array_equal(nx, ox) for nx, ox in zip(gtsrb.data, x)])
    assert gtsrb.targets == y
    assert gtsrb.num_classes == len(set(y))
    assert len(gtsrb.data.shape) == 4


@pytest.mark.parametrize(['start_idx', 'end_idx', 'dataset_type'],
                         [(0, 9, 'IndexGTSRB'), (-10, 8, 'IndexGTSRB'),
                          (2, 12, 'IndexGTSRB'), (4, 4, 'IndexGTSRB'),
                          (4, 3, 'IndexGTSRB'), (0, 99, 'IndexGTSRB'),
                          (-10, 8, 'IndexGTSRB'), (40, 50, 'IndexGTSRB')])
def test_index(start_idx: int, end_idx: int, dataset_type: str) -> None:
    kwargs = dict(start_idx=start_idx, end_idx=end_idx)

    if start_idx > end_idx:
        with pytest.raises(ValueError):
            _ = build_gtsrb_fake_dataset(dataset_type, **kwargs)
        return
    gtsrb = build_gtsrb_fake_dataset(dataset_type, **kwargs)
    assert gtsrb.start_idx == start_idx
    assert gtsrb.end_idx == end_idx

    for y in gtsrb.targets:
        assert start_idx <= y <= end_idx

    assert gtsrb.num_classes == min(
        gtsrb.end_idx, gtsrb.raw_num_classes - 1) - max(gtsrb.start_idx, 0) + 1
    assert len(gtsrb.data.shape) == 4


@pytest.mark.parametrize(['ratio', 'dataset_type'], [(-1, 'RatioGTSRB'),
                                                     (0, 'RatioGTSRB'),
                                                     (0.1, 'RatioGTSRB'),
                                                     (0.5, 'RatioGTSRB'),
                                                     (1, 'RatioGTSRB'),
                                                     (2, 'RatioGTSRB'),
                                                     (0.4, 'RatioGTSRB')])
def test_ratio(ratio: float, dataset_type: str) -> None:
    kwargs = dict(ratio=ratio)

    if ratio <= 0 or ratio > 1:
        with pytest.raises(ValueError):
            _ = build_gtsrb_fake_dataset(dataset_type, **kwargs)
        return
    gtsrb = build_gtsrb_fake_dataset(dataset_type, **kwargs)

    assert gtsrb.num_classes == gtsrb.raw_num_classes
    assert len(gtsrb.targets) == \
        int(GTSRB_TESTSET_NUM / gtsrb.num_classes * ratio) * gtsrb.num_classes
    assert len(gtsrb.data.shape) == 4


@pytest.mark.parametrize('range_ratio', [(-1, 0.2), (0, 2), (0.1, 0.1),
                                         (0.5, 0.2), (0.1, 0.5), (0, 1)])
@pytest.mark.parametrize('dataset_type', ['RangeRatioGTSRB'])
def test_range_ratio(range_ratio: tuple[float, float],
                     dataset_type: str) -> None:
    kwargs = dict(range_ratio=range_ratio)

    start_ratio = range_ratio[0]
    end_ratio = range_ratio[1]
    if not (0 <= start_ratio < end_ratio <= 1):
        with pytest.raises(ValueError):
            _ = build_gtsrb_fake_dataset(dataset_type, **kwargs)
        return

    gtsrb = build_gtsrb_fake_dataset(dataset_type, **kwargs)
    assert gtsrb.num_classes == gtsrb.raw_num_classes
    assert len(gtsrb.targets) == \
        round(GTSRB_TESTSET_NUM * (end_ratio - start_ratio))
    assert len(gtsrb.data.shape) == 4


@pytest.mark.parametrize(['range_ratio1', 'range_ratio2'],
                         [((0, 0.5), (0.5, 1)), ((0, 0.6), (0.4, 1)),
                          ((0, 0.7), (0.3, 1)), ((0, 0.5), (0, 1))])
@pytest.mark.parametrize('dataset_type', ['RangeRatioGTSRB'])
def test_range_ratio_intersection(range_ratio1: tuple[float, float],
                                  range_ratio2: tuple[float, float],
                                  dataset_type: str) -> None:

    gtsrb1 = build_gtsrb_fake_dataset(dataset_type=dataset_type,
                                      range_ratio=range_ratio1,
                                      cache_xy=True)
    gtsrb2 = build_gtsrb_fake_dataset(dataset_type=dataset_type,
                                      range_ratio=range_ratio2,
                                      cache_xy=True)

    cat_x = np.concatenate([gtsrb1.data, gtsrb2.data], axis=0)
    unique_x = np.unique(cat_x, axis=0)
    intersection_number = cat_x.shape[0] - unique_x.shape[0]

    intersection_ratio = max(0, range_ratio1[1] - range_ratio2[0])
    assert round(intersection_ratio * GTSRB_TESTSET_NUM) == intersection_number


@pytest.mark.parametrize('dataset_type', ['IndexRatioGTSRB'])
@pytest.mark.parametrize(['start_idx', 'end_idx', 'ratio'], [(4, 3, 0.5),
                                                             (3, 4, 0),
                                                             (3, 4, 2),
                                                             (1, 4, 0.1)])
def test_index_ratio(start_idx: int, end_idx: int, ratio: float,
                     dataset_type: str) -> None:
    kwargs = dict(start_idx=start_idx, end_idx=end_idx, ratio=ratio)

    if ratio <= 0 or ratio > 1 or start_idx > end_idx:
        with pytest.raises(ValueError):
            _ = build_gtsrb_fake_dataset(dataset_type, **kwargs)
        return
    gtsrb = build_gtsrb_fake_dataset(dataset_type, **kwargs)
    assert gtsrb.start_idx == start_idx
    assert gtsrb.end_idx == end_idx

    for y in gtsrb.targets:
        assert start_idx <= y <= end_idx
    assert len(gtsrb.targets) == \
        gtsrb.num_classes / gtsrb.raw_num_classes * GTSRB_TESTSET_NUM * ratio
    assert len(gtsrb.data.shape) == 4


@pytest.mark.parametrize('dataset_type', ['PoisonLabelGTSRB'])
@pytest.mark.parametrize('poison_label', [-1, 5, 101])
def test_poison_label(poison_label: int, dataset_type: str) -> None:
    kwargs = dict(poison_label=poison_label)

    if poison_label < 0 or poison_label >= 100:
        with pytest.raises(ValueError):
            _ = build_gtsrb_fake_dataset(dataset_type, **kwargs)
        return
    gtsrb = build_gtsrb_fake_dataset(dataset_type, **kwargs)
    assert gtsrb.poison_label == poison_label

    assert gtsrb.num_classes == 1
    assert all(map(lambda x: x == poison_label, gtsrb.targets))
    assert len(gtsrb.data.shape) == 4


@pytest.mark.parametrize('dataset_type', ['RatioPoisonLabelGTSRB'])
@pytest.mark.parametrize('poison_label', [-1, 5, 101])
@pytest.mark.parametrize('ratio', [0, 0.2, 1, 1.2])
def test_ratio_poison_label(ratio: float, poison_label: int,
                            dataset_type: str) -> None:
    kwargs = dict(ratio=ratio, poison_label=poison_label)

    if (poison_label < 0 or poison_label >= 100) or \
            (ratio <= 0 or ratio > 1):
        with pytest.raises(ValueError):
            _ = build_gtsrb_fake_dataset(dataset_type, **kwargs)
        return
    gtsrb = build_gtsrb_fake_dataset(dataset_type, **kwargs)
    assert gtsrb.poison_label == poison_label

    assert len(gtsrb) == round(GTSRB_TESTSET_NUM * ratio)
    assert gtsrb.num_classes == 1
    assert all(map(lambda x: x == poison_label, gtsrb.targets))
    assert len(gtsrb.data.shape) == 4


@pytest.mark.parametrize('dataset_type', ['RangeRatioPoisonLabelGTSRB'])
@pytest.mark.parametrize('poison_label', [-1, 1, 101])
@pytest.mark.parametrize('range_ratio', [(0, 0.2), (0.2, 0.5), (0.5, 1),
                                         (0.5, 0.2)])
def test_range_ratio_poison_label(range_ratio: tuple[float,
                                                     float], poison_label: int,
                                  dataset_type: str) -> None:
    kwargs = dict(range_ratio=range_ratio, poison_label=poison_label)

    if poison_label < 0 or poison_label >= 100 or \
            not (0 <= range_ratio[0] < range_ratio[1] <= 1):
        with pytest.raises(ValueError):
            _ = build_gtsrb_fake_dataset(dataset_type, **kwargs)
        return
    gtsrb = build_gtsrb_fake_dataset(dataset_type, **kwargs)
    assert gtsrb.poison_label == poison_label

    assert len(gtsrb) == round(GTSRB_TESTSET_NUM *
                               (range_ratio[1] - range_ratio[0]))
    assert gtsrb.num_classes == 1
    assert all(map(lambda x: x == poison_label, gtsrb.targets))
    assert len(gtsrb.data.shape) == 4
