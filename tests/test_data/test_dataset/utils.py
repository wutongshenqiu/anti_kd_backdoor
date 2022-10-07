import numpy as np
import torch
from torch.utils.data import Dataset

from anti_kd_backdoor.data.dataset.base import (DatasetInterface, IndexDataset,
                                                IndexRatioDataset,
                                                PoisonLabelDataset,
                                                RatioDataset,
                                                RatioPoisonLabelDataset)
from anti_kd_backdoor.data.dataset.types import XY_TYPE


class FakeDataset(DatasetInterface, Dataset):

    def __init__(self,
                 *,
                 x_shape: tuple[int, int, int] = (3, 32, 32),
                 y_range: tuple[int, int] = (0, 9),
                 nums: int = 10000) -> None:
        self._nums = nums
        self._x_shape = x_shape
        self._y_range = y_range
        self._raw_num_classes = y_range[1] - y_range[0] + 1

        self.data = torch.rand((nums, *x_shape))
        num_per_class = nums // self._raw_num_classes
        self.targets = [
            i for _ in range(num_per_class)
            for i in range(self._raw_num_classes)
        ]
        self.targets.extend([y_range[0]] * (nums - len(self.targets)))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def get_xy(self) -> XY_TYPE:
        return self.data.tolist(), self.targets.copy()

    def set_xy(self, xy: XY_TYPE) -> None:
        x, y = xy
        assert len(x) == len(y)

        self.data = np.stack(x, axis=0)
        self.targets = y.copy()

    @property
    def num_classes(self) -> int:
        return len(set(self.targets))

    @property
    def raw_num_classes(self) -> int:
        return self._raw_num_classes


class IndexFakeDataset(FakeDataset, IndexDataset):

    def __init__(self, *, start_idx: int, end_idx: int, **kwargs) -> None:
        FakeDataset.__init__(self, **kwargs)
        IndexDataset.__init__(self, start_idx=start_idx, end_idx=end_idx)


class RatioFakeDataset(FakeDataset, RatioDataset):

    def __init__(self, *, ratio: float, **kwargs) -> None:
        FakeDataset.__init__(self, **kwargs)
        RatioDataset.__init__(self, ratio=ratio)


class IndexRatioFakeDataset(FakeDataset, IndexRatioDataset):

    def __init__(self, *, start_idx: int, end_idx: int, ratio: float,
                 **kwargs) -> None:
        FakeDataset.__init__(self, **kwargs)
        IndexRatioDataset.__init__(self,
                                   start_idx=start_idx,
                                   end_idx=end_idx,
                                   ratio=ratio)


class PoisonLabelFakeDataset(FakeDataset, PoisonLabelDataset):

    def __init__(self, *, poison_label: int, **kwargs) -> None:
        FakeDataset.__init__(self, **kwargs)
        PoisonLabelDataset.__init__(self, poison_label=poison_label)


class RatioPoisonLabelFakeDataset(FakeDataset, RatioPoisonLabelDataset):

    def __init__(self, *, ratio: float, poison_label: int, **kwargs) -> None:
        FakeDataset.__init__(self, **kwargs)
        RatioPoisonLabelDataset.__init__(self,
                                         ratio=ratio,
                                         poison_label=poison_label)


FAKE_DATASETS_MAPPING = {
    'FakeDataset': FakeDataset,
    'IndexFakeDataset': IndexFakeDataset,
    'RatioFakeDataset': RatioFakeDataset,
    'IndexRatioFakeDataset': IndexRatioFakeDataset,
    'PoisonLabelFakeDataset': PoisonLabelFakeDataset,
    'RatioPoisonLabelFakeDataset': RatioPoisonLabelFakeDataset
}


def build_fake_dataset(dataset_cfg: dict) -> FakeDataset:
    if 'type' not in dataset_cfg:
        raise ValueError('Dataset config must have `type` field')
    dataset_type = dataset_cfg.pop('type')
    if dataset_type not in FAKE_DATASETS_MAPPING:
        raise ValueError(
            f'Dataset `{dataset_type}` is not support, '
            f'available datasets: {list(FAKE_DATASETS_MAPPING.keys())}')
    dataset = FAKE_DATASETS_MAPPING[dataset_type]

    return dataset(**dataset_cfg)
