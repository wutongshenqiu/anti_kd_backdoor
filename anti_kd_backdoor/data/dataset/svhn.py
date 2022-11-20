import numpy as np
from torchvision.datasets import SVHN as TORCH_SVHN

from .base import DatasetInterface, PoisonLabelDataset, RatioPoisonLabelDataset
from .types import XY_TYPE


class SVHN(TORCH_SVHN, DatasetInterface):
    data: np.ndarray
    labels: np.ndarray

    def get_xy(self) -> XY_TYPE:
        return list(self.data), self.labels.tolist()

    def set_xy(self, xy: XY_TYPE) -> None:
        x, y = xy
        assert len(x) == len(y)

        self.data = np.stack(x, axis=0)
        self.labels = np.stack(y, axis=0)

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))

    @property
    def raw_num_classes(self) -> int:
        return 10


class PoisonLabelSVHN(SVHN, PoisonLabelDataset):

    def __init__(self, *, poison_label: int, **kwargs) -> None:
        SVHN.__init__(self, **kwargs)
        PoisonLabelDataset.__init__(self, poison_label=poison_label)


class RatioPoisonLabelSVHN(SVHN, PoisonLabelDataset):

    def __init__(self, *, ratio: float, poison_label: int, **kwargs) -> None:
        SVHN.__init__(self, **kwargs)
        RatioPoisonLabelDataset.__init__(self,
                                         ratio=ratio,
                                         poison_label=poison_label)
