import numpy as np
from torchvision.datasets import CIFAR10 as TORCH_CIFAR10
from torchvision.datasets import CIFAR100 as TORCH_CIFAR100
from torchvision.datasets import GTSRB as TORCH_GTSRB

from .base import (DatasetInterface, IndexDataset, IndexRatioDataset,
                   PoisonLabelDataset, RatioDataset, RatioPoisonLabelDataset)
from .types import XY_TYPE


class CIFAR10(TORCH_CIFAR10, DatasetInterface):
    data: np.ndarray
    targets: list[int]

    def get_xy(self) -> XY_TYPE:
        return list(self.data), self.targets.copy()

    def set_xy(self, xy: XY_TYPE):
        x, y = xy
        assert len(x) == len(y)

        self.data = np.stack(x, axis=0)
        self.targets = y.copy()

    @property
    def num_classes(self) -> int:
        return len(set(self.targets))

    @property
    def raw_num_classes(self) -> int:
        return 10


class IndexCIFAR10(CIFAR10, IndexDataset):

    def __init__(self, *, start_idx: int, end_idx: int, **kwargs) -> None:
        CIFAR10.__init__(self, **kwargs)
        IndexDataset.__init__(self, start_idx=start_idx, end_idx=end_idx)


class RatioCIFAR10(CIFAR10, RatioDataset):

    def __init__(self, *, ratio: float, **kwargs) -> None:
        CIFAR10.__init__(self, **kwargs)
        RatioDataset.__init__(self, ratio=ratio)


class IndexRatioCIFAR10(CIFAR10, IndexRatioDataset):

    def __init__(self, *, start_idx: int, end_idx: int, ratio: float,
                 **kwargs) -> None:
        CIFAR10.__init__(self, **kwargs)
        IndexRatioDataset.__init__(self,
                                   start_idx=start_idx,
                                   end_idx=end_idx,
                                   ratio=ratio)


class PoisonLabelCIFAR10(CIFAR10, PoisonLabelDataset):

    def __init__(self, *, poison_label: int, **kwargs) -> None:
        CIFAR10.__init__(self, **kwargs)
        PoisonLabelDataset.__init__(self, poison_label=poison_label)


class RatioPoisonLabelCIFAR10(CIFAR10, RatioPoisonLabelDataset):

    def __init__(self, *, ratio: float, poison_label: int, **kwargs) -> None:
        CIFAR10.__init__(self, **kwargs)
        RatioPoisonLabelDataset.__init__(self,
                                         ratio=ratio,
                                         poison_label=poison_label)


class CIFAR100(CIFAR10, TORCH_CIFAR100):

    @property
    def raw_num_classes(self) -> int:
        return 100


class IndexCIFAR100(CIFAR100, IndexDataset):

    def __init__(self, *, start_idx: int, end_idx: int, **kwargs) -> None:
        CIFAR100.__init__(self, **kwargs)
        IndexDataset.__init__(self, start_idx=start_idx, end_idx=end_idx)


class RatioCIFAR100(CIFAR100, RatioDataset):

    def __init__(self, *, ratio: float, **kwargs) -> None:
        CIFAR100.__init__(self, **kwargs)
        RatioDataset.__init__(self, ratio=ratio)


class IndexRatioCIFAR100(CIFAR100, IndexRatioDataset):

    def __init__(self, *, start_idx: int, end_idx: int, ratio: float,
                 **kwargs) -> None:
        CIFAR100.__init__(self, **kwargs)
        IndexRatioDataset.__init__(self,
                                   start_idx=start_idx,
                                   end_idx=end_idx,
                                   ratio=ratio)


class PoisonLabelCIFAR100(CIFAR100, PoisonLabelDataset):

    def __init__(self, *, poison_label: int, **kwargs) -> None:
        CIFAR100.__init__(self, **kwargs)
        PoisonLabelDataset.__init__(self, poison_label=poison_label)


class RatioPoisonLabelCIFAR100(CIFAR100, RatioPoisonLabelDataset):

    def __init__(self, *, ratio: float, poison_label: int, **kwargs) -> None:
        CIFAR100.__init__(self, **kwargs)
        RatioPoisonLabelDataset.__init__(self,
                                         ratio=ratio,
                                         poison_label=poison_label)
                              
class GTSRB(TORCH_GTSRB, DatasetInterface):
    data: np.ndarray
    targets: list[int]

def get_xy(self) -> XY_TYPE:
    return list(self.data), self.targets.copy()

def set_xy(self, xy: XY_TYPE):
    x, y = xy
    assert len(x) == len(y)

    self.data = np.stack(x, axis=0)
    self.targets = y.copy()

@property
def num_classes(self) -> int:
    return len(set(self.targets))

@property
def raw_num_classes(self) -> int:
    return 43

class IndexGTSRB(GTSRB, IndexDataset):

def __init__(self, *, start_idx: int, end_idx: int, **kwargs) -> None:
    GTSRB.__init__(self, **kwargs)
    IndexDataset.__init__(self, start_idx=start_idx, end_idx=end_idx)


class RatioGTSRB(GTSRB, RatioDataset):

def __init__(self, *, ratio: float, **kwargs) -> None:
    GTSRB.__init__(self, **kwargs)
    RatioDataset.__init__(self, ratio=ratio)


class IndexRatioGTSRB(GTSRB, IndexRatioDataset):

def __init__(self, *, start_idx: int, end_idx: int, ratio: float,
             **kwargs) -> None:
    GTSRB.__init__(self, **kwargs)
    IndexRatioDataset.__init__(self,
                               start_idx=start_idx,
                               end_idx=end_idx,
                               ratio=ratio)


class PoisonLabelGTSRB(GTSRB, PoisonLabelDataset):

def __init__(self, *, poison_label: int, **kwargs) -> None:
    GTSRB.__init__(self, **kwargs)
    PoisonLabelDataset.__init__(self, poison_label=poison_label)


class RatioPoisonLabelGTSRB(GTSRB, RatioPoisonLabelDataset):

def __init__(self, *, ratio: float, poison_label: int, **kwargs) -> None:
    GTSRB.__init__(self, **kwargs)
    RatioPoisonLabelDataset.__init__(self,
                                     ratio=ratio,
                                     poison_label=poison_label)
 