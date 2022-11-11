import numpy as np
from torchvision.datasets import GTSRB as TORCH_GTSRB

from .base import (
    DatasetInterface,
    IndexDataset,
    IndexRatioDataset,
    PoisonLabelDataset,
    RangeRatioDataset,
    RangeRatioPoisonLabelDataset,
    RatioDataset,
    RatioPoisonLabelDataset,
)
from .types import XY_TYPE


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


class RangeRatioGTSRB(GTSRB, RangeRatioDataset):

    def __init__(self, *, range_ratio: tuple[float, float], **kwargs) -> None:
        GTSRB.__init__(self, **kwargs)
        RangeRatioDataset.__init__(self, range_ratio=range_ratio)


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


class RangeRatioPoisonLabelGTSRB(GTSRB, RangeRatioPoisonLabelDataset):

    def __init__(self, *, range_ratio: tuple[float, float], poison_label: int,
                 **kwargs) -> None:
        GTSRB.__init__(self, **kwargs)
        RangeRatioPoisonLabelDataset.__init__(self,
                                              range_ratio=range_ratio,
                                              poison_label=poison_label)
