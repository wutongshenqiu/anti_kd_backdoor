from torchvision.datasets import GTSRB as TORCH_GTSRB

from .base import DatasetInterface, PoisonLabelDataset, RatioPoisonLabelDataset
from .types import XY_TYPE


class GTSRB(TORCH_GTSRB, DatasetInterface):
    _samples: list[tuple[str, int]]

    def get_xy(self) -> XY_TYPE:
        x = [sample[0] for sample in self._samples]
        y = [sample[1] for sample in self._samples]

        return x, y

    def set_xy(self, xy: XY_TYPE) -> None:
        x, y = xy
        assert len(x) == len(y)

        samples = list(zip(x, y))
        self._samples = samples

    @property
    def num_classes(self) -> int:
        return len(set((sample[1] for sample in self._samples)))

    @property
    def raw_num_classes(self) -> int:
        return 43


class PoisonLabelGTSRB(GTSRB, PoisonLabelDataset):

    def __init__(self, *, poison_label: int, **kwargs) -> None:
        GTSRB.__init__(self, **kwargs)
        PoisonLabelDataset.__init__(self, poison_label=poison_label)


class RatioPoisonLabelGTSRB(GTSRB, PoisonLabelDataset):

    def __init__(self, *, ratio: float, poison_label: int, **kwargs) -> None:
        GTSRB.__init__(self, **kwargs)
        RatioPoisonLabelDataset.__init__(self,
                                         ratio=ratio,
                                         poison_label=poison_label)
