from pathlib import Path

from torchvision.datasets import Flowers102 as TORCH_Flowers102

from .base import DatasetInterface, PoisonLabelDataset, RatioPoisonLabelDataset
from .types import XY_TYPE


class Flowers102(TORCH_Flowers102, DatasetInterface):
    _image_files: list[Path]
    _labels: list[int]

    def get_xy(self) -> XY_TYPE:
        return self._image_files.copy(), self._labels.copy()

    def set_xy(self, xy: XY_TYPE) -> None:
        x, y = xy
        assert len(x) == len(y)

        self._image_files = x
        self._labels = y

    @property
    def num_classes(self) -> int:
        return len(set(self._labels))

    @property
    def raw_num_classes(self) -> int:
        return 102


class PoisonLabelFlowers102(Flowers102, PoisonLabelDataset):

    def __init__(self, *, poison_label: int, **kwargs) -> None:
        Flowers102.__init__(self, **kwargs)
        PoisonLabelDataset.__init__(self, poison_label=poison_label)


class RatioPoisonLabelFlowers102(Flowers102, PoisonLabelDataset):

    def __init__(self, *, ratio: float, poison_label: int, **kwargs) -> None:
        Flowers102.__init__(self, **kwargs)
        RatioPoisonLabelDataset.__init__(self,
                                         ratio=ratio,
                                         poison_label=poison_label)
