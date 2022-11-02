import abc

from .mixins import (
    IndexFilterMixin,
    PoisonLabelMixin,
    RangeRatioFilterMixin,
    RatioFilterMixin,
)
from .types import XY_TYPE


class DatasetInterface:

    @abc.abstractmethod
    def get_xy(self) -> XY_TYPE:
        ...

    @abc.abstractmethod
    def set_xy(self, xy: XY_TYPE) -> None:
        ...

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def raw_num_classes(self) -> int:
        ...


class IndexDataset(DatasetInterface, IndexFilterMixin):

    def __init__(self, *, start_idx: int, end_idx: int) -> None:
        self.start_idx = start_idx
        self.end_idx = end_idx

        xy = self.get_xy()
        filtered_xy = self.filter_by_index(xy)
        self.set_xy(filtered_xy)


class RatioDataset(DatasetInterface, RatioFilterMixin):

    def __init__(self, *, ratio: float) -> None:
        self.ratio = ratio

        xy = self.get_xy()
        filtered_xy = self.filter_by_ratio(xy)
        self.set_xy(filtered_xy)


class RangeRatioDataset(DatasetInterface, RangeRatioFilterMixin):

    def __init__(self, *, range_ratio: tuple[float, float]) -> None:
        self.range_ratio = range_ratio

        xy = self.get_xy()
        filtered_xy = self.filter_by_range_ratio(xy)
        self.set_xy(filtered_xy)


class IndexRatioDataset(DatasetInterface, IndexFilterMixin, RatioFilterMixin):

    def __init__(self, *, start_idx: int, end_idx: int, ratio: float) -> None:
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.ratio = ratio

        xy = self.get_xy()
        filtered_xy = self.filter_by_index(xy)
        filtered_xy = self.filter_by_ratio(filtered_xy)
        self.set_xy(filtered_xy)


class PoisonLabelDataset(DatasetInterface, PoisonLabelMixin):

    def __init__(self, *, poison_label: int) -> None:
        self.poison_label = poison_label

        xy = self.get_xy()
        filtered_xy = self.to_target_label(xy)
        self.set_xy(filtered_xy)


class RatioPoisonLabelDataset(DatasetInterface, RatioFilterMixin,
                              PoisonLabelMixin):

    def __init__(self, *, ratio: float, poison_label: int) -> None:
        self.ratio = ratio
        self.poison_label = poison_label

        xy = self.get_xy()
        filtered_xy = self.filter_by_ratio(xy)
        filtered_xy = self.to_target_label(filtered_xy)
        self.set_xy(filtered_xy)
