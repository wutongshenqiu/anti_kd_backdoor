import typing

from .types import XY_TYPE


class IndexFilterMixin:
    start_idx: int
    end_idx: int

    def filter_by_index(self, xy: XY_TYPE) -> XY_TYPE:
        if self.start_idx > self.end_idx:
            raise ValueError('`start_idx` must be less or equal than `end_idx`'
                             f', but got {self.start_idx} > {self.end_idx}')
        x, y = xy

        filtered_x = []
        filtered_y = []

        for itx, ity in zip(x, y):
            if self.start_idx <= ity <= self.end_idx:
                filtered_x.append(itx)
                filtered_y.append(ity)

        return filtered_x, filtered_y


class RatioFilterMixin:
    ratio: float

    def filter_by_ratio(self, xy: XY_TYPE) -> XY_TYPE:
        if not (0 < self.ratio <= 1):
            raise ValueError('`ratio` must be in range (0, 1]'
                             f', but got: {self.ratio}')

        x, y = xy

        filtered_x = []
        filtered_y = []

        num_classes = len(set(y))
        num_per_class = int(len(x) / num_classes * self.ratio)
        class2num: dict[int, int] = dict()
        for itx, ity in zip(x, y):
            try:
                class2num[ity] += 1
            except KeyError:
                class2num[ity] = 1

            if class2num[ity] <= num_per_class:
                filtered_x.append(itx)
                filtered_y.append(ity)

        return filtered_x, filtered_y


# TODO: name
class PoisonLabelMixin:
    poison_label: int

    @typing.no_type_check
    def to_target_label(self, xy: XY_TYPE) -> XY_TYPE:
        if not (0 <= self.poison_label < self.raw_num_classes):
            raise ValueError(
                '`target` must between 0 and '
                f'{self.raw_num_classes - 1}, but got {self.poison_label}')

        x, _ = xy

        return x.copy(), [self.poison_label] * len(x)
