from __future__ import annotations

from typing import Any, Callable, Protocol

X_TYPE = list[Any]
Y_TYPE = list[int]
XY_TYPE = tuple[X_TYPE, Y_TYPE]

GET_XY_FN = Callable[[], XY_TYPE]
SET_XY_FN = Callable[[XY_TYPE], None]


class DatasetProtocol(Protocol):

    def __getitem__(self, other: DatasetProtocol) -> Any:
        ...

    def get_xy(self) -> XY_TYPE:
        ...

    def set_xy(self, xy: XY_TYPE) -> None:
        ...

    def num_classes(self) -> int:
        ...

    def raw_num_classes(self) -> int:
        ...
