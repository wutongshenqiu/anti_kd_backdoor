from typing import Any, Callable

X_TYPE = list[Any]
Y_TYPE = list[int]
XY_TYPE = tuple[X_TYPE, Y_TYPE]

GET_XY_FN = Callable[[], XY_TYPE]
SET_XY_FN = Callable[[XY_TYPE], None]
