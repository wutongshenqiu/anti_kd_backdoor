import copy
import inspect
from types import FrameType
from typing import Any, Iterable, Optional


def collect_hyperparameters(
        frame: Optional[FrameType] = None,
        ignored: Iterable[str] = ['self']) -> dict[str, Any]:
    if frame is None:
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
    assert isinstance(frame, FrameType)

    hyperparameters: dict[str, Any] = dict()

    args, varargs, keywords, locals = inspect.getargvalues(frame)
    # collect args
    for arg in args:
        hyperparameters[arg] = locals[arg]
    # collect varargs
    if varargs is not None and len(locals[varargs]) != 0:
        hyperparameters[varargs] = locals[varargs]
    # collect keywords
    if keywords is not None:
        hyperparameters.update(locals[keywords])

    hp = dict()
    for k, v in hyperparameters.items():
        if k in ignored:
            continue
        hp[k] = copy.deepcopy(v)

    return hp
