import copy
import functools
import inspect
from types import FrameType
from typing import Any, Iterable, Optional

from torch import Tensor
from torch.nn import Module


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


def get_module_by_name(base_module: Tensor | Module,
                       access_string: str) -> Tensor | Module:
    if access_string == '':
        return Module

    names = access_string.split(sep='.')
    return functools.reduce(getattr, names, base_module)


def get_parent_module_name(s: str) -> str:
    s_module_list = s.split('.')
    if len(s_module_list) == 1:
        return ''

    return '.'.join(s_module_list[:-1])
