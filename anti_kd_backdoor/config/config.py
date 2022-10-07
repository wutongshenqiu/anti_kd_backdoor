from __future__ import annotations

import copy
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, Optional

from addict import Dict
from yapf.yapflib.yapf_api import FormatCode


class ConfigDict(Dict):
    """Modify from mmengine"""

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no "
                                 f"attribute '{name}'")
        except Exception as e:
            raise e
        else:
            return value


class Config:
    """Modified from mmengine"""

    valid_suffix = {'.py'}

    def __init__(self,
                 cfg_dict: Optional[dict | ConfigDict] = None,
                 filename: Optional[str | Path] = None) -> None:
        if isinstance(filename, Path):
            filename = str(filename)
        if cfg_dict is None:
            cfg_dict = dict()
        if isinstance(cfg_dict, dict):
            cfg_dict = ConfigDict(cfg_dict)
        assert isinstance(cfg_dict, ConfigDict)

        super().__setattr__('_cfg_dict', cfg_dict)
        super().__setattr__('_filename', filename)

    @classmethod
    def fromfile(cls, config_path: str | Path) -> Config:
        if isinstance(config_path, str):
            config_path = Path(config_path)
        if not config_path.exists():
            raise ValueError(f'`{config_path}` does not exist')
        if (suffix := config_path.suffix) not in cls.valid_suffix:
            raise ValueError(f'Suffix `{suffix}` is not support, Expect '
                             f'one among {list(cls.valid_suffix)}')

        global_local_variables: dict[str, Any] = dict()
        with config_path.open('r', encoding='utf8') as f:
            codes = f.read()
            exec(codes, global_local_variables, global_local_variables)

        cfg_dict = {
            k: v
            for k, v in global_local_variables.items()
            if not (k.startswith('__') or isinstance(v, (FunctionType,
                                                         ModuleType)))
        }

        return cls(cfg_dict=cfg_dict, filename=str(config_path))

    def __repr__(self):
        return f'Config (path: {self.filename}): {self._cfg_dict.__repr__()}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)

        return other

    @property
    def pretty_text(self) -> str:
        """get formatted python config text."""

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = repr(v)
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(based_on_style='pep8',
                          blank_line_before_nested_class_or_def=True,
                          split_before_expression_after_opening_paren=True)
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text
