import sys
from pathlib import Path

# HACK
project_path = Path(__file__).parent.parent.parent.parent
third_party_path = project_path / 'third_party'
cifar_submodule_path = third_party_path / 'pytorch-cifar-models'

sys.path.append(str(cifar_submodule_path))

try:
    from pytorch_cifar_models import *  # noqa: F403, F401
except ModuleNotFoundError:
    import warnings
    warnings.warn(f'Third party path `{cifar_submodule_path}` does not exist, '
                  'Please using `git submodule init && git submodule update` '
                  'before using third party models')
