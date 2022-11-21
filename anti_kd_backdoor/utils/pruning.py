import copy

from torch import nn
from torch.nn.utils import prune


def ln_pruning(model: nn.Module, *, pruning_ratio: float, n: int,
               dim: int) -> nn.Module:
    model = copy.deepcopy(model)

    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    for module, name in parameters_to_prune:
        prune.ln_structured(module,
                            name=name,
                            amount=pruning_ratio,
                            n=n,
                            dim=dim)

    for module, name in parameters_to_prune:
        prune.remove(module, name)

    return model
