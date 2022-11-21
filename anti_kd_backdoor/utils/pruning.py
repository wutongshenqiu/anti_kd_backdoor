import copy

from torch import nn
from torch.nn.utils import prune


def ln_pruning(model: nn.Module,
               *,
               pruning_ratio: float,
               n: int,
               dim: int,
               verbose: bool = False) -> nn.Module:
    if pruning_ratio <= 0:
        print(f'Pruning for ratio `{pruning_ratio}` will be ignored')
        return model

    model = copy.deepcopy(model)

    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            if verbose:
                print(f'`{module}` will be pruned')

    for module, name in parameters_to_prune:
        prune.ln_structured(module,
                            name=name,
                            amount=pruning_ratio,
                            n=n,
                            dim=dim)

    for module, name in parameters_to_prune:
        prune.remove(module, name)

    return model
