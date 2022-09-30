from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


def evaluate_accuracy(
        model: Module,
        dataloader: DataLoader,
        device: str | torch.device = 'cuda',
        top_k_list: Sequence[int] = (1, ),
        before_forward_fn: Optional[Callable] = None) -> list[float]:
    # HACK
    is_model_training = model.training

    model.eval()
    model.to(device)
    correct_list = [0 for _ in range(len(top_k_list))]
    max_k = max(top_k_list)
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            if before_forward_fn is not None:
                x = before_forward_fn(x)
            y = y.to(device)
            logits: Tensor = model(x)

            _, pred_y = logits.topk(max_k, dim=1)
            pred_y = pred_y.t()
            reshaped_y = y.view(1, -1).expand_as(pred_y)
            correct: Tensor = (pred_y == reshaped_y)

            for i, top_k in enumerate(top_k_list):
                top_k_correct = correct[:top_k].sum().item()
                correct_list[i] += top_k_correct

    if is_model_training:
        model.train()

    dataset_len = len(dataloader.dataset)
    return [correct / dataset_len for correct in correct_list]
