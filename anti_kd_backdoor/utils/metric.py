from typing import Callable, Optional, Sequence

import torch
from torch.nn import Module
from torch.utils.data import DataLoader


def calc_batch_correct(logits: torch.Tensor,
                       y: torch.Tensor,
                       top_k_list: Sequence[int] = [1]) -> list[int]:
    max_k = max(top_k_list)

    _, pred_y = logits.topk(max_k, dim=1)
    pred_y = pred_y.t()
    reshaped_y = y.view(1, -1).expand_as(pred_y)
    correct: torch.Tensor = (pred_y == reshaped_y)

    correct_list = []
    for top_k in top_k_list:
        top_k_correct = correct[:top_k].sum().item()
        correct_list.append(top_k_correct)

    return correct_list


def calc_batch_acc(logits: torch.Tensor,
                   y: torch.Tensor,
                   top_k_list: Sequence[int] = [1]) -> list[float]:
    batch_correct_list = calc_batch_correct(logits, y, top_k_list)
    batch_size = logits.size(0)

    return [batch_correct / batch_size for batch_correct in batch_correct_list]


def evaluate_accuracy(
        model: Module,
        dataloader: DataLoader,
        device: str | torch.device = 'cuda',
        top_k_list: Sequence[int] = [1],
        before_forward_fn: Optional[Callable] = None) -> dict[str, float]:
    # HACK
    is_model_training = model.training

    model.eval()
    model.to(device)
    correct_list = [0 for _ in range(len(top_k_list))]
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            if before_forward_fn is not None:
                x = before_forward_fn(x)
            y = y.to(device)
            logits: torch.Tensor = model(x)

            batch_correct_list = calc_batch_correct(logits,
                                                    y,
                                                    top_k_list=top_k_list)
            for i, batch_correct in enumerate(batch_correct_list):
                correct_list[i] += batch_correct

    if is_model_training:
        model.train()

    dataset_len = len(dataloader.dataset)

    return {
        f'top{k}': v / dataset_len
        for k, v in zip(top_k_list, correct_list)
    }
