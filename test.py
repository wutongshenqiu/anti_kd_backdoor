# how to preprocess imagenet
# https://csinva.io/blog/misc/imagenet_quickstart/readme
from pathlib import Path
from typing import Iterable, Optional, Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models


def evaluate_accuracy(
    model: Module,
    dataloader: DataLoader,
    device: str | torch.device = 'cuda',
    top_k_list: Iterable[int] = (1, ),
    x_transform: Optional[Callable] = None
) -> list[int]:
    # HACK
    is_model_training = model.training

    model.eval()
    model.to(device)
    correct_list = [0 for _ in range(len(top_k_list))]
    max_k = max(top_k_list)
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            if x_transform is not None:
                x = x_transform(x)
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


if __name__ == "__main__":
    root_dir = Path('data/imagenet')
    val_dir = root_dir / 'val'

    device = 'cuda'
    batch_size = 256
    num_workers = 8

    from datasets import get_dataloader
    
    train_loader, val_loader = get_dataloader(
        'imagenet', 
        batch_size=batch_size,
        num_workers=num_workers)
    
    print(val_loader.dataset.transform)
    
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    
    # val_dataset = ImageFolder(val_dir, transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     normalize,
    # ]))

    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=batch_size, shuffle=False,
    #     num_workers=num_workers, pin_memory=True)

    resnet34 = models.resnet34(pretrained=True)
    print(evaluate_accuracy(
        resnet34, val_loader, device=device, top_k_list=(1, 5)))
