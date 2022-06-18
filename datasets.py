import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms


class CustomCIFAR10All(datasets.CIFAR10):
    def __init__(
        self,
        root,
        train,
        download,
        transform=None,
        target=0,
        ratio=0.1,
        use_ratio=True,
        change_label=True,
    ):
        super(CustomCIFAR10All, self).__init__(
            root=root, train=train, download=download, transform=transform
        )
        self.ratio = ratio
        data, targets = [], []
        count = [0] * 10
        if use_ratio:
            for (d, t) in zip(self.data, self.targets):
                count[t] += 1
                if count[t] <= int(len(self.data) // len(self.classes) * ratio):
                    data.append(np.expand_dims(d, 0))
                    if change_label:
                        targets.append(target)
                    else:
                        targets.append(t)
        else:
            # remaining part (original label)
            for (d, t) in zip(self.data, self.targets):
                count[t] += 1
                if count[t] > int(len(self.data) // len(self.classes) * ratio):
                    data.append(np.expand_dims(d, 0))
                    targets.append(t)
        self.data = np.concatenate(data, 0)
        self.targets = targets


def get_transform(dataset_name):
    if dataset_name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    elif dataset_name == "cifar100":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    elif dataset_name == "svhn":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    elif dataset_name == "tiny_imagenet":
        train_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        raise ValueError("Dataset {} not found!".format(dataset_name))

    return train_transform, test_transform


def get_dataset(dataset_name):
    train_transform, test_transform = get_transform(dataset_name)
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=r"./data", train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=r"./data", train=False, download=True, transform=test_transform
        )
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=r"./data", train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=r"./data", train=False, download=True, transform=test_transform
        )
    elif dataset_name == "svhn":
        train_dataset = datasets.SVHN(
            root=r"./data", split="train", download=True, transform=train_transform
        )
        test_dataset = datasets.SVHN(
            root=r"./data", split="test", download=True, transform=test_transform
        )
    elif dataset_name == "tiny_imagenet":
        train_dataset = datasets.ImageFolder(
            root=r"/mnt/data/datasets/tiny-imagenet/tiny-imagenet-200/train",
            transform=train_transform,
        )
        test_dataset = datasets.ImageFolder(
            root=r"/mnt/data/datasets/tiny-imagenet/tiny-imagenet-200/val",
            transform=test_transform,
        )
    else:
        raise ValueError("Dataset {} not found!".format(dataset_name))
    return train_dataset, test_dataset


def get_dataloader(dataset_name, batch_size):
    train_dataset, test_dataset = get_dataset(dataset_name)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False
    )
    return train_dataloader, test_dataloader


def get_bd_dataset_all(dataset_name, target, ratio):
    train_transform, test_transform = get_transform(dataset_name)
    if dataset_name == "cifar10":
        train_dataset = CustomCIFAR10All(
            root=r"./data",
            train=True,
            download=True,
            transform=train_transform,
            target=target,
            ratio=ratio,
        )
        test_dataset = CustomCIFAR10All(
            root=r"./data",
            train=False,
            download=True,
            transform=test_transform,
            target=target,
            ratio=1.0,
        )
    else:
        raise ValueError("Dataset {} not found!".format(dataset_name))
    return train_dataset, test_dataset


def get_bd_dataloader_all(dataset_name, batch_size, target, ratio):
    train_bd_dataset, test_bd_dataset = get_bd_dataset_all(dataset_name, target, ratio)
    train_bd_dataloader = torch.utils.data.DataLoader(
        train_bd_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )
    test_bd_dataloader = torch.utils.data.DataLoader(
        test_bd_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    return train_bd_dataloader, test_bd_dataloader
