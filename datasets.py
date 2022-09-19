from typing import Optional, Callable
import copy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision import datasets, transforms


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


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


class Imagenet(datasets.ImageFolder):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            start_idx: int = 0,
            end_idx: int = 0x3f3f3f3f) -> None:
        assert end_idx >= start_idx

        super().__init__(root, transform, target_transform)

        if start_idx > 0 or end_idx < 999:
            filtered_samples = []
            for x, y in self.samples:
                if start_idx <= y <= end_idx:
                    filtered_samples.append((x, y))
            self.samples = filtered_samples
            self.targets = [s[1] for s in filtered_samples]

            filtered_class_to_idx = dict()
            for class_, idx in self.class_to_idx.items():
                if start_idx <= idx <= end_idx:
                    filtered_class_to_idx[class_] = idx
            self.class_to_idx = filtered_class_to_idx

            filtered_classes = []
            for class_ in self.classes:
                if class_ in filtered_class_to_idx:
                    filtered_classes.append(class_)
            self.classes = filtered_classes
            
            print(f'start_idx: {start_idx}, end_idx: {end_idx}, '
                  f'total classes number: {len(self.classes)}')
        else:
            print(f'start_idx: {start_idx}, end_idx: {end_idx}, '
                  'Ignore process.')


class PoisonImagenet(Imagenet):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            start_idx: int = 0,
            end_idx: int = 1061109567,
            target: int = 0,
            ratio: float = 0.1) -> None:
        assert 0 < ratio <= 1

        super().__init__(root, transform, target_transform, start_idx, end_idx)

        self.ratio = ratio
        self.target = target

        nums_per_class = int((len(self) // len(self.classes)) * ratio)
        class2nums = [0 for _ in range(len(self.classes))]
        filtered_samples = list()
        for x, y in self.samples:
            class_idx = y - start_idx
            if class2nums[class_idx] < nums_per_class:
                class2nums[class_idx] += 1
                filtered_samples.append((x, target))
        self.samples = filtered_samples
        self.target = [s[1] for s in filtered_samples]


def get_transform(dataset_name: str) -> tuple[Callable, Callable]:
    if dataset_name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )
    elif dataset_name == 'imagenet':
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    else:
        raise ValueError("Dataset {} not found!".format(dataset_name))

    if get_rank() == 0:
        print(f"Get transform of {dataset_name}")
        print("train transform")
        print(train_transform)
        print("test transform")
        print(test_transform)

    return train_transform, test_transform


def get_dataset(
        dataset_name: str,
        start_idx: int = 0,
        end_idx: int = 0x3f3f3f3f) -> tuple[Dataset, Dataset]:
    train_transform, test_transform = get_transform(dataset_name)
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=r"./data/cifar10", train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=r"./data/cifar10", train=False, download=True, transform=test_transform
        )
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=r"./data/cifar100", train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=r"./data/cifar100", train=False, download=True, transform=test_transform
        )
    elif dataset_name == "svhn":
        train_dataset = datasets.SVHN(
            root=r"./data/svhn", split="train", download=True, transform=train_transform
        )
        test_dataset = datasets.SVHN(
            root=r"./data/svhn", split="test", download=True, transform=test_transform
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
    elif dataset_name == 'imagenet':
        train_dataset = Imagenet(
            root='data/imagenet/train',
            transform=train_transform,
            start_idx=start_idx,
            end_idx=end_idx
        )
        test_dataset = Imagenet(
            root='data/imagenet/val',
            transform=test_transform,
            start_idx=start_idx,
            end_idx=end_idx
        )
    else:
        raise ValueError("Dataset {} not found!".format(dataset_name))
    return train_dataset, test_dataset


def get_dataloader(
        dataset_name: str, 
        batch_size: int = 64,
        num_workers: int = 8,
        drop_last: bool = False,
        pin_memory: bool = True,
        start_idx: int = 0,
        end_idx: int = 0x3f3f3f3f,
        use_ddp: bool = False):
    train_dataset, test_dataset = get_dataset(dataset_name, start_idx, end_idx)
    
    default_loader_cfg = dict(        
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory)
    
    train_loader_cfg = copy.deepcopy(default_loader_cfg)
    train_loader_cfg['dataset'] = train_dataset
    test_loader_cfg = copy.deepcopy(default_loader_cfg)
    test_loader_cfg['dataset'] = test_dataset

    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, drop_last=drop_last)
        test_sampler = DistributedSampler(
            test_dataset, shuffle=False, drop_last=False)
        train_loader_cfg['sampler'] = train_sampler
        test_loader_cfg['sampler'] = test_sampler
    else:
        train_loader_cfg['shuffle'] = True
        train_loader_cfg['drop_last'] = drop_last
        test_loader_cfg['shuffle'] = False
        test_loader_cfg['drop_last'] = False
        
    if get_rank() == 0:
        print('train loader config')
        print(train_loader_cfg)
        print(f'test loader config')
        print(test_loader_cfg)
        
    train_dataloader = DataLoader(**train_loader_cfg)
    test_dataloader = DataLoader(**test_loader_cfg)

    return train_dataloader, test_dataloader


def get_bd_dataset_all(
        dataset_name: str,
        target: int,
        ratio: float,
        start_idx: int = 0,
        end_idx: int = 0x3f3f3f3f) -> tuple[Dataset, Dataset]:
    train_transform, test_transform = get_transform(dataset_name)
    if dataset_name == "cifar10":
        train_dataset = CustomCIFAR10All(
            root=r"./data/cifar10",
            train=True,
            download=True,
            transform=train_transform,
            target=target,
            ratio=ratio
        )
        test_dataset = CustomCIFAR10All(
            root=r"./data/cifar10",
            train=False,
            download=True,
            transform=test_transform,
            target=target,
            ratio=1.0
        )
    elif dataset_name == 'imagenet':
        train_dataset = PoisonImagenet(
            root='data/imagenet/train',
            transform=train_transform,
            target=target,
            ratio=ratio,
            start_idx=start_idx,
            end_idx=end_idx
        )
        test_dataset = PoisonImagenet(
            root='data/imagenet/val',
            transform=test_transform,
            target=target,
            ratio=1.0,
            start_idx=start_idx,
            end_idx=end_idx
        )
    else:
        raise ValueError("Dataset {} not found!".format(dataset_name))
    return train_dataset, test_dataset


def get_bd_dataloader_all(
        dataset_name: str,
        target: int,
        ratio: float,
        batch_size: int = 64,
        num_workers: int = 8,
        drop_last: bool = False,
        pin_memory: bool = True,
        start_idx: int = 0,
        end_idx: int = 0x3f3f3f3f,
        use_ddp: bool = False) -> tuple[DataLoader, DataLoader]:
    train_bd_dataset, test_bd_dataset = get_bd_dataset_all(
        dataset_name, target, ratio, start_idx, end_idx)
    
    default_loader_cfg = dict(        
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory)
    
    train_bd_loader_cfg = copy.deepcopy(default_loader_cfg)
    train_bd_loader_cfg['dataset'] = train_bd_dataset
    test_bd_loader_cfg = copy.deepcopy(default_loader_cfg)
    test_bd_loader_cfg['dataset'] = test_bd_dataset

    if use_ddp:
        train_sampler = DistributedSampler(
            train_bd_dataset, shuffle=True, drop_last=drop_last)
        test_sampler = DistributedSampler(
            test_bd_dataset, shuffle=False, drop_last=False)
        train_bd_loader_cfg['sampler'] = train_sampler
        test_bd_loader_cfg['sampler'] = test_sampler
    else:
        train_bd_loader_cfg['shuffle'] = True
        train_bd_loader_cfg['drop_last'] = drop_last
        test_bd_loader_cfg['shuffle'] = False
        test_bd_loader_cfg['drop_last'] = False

        
    if get_rank() == 0:
        print('poison train loader config')
        print(train_bd_loader_cfg)
        print(f'poison test loader config')
        print(test_bd_loader_cfg)

    train_bd_dataloader = DataLoader(**train_bd_loader_cfg)
    test_bd_dataloader = DataLoader(**test_bd_loader_cfg)

    return train_bd_dataloader, test_bd_dataloader


if __name__ == '__main__':
    from pathlib import Path
    root_dir = Path('/home/ubuntu/workspace/dataset/imagenet')
    train_dir = root_dir / 'train'
    ds = PoisonImagenet(train_dir)
    print(len(ds))

    ds = PoisonImagenet(train_dir, start_idx=0, end_idx=9)
    print(ds.classes[1])
    print(len(ds))
    print(len(ds.classes))
    print(ds.class_to_idx)

    class2nums = dict()
    for i in range(len(ds)):
        x, y = ds[i]
        try:
            class2nums[y] += 1
        except KeyError:
            class2nums[y] = 1
    print(class2nums)
