import copy

from torch.utils.data import DataLoader

from .dataset import build_dataset


def build_dataloader(dataload_cfg: dict) -> DataLoader:
    dataload_cfg = copy.deepcopy(dataload_cfg)

    if 'dataset' not in dataload_cfg:
        raise ValueError('Dataloader config must have `dataset` field')
    dataset_cfg = dataload_cfg.pop('dataset')
    dataset = build_dataset(dataset_cfg)

    return DataLoader(dataset=dataset, **dataload_cfg)
