import copy
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import tqdm
from torch import nn, optim
from torch.nn import Module, Parameter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from anti_kd_backdoor.data import build_dataloader
from anti_kd_backdoor.network import build_network
from anti_kd_backdoor.utils import (
    AverageMeter,
    build_optimizer,
    build_scheduler,
    calc_batch_acc,
    collect_hyperparameters,
    evaluate_accuracy,
)

from .anti_kd import BaseWrapper


def _build_module(module_cfg: dict) -> Module:
    if 'type' not in module_cfg:
        raise ValueError('Config must have `type` field')

    module_cfg = copy.deepcopy(module_cfg)

    module_type = module_cfg.pop('type')
    if (module := getattr(nn, module_type)) is None:
        raise ValueError(f'`{module_type}` is not supported!')

    return module(**module_cfg)


def _replace_module(model: Module, mapping_cfg: dict[str, dict]) -> None:
    used_modules = set()

    def traverse_children(module: Module, prefix: str) -> None:
        for name, child in module.named_children():
            if prefix == '':
                child_name = name
            else:
                child_name = f'{prefix}.{name}'
            if child_name in mapping_cfg:
                new_module = _build_module(mapping_cfg[child_name])
                print(f'Module `{child_name}` will be replaced')
                print(f'Original: {child}')
                print(f'New: {new_module}')
                setattr(module, name, new_module)
                used_modules.add(child_name)
            else:
                traverse_children(child, child_name)

    traverse_children(model, '')

    if (unused_modules := set(mapping_cfg.keys()) - used_modules):
        raise ValueError(f'Unexpected modules: {unused_modules}')


class FinetuneWrapper(BaseWrapper):

    @classmethod
    def build_from_cfg(cls, cfg: dict) -> BaseWrapper:
        print(cfg)
        if 'network' not in cfg:
            raise ValueError('Config must have `network` field')
        if 'optimizer' not in cfg:
            raise ValueError('Config must have `optimizer` field')
        if 'finetune' not in cfg:
            raise ValueError('Config must have `finetune` field')

        network_cfg = cfg.pop('network')
        network: Module = build_network(network_cfg)

        if 'mapping' in cfg:
            mapping_cfg: dict[str, dict] = cfg.pop('mapping')
            _replace_module(network, mapping_cfg)

        finetune_cfg = cfg.pop('finetune')
        training_mode: bool = finetune_cfg.get('training', False)
        network.train(training_mode)

        # freeze all parameters
        for p in network.parameters():
            p.requires_grad = False

        # collect trainable parameters
        name2trainable_paramter: dict[str, Parameter] = \
            cls.collect_trainable_parameters(
                network, finetune_cfg['trainable_modules'])
        for name, parameter in network.named_parameters():
            if name in name2trainable_paramter:
                parameter.requires_grad = True

        print('Trainable parameters')
        for name, parameter in name2trainable_paramter.items():
            print(f'name: `{name}`, shape: {parameter.shape}')

        optimizer_cfg = cfg.pop('optimizer')
        optimizer: optim.Optimizer = build_optimizer(
            params=name2trainable_paramter.values(),
            optimizer_cfg=optimizer_cfg)

        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        if 'scheduler' in cfg:
            scheduler_cfg = cfg.pop('scheduler')
            scheduler = build_scheduler(optimizer=optimizer,
                                        scheduler_cfg=scheduler_cfg)

        # HACK: inconsistent with `__init__` signature
        return cls(network=network,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   **cfg)

    @staticmethod
    def collect_trainable_parameters(
            model: Module,
            trainable_module_names: Iterable[str]) -> dict[str, Parameter]:
        name2trainable_paramter: dict[str, Parameter] = dict()

        trainable_module_name_set = set(trainable_module_names)
        used_module_set = set()
        for module_name, module in model.named_modules():
            if module_name in trainable_module_name_set:
                for parameter_name, parameter in module.named_parameters():
                    name2trainable_paramter[
                        f'{module_name}.{parameter_name}'] = parameter

                used_module_set.add(module_name)

        if used_module_set != trainable_module_name_set:
            print('Unexpected module: '
                  f'{trainable_module_name_set - used_module_set}')

        return name2trainable_paramter


# FIXME: should be more elegant
class FinetuneTrainer(Module):

    def __init__(self,
                 *,
                 network: dict,
                 train_dataloader: dict,
                 test_dataloader: dict,
                 epochs: int,
                 save_interval: int,
                 work_dirs: str,
                 device: str = 'cuda',
                 epochs_per_validation: int = 1) -> None:
        hyperparameters = collect_hyperparameters()
        self._hp = hyperparameters

        super().__init__()

        self._network_wrapper = FinetuneWrapper.build_from_cfg(network)
        self._train_dataloader = build_dataloader(train_dataloader)
        self._test_dataloader = build_dataloader(test_dataloader)

        self._epochs = epochs
        self._save_interval = save_interval
        self._device = device
        self._epochs_per_validation = epochs_per_validation

        self._work_dirs = Path(work_dirs)
        self._ckpt_dirs = self._work_dirs / 'ckpt'
        self._log_dir = self._work_dirs / 'logs'
        self._tb_writer = SummaryWriter(log_dir=self._log_dir)
        hyperparameters['work_dirs'] = str(hyperparameters['work_dirs'])
        with (self._work_dirs / 'hparams.json').open('w',
                                                     encoding='utf8') as f:
            f.write(json.dumps(hyperparameters))

        self._current_epoch = 0

    def train(self) -> None:
        pbar = tqdm.tqdm(total=self._epochs)
        prev_ckpt_path: Optional[Path] = None

        while self._current_epoch < self._epochs:
            self._current_epoch += 1

            self.before_train_epoch()
            self.train_epoch()
            self.after_train_epoch()

            if self._current_epoch == 1 or \
                    self._current_epoch % self._epochs_per_validation == 0:
                self.validation()

            if self._current_epoch % self._save_interval == 0:
                self.save_checkpoint(ckpt_path=self._ckpt_dirs /
                                     f'epoch={self._current_epoch}.pth')

            if prev_ckpt_path is not None:
                prev_ckpt_path.unlink()
            prev_ckpt_path = self._ckpt_dirs \
                / f'all-epoch={self._current_epoch}.pth'
            self.save_checkpoint(ckpt_path=prev_ckpt_path)

            pbar.update(1)

        if prev_ckpt_path is not None:
            prev_ckpt_path.unlink()
        self.save_checkpoint(ckpt_path=self._ckpt_dirs / 'lastest.pth')
        torch.save(self._network_wrapper.network.state_dict(),
                   self._ckpt_dirs / 'latest_network.pth')

    def before_train_epoch(self) -> None:
        ...

    def train_epoch(self) -> None:
        tag_scalar_dict = self._train_network(
            network_wrapper=self._network_wrapper,
            dataloader=self._train_dataloader,
            device=self._device)
        self._tb_writer.add_scalars(
            f'{type(self._network_wrapper.network).__name__} training '
            f'on {type(self._train_dataloader.dataset).__name__}',
            tag_scalar_dict, self._current_epoch)

    def after_train_epoch(self) -> None:
        # Step schedulers
        if (n_scheduler := self._network_wrapper.scheduler) is not None:
            n_scheduler.step()

        # log learning rate
        network_lr = {
            type(self._network_wrapper.network).__name__.lower():
            self._network_wrapper.optimizer.param_groups[0]['lr']
        }
        self._tb_writer.add_scalars('Learning rate', {**network_lr},
                                    self._current_epoch)

    def validation(self) -> None:
        # acc of clean test data on teacher & student
        tag_scalar_dict = evaluate_accuracy(
            model=self._network_wrapper.network,
            dataloader=self._test_dataloader,
            device=self._device,
            top_k_list=(1, 5))
        self._tb_writer.add_scalars(
            f'{type(self._network_wrapper.network).__name__} validation on '
            f'{type(self._test_dataloader.dataset).__name__}', tag_scalar_dict,
            self._current_epoch)

    @staticmethod
    def _train_network(network_wrapper: BaseWrapper, dataloader: DataLoader,
                       device: str) -> dict[str, float]:
        network = network_wrapper.network
        optimizer = network_wrapper.optimizer

        network.train()
        network.to(device)

        loss_meter = AverageMeter(name='loss')
        top1_meter = AverageMeter(name='top1_acc')
        top5_meter = AverageMeter(name='top5_acc')

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = network(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            loss_meter.update(loss.item(), batch_size)
            top1_acc, top5_acc = calc_batch_acc(logits, y, (1, 5))
            top1_meter.update(top1_acc, batch_size)
            top5_meter.update(top5_acc, batch_size)

        return {
            'loss': loss_meter.avg,
            'top1_acc': top1_meter.avg,
            'top5_acc': top5_meter.avg
        }

    @property
    def stats(self) -> dict[str, Any]:
        return dict(hyperparameters=self._hp,
                    runtime_stats=self._current_epoch)

    def save_checkpoint(self, ckpt_path: str | Path) -> None:
        if isinstance(ckpt_path, str):
            ckpt_path = Path(ckpt_path)
        if not ckpt_path.parent.exists():
            ckpt_path.parent.mkdir(parents=True)

        save_obj = dict(stats=self.stats, state_dict=self.state_dict())
        torch.save(save_obj, ckpt_path)

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        if isinstance(ckpt_path, str):
            ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise ValueError(f'Path `{ckpt_path}` does not exist')

        print(f'Load from checkpoint: {ckpt_path}')

        load_obj = torch.load(ckpt_path, map_location='cpu')
        load_stats = load_obj['stats']
        if (old_hp := load_stats['hyperparameters']) != self._hp:
            print(f'Hyperparameters not same.\n '
                  f'Previous: {old_hp} \n'
                  f'new: {self._hp}')

        self._current_epoch = load_stats['runtime_stats']
        self.load_state_dict(load_obj['state_dict'])
