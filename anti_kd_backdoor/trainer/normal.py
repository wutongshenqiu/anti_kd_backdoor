import json
from pathlib import Path
from typing import Any, Optional

import torch
import tqdm
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from anti_kd_backdoor.data import build_dataloader
from anti_kd_backdoor.utils import (AverageMeter, calc_batch_acc,
                                    collect_hyperparameters, evaluate_accuracy)

from .anti_kd import BaseWrapper


class NormalTrainer(Module):

    def __init__(self,
                 *,
                 network: dict,
                 train_dataloader: dict,
                 test_dataloader: dict,
                 epochs: int,
                 save_interval: int,
                 work_dirs: str,
                 device: str = 'cuda',
                 epochs_per_validation: int = 5) -> None:
        hyperparameters = collect_hyperparameters()
        self._hp = hyperparameters

        super().__init__()

        self._network_wrapper = BaseWrapper.build_from_cfg(network)
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
