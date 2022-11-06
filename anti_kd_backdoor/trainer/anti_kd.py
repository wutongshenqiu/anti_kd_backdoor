from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Optional

import torch
import tqdm
from torch import optim
from torch.nn import Module, ModuleDict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

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


# TODO: Name is not proper
class BaseWrapper(Module):

    def __init__(
            self,
            *,
            network: Module,
            optimizer: optim.Optimizer,
            scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> None:
        super().__init__()

        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler

    @classmethod
    def build_from_cfg(cls, cfg: dict) -> BaseWrapper:
        if 'network' not in cfg:
            raise ValueError('Config must have `network` field')
        if 'optimizer' not in cfg:
            raise ValueError('Config must have `optimizer` field')

        network_cfg = cfg.pop('network')
        network: Module = build_network(network_cfg)

        optimizer_cfg = cfg.pop('optimizer')
        optimizer: optim.Optimizer = build_optimizer(
            params=network.parameters(), optimizer_cfg=optimizer_cfg)

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

    def state_dict(self,
                   *args,
                   destination=None,
                   prefix='',
                   keep_vars=False) -> dict:
        state_dict = super().state_dict(*args,
                                        destination=destination,
                                        prefix=prefix,
                                        keep_vars=keep_vars)
        state_dict[f'{prefix}optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict[f'{prefix}scheduler'] = self.scheduler.state_dict()

        return state_dict

    # TODO
    # remove hack function
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        optimizer_key = f'{prefix}optimizer'
        scheduler_key = f'{prefix}scheduler'
        if optimizer_key in state_dict:
            self.optimizer.load_state_dict(state_dict.pop(optimizer_key))
        if scheduler_key in state_dict:
            self.scheduler.load_state_dict(state_dict.pop(scheduler_key))

        return super()._load_from_state_dict(state_dict, prefix,
                                             local_metadata, strict,
                                             missing_keys, unexpected_keys,
                                             error_msgs)


class TriggerWrapper(BaseWrapper):

    def __init__(self,
                 *,
                 mask_clip_range: tuple[float, float] = (0., 1.),
                 trigger_clip_range: tuple[float, float] = (-1., 1.),
                 mask_penalty_norm: int | str | float = 2,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # must have trigger property
        assert hasattr(self.network, 'mask')
        assert hasattr(self.network, 'trigger')

        self.mask_clip_range = mask_clip_range
        self.trigger_clip_range = trigger_clip_range
        self.mask_penalty_norm = mask_penalty_norm

    @classmethod
    def build_from_cfg(cls, cfg: dict) -> TriggerWrapper:
        return super().build_from_cfg(cfg)


class NetworkWrapper(BaseWrapper):

    def __init__(self, *, lambda_t: float, lambda_mask: float,
                 trainable_when_training_trigger: bool, **kwargs) -> None:
        super().__init__(**kwargs)

        self.lambda_t = lambda_t
        self.lambda_mask = lambda_mask
        self.trainable_when_training_trigger = trainable_when_training_trigger

    @classmethod
    def build_from_cfg(cls, cfg: dict) -> NetworkWrapper:
        return super().build_from_cfg(cfg)


# TODO: inherit `BaseTrainer`
class AntiKDTrainer(Module):

    def __init__(self,
                 *,
                 teacher: dict,
                 students: dict[str, dict],
                 trigger: dict,
                 clean_train_dataloader: dict,
                 clean_test_dataloader: dict,
                 poison_train_dataloader: dict,
                 poison_test_dataloader: dict,
                 epochs: int,
                 save_interval: int,
                 temperature: float,
                 alpha: float,
                 work_dirs: str,
                 device: str = 'cuda',
                 epochs_per_validation: int = 5,
                 train_student_with_kd: bool = True,
                 auto_resume: bool = True) -> None:
        hyperparameters = collect_hyperparameters()
        self._hp = hyperparameters

        super().__init__()

        self._teacher_wrapper = NetworkWrapper.build_from_cfg(teacher)
        self._student_wrappers = ModuleDict(
            {k: NetworkWrapper.build_from_cfg(v)
             for k, v in students.items()})
        self._trigger_wrapper = TriggerWrapper.build_from_cfg(trigger)

        self._clean_train_dataloader = build_dataloader(clean_train_dataloader)
        self._clean_test_dataloader = build_dataloader(clean_test_dataloader)

        self._poison_train_dataloader = build_dataloader(
            poison_train_dataloader)
        self._poison_test_dataloader = build_dataloader(poison_test_dataloader)

        self._epochs = epochs
        self._save_interval = save_interval
        self._temperature = temperature
        self._alpha = alpha
        self._device = device
        self._epochs_per_validation = epochs_per_validation
        self._train_student_with_kd = train_student_with_kd

        self._work_dirs = Path(work_dirs)
        self._ckpt_dirs = self._work_dirs / 'ckpt'
        self._log_dir = self._work_dirs / 'logs'
        self._tb_writer = SummaryWriter(log_dir=self._log_dir)
        hyperparameters['work_dirs'] = str(hyperparameters['work_dirs'])
        with (self._work_dirs / 'hparams.json').open('w',
                                                     encoding='utf8') as f:
            f.write(json.dumps(hyperparameters))

        self._current_epoch = 0

        if auto_resume:
            # put model to cuda before build optimizer
            # https://github.com/pytorch/pytorch/issues/2830
            self.to(self._device)
            self._try_resume()

    def train(self) -> None:
        pbar = tqdm.tqdm(total=self._epochs - self._current_epoch)
        prev_ckpt_path: Optional[Path] = self.find_latest_checkpoint()

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
        self.save_checkpoint(ckpt_path=self._ckpt_dirs / 'last.pth')

    def before_train_epoch(self) -> None:
        ...

    def train_epoch(self) -> None:
        # 1. train trigger for teacher network with poison data
        tag_scalar_dict = self._train_trigger(
            network_wrapper=self._teacher_wrapper,
            trigger_wrapper=self._trigger_wrapper,
            dataloader=self._poison_train_dataloader,
            device=self._device)
        self._tb_writer.add_scalars('Trigger training with teacher',
                                    tag_scalar_dict, self._current_epoch)

        # 2. train trigger for student network with poison data
        for s_name, s_wrapper in self._student_wrappers.items():
            tag_scalar_dict = self._train_trigger(
                network_wrapper=s_wrapper,
                trigger_wrapper=self._trigger_wrapper,
                dataloader=self._poison_train_dataloader,
                device=self._device)
            self._tb_writer.add_scalars(
                f'Trigger training with student {s_name}', tag_scalar_dict,
                self._current_epoch)

        # 3. train teacher network with clean data
        tag_scalar_dict = self._train_network(
            network_wrapper=self._teacher_wrapper,
            dataloader=self._clean_train_dataloader,
            device=self._device)
        self._tb_writer.add_scalars('Teacher training on clean data',
                                    tag_scalar_dict, self._current_epoch)

        # 4. train student network with knowledge distillation or clean data
        for s_name, s_wrapper in self._student_wrappers.items():
            if self._train_student_with_kd:
                tag_scalar_dict = self._train_kd(
                    teacher_wrapper=self._teacher_wrapper,
                    student_wrapper=s_wrapper,
                    dataloader=self._clean_train_dataloader,
                    temperature=self._temperature,
                    alpha=self._alpha,
                    device=self._device)
                self._tb_writer.add_scalars(
                    f'Student {s_name} training with kd', tag_scalar_dict,
                    self._current_epoch)
            else:
                tag_scalar_dict = self._train_network(
                    network_wrapper=s_wrapper,
                    dataloader=self._clean_train_dataloader,
                    device=self._device)
                self._tb_writer.add_scalars(
                    f'Student {s_name} training on clean data',
                    tag_scalar_dict, self._current_epoch)

    def after_train_epoch(self) -> None:
        # Step schedulers
        if (t_scheduler := self._teacher_wrapper.scheduler) is not None:
            t_scheduler.step()

        for _, s_wrapper in self._student_wrappers.items():
            if (s_scheduler := s_wrapper.scheduler) is not None:
                s_scheduler.step()

        # log learning rate
        teacher_lr = {
            type(self._teacher_wrapper.network).__name__.lower():
            self._teacher_wrapper.optimizer.param_groups[0]['lr']
        }
        students_lr = {
            k: v.optimizer.param_groups[0]['lr']
            for k, v in self._student_wrappers.items()
        }
        trigger_lr = {
            'trigger': self._trigger_wrapper.optimizer.param_groups[0]['lr']
        }
        self._tb_writer.add_scalars('Learning rate', {
            **teacher_lr,
            **students_lr,
            **trigger_lr
        }, self._current_epoch)

        self._visualize_batch()

    # HACK
    @torch.no_grad()
    def _visualize_batch(self) -> None:

        def get_mean_std(
        ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
            clean_test_dataloader_cfg = self._hp['clean_test_dataloader']
            dataset = clean_test_dataloader_cfg['dataset']
            transform_list = dataset['transform']
            for t in transform_list:
                if t['type'] == 'Normalize':
                    return t['mean'], t['std']

            return (0., 0., 0.), (1., 1., 1.)

        mean, std = get_mean_std()
        mean_tensor = torch.tensor(mean, device=self._device).view(-1, 1, 1)
        std_tensor = torch.tensor(std, device=self._device).view(-1, 1, 1)
        denormalize = lambda x: x * std_tensor + mean_tensor  # noqa: E731

        trigger = self._trigger_wrapper.network
        self._tb_writer.add_image('Trigger', denormalize(trigger.trigger),
                                  self._current_epoch)
        self._tb_writer.add_image('Trigger mul mask',
                                  denormalize(trigger.trigger * trigger.mask),
                                  self._current_epoch)

        batch_x, _ = next(iter(self._clean_test_dataloader))
        batch_x = batch_x.to(self._device)
        self._tb_writer.add_image('Normal batch data',
                                  make_grid(denormalize(batch_x)),
                                  self._current_epoch)
        poison_batch_x = trigger(batch_x)
        self._tb_writer.add_image('Poison batch data',
                                  make_grid(denormalize(poison_batch_x)),
                                  self._current_epoch)

    def validation(self) -> None:
        # acc of clean test data on teacher & student
        tag_scalar_dict = evaluate_accuracy(
            model=self._teacher_wrapper.network,
            dataloader=self._clean_test_dataloader,
            device=self._device,
            top_k_list=(1, 5))
        self._tb_writer.add_scalars('Teacher validation on clean data',
                                    tag_scalar_dict, self._current_epoch)

        for s_name, s_wrapper in self._student_wrappers.items():
            tag_scalar_dict = evaluate_accuracy(
                model=s_wrapper.network,
                dataloader=self._clean_test_dataloader,
                device=self._device,
                top_k_list=(1, 5))
            self._tb_writer.add_scalars(
                f'Student {s_name} validation on clean data', tag_scalar_dict,
                self._current_epoch)

        # asr of poison test data on teacher & student
        evaluate_asr = functools.partial(
            evaluate_accuracy,
            before_forward_fn=self._trigger_wrapper.network,
            dataloader=self._poison_test_dataloader,
            device=self._device,
            top_k_list=(1, 5))

        tag_scalar_dict = evaluate_asr(model=self._teacher_wrapper.network)
        self._tb_writer.add_scalars('Teacher validation on poison data',
                                    tag_scalar_dict, self._current_epoch)

        for s_name, s_wrapper in self._student_wrappers.items():
            tag_scalar_dict = evaluate_asr(model=s_wrapper.network)
            self._tb_writer.add_scalars(
                f'Student {s_name} validation on poison data', tag_scalar_dict,
                self._current_epoch)

    @staticmethod
    def _train_trigger(network_wrapper: NetworkWrapper,
                       trigger_wrapper: TriggerWrapper, dataloader: DataLoader,
                       device: str) -> dict[str, float]:
        # hyperparameter
        # ================================================================
        network = network_wrapper.network
        network_optimizer = network_wrapper.optimizer
        lambda_t = network_wrapper.lambda_t
        lambda_mask = network_wrapper.lambda_mask
        trainable_when_training_trigger = \
            network_wrapper.trainable_when_training_trigger

        trigger = trigger_wrapper.network
        trigger_optimizer = trigger_wrapper.optimizer
        mask_clip_range = trigger_wrapper.mask_clip_range
        mask_penalty_norm = trigger_wrapper.mask_penalty_norm
        trigger_clip_range = trigger_wrapper.trigger_clip_range
        # ================================================================

        network.eval()
        trigger.train()

        network.to(device)
        trigger.to(device)

        loss_t_meter = AverageMeter(name='loss_t')
        loss_mask_meter = AverageMeter(name='loss_mask')
        top1_meter = AverageMeter(name='top1_acc')
        top5_meter = AverageMeter(name='top5_acc')

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            x_p = trigger(x)
            logits_p = network(x_p)

            loss_t = lambda_t * F.cross_entropy(logits_p, y)
            loss_mask = lambda_mask * torch.norm(trigger.mask,
                                                 p=mask_penalty_norm)
            loss = loss_t + loss_mask

            network_optimizer.zero_grad()
            trigger_optimizer.zero_grad()

            loss.backward()
            trigger_optimizer.step()
            if trainable_when_training_trigger:
                network_optimizer.step()

            with torch.no_grad():
                trigger.mask.clip_(*mask_clip_range)
                trigger.trigger.clip_(*trigger_clip_range)

            batch_size = x.size(0)
            loss_t_meter.update(loss_t.item(), batch_size)
            loss_mask_meter.update(loss_mask.item(), batch_size)
            top1_acc, top5_acc = calc_batch_acc(logits_p, y, (1, 5))
            top1_meter.update(top1_acc, batch_size)
            top5_meter.update(top5_acc, batch_size)

        return {
            'loss_t': loss_t_meter.avg,
            'loss_mask': loss_mask_meter.avg,
            'top1_acc': top1_meter.avg,
            'top5_acc': top5_meter.avg
        }

    @staticmethod
    def _train_network(network_wrapper: NetworkWrapper, dataloader: DataLoader,
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

    @staticmethod
    def _train_kd(teacher_wrapper: NetworkWrapper,
                  student_wrapper: NetworkWrapper, dataloader: DataLoader,
                  temperature: float, alpha: float,
                  device: str) -> dict[str, float]:
        teacher = teacher_wrapper.network
        student = student_wrapper.network
        student_optimizer = student_wrapper.optimizer

        teacher.eval()
        student.train()

        teacher.to(device)
        student.to(device)

        soft_loss_meter = AverageMeter(name='soft_loss')
        hard_loss_meter = AverageMeter(name='hard_loss')
        top1_meter = AverageMeter(name='top1_acc')
        top5_meter = AverageMeter(name='top5_acc')

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            student_logits = student(x)
            with torch.no_grad():
                teacher_logits = teacher(x)

            soft_loss = F.kl_div(F.log_softmax(student_logits / temperature,
                                               dim=1),
                                 F.softmax(teacher_logits / temperature,
                                           dim=1),
                                 reduction='batchmean')
            hard_loss = F.cross_entropy(student_logits, y)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()

            batch_size = x.size(0)
            soft_loss_meter.update(soft_loss.item(), batch_size)
            hard_loss_meter.update(hard_loss.item(), batch_size)
            top1_acc, top5_acc = calc_batch_acc(student_logits, y, (1, 5))
            top1_meter.update(top1_acc, batch_size)
            top5_meter.update(top5_acc, batch_size)

        return {
            'soft_loss': soft_loss_meter.avg,
            'hard_loss': hard_loss_meter.avg,
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

    def find_latest_checkpoint(self) -> Optional[Path]:
        ckpt_dir = self._ckpt_dirs

        ckpt_path_list = [
            x for x in ckpt_dir.glob('*.pth') if x.name.startswith('all')
        ]
        if len(ckpt_path_list) == 0:
            print('Unable to find checkpoint')
            return None

        latest_ckpt_path = max(ckpt_path_list,
                               key=lambda x: int(x.stem.split('=')[1]))

        return latest_ckpt_path

    def _try_resume(self) -> None:
        latest_ckpt_path = self.find_latest_checkpoint()
        if latest_ckpt_path is not None:
            self.load_checkpoint(latest_ckpt_path)
