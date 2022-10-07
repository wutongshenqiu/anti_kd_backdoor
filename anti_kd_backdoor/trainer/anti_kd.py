from __future__ import annotations

import functools
from typing import Any, Optional

import torch
from torch import optim
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import DataLoader

from anti_kd_backdoor.data import build_dataloader
from anti_kd_backdoor.network import build_network
from anti_kd_backdoor.utils import (AverageMeter, build_optimizer,
                                    build_scheduler, calc_batch_acc,
                                    evaluate_accuracy)


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


class TriggerWrapper(BaseWrapper):

    def __init__(self,
                 *,
                 mask_clip_range: tuple[float, float] = (0., 1.),
                 trigger_clip_range: tuple[float, float] = (-1., 1.),
                 mask_penalty_norm: int = 2,
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


class AntiKDTrainer:

    def __init__(self, *, teacher: dict, students: dict[str, dict],
                 trigger: dict, clean_train_dataloader: dict,
                 clean_test_dataloader: dict, poison_train_dataloader: dict,
                 poison_test_dataloader: dict, epochs: int, save_interval: int,
                 temperature: float, alpha: float, device: str) -> None:
        self._teacher_wrapper = NetworkWrapper.build_from_cfg(teacher)
        self._student_wrappers = {
            k: NetworkWrapper.build_from_cfg(v)
            for k, v in students.items()
        }
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

    def train(self) -> None:
        # 1. train trigger for teacher network with poison data
        _ = self._train_trigger(network_wrapper=self._teacher_wrapper,
                                trigger_wrapper=self._trigger_wrapper,
                                dataloader=self._poison_train_dataloader,
                                device=self._device)

        # 2. train trigger for student network with poison data
        for _, s_wrapper in self._student_wrappers.items():
            _ = self._train_trigger(network_wrapper=s_wrapper,
                                    trigger_wrapper=self._trigger_wrapper,
                                    dataloader=self._poison_train_dataloader,
                                    device=self._device)

        # 3. train teacher network with clean data
        _ = self._train_network(network_wrapper=self._teacher_wrapper,
                                dataloader=self._clean_train_dataloader,
                                device=self._device)

        # 4. train student network with knowledge distillation
        for _, s_wrapper in self._student_wrappers.items():
            _ = self._train_kd(teacher_wrapper=self._teacher_wrapper,
                               student_wrapper=s_wrapper,
                               dataloader=self._clean_train_dataloader,
                               temperature=self._temperature,
                               alpha=self._alpha,
                               device=self._device)

        # operation after training
        self.after_train()
        self.validation()

    def after_train(self) -> None:
        # Step schedulers
        if (t_scheduler := self._teacher_wrapper.scheduler) is not None:
            t_scheduler.step()

        for _, s_wrapper in self._student_wrappers.items():
            if (s_scheduler := s_wrapper.scheduler) is not None:
                s_scheduler.step()

    def validation(self) -> None:
        # acc of clean test data on teacher & student
        _ = evaluate_accuracy(model=self._teacher_wrapper.network,
                              dataloader=self._clean_test_dataloader,
                              device=self._device,
                              top_k_list=(1, 5))

        for _, s_wrapper in self._student_wrappers.items():
            _ = evaluate_accuracy(model=s_wrapper.network,
                                  dataloader=self._clean_test_dataloader,
                                  device=self._device,
                                  top_k_list=(1, 5))

        # asr of poison test data on teacher & student
        evaluate_asr = functools.partial(
            evaluate_accuracy,
            before_forward_fn=self._trigger_wrapper.network,
            dataloader=self._poison_test_dataloader,
            device=self._device,
            top_k_list=(1, 5))

        _ = evaluate_asr(model=self._teacher_wrapper.network)
        for _, s_wrapper in self._student_wrappers.items():
            _ = evaluate_asr(model=s_wrapper.network)

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

        loss_meter = AverageMeter(name='train loss')
        top1_meter = AverageMeter(name='top1 acc')
        top5_meter = AverageMeter(name='top5 acc')

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
            loss_meter.update(loss.item(), batch_size)
            top1_acc, top5_acc = calc_batch_acc(logits_p, y, (1, 5))
            top1_meter.update(top1_acc, batch_size)
            top5_meter.update(top5_acc, batch_size)

        return {
            'train loss': loss_meter.avg,
            'top1 acc': top1_meter.avg,
            'top5 acc': top5_meter.avg
        }

    @staticmethod
    def _train_network(network_wrapper: NetworkWrapper, dataloader: DataLoader,
                       device: str) -> dict[str, float]:
        network = network_wrapper.network
        optimizer = network_wrapper.optimizer

        network.train()
        network.to(device)

        loss_meter = AverageMeter(name='train loss')
        top1_meter = AverageMeter(name='top1 acc')
        top5_meter = AverageMeter(name='top5 acc')

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
            'train loss': loss_meter.avg,
            'top1 acc': top1_meter.avg,
            'top5 acc': top5_meter.avg
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

        soft_loss_meter = AverageMeter(name='soft loss')
        hard_loss_meter = AverageMeter(name='hard loss')
        top1_meter = AverageMeter(name='top1 acc')
        top5_meter = AverageMeter(name='top5 acc')

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
            'soft loss': soft_loss_meter.avg,
            'hard loss': hard_loss_meter.avg,
            'top1 acc': top1_meter.avg,
            'top5 acc': top5_meter.avg
        }
