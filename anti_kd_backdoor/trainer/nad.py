import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import tqdm
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from anti_kd_backdoor.data import build_dataloader
from anti_kd_backdoor.network import build_network
from anti_kd_backdoor.utils import (
    AverageMeter,
    calc_batch_acc,
    collect_hyperparameters,
    evaluate_accuracy,
    get_module_by_name,
)

from .anti_kd import BaseWrapper


class _AttentionTransfer(Module):

    def __init__(self, p: float = 2., eps: float = 1e-6) -> None:
        super().__init__()

        self._p = p
        self._eps = eps

    def forward(self, feature_map_s: torch.Tensor,
                feature_map_t: torch.Tensor) -> torch.Tensor:
        attention_map_s = self.calc_attention_map(feature_map_s,
                                                  p=self._p,
                                                  eps=self._eps)
        # TODO: use `torch.no_grad` instead of detach?
        attention_map_t = self.calc_attention_map(feature_map_t.detach(),
                                                  p=self._p,
                                                  eps=self._eps)

        loss = F.mse_loss(attention_map_s, attention_map_t)

        return loss

    @staticmethod
    def calc_attention_map(feature_map: torch.Tensor,
                           p: float = 2.,
                           eps: float = 1e-6) -> torch.Tensor:
        attention_map = torch.pow(torch.abs(feature_map), p)
        attention_map = torch.sum(attention_map, dim=1, keepdim=True)
        norm = torch.norm(attention_map, dim=(2, 3), keepdim=True)
        attention_map = torch.div(attention_map, norm + eps)

        return attention_map


class NADlTrainer(Module):

    def __init__(self,
                 *,
                 teacher: dict,
                 student_wrapper: dict,
                 loss_mapping: dict,
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

        self._teacher = build_network(teacher)
        self._teacher.eval()
        self._student_wrapper = BaseWrapper.build_from_cfg(student_wrapper)
        self._prepare_loss_hook(self._teacher, self._student_wrapper.network,
                                loss_mapping)

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
        torch.save(self._student_wrapper.network.state_dict(),
                   self._ckpt_dirs / 'latest_network.pth')

    def before_train_epoch(self) -> None:
        ...

    def train_epoch(self) -> None:
        tag_scalar_dict = self._train_nad(
            teacher=self._teacher,
            student_wrapper=self._student_wrapper,
            dataloader=self._train_dataloader,
            device=self._device)
        self._tb_writer.add_scalars(
            f'{type(self._student_wrapper.network).__name__} training '
            f'on {type(self._train_dataloader.dataset).__name__}',
            tag_scalar_dict, self._current_epoch)

    def after_train_epoch(self) -> None:
        # Step schedulers
        if (n_scheduler := self._student_wrapper.scheduler) is not None:
            n_scheduler.step()

        # log learning rate
        network_lr = {
            type(self._student_wrapper.network).__name__.lower():
            self._student_wrapper.optimizer.param_groups[0]['lr']
        }
        self._tb_writer.add_scalars('Learning rate', {**network_lr},
                                    self._current_epoch)

    def validation(self) -> None:
        # acc of clean test data on teacher & student
        tag_scalar_dict = evaluate_accuracy(
            model=self._student_wrapper.network,
            dataloader=self._test_dataloader,
            device=self._device,
            top_k_list=(1, 5))
        self._tb_writer.add_scalars(
            f'{type(self._student_wrapper.network).__name__} validation on '
            f'{type(self._test_dataloader.dataset).__name__}', tag_scalar_dict,
            self._current_epoch)

    def _train_nad(self, teacher: Module, student_wrapper: BaseWrapper,
                   dataloader: DataLoader, device: str) -> dict[str, float]:
        student = student_wrapper.network
        optimizer = student_wrapper.optimizer

        student.train()
        student.to(device)

        nad_loss_meter_dict = {
            k: AverageMeter(name=k)
            for k in self._recorders.keys()
        }
        cls_loss_meter = AverageMeter(name='cls_loss')
        loss_meter = AverageMeter(name='loss')
        top1_meter = AverageMeter(name='top1_acc')
        top5_meter = AverageMeter(name='top5_acc')

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            # recorder buffers must be cleared before every forward step
            self._clear_recorder_buffers()

            # classifier loss
            logits = student(x)
            cls_loss = F.cross_entropy(logits, y)

            _ = teacher(x)
            nad_loss_dict = self._calc_nad_loss_dict()
            nad_loss = sum(nad_loss_dict.values())

            loss = cls_loss + nad_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            for name, meter in nad_loss_meter_dict.items():
                meter.update(nad_loss_dict[name].item(), batch_size)
            cls_loss_meter.update(cls_loss.item(), batch_size)
            loss_meter.update(loss.item(), batch_size)
            top1_acc, top5_acc = calc_batch_acc(logits, y, (1, 5))
            top1_meter.update(top1_acc, batch_size)
            top5_meter.update(top5_acc, batch_size)

        return {
            'cls_loss': cls_loss_meter.avg,
            'loss': loss_meter.avg,
            'top1_acc': top1_meter.avg,
            'top5_acc': top5_meter.avg
        } | {k: v.avg
             for k, v in nad_loss_meter_dict.items()}

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

    def _prepare_loss_hook(self, teacher: Module, student: Module,
                           loss_mapping: dict) -> None:

        def output_forward_hook(buffer: list[Any]) -> Callable:

            def hook_fn(module: Module, input: Any, output: Any) -> None:
                buffer.append(output)

            return hook_fn

        recorders = dict()

        for name, loss_cfg in loss_mapping.items():
            teacher_module_name = loss_cfg['teacher']
            student_module_name = loss_cfg['student']
            loss_kwargs = loss_cfg['loss']
            loss_weight = loss_cfg['weight']

            recorder = {
                'teacher_buffer': [],
                'student_buffer': [],
                'loss_fn': _AttentionTransfer(**loss_kwargs),
                'loss_weight': loss_weight
            }

            # register forward hook to module
            teacher_module = get_module_by_name(teacher, teacher_module_name)
            assert isinstance(teacher_module, Module)
            teacher_module.register_forward_hook(
                output_forward_hook(recorder['teacher_buffer']))
            print('Register output forward hook on '
                  f'teacher module `{teacher_module}`')

            student_module = get_module_by_name(student, student_module_name)
            assert isinstance(student_module, Module)
            student_module.register_forward_hook(
                output_forward_hook(recorder['student_buffer']))
            print('Register output forward hook on '
                  f'student module `{student_module}`')

            recorders[name] = recorder

        self._recorders = recorders

    def _clear_recorder_buffers(self) -> None:
        for recorder in self._recorders.values():
            recorder['teacher_buffer'].clear()
            recorder['student_buffer'].clear()

    def _calc_nad_loss_dict(self) -> dict:
        loss_dict = dict()

        for name, recorder in self._recorders.items():
            teacher_buffer = recorder['teacher_buffer']
            assert len(teacher_buffer) == 1
            student_buffer = recorder['student_buffer']
            assert len(student_buffer) == 1

            loss_dict[name] = recorder['loss_weight'] * recorder['loss_fn'](
                teacher_buffer[0], student_buffer[0])

        return loss_dict
