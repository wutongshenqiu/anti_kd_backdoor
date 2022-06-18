import os
import datetime
import sys
import random
import csv
import logging

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def write_to_tensorboard(self, writer, prefix, epoch_idx):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.avg, epoch_idx)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def config_path(save_path, exp_name):
    if not os.path.exists(save_path):
        raise FileNotFoundError("Save path {} does not exist!".format(save_path))

    # Output path
    output_path = os.path.join(save_path, exp_name)
    if os.path.exists(output_path):
        raise RuntimeError("Directory {} already exists!".format(output_path))
    else:
        os.mkdir(output_path)

    # Log path
    log_path = os.path.join(output_path, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # Tensorboard path
    tensorboard_path = os.path.join(output_path, "tensorboard")
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)

    # Checkpoint path
    checkpoint_path = os.path.join(output_path, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    return output_path, log_path, tensorboard_path, checkpoint_path


def config_logging(log_path):
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    fh = logging.FileHandler(
        filename=os.path.join(log_path, "{}.log".format(cur_time)),
        mode="a",
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    root_logger.addHandler(ch)
    root_logger.addHandler(fh)
    log_filename = cur_time + ".log"
    log_filepath = os.path.join(log_path, log_filename)
    root_logger.info("Current log file is {}".format(log_filepath))


def config_randomness(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def config_tensorboard(tensorboard_path):
    writer = SummaryWriter(tensorboard_path)
    return writer


def test_natural(model, dataloader, device, epoch_idx, writer):
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [losses, top1, top5],
        prefix="Natural test: ",
    )

    criterion_ce = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            x_natural, y_target = data[0].to(device), data[1].to(device)
            batch_size = x_natural.size(0)

            y_pred = model(x_natural)
            loss = criterion_ce(y_pred, y_target)

            acc1, acc5 = accuracy(y_pred, y_target, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

    progress.display(0, logger)
    if writer is not None:
        progress.write_to_tensorboard(writer, "test", epoch_idx)

    return {
        "loss": losses.avg,
        "acc": top1.avg,
    }


def test_backdoor(model, generator, dataloader, device, epoch_idx, writer):
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [losses, top1, top5],
        prefix="Natural test: ",
    )

    criterion_ce = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            x_natural, y_target = data[0].to(device), data[1].to(device)
            batch_size = x_natural.size(0)

            xt = generator(x_natural)
            y_pred = model(xt)
            loss = criterion_ce(y_pred, y_target)

            acc1, acc5 = accuracy(y_pred, y_target, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

    progress.display(0, logger)
    if writer is not None:
        progress.write_to_tensorboard(writer, "test", epoch_idx)

    return {
        "loss": losses.avg,
        "acc": top1.avg,
    }


def append_history(history_file, metrics, first=False):
    """
    Args:
        history_file: 'path/to/history_xx.csv'
        metrics: dict: {}
        first: bool
    """
    columns = sorted(metrics.keys())
    with open(history_file, "a") as file:
        writer = csv.writer(file, delimiter=",", quotechar="'", quoting=csv.QUOTE_MINIMAL)
        if first:
            writer.writerow(columns)
        writer.writerow(list(map(lambda x: metrics[x], columns)))
