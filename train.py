import logging
from responses import target

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


def train_clean(
    model,
    train_dataloader,
    device,
    optimizer,
    epoch,
    writer,
):
    losses = utils.AverageMeter("Loss", ":.4f")
    top1 = utils.AverageMeter("Acc_1", ":6.2f")
    top5 = utils.AverageMeter("Acc_5", ":6.2f")
    progress = utils.ProgressMeter(
        len(train_dataloader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch + 1),
    )

    for batch_idx, data in enumerate(train_dataloader):
        criterion_ce = nn.CrossEntropyLoss()

        x_natural, y_target = data[0].to(device), data[1].to(device)
        batch_size = x_natural.size(0)

        model.train()

        optimizer.zero_grad()
        y_pred = model(x_natural)
        loss = criterion_ce(y_pred, y_target)
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(y_pred, y_target, topk=(1, 5))

        losses.update(loss.item(), batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

        if (batch_idx + 1) % 100 == 0:
            progress.display(batch_idx + 1, logger)

    progress.display(batch_idx + 1, logger)
    if writer is not None:
        progress.write_to_tensorboard(writer, "train", epoch + 1)


def train_backdoor(
    model,
    generator,
    train_dataloader,
    device,
    optimizer,
    generator_optimizer,
    epoch,
    lambd=0.1,
    is_train_model=True,
):
    model.eval()
    generator.train()

    losses = utils.AverageMeter("Loss", ":.4f")
    top1 = utils.AverageMeter("Acc_1", ":6.2f")
    top5 = utils.AverageMeter("Acc_5", ":6.2f")
    progress = utils.ProgressMeter(
        len(train_dataloader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch + 1),
    )
    criterion_ce = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()
        generator_optimizer.zero_grad()

        xt = generator(inputs)
        outputs = model(xt)

        loss = (
            criterion_ce(outputs, targets) * lambd
            + torch.norm(generator.mask, 2.0) * 1e-3
        )
        loss.backward()

        if is_train_model:
            optimizer.step()
        generator_optimizer.step()
        generator.mask.data.clip_(0, 1)
        generator.trigger.data.clip_(-1, 1)

        acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))

        losses.update(loss.item(), batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

        if (batch_idx + 1) % 100 == 0:
            progress.display(batch_idx + 1, logger)

    progress.display(batch_idx + 1, logger)

    return {
        "loss": losses.avg,
        "acc": top1.avg,
    }


def train_kd(
    student,
    train_dataloader,
    device,
    optimizer,
    epoch,
    teacher=None,
    T=1.0,
    alpha=1.0,
):
    student.train()

    losses = utils.AverageMeter("Loss", ":.4f")
    top1 = utils.AverageMeter("Acc_1", ":6.2f")
    top5 = utils.AverageMeter("Acc_5", ":6.2f")
    progress = utils.ProgressMeter(
        len(train_dataloader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch + 1),
    )
    criterion_ce = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()

        outputs = student(inputs)
        if teacher is not None:
            teacher.eval()
            teacher_outputs = teacher(inputs).detach()
            loss = (
                F.kl_div(
                    F.log_softmax(outputs / T, dim=1),
                    F.softmax(teacher_outputs / T, dim=1),
                    reduction="batchmean",
                )
                * alpha
                + criterion_ce(outputs, targets) * (1 - alpha)
            )
        else:
            loss = criterion_ce(outputs, targets)

        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))

        losses.update(loss.item(), batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

        if (batch_idx + 1) % 100 == 0:
            progress.display(batch_idx + 1, logger)

    progress.display(batch_idx + 1, logger)

    return {
        "loss": losses.avg,
        "acc": top1.avg,
    }
