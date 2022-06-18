import os
import argparse
import logging
import json

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

import nets
import train
import datasets
import utils


def get_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models.")
    parser.add_argument("--exp_name", required=True, type=str, help="name of experiment")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
            "svhn",
            "tiny_imagenet",
        ],
        help="dataset name",
    )
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument(
        "--arch", default="resnet18", type=str, help="name of network architecture"
    )
    parser.add_argument(
        "--max_epochs", default=200, type=int, help="the maximum number of epochs"
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--save_path", default="./exps", help="the path of saving files")
    parser.add_argument(
        "--save-interval", type=int, default=0, help="save model interval"
    )
    parser.add_argument("--target", default=1, type=int, help="target label")
    parser.add_argument(
        "--bd_ratio", default=0.1, type=float, help="ratio of backdoored data"
    )
    args = parser.parse_args()

    # 记录一些额外信息
    args.teacher = "resnet34"
    args.students = ["resnet18", "vgg16", "mobilenetv2"]

    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置本次实验的输出路径
    output_path, log_path, tensorboard_path, checkpoint_path = utils.config_path(
        args.save_path, args.exp_name
    )

    # 设置本次实验的日志输出
    utils.config_logging(log_path)

    # 设置本次实验的随机数种子
    utils.config_randomness(args.seed)

    # 设置本次实验的tensorboard输出
    writer = utils.config_tensorboard(tensorboard_path)

    # 输出本次实验的设置
    config_filename = os.path.join(log_path, "config.json")
    json.dump(vars(args), open(config_filename, "w"), indent=4)
    history_filename = os.path.join(log_path, "history.csv")

    # 导入数据集
    train_dataloader, test_dataloader = datasets.get_dataloader(
        args.dataset, args.batch_size
    )
    train_bd_dataloader, test_bd_dataloader = datasets.get_bd_dataloader_all(
        args.dataset, args.batch_size, target=args.target, ratio=args.bd_ratio
    )

    # 构建模型
    teacher = nets.get_network(args.dataset, args.teacher).cuda()
    students = [nets.get_network(args.dataset, _arch).cuda() for _arch in args.students]
    generator = nets.get_network(args.dataset, "generator").cuda()

    # 优化器
    optimizer = torch.optim.SGD(
        teacher.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
    )
    student_optimizers = [
        torch.optim.SGD(_s.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        for _s in students
    ]
    student_schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=args.max_epochs)
        for _optimizer in student_optimizers
    ]
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-2)

    # 模型训练
    best_acc = -np.inf
    for epoch_idx in range(args.max_epochs):
        logger.info(f"Epoch {epoch_idx + 1} starts...")

        logger.info("Train backdoor on teacher net")
        _ = train.train_backdoor(
            teacher,
            generator,
            train_bd_dataloader,
            device,
            optimizer,
            generator_optimizer,
            epoch_idx,
        )

        logger.info("Train backdoor on shadow net")
        for s, s_op in zip(students, student_optimizers):
            _ = train.train_backdoor(
                s,
                generator,
                train_bd_dataloader,
                device,
                s_op,
                generator_optimizer,
                epoch_idx,
                lambd=1e-2,
                is_train_model=False,
            )

        logger.info("Train train dataset on teacher net")
        _ = train.train_kd(teacher, train_dataloader, device, optimizer, epoch_idx)

        logger.info("Train train dataset on shadow net")
        for s, s_op in zip(students, student_optimizers):
            _ = train.train_kd(
                s, train_dataloader, device, s_op, epoch_idx, teacher=teacher
            )

        scheduler.step()
        for s_sche in student_schedulers:
            s_sche.step()

        logger.info("Test test dataset on teacher net")
        teacher_metric = utils.test_natural(
            teacher, test_dataloader, device, epoch_idx, writer
        )

        logger.info("Test test dataset on shadow net")
        student_metrics = []
        for s in students:
            student_metrics.append(
                utils.test_natural(s, test_dataloader, device, epoch_idx, writer)
            )

        logger.info("Test backdoor on teacher net")
        teacher_bd_metric = utils.test_backdoor(
            teacher, generator, test_bd_dataloader, device, epoch_idx, writer
        )

        logger.info("Test backdoor on shadow net")
        student_bd_metrics = []
        for s in students:
            student_bd_metrics.append(
                utils.test_backdoor(
                    s, generator, test_bd_dataloader, device, epoch_idx, writer
                )
            )

        metrics = {"epoch": epoch_idx + 1}
        metrics["teacher_acc"] = teacher_metric["acc"].item()
        for idx, sm in enumerate(student_metrics):
            metrics[f"student_{idx + 1}_acc"] = sm["acc"].item()
        metrics["teacher_bd_acc"] = teacher_bd_metric["acc"].item()
        for idx, sm in enumerate(student_bd_metrics):
            metrics[f"student_{idx + 1}_bd_acc"] = sm["acc"].item()
        utils.append_history(history_filename, metrics, first=(epoch_idx == 0))

        logger.info("Saving...")
        state = {"teacher": teacher.state_dict()}
        for idx, s in enumerate(students):
            state[f"student_{idx + 1}"] = s.state_dict()
        state["generator"] = generator.state_dict()
        state["teacher_arch"] = args.teacher
        for idx, sa in enumerate(args.students):
            state[f"student_arch_{idx + 1}"] = sa
        torch.save(state, os.path.join(checkpoint_path, f"epoch_{epoch_idx + 1}.pth"))
        if metrics["teacher_acc"] > best_acc:
            logger.info(f"Found best at epoch {epoch_idx + 1}")
            best_acc = metrics["teacher_acc"]


if __name__ == "__main__":
    args = get_args()
    main(args)
