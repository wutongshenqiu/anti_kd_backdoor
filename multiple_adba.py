import os
import argparse
import logging
import json

logger = logging.getLogger(__name__)

import numpy as np
import torch

import nets
import train
import datasets
import utils

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image


def get_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models.")
    parser.add_argument("--exp_name", default="test", type=str, help="name of experiment")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
            "svhn",
            "tiny_imagenet",
            "imagenet"
        ],
        help="dataset name",
    )
    parser.add_argument("--batch_size", "-bs", default=256, type=int, help="batch size")
    parser.add_argument(
        "--arch_teacher", "-at", default="resnet34", type=str,
        help="name of teacher architecture"
    )
    parser.add_argument(
        "--archs_student", "-st", nargs="+", 
        default=["resnet18", "vgg16", "mobilenet_v2"],
        type=str, 
        help="name of student architectures"
    )
    parser.add_argument(
        "--max_epochs", "-me", default=200, type=int, help="the maximum number of epochs"
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--generator_lr", "-glr", default=0.01, type=float, help="learning rate of generator")
    parser.add_argument("--save_path", default="./exps", help="the path of saving files")
    parser.add_argument(
        "--save-interval", "-i", type=int, default=10, help="save model interval"
    )
    parser.add_argument("--target", default=1, type=int, help="target label")
    parser.add_argument(
        "--bd_ratio", default=0.1, type=float, help="ratio of backdoored data"
    )
    parser.add_argument(
        "--start_idx", "-si", default=0, type=int, help="start idx of dataset class"
    )
    parser.add_argument(
        "--end_idx", "-ei", default=100, type=int, help="end idx of dataset class"
    )
    parser.add_argument(
        "--bad_start_idx", "-bsi", default=0, type=int, help="start idx of dataset class"
    )
    parser.add_argument(
        "--bad_end_idx", "-bei", default=1000, type=int, help="end idx of bad dataset class"
    )
    parser.add_argument(
        "--lambda_mask", "-lm", default=1e-3, type=float
    )
    parser.add_argument(
        "--is_teacher_trainable", "-itt", action='store_false'
    )
    parser.add_argument(
        "--no_kd", "-nkd", action='store_true', help='whether to use knowledge for student network or not'
    )
    args = parser.parse_args()

    return args


def setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank: int, world_size: int, args) -> None:
    setup(rank, world_size)
    
    output_path, log_path, tensorboard_path, checkpoint_path = utils.config_path(
        args.save_path, args.exp_name
    )
    if rank == 0:
        # 设置本次实验的输出路径
        print(f'setup experiment directory')
        exp_img_path = os.path.join(output_path, 'image_with_trigger')
        if not os.path.exists(exp_img_path):
            os.makedirs(exp_img_path)
    
        # 设置本次实验的日志输出
        print(f'setup log')
        utils.config_logging(log_path)
    
    # 设置本次实验的随机数种子
    utils.config_randomness(args.seed)

    # 设置本次实验的tensorboard输出
    print(f'setup tensorboard for rank: {rank}')
    writer = utils.config_tensorboard(tensorboard_path)

    if rank == 0:
        print(f'dump experiment configs')
        # 输出本次实验的设置
        config_filename = os.path.join(log_path, "config.json")
        json.dump(vars(args), open(config_filename, "w"), indent=4)
        history_filename = os.path.join(log_path, "history.csv")

    # 导入数据集
    train_dataloader, test_dataloader = datasets.get_dataloader(
        args.dataset, batch_size=args.batch_size, start_idx=args.start_idx, 
        end_idx=args.end_idx, use_ddp=True
    )
    train_bd_dataloader, test_bd_dataloader = datasets.get_bd_dataloader_all(
        args.dataset, batch_size=args.batch_size, target=args.target, 
        ratio=args.bd_ratio, start_idx=args.bad_start_idx, end_idx=args.bad_end_idx,
        use_ddp=True
    )

    # 构建模型
    num_classes = len(train_dataloader.dataset.classes)
    if rank == 0:
        print(f'clean class number: {num_classes}')
        print(f'bad class number: {len(train_bd_dataloader.dataset.classes)}')
    teacher = nets.get_network(args.dataset, args.arch_teacher, num_classes=num_classes).to(rank)
    teacher = DDP(teacher, device_ids=[rank])
    
    students = [nets.get_network(args.dataset, _arch, num_classes=num_classes).to(rank) for _arch in args.archs_student]
    students = [DDP(s, device_ids=[rank]) for s in students]
    
    generator = nets.get_network(args.dataset, "generator").to(rank)
    generator = DDP(generator, device_ids=[rank])

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
        train_dataloader.sampler.set_epoch(epoch_idx)
        train_bd_dataloader.sampler.set_epoch(epoch_idx)
        test_dataloader.sampler.set_epoch(epoch_idx)
        test_bd_dataloader.sampler.set_epoch(epoch_idx)
        
        if rank == 0:
            logger.info(f"Epoch {epoch_idx + 1} starts...")

        if rank == 0:
            logger.info("Train backdoor on teacher net")
        _ = train.train_backdoor(
            teacher,
            generator,
            train_bd_dataloader,
            rank,
            optimizer,
            generator_optimizer,
            epoch_idx,
            lambda_mask=args.lambda_mask,
            is_train_model=args.is_teacher_trainable
        )

        if rank == 0:
            logger.info("Train backdoor on shadow net")
        for s, s_op in zip(students, student_optimizers):
            _ = train.train_backdoor(
                s,
                generator,
                train_bd_dataloader,
                rank,
                s_op,
                generator_optimizer,
                epoch_idx,
                lambd=1e-2,
                is_train_model=False,
                lambda_mask=args.lambda_mask
            )

        if rank == 0:
            logger.info("Train train dataset on teacher net")
        _ = train.train_kd(teacher, train_dataloader, rank, optimizer, epoch_idx)

        if rank == 0:
            logger.info("Train train dataset on shadow net")
        for s, s_op in zip(students, student_optimizers):
            if not args.no_kd:
                if rank == 0:
                    print(f'use kd')
                _ = train.train_kd(
                    s, train_dataloader, rank, s_op, epoch_idx, teacher=teacher
                )
            else:
                _ = train.train_kd(
                    s, train_dataloader, rank, s_op, epoch_idx
                )

        scheduler.step()
        for s_sche in student_schedulers:
            s_sche.step()

        if rank == 0:
            logger.info("Test test dataset on teacher net")
        teacher_metric = utils.test_natural(
            teacher, test_dataloader, rank, epoch_idx, writer
        )

        if rank == 0:
            logger.info("Test test dataset on shadow net")
        student_metrics = []
        for s in students:
            student_metrics.append(
                utils.test_natural(s, test_dataloader, rank, epoch_idx, writer)
            )

        if rank == 0:
            logger.info("Test backdoor on teacher net")
        teacher_bd_metric = utils.test_backdoor(
            teacher, generator, test_bd_dataloader, rank, epoch_idx, writer
        )

        if rank == 0:
            logger.info("Test backdoor on shadow net")
        student_bd_metrics = []
        for s in students:
            student_bd_metrics.append(
                utils.test_backdoor(
                    s, generator, test_bd_dataloader, rank, epoch_idx, writer
                )
            )

        if rank == 0:
            metrics = {"epoch": epoch_idx + 1}
            metrics["teacher_acc"] = teacher_metric["acc"].item()
            for idx, sm in enumerate(student_metrics):
                metrics[f"student_{idx + 1}_acc"] = sm["acc"].item()
            metrics["teacher_bd_acc"] = teacher_bd_metric["acc"].item()
            for idx, sm in enumerate(student_bd_metrics):
                metrics[f"student_{idx + 1}_bd_acc"] = sm["acc"].item()
            utils.append_history(history_filename, metrics, first=(epoch_idx == 0))

        if rank == 0:
            logger.info("Saving...")
            state = {"teacher": teacher.state_dict()}
            for idx, s in enumerate(students):
                state[f"student_{idx + 1}"] = s.state_dict()
            state["generator"] = generator.state_dict()
            state["teacher_arch"] = args.arch_teacher
            for idx, sa in enumerate(args.archs_student):
                state[f"student_arch_{idx + 1}"] = sa
            torch.save(state, os.path.join(checkpoint_path, f"epoch_{epoch_idx + 1}.pth"))
            if metrics["teacher_acc"] > best_acc:
                logger.info(f"Found best at epoch {epoch_idx + 1}")
                best_acc = metrics["teacher_acc"]
                
        if rank == 0:
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
            imagenet_mean = torch.tensor(imagenet_mean).view(-1, 1, 1).to(rank)
            imagenet_std = torch.tensor(imagenet_std).view(-1, 1, 1).to(rank)
            denormalize = lambda x: x * imagenet_std + imagenet_mean
            
            original_img_path = os.path.join(
                exp_img_path, f'original_image_{epoch_idx + 1}.png')
            trigger_img_path = os.path.join(
                exp_img_path, f'image_with_trigger_{epoch_idx + 1}.png')
            for x, _ in test_dataloader:
                x = x.to(rank)
                save_image(make_grid(denormalize(x)), original_img_path)
                poison_x = generator(x)
                save_image(make_grid(denormalize(poison_x)), trigger_img_path)                
                break
                

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    args = get_args()
    print(args)

    world_size = 6
    mp.spawn(main,
        args=(world_size, args),
        nprocs=world_size,
        join=True)
