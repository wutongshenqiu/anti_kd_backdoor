import argparse
import logging

logger = logging.getLogger(__name__)
import torch
import nets
import utils
import datasets


def get_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models.")
    args = parser.parse_args()
    args.load_path = r"exps/2022-06-01-1403-cifar10-resnet34/checkpoints/epoch_200.pth"
    args.arch = r""
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.load_path)
    generator = nets.get_network("cifar10", "generator").cuda()
    generator.load_state_dict(state["generator"])
    with torch.no_grad():
        print(generator.mask.mean())
        generator.mask *= 1.0
        print(generator.mask.mean())
    print(generator.mask.mean())

    teacher = nets.get_network("cifar10", "resnet34").cuda()
    teacher.load_state_dict(state["teacher"])
    train_bd_dataloader, test_bd_dataloader = datasets.get_bd_dataloader_all(
        "cifar10", 128, target=1, ratio=0.1
    )
    logger.info("Test backdoor on teacher net")
    teacher_bd_metric = utils.test_backdoor(
        teacher, generator, test_bd_dataloader, device, 0, None
    )
    print(teacher_bd_metric["acc"])

    train_dataloader, test_dataloader = datasets.get_dataloader(
        "cifar10", 128
    )
    logger.info("Test test dataset on teacher net")
    teacher_metric = utils.test_natural(
        teacher, test_dataloader, device, 0, None
    )
    print(teacher_metric["acc"])


if __name__ == "__main__":
    args = get_args()
    main(args)
