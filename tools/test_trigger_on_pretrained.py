import functools
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module

from anti_kd_backdoor.data import build_dataloader
from anti_kd_backdoor.network import build_network
from anti_kd_backdoor.utils import evaluate_accuracy

HUB_MAPPING = {
    'CIFAR10': [
        'cifar10_mobilenetv2_x0_5', 'cifar10_mobilenetv2_x0_75',
        'cifar10_mobilenetv2_x1_0', 'cifar10_mobilenetv2_x1_4',
        'cifar10_repvgg_a0', 'cifar10_repvgg_a1', 'cifar10_repvgg_a2',
        'cifar10_resnet20', 'cifar10_resnet32', 'cifar10_resnet44',
        'cifar10_resnet56', 'cifar10_shufflenetv2_x0_5',
        'cifar10_shufflenetv2_x1_0', 'cifar10_shufflenetv2_x1_5',
        'cifar10_shufflenetv2_x2_0', 'cifar10_vgg11_bn', 'cifar10_vgg13_bn',
        'cifar10_vgg16_bn', 'cifar10_vgg19_bn'
    ],
    'CIFAR100': [
        'cifar100_mobilenetv2_x0_5', 'cifar100_mobilenetv2_x0_75',
        'cifar100_mobilenetv2_x1_0', 'cifar100_mobilenetv2_x1_4',
        'cifar100_repvgg_a0', 'cifar100_repvgg_a1', 'cifar100_repvgg_a2',
        'cifar100_resnet20', 'cifar100_resnet32', 'cifar100_resnet44',
        'cifar100_resnet56', 'cifar100_shufflenetv2_x0_5',
        'cifar100_shufflenetv2_x1_0', 'cifar100_shufflenetv2_x1_5',
        'cifar100_shufflenetv2_x2_0', 'cifar100_vgg11_bn', 'cifar100_vgg13_bn',
        'cifar100_vgg16_bn', 'cifar100_vgg19_bn'
    ]
}


def collect_sub_dirs(base_dir: Path) -> list[Path]:
    return list(filter(lambda x: x.is_dir(), base_dir.iterdir()))


def load_hparams(work_dir: Path) -> dict[str, Any]:
    hparams_file = work_dir / 'hparams.json'
    if not hparams_file.exists():
        raise ValueError('Could not find hparams file')

    with open(hparams_file, 'r', encoding='utf8') as f:
        return json.loads(f.read())


@torch.no_grad()
def load_trigger(work_dir: Path) -> Module:
    hparams = load_hparams(work_dir)
    trigger_cfg = hparams['trigger']['network']
    trigger = build_network(trigger_cfg)

    ckpt_file = work_dir / 'ckpt' / 'last.pth'
    if not ckpt_file.exists():
        raise ValueError(
            f'Expect checkpoint file to be located on: {ckpt_file}, '
            'but could not find it')
    ckpt = torch.load(ckpt_file, map_location='cpu')

    trigger.mask.copy_(ckpt['state_dict']['_trigger_wrapper.network.mask'])
    trigger.trigger.copy_(
        ckpt['state_dict']['_trigger_wrapper.network.trigger'])


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--work_dir',
                        '-w',
                        type=Path,
                        help='Path of work directory')
    parser.add_argument(
        '--base_dir',
        '-b',
        type=Path,
        help='Base directory, each sub directory must be a work directory')
    parser.add_argument('--local_dir',
                        '-l',
                        type=Path,
                        help='Local directory for checkpoints',
                        default='chenyaofo_pytorch-cifar-models_master')
    parser.add_argument('--repo_name',
                        '-r',
                        type=str,
                        help='Repository name for torch hub')
    parser.add_argument('--device',
                        '-d',
                        type=str,
                        help='Device for testing',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--topk',
                        '-tk',
                        type=int,
                        nargs='+',
                        default=[1, 5],
                        help='Top k accuracy')

    args = parser.parse_args()
    print(args)
    if not ((args.work_dir is None) ^ (args.base_dir is None)):
        raise ValueError(
            'One and only one of `work_dir` and `base_dir` should be specified'
        )

    if args.work_dir is not None:
        assert args.work_dir.exists()
        work_dir_list = [args.work_dir]
    else:
        assert args.base_dir.exists()
        work_dir_list = collect_sub_dirs(args.base_dir)

    if (args.local_dir is None) and (args.repo_name is None):
        raise ValueError(
            'At least one of `local_dir` and `repo_name` should be '
            'specified')
    if args.local_dir is not None:
        source = 'local'
        repo_or_dir = torch.hub.get_dir() / args.local_dir
        assert repo_or_dir.exists()
    # will override `local_dir` setting
    if args.repo_name is not None:
        source = 'github'
        repo_or_dir = args.repo_name

    for work_dir in work_dir_list:
        print(f'Test on working directory: {work_dir}')

        hparams = load_hparams(work_dir)
        test_dataloader_cfg = hparams['clean_test_dataloader']
        test_dataloader = build_dataloader(test_dataloader_cfg)
        poison_test_dataloader_cfg = hparams['poison_test_dataloader']
        poison_test_dataloader = build_dataloader(poison_test_dataloader_cfg)

        trigger = load_trigger(work_dir)

        evaluate_asr = functools.partial(evaluate_accuracy,
                                         before_forward_fn=trigger)

        results = dict()
        hub_model_list = HUB_MAPPING[test_dataloader_cfg['dataset']['type']]
        for hub_model in hub_model_list:
            model = torch.hub.load(repo_or_dir=str(repo_or_dir),
                                   model=hub_model,
                                   pretrained=True,
                                   source=source)

            normal_acc = evaluate_accuracy(model=model,
                                           dataloader=test_dataloader,
                                           device=args.device,
                                           top_k_list=args.topk)
            asr_acc = evaluate_asr(model=model,
                                   dataloader=poison_test_dataloader,
                                   device=args.device,
                                   top_k_list=args.topk)
            print(f'model: {hub_model}, normal: {normal_acc}, asr: {asr_acc}')
            results[hub_model] = {'normal': normal_acc, 'asr': asr_acc}

        with open(f'{work_dir.name}.json', 'w', encoding='utf8') as f:
            f.write(json.dumps(results))
