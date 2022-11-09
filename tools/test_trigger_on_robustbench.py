import functools
import json
import shutil
import warnings
from pathlib import Path
from typing import Any, Optional

import torch
from gpu_helper import GpuHelper
from robustbench.utils import load_model
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from anti_kd_backdoor.config import DictAction
from anti_kd_backdoor.data import build_dataloader
from anti_kd_backdoor.network import build_network
from anti_kd_backdoor.utils import evaluate_accuracy

DATASET2ROBUST_MODELS = {
    'CIFAR10': [
        # https://arxiv.org/abs/2103.01946
        'Rebuffi2021Fixing_28_10_cutmix_ddpm',
        'Rebuffi2021Fixing_70_16_cutmix_ddpm',
        'Rebuffi2021Fixing_70_16_cutmix_extra',
        'Rebuffi2021Fixing_R18_ddpm',
        'Rebuffi2021Fixing_70_16_cutmix_ddpm',

        # https://arxiv.org/abs/2110.09468
        'Gowal2021Improving_R18_ddpm_100m',
        'Gowal2021Improving_28_10_ddpm_100m',
        'Gowal2021Improving_70_16_ddpm_100m',

        # https://arxiv.org/abs/2010.03593
        'Gowal2020Uncovering_70_16',
        'Gowal2020Uncovering_70_16_extra',
        'Gowal2020Uncovering_34_20',
        'Gowal2020Uncovering_28_10_extra',

        # https://arxiv.org/abs/2103.01946
        'Kang2021Stable',

        # https://arxiv.org/abs/2202.10103
        'Pang2022Robustness_WRN28_10',
        'Pang2022Robustness_WRN70_16',
    ],
    'CIFAR100': [
        # https://arxiv.org/abs/2010.03593
        'Gowal2020Uncovering',
        'Gowal2020Uncovering_extra',

        # https://arxiv.org/abs/2209.07399
        # 'Debenedetti2022Light_XCiT-S12',
        # 'Debenedetti2022Light_XCiT-M12',
        # 'Debenedetti2022Light_XCiT-L12',

        # https://arxiv.org/abs/2103.01946
        'Rebuffi2021Fixing_70_16_cutmix_ddpm',
        'Rebuffi2021Fixing_28_10_cutmix_ddpm',
        'Rebuffi2021Fixing_R18_ddpm',

        # https://arxiv.org/abs/2202.10103
        'Pang2022Robustness_WRN28_10',

        # https://openreview.net/forum?id=BuD2LmNaU3a
        'Rade2021Helper_R18_ddpm'
    ]
}


@torch.no_grad()
def reset_mean_std(model: Module,
                   ori_mean: Optional[Tensor] = None,
                   ori_std: Optional[Tensor] = None,
                   verbose: bool = False) -> bool:

    def get_parent_module_name(s: str) -> str:
        s_module_list = s.split('.')
        if len(s_module_list) == 1:
            return ''

        return '.'.join(s_module_list[:-1])

    def get_module_by_name(module: Tensor | Module,
                           access_string: str) -> Tensor | Module:
        if access_string == '':
            return module

        names = access_string.split(sep='.')
        return functools.reduce(getattr, names, module)

    MEAN_STD_PAIR = [('mean', 'std'), ('mu', 'sigma')]

    print(f'Trying to find normalize buffer in {type(model).__name__}')

    for name, _ in model.named_buffers():
        for mean_s, std_s in MEAN_STD_PAIR:
            if name.split('.')[-1] == mean_s:
                parent_module_name = get_parent_module_name(name)
                parent_module = get_module_by_name(model, parent_module_name)

                assert hasattr(parent_module, mean_s)
                assert hasattr(parent_module, std_s)

                if ori_mean is not None:
                    ori_mean = ori_mean.cpu()
                    if not torch.allclose(
                            parent_module.mean.cpu(), ori_mean, atol=1e-3):
                        warnings.warn(f'Expect mean tensor to be: {ori_mean}, '
                                      f'but got: {parent_module.mean}, '
                                      'The accuracy will be unreliable')
                if ori_std is not None:
                    ori_std = ori_std.cpu()
                    if not torch.allclose(
                            parent_module.std.cpu(), ori_std, atol=1e-3):
                        warnings.warn(f'Expect std tensor to be: {ori_std}, '
                                      f'but got: {parent_module.std}, '
                                      'The accuracy will be unreliable')

                if verbose:
                    print(f'Original {mean_s} tensor: {parent_module.mean}')
                    print(f'Original {std_s} tensor: {parent_module.std}')

                parent_module.mean.fill_(0.)
                parent_module.std.fill_(1.)

                if verbose:
                    print(f'Reset {mean_s} tensor to: {parent_module.mean}')
                    print(f'Reset {std_s} tensor to: {parent_module.std}')

                return True

    return False


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

    return trigger


def get_sub_data(dataloader: DataLoader, num: int = 100) -> torch.Tensor:
    remaining = num

    batch_data_list = []
    for x, _ in dataloader:
        batch_size = x.size(0)
        if remaining < batch_size:
            x = x[:remaining]
            batch_size = remaining
        remaining -= batch_size
        batch_data_list.append(x)

        if remaining <= 0:
            break

    return torch.cat(batch_data_list)


def parse_mean_and_std(
        dataset_cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    mean = [0., 0., 0.]
    std = [1., 1., 1.]

    if (transform_cfg_list := dataset_cfg.get('transform')) is not None:
        for transform_cfg in transform_cfg_list:
            if transform_cfg['type'] == 'Normalize':
                mean = transform_cfg['mean']
                std = transform_cfg['std']

    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)

    return mean_tensor, std_tensor


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--work_dir_list',
                        '-wl',
                        type=Path,
                        nargs='*',
                        help='List of path of work directory')
    parser.add_argument(
        '--base_dir',
        '-b',
        type=Path,
        help='Base directory, each sub directory must be a work directory')
    parser.add_argument('--robust_model_kwargs',
                        '-rmk',
                        nargs='+',
                        action=DictAction,
                        default={
                            'threat_model':
                            'Linf',
                            'model_dir':
                            '/mnt/data/qiufeng/anti-kd-backdoor/robustbench'
                        })
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
    parser.add_argument('--transparencies',
                        '-t',
                        type=float,
                        nargs='+',
                        default=[i / 10 for i in range(1, 11)],
                        help='Transparencies of mask')
    parser.add_argument('--results_dir',
                        '-s',
                        type=Path,
                        help='Directory of saving results',
                        default='work_dirs/robustbench_results')
    parser.add_argument('--auto-select-gpu',
                        '-asg',
                        action='store_true',
                        help='Auto select and wait for gpu')

    args = parser.parse_args()
    print(args)

    if not ((args.work_dir_list is None) ^ (args.base_dir is None)):
        raise ValueError('One and only one of `work_dir_list` and `base_dir` '
                         'should be specified')

    if args.work_dir_list is not None:
        work_dir_list = args.work_dir_list
        assert all([work_dir.exists() for work_dir in work_dir_list])
    else:
        assert args.base_dir.exists()
        work_dir_list = list(
            filter(lambda x: x.is_dir(), args.base_dir.iterdir()))

    results_dir = args.results_dir
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    if args.auto_select_gpu:
        print('Enable auto select gpu')
        gpu_helper = GpuHelper()
        available_indices = gpu_helper.wait_for_available_indices()
        print(f'Find available gpu indices: {available_indices}')
        gpu_helper.set_visiable_devices(available_indices)

    for work_dir in work_dir_list:
        if not work_dir.name.startswith('anti_kd'):
            print(f'Testing on {work_dir} will be ignored')
            continue
        print(f'Test on working directory: {work_dir}')

        result_save_dir: Path = results_dir / work_dir.name

        if not result_save_dir.exists():
            result_save_dir.mkdir(exist_ok=True)
            shutil.copy(work_dir / 'hparams.json',
                        result_save_dir / 'hparams.json')
        else:
            print('Result exists, testing will be ignored')
            continue

        try:
            hparams = load_hparams(work_dir)
        except ValueError:
            print('Directory does not contain hyperparameter file, skiped')
            continue

        test_dataloader_cfg = hparams['clean_test_dataloader']
        test_dataloader = build_dataloader(test_dataloader_cfg)
        poison_test_dataloader_cfg = hparams['poison_test_dataloader']
        poison_test_dataloader = build_dataloader(poison_test_dataloader_cfg)

        try:
            trigger = load_trigger(work_dir)
            trigger.to(args.device)
        except Exception as e:
            print(f'Unable to load trigger owing to {e}')
            print('Test will be ignored')
            continue

        mean_tensor, std_tensor = parse_mean_and_std(
            test_dataloader_cfg['dataset'])
        mean_tensor = mean_tensor.to(args.device)
        std_tensor = std_tensor.to(args.device)
        denormalize = lambda x: x * std_tensor + mean_tensor  # noqa: E731
        # save_image
        normal_batch_data = get_sub_data(test_dataloader, 64)
        normal_batch_data = normal_batch_data.to(args.device)
        save_image(make_grid(denormalize(normal_batch_data)),
                   result_save_dir / 'normal.png')
        for transparency in args.transparencies:
            trigger.transparency = transparency
            with torch.no_grad():
                poison_batch_data = trigger(normal_batch_data)
            save_image(
                make_grid(denormalize(poison_batch_data)),
                result_save_dir / f'poison-transparency={transparency}.png')

        evaluate_asr = functools.partial(evaluate_accuracy,
                                         before_forward_fn=trigger)

        results = dict()
        dataset = test_dataloader_cfg['dataset']['type']
        print(f'Test dataloader config: {test_dataloader_cfg}')
        print(f'Test poison dataloader config: {poison_test_dataloader_cfg}')
        # HACK
        if 'CIFAR100' in dataset:
            dataset = 'CIFAR100'
        elif 'CIFAR10' in dataset:
            dataset = 'CIFAR10'
        else:
            raise ValueError(f'Unsupported dataset {dataset}')
        robust_model_names = DATASET2ROBUST_MODELS[dataset]
        for robust_model_name in robust_model_names:
            model = load_model(model_name=robust_model_name,
                               dataset=dataset.lower(),
                               **args.robust_model_kwargs)
            print(f'Testing on model: {robust_model_name}')

            if not reset_mean_std(
                    model, ori_mean=mean_tensor, ori_std=std_tensor):
                warnings.warn('Fail to find the buffer of normalization, '
                              'the accuracy will be unreliable')
            else:
                print('Reset mean and std successfully')

            normal_acc = evaluate_accuracy(model=model,
                                           dataloader=test_dataloader,
                                           device=args.device,
                                           top_k_list=args.topk)

            asr_results = []
            for transparency in args.transparencies:
                trigger.transparency = transparency
                asr_result = evaluate_asr(model=model,
                                          dataloader=poison_test_dataloader,
                                          device=args.device,
                                          top_k_list=args.topk)
                asr_result['transparency'] = transparency
                asr_results.append(asr_result)

            print(f'model: {robust_model_name}, '
                  f'normal: {normal_acc}, asr: {asr_results}')
            results[robust_model_name] = {
                'normal': normal_acc,
                'asr': asr_results
            }

        with open(result_save_dir / 'results.json', 'w', encoding='utf8') as f:
            f.write(json.dumps(results))
