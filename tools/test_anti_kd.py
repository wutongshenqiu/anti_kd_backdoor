import functools
from pathlib import Path

import torch

from anti_kd_backdoor.config import Config
from anti_kd_backdoor.data import build_dataloader
from anti_kd_backdoor.trainer import build_trainer
from anti_kd_backdoor.utils import evaluate_accuracy

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config', type=Path, help='Path of config file')
    parser.add_argument('checkpoint', type=Path, help='Path of checkpoint')
    parser.add_argument('--topk',
                        '-tk',
                        type=int,
                        nargs='+',
                        default=[1],
                        help='Top k accuracy')
    parser.add_argument('--device',
                        '-d',
                        type=str,
                        required=False,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device used for testing')

    args = parser.parse_args()
    print(args)

    config_path: Path = args.config
    config = Config.fromfile(config_path)
    config.trainer.auto_resume = False
    work_dirs: Path = Path('work_dirs/tmp') / config_path.stem
    if not work_dirs.exists():
        work_dirs.mkdir(parents=True)
    config.trainer.work_dirs = str(work_dirs)

    trainer = build_trainer(config.trainer)
    print(trainer.find_latest_checkpoint())
    trainer.load_checkpoint(args.checkpoint)

    test_dataloader = build_dataloader(config.trainer.clean_test_dataloader)

    metrics = evaluate_accuracy(model=trainer._teacher_wrapper.network,
                                dataloader=trainer._clean_test_dataloader,
                                device=args.device,
                                top_k_list=(1, 5))
    print(f'Teacher validation on clean data: {metrics}')

    for s_name, s_wrapper in trainer._student_wrappers.items():
        metrics = evaluate_accuracy(model=s_wrapper.network,
                                    dataloader=trainer._clean_test_dataloader,
                                    device=args.device,
                                    top_k_list=(1, 5))
        print(f'Student {s_name} validation on clean data: {metrics}')

    # asr of poison test data on teacher & student
    trainer._trigger_wrapper.network.to(args.device)
    evaluate_asr = functools.partial(
        evaluate_accuracy,
        before_forward_fn=trainer._trigger_wrapper.network,
        dataloader=trainer._poison_test_dataloader,
        device=args.device,
        top_k_list=(1, 5))

    metrics = evaluate_asr(model=trainer._teacher_wrapper.network)
    print(f'Teacher validation on poison data: {metrics}')

    for s_name, s_wrapper in trainer._student_wrappers.items():
        metrics = evaluate_asr(model=s_wrapper.network)
        print(f'Student {s_name} validation on poison data: {metrics}')
