from pathlib import Path

import torch

from anti_kd_backdoor.config import Config
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
    work_dirs: Path = Path('work_dirs/tmp') / config_path.stem
    if not work_dirs.exists():
        work_dirs.mkdir(parents=True)
    config.trainer.work_dirs = str(work_dirs)

    trainer = build_trainer(config.trainer)
    trainer.load_checkpoint(args.checkpoint)

    metrics = evaluate_accuracy(model=trainer._network_wrapper.network,
                                dataloader=trainer._test_dataloader,
                                device=args.device,
                                top_k_list=(1, 5))
    print(f'validation on clean data: {metrics}')
