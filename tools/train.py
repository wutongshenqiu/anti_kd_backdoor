from pathlib import Path

import torch

from anti_kd_backdoor.config import Config
from anti_kd_backdoor.trainer import build_trainer

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config', type=Path, help='Path of config file')
    parser.add_argument('--work_dirs',
                        type=Path,
                        default='work_dirs',
                        help='Path of work directory')
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

    trainer_config = config.trainer
    work_dirs: Path = args.work_dirs / config_path.stem
    if not work_dirs.exists():
        work_dirs.mkdir(parents=True)
    trainer_config.work_dirs = str(work_dirs)
    print(config.pretty_text)

    trainer = build_trainer(trainer_config)
    trainer.train()
