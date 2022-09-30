from pathlib import Path

from anti_kd_backdoor.config import Config

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config', type=Path, help='Path of config file')

    args = parser.parse_args()

    config_path: Path = args.config
    config = Config.fromfile(config_path)
