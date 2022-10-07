from pathlib import Path

import torch
from torch.nn.modules import utils

from anti_kd_backdoor.config import Config
from anti_kd_backdoor.data import build_dataloader
from anti_kd_backdoor.network import build_network
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

    model = build_network(config.trainer.teacher.network)
    print(model)

    ckpt = torch.load(args.checkpoint, map_location='cpu')['teacher']
    utils.consume_prefix_in_state_dict_if_present(ckpt, 'module.')

    model.load_state_dict(ckpt)
    test_dataloader = build_dataloader(config.trainer.clean_test_dataloader)

    print(
        evaluate_accuracy(model,
                          test_dataloader,
                          device=args.device,
                          top_k_list=args.topk))
