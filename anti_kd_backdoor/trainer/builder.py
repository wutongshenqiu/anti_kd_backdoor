from .anti_kd import AntiKDTrainer
from .finetune import FinetuneTrainer
from .normal import NormalTrainer

TRAINERS_MAPPING = {
    'AntiKDTrainer': AntiKDTrainer,
    'NormalTrainer': NormalTrainer,
    'FinetuneTrainer': FinetuneTrainer
}


def build_trainer(trainer_cfg: dict):
    if 'type' not in trainer_cfg:
        raise ValueError('Trainer config must have `type` field')
    trainer_type = trainer_cfg.pop('type')
    if trainer_type not in TRAINERS_MAPPING:
        raise ValueError(
            f'trainer `{trainer_type}` is not support, '
            f'available trainers: {list(TRAINERS_MAPPING.keys())}')
    trainer = TRAINERS_MAPPING[trainer_type]

    return trainer(**trainer_cfg)
