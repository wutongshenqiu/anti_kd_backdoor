from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100, GTSRB

DATA_ROOT: Path = Path('data')


def cal_file_sha256(file_path: str | Path) -> str:
    import hashlib

    BLOCKSIZE = 65536
    sha256_hash = hashlib.sha256()

    with open(file_path, 'rb') as f:
        block = f.read(BLOCKSIZE)
        while block:
            sha256_hash.update(block)
            block = f.read(BLOCKSIZE)

    return sha256_hash.hexdigest()


def download_cifar10() -> dict:
    CIFAR10(root=DATA_ROOT, download=True, train=True)
    CIFAR10(root=DATA_ROOT, download=True, train=False)

    path = DATA_ROOT / CIFAR10.filename

    return dict(name='cifar10',
                path=path.as_posix(),
                sha256=cal_file_sha256(path))


def download_cifar100() -> dict:
    CIFAR100(root=DATA_ROOT, download=True, train=True)
    CIFAR100(root=DATA_ROOT, download=True, train=False)

    path = DATA_ROOT / CIFAR100.filename

    return dict(name='cifar100',
                path=path.as_posix(),
                sha256=cal_file_sha256(path))


def download_gtsrb() -> dict:
    GTSRB(root=DATA_ROOT, download=True, split='train')
    GTSRB(root=DATA_ROOT, download=True, split='test')

    train_path = DATA_ROOT / 'gtsrb' / 'GTSRB-Training_fixed.zip'
    test_images_path = DATA_ROOT / 'gtsrb' / 'GTSRB_Final_Test_Images.zip'
    test_gt_path = DATA_ROOT / 'gtsrb' / 'GTSRB_Final_Test_GT.zip'

    return dict(name='gtsrb',
                train=dict(path=train_path.as_posix(),
                           sha256=cal_file_sha256(train_path)),
                test=[
                    dict(path=test_images_path.as_posix(),
                         sha256=cal_file_sha256(test_images_path)),
                    dict(path=test_gt_path.as_posix(),
                         sha256=cal_file_sha256(test_gt_path))
                ])


if __name__ == '__main__':
    import json

    meta_dict = []
    meta_dict.append(download_cifar10())
    meta_dict.append(download_cifar100())
    meta_dict.append(download_gtsrb())

    with open(DATA_ROOT / 'meta.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(meta_dict))
