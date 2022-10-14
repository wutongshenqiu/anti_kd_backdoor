import torch

from anti_kd_backdoor.data import build_dataloader
from anti_kd_backdoor.utils import evaluate_accuracy

hub_list = hub_list = [
    'cifar100_mobilenetv2_x0_5', 'cifar100_mobilenetv2_x0_75',
    'cifar100_mobilenetv2_x1_0', 'cifar100_mobilenetv2_x1_4',
    'cifar100_repvgg_a0', 'cifar100_repvgg_a1', 'cifar100_repvgg_a2',
    'cifar100_resnet20', 'cifar100_resnet32', 'cifar100_resnet44',
    'cifar100_resnet56', 'cifar100_shufflenetv2_x0_5',
    'cifar100_shufflenetv2_x1_0', 'cifar100_shufflenetv2_x1_5',
    'cifar100_shufflenetv2_x2_0', 'cifar100_vgg11_bn', 'cifar100_vgg13_bn',
    'cifar100_vgg16_bn', 'cifar100_vgg19_bn', 'cifar100_vit_b16',
    'cifar100_vit_b32', 'cifar100_vit_h14', 'cifar100_vit_l16',
    'cifar100_vit_l32', 'cifar10_mobilenetv2_x0_5',
    'cifar10_mobilenetv2_x0_75', 'cifar10_mobilenetv2_x1_0',
    'cifar10_mobilenetv2_x1_4', 'cifar10_repvgg_a0', 'cifar10_repvgg_a1',
    'cifar10_repvgg_a2', 'cifar10_resnet20', 'cifar10_resnet32',
    'cifar10_resnet44', 'cifar10_resnet56', 'cifar10_shufflenetv2_x0_5',
    'cifar10_shufflenetv2_x1_0', 'cifar10_shufflenetv2_x1_5',
    'cifar10_shufflenetv2_x2_0', 'cifar10_vgg11_bn', 'cifar10_vgg13_bn',
    'cifar10_vgg16_bn', 'cifar10_vgg19_bn', 'cifar10_vit_b16',
    'cifar10_vit_b32', 'cifar10_vit_h14', 'cifar10_vit_l16', 'cifar10_vit_l32'
]

for hub_model in hub_list:

    if hub_model.startswith('cifar10'):
        dataset = 'CIFAR10'
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        dataset = 'CIFAR100'
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    dataloader_cfg = dict(dataset=dict(type=dataset,
                                       root='data',
                                       train=False,
                                       download=True,
                                       transform=[
                                           dict(type='ToTensor'),
                                           dict(type='Normalize',
                                                mean=mean,
                                                std=std)
                                       ]),
                          batch_size=128,
                          num_workers=4,
                          pin_memory=True)
    dataloader = build_dataloader(dataloader_cfg)
    model = torch.hub.load('chenyaofo/pytorch-cifar-models',
                           hub_model,
                           pretrained=True)

    acc = evaluate_accuracy(
        model=model,
        dataloader=dataloader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        top_k_list=(1, 5))
    print(f'{hub_model}: {acc}')
