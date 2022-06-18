from . import resnet_cifar
from . import vgg_cifar
from . import densenet_cifar
from . import mobilenetv2_cifar
from . import generator
import torch.nn as nn
import torchvision.models as models


def get_network(dataset_name, arch_name):
    if dataset_name in ("cifar10", "svhn") and arch_name == "resnet18":
        network = resnet_cifar.resnet18()
    elif dataset_name in ("cifar10", "svhn") and arch_name == "resnet34":
        network = resnet_cifar.resnet34()
    elif dataset_name in ("cifar10", "svhn") and arch_name == "vgg16":
        network = vgg_cifar.VGG("VGG16")
    elif dataset_name in ("cifar10", "svhn") and arch_name == "densenet121":
        network = densenet_cifar.DenseNet121()
    elif dataset_name in ("cifar10", "svhn") and arch_name == "mobilenetv2":
        network = mobilenetv2_cifar.MobileNetV2(num_classes=10)
    elif dataset_name == "cifar100" and arch_name == "resnet18":
        network = resnet_cifar.resnet18(num_classes=100)
    elif dataset_name == "cifar100" and arch_name == "vgg16":
        network = vgg_cifar.VGG("VGG16", num_classes=100)
    elif dataset_name == "cifar100" and arch_name == "densenet121":
        network = densenet_cifar.DenseNet121(num_classes=100)
    elif dataset_name == "cifar100" and arch_name == "mobilenetv2":
        network = mobilenetv2_cifar.MobileNetV2(num_classes=100)
    elif dataset_name == "tiny_imagenet" and arch_name == "resnet18":
        network = models.resnet.resnet18(pretrained=False, progress=True, num_classes=200)
        # network = models.resnet.resnet18(pretrained=True, progress=True)
        # network.fc = nn.Linear(512, 200)
    elif dataset_name == "tiny_imagenet" and arch_name == "vgg16":
        network = models.vgg.vgg16_bn(pretrained=False, progress=True, num_classes=200)
    elif dataset_name == "tiny_imagenet" and arch_name == "densenet121":
        network = models.densenet.densenet121(
            pretrained=False, progress=True, num_classes=200
        )
    elif dataset_name in ("cifar10", "cifar100", "svhn") and arch_name == "generator":
        network = generator.TriggerGenerator()
    else:
        raise ValueError(
            "arch name {} for {} dataset has not been implemented yet!".format(
                arch_name, dataset_name
            )
        )
    return network
