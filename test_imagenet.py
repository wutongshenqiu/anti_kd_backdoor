import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from torchvision import models
from torchvision.utils import save_image, make_grid

import nets
import datasets


@torch.no_grad()
def slim_linear_layer(linear: torch.nn.Linear, num_classes: int) -> None:
    if linear.out_features == num_classes:
        return

    linear.out_features = num_classes
    linear.weight = torch.nn.Parameter(linear.weight[:num_classes])
    if linear.bias is not None:
        linear.bias = torch.nn.Parameter(linear.bias[:num_classes])


if __name__ == "__main__":
    from test import evaluate_accuracy

    device = 'cuda'
    ckpt_path = 'exps/imagenet_ei=100_bei=1000_ratio=0.1_glr=0.001_lm=0.0001_itt=True/checkpoints/epoch_200.pth'
    state_dict = torch.load(ckpt_path)
    network = 'mobilenet_v2'
    network_type = 'student_3'
    start_idx = 0
    end_idx = 99
    num_classes = end_idx - start_idx + 1

    generator = nets.get_network('imagenet', 'generator')
    generator_state_dict = state_dict['generator']
    consume_prefix_in_state_dict_if_present(generator_state_dict, 'module.')
    generator.load_state_dict(generator_state_dict)
    generator.eval()
    generator.to(device)
    print(f'mean of generator mask: {generator.mask.mean()}')

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    imagenet_mean = torch.tensor(imagenet_mean).view(-1, 1, 1).to(device)
    imagenet_std = torch.tensor(imagenet_std).view(-1, 1, 1).to(device)
    def denormalize(x): return x * imagenet_std + imagenet_mean

    _, test_poison_loader = datasets.get_bd_dataloader_all(
        'imagenet',
        target=1,
        ratio=1.0,
        start_idx=start_idx,
        end_idx=end_idx)
    for x, _ in test_poison_loader:
        x = x.to(device)
        x = generator(x)
        x = denormalize(x)
        save_image(make_grid(x), 'imaganet_with_trigger.png')
        break

    _, test_loader = datasets.get_dataloader(
        'imagenet',
        start_idx=start_idx,
        end_idx=end_idx)
    for x, _ in test_loader:
        x = x.to(device)
        x = denormalize(x)
        save_image(make_grid(x), 'imagenet_original.png')
        break

    # model = nets.get_network('imagenet', network, num_classes=num_classes)
    # model_state_dict = state_dict[network_type]
    # consume_prefix_in_state_dict_if_present(model_state_dict, 'module.')
    # model.load_state_dict(model_state_dict)

    network2linear = {
        'resnet18': 'fc',
        'resnet34': 'fc',
        'resnet50': 'fc',
        'resnet101': 'fc',
        'wide_resnet50_2': 'fc',
        'wide_resnet101_2': 'fc',
        'vgg11': 'classifier.-1',
        'vgg11_bn': 'classifier.-1',
        'vgg16': 'classifier.-1',
        'vgg16_bn': 'classifier.-1',
        'mobilenet_v2': 'classifier.-1',
        'shufflenet_v2_x1_0': 'fc',
        'efficientnet_b0': 'classifier.-1',
        'densenet121': 'classifier',
        'densenet169': 'classifier',
        'densenet201': 'classifier',
    }

    top_k_list = (1, 5)
    results = {}
    for model_name, linear_name in network2linear.items():
        model = getattr(models, model_name)(pretrained=True)
        model.eval()

        print(f'Pytorch model: {model_name}')

        linear_name_list = linear_name.split('.')
        if len(linear_name_list) == 1:
            linear_layer = getattr(model, linear_name)
        elif len(linear_name_list) == 2:
            classifier_name, idx = linear_name_list
            idx = int(idx)
            classifier = getattr(model, classifier_name)
            linear_layer = classifier[idx]
        else:
            raise ValueError()

        for i in range(2):
            if i == 1:
                print(f'slim linear layer: {linear_layer}')
                slim_linear_layer(linear_layer, num_classes)

            print(f'test backdoor on teacher network')
            asr = evaluate_accuracy(
                model=model,
                dataloader=test_poison_loader,
                device=device,
                top_k_list=top_k_list,
                x_transform=generator)
            asr = {f'top{topk}': v for topk, v in zip(top_k_list, asr)}
            print(asr)

            print(f'test benign on teacher network')
            acc = evaluate_accuracy(
                model=model,
                dataloader=test_loader,
                device=device,
                top_k_list=top_k_list)
            acc = {f'top{topk}': v for topk, v in zip(top_k_list, acc)}
            print(acc)
            
            if i == 0:
                results[model_name] = {1000: {'asr': asr, 'acc': acc}}
            else:
                results[model_name][num_classes] = {'asr': asr, 'acc': acc}
    
    import json
    with open('bei=1000_result_itt=True_with-kd.json', 'w') as f:
        f.write(json.dumps(results))
    from pprint import pprint         
    pprint(results)
