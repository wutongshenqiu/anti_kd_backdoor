num_classes = 100
dataset_mean = (0.5071, 0.4867, 0.4408)
dataset_std = (0.2675, 0.2565, 0.2761)

dataest_type = 'RangeRatioCIFAR100'
range_ratio = (0.3, 1)

poison_dataset_type = 'RangeRatioPoisonLabelCIFAR100'
poison_label = 1

num_workers = 2

trainer = dict(
    type='AntiKDTrainer',
    teacher=dict(network=dict(arch='cifar',
                              type='resnet34',
                              num_classes=num_classes),
                 optimizer=dict(type='SGD',
                                lr=0.2,
                                momentum=0.9,
                                weight_decay=5e-4),
                 scheduler=dict(type='CosineAnnealingLR', T_max=100),
                 lambda_t=0.1,
                 lambda_mask=1e-4,
                 trainable_when_training_trigger=False),
    students=dict(
        resnet18=dict(network=dict(arch='cifar',
                                   type='resnet18',
                                   num_classes=num_classes),
                      optimizer=dict(type='SGD',
                                     lr=0.2,
                                     momentum=0.9,
                                     weight_decay=5e-4),
                      scheduler=dict(type='CosineAnnealingLR', T_max=100),
                      lambda_t=1e-2,
                      lambda_mask=1e-4,
                      trainable_when_training_trigger=False),
        vgg16=dict(network=dict(arch='cifar',
                                type='vgg16',
                                num_classes=num_classes),
                   optimizer=dict(type='SGD',
                                  lr=0.2,
                                  momentum=0.9,
                                  weight_decay=5e-4),
                   scheduler=dict(type='CosineAnnealingLR', T_max=100),
                   lambda_t=1e-2,
                   lambda_mask=1e-4,
                   trainable_when_training_trigger=False),
        mobilenet_v2=dict(network=dict(arch='cifar',
                                       type='mobilenet_v2',
                                       num_classes=num_classes),
                          optimizer=dict(type='SGD',
                                         lr=0.2,
                                         momentum=0.9,
                                         weight_decay=5e-4),
                          scheduler=dict(type='CosineAnnealingLR', T_max=100),
                          lambda_t=1e-2,
                          lambda_mask=1e-4,
                          trainable_when_training_trigger=False),
    ),
    trigger=dict(network=dict(arch='trigger', type='trigger', size=32),
                 optimizer=dict(type='Adam', lr=1e-2),
                 mask_clip_range=(0., 1.),
                 trigger_clip_range=(-1., 1.),
                 mask_penalty_norm=2),
    clean_train_dataloader=dict(dataset=dict(
        type=dataest_type,
        range_ratio=range_ratio,
        root='data',
        train=True,
        download=True,
        transform=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomHorizontalFlip'),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=dataset_mean, std=dataset_std)
        ]),
                                batch_size=128,
                                num_workers=num_workers,
                                pin_memory=True,
                                shuffle=True),
    clean_test_dataloader=dict(dataset=dict(type='CIFAR100',
                                            root='data',
                                            train=False,
                                            download=True,
                                            transform=[
                                                dict(type='ToTensor'),
                                                dict(type='Normalize',
                                                     mean=dataset_mean,
                                                     std=dataset_std)
                                            ]),
                               batch_size=128,
                               num_workers=num_workers,
                               pin_memory=True),
    poison_train_dataloader=dict(dataset=dict(
        type=poison_dataset_type,
        range_ratio=range_ratio,
        poison_label=poison_label,
        root='data',
        train=True,
        download=True,
        transform=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomHorizontalFlip'),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=dataset_mean, std=dataset_std)
        ]),
                                 batch_size=128,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 shuffle=True),
    poison_test_dataloader=dict(dataset=dict(type='PoisonLabelCIFAR100',
                                             poison_label=poison_label,
                                             root='data',
                                             train=False,
                                             download=True,
                                             transform=[
                                                 dict(type='ToTensor'),
                                                 dict(type='Normalize',
                                                      mean=dataset_mean,
                                                      std=dataset_std)
                                             ]),
                                batch_size=128,
                                num_workers=num_workers,
                                pin_memory=True),
    epochs=100,
    save_interval=5,
    temperature=1.0,
    alpha=1.0,
    device='cuda')
