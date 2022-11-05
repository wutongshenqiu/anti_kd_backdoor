num_classes = 100
dataest_type = 'RangeRatioCIFAR100'
dataset_mean = (0.5071, 0.4867, 0.4408)
dataset_std = (0.2675, 0.2565, 0.2761)

trainer = dict(
    type='NormalTrainer',
    network=dict(network=dict(arch='chenyaofo_cifar_models',
                              type='cifar100_repvgg_a1'),
                 optimizer=dict(type='SGD',
                                lr=0.2,
                                momentum=0.9,
                                weight_decay=5e-4),
                 scheduler=dict(type='CosineAnnealingLR', T_max=100)),
    train_dataloader=dict(dataset=dict(type=dataest_type,
                                       range_ratio=(0, 0.7),
                                       root='data',
                                       train=True,
                                       download=True,
                                       transform=[
                                           dict(type='RandomCrop',
                                                size=32,
                                                padding=4),
                                           dict(type='RandomHorizontalFlip'),
                                           dict(type='ToTensor'),
                                           dict(type='Normalize',
                                                mean=dataset_mean,
                                                std=dataset_std)
                                       ]),
                          batch_size=128,
                          num_workers=2,
                          pin_memory=True,
                          shuffle=True),
    test_dataloader=dict(dataset=dict(type='CIFAR100',
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
                         num_workers=2,
                         pin_memory=True),
    epochs=100,
    save_interval=5,
    device='cuda')
