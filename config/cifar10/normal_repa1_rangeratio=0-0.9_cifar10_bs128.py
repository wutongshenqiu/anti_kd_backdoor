num_classes = 10
dataest_type = 'RangeRatioCIFAR10'
range_ratio = (0, 0.9)
dataset_mean = (0.4914, 0.4822, 0.4465)
dataset_std = (0.2023, 0.1994, 0.2010)

trainer = dict(
    type='NormalTrainer',
    network=dict(network=dict(arch='chenyaofo_cifar_models',
                              type='cifar10_repvgg_a1'),
                 optimizer=dict(type='SGD',
                                lr=0.2,
                                momentum=0.9,
                                weight_decay=5e-4),
                 scheduler=dict(type='CosineAnnealingLR', T_max=100)),
    train_dataloader=dict(dataset=dict(type=dataest_type,
                                       range_ratio=range_ratio,
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
    test_dataloader=dict(dataset=dict(type='CIFAR10',
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
