trainer = dict(
    type='NormalTrainer',
    network=dict(network=dict(arch='cifar', type='resnet34', num_classes=10),
                 optimizer=dict(type='SGD',
                                lr=0.2,
                                momentum=0.9,
                                weight_decay=5e-4),
                 scheduler=dict(type='CosineAnnealingLR', T_max=100)),
    train_dataloader=dict(dataset=dict(type='CIFAR10',
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
                                                mean=(0.4914, 0.4822, 0.4465),
                                                std=(0.2023, 0.1994, 0.2010))
                                       ]),
                          batch_size=128,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True),
    test_dataloader=dict(dataset=dict(type='CIFAR10',
                                      root='data',
                                      train=False,
                                      download=True,
                                      transform=[
                                          dict(type='ToTensor'),
                                          dict(type='Normalize',
                                               mean=(0.4914, 0.4822, 0.4465),
                                               std=(0.2023, 0.1994, 0.2010))
                                      ]),
                         batch_size=128,
                         num_workers=4,
                         pin_memory=True),
    epochs=100,
    save_interval=5,
    device='cuda')
