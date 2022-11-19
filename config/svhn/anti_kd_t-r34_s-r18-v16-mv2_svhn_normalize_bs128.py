num_classes = 10
dataest_type = 'SVHN'
poison_dataest_type = 'PoisonLabelSVHN'
dataset_mean = (0.4377, 0.4438, 0.4728)
dataset_std = (0.1980, 0.2010, 0.1970)
max_epochs = 100
batch_size = 128
num_workers = 4
pin_memory = True

trainer = dict(
    type='AntiKDTrainer',
    teacher=dict(network=dict(arch='cifar',
                              type='resnet34',
                              num_classes=num_classes),
                 optimizer=dict(type='SGD',
                                lr=0.2,
                                momentum=0.9,
                                weight_decay=5e-4),
                 scheduler=dict(type='CosineAnnealingLR', T_max=max_epochs),
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
                      scheduler=dict(type='CosineAnnealingLR',
                                     T_max=max_epochs),
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
                   scheduler=dict(type='CosineAnnealingLR', T_max=max_epochs),
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
                          scheduler=dict(type='CosineAnnealingLR',
                                         T_max=max_epochs),
                          lambda_t=1e-2,
                          lambda_mask=1e-4,
                          trainable_when_training_trigger=False),
    ),
    trigger=dict(network=dict(arch='trigger', type='trigger', size=32),
                 optimizer=dict(type='Adam', lr=1e-2),
                 mask_clip_range=(0., 1.),
                 trigger_clip_range=(-1., 1.),
                 mask_penalty_norm=2),
    clean_train_dataloader=dict(dataset=dict(type=dataest_type,
                                             root='data/svhn',
                                             split='train',
                                             download=True,
                                             transform=[
                                                 dict(type='Resize',
                                                      size=(32, 32)),
                                                 dict(type='ToTensor'),
                                                 dict(type='Normalize',
                                                      mean=dataset_mean,
                                                      std=dataset_std)
                                             ]),
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                shuffle=True),
    clean_test_dataloader=dict(dataset=dict(type=dataest_type,
                                            root='data/svhn',
                                            split='test',
                                            download=True,
                                            transform=[
                                                dict(type='Resize',
                                                     size=(32, 32)),
                                                dict(type='ToTensor'),
                                                dict(type='Normalize',
                                                     mean=dataset_mean,
                                                     std=dataset_std)
                                            ]),
                               batch_size=batch_size,
                               num_workers=num_workers,
                               pin_memory=pin_memory),
    poison_train_dataloader=dict(dataset=dict(type=poison_dataest_type,
                                              poison_label=1,
                                              root='data/svhn',
                                              split='train',
                                              download=True,
                                              transform=[
                                                  dict(type='Resize',
                                                       size=(32, 32)),
                                                  dict(type='ToTensor'),
                                                  dict(type='Normalize',
                                                       mean=dataset_mean,
                                                       std=dataset_std)
                                              ]),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 shuffle=True),
    poison_test_dataloader=dict(dataset=dict(type=poison_dataest_type,
                                             poison_label=1,
                                             root='data/svhn',
                                             split='test',
                                             download=True,
                                             transform=[
                                                 dict(type='Resize',
                                                      size=(32, 32)),
                                                 dict(type='ToTensor'),
                                                 dict(type='Normalize',
                                                      mean=dataset_mean,
                                                      std=dataset_std)
                                             ]),
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory),
    epochs=max_epochs,
    save_interval=5,
    temperature=1.0,
    alpha=1.0,
    device='cuda')
