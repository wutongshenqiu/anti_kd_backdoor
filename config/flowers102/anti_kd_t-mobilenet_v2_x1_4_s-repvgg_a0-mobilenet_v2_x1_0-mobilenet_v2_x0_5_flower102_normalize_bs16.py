num_classes = 102
dataest_type = 'Flowers102'
poison_dataest_type = 'PoisonLabelFlowers102'
dataset_mean = (0.485, 0.456, 0.406)
dataset_std = (0.229, 0.224, 0.225)
max_epochs = 100
batch_size = 16
lr = 0.1 / 256 * batch_size
num_workers = 4
pin_memory = True

trainer = dict(
    type='AntiKDTrainer',
    teacher=dict(network=dict(arch='cifar',
                              type='mobilenetv2_x1_4',
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
        repvgg_a0=dict(network=dict(arch='cifar',
                                    type='repvgg_a0',
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
                                type='mobilenetv2_x1_0',
                                num_classes=num_classes),
                   optimizer=dict(type='SGD',
                                  lr=0.2,
                                  momentum=0.9,
                                  weight_decay=5e-4),
                   scheduler=dict(type='CosineAnnealingLR', T_max=max_epochs),
                   lambda_t=1e-2,
                   lambda_mask=1e-4,
                   trainable_when_training_trigger=False),
        mobilenetv2_x0_5=dict(network=dict(arch='cifar',
                                           type='mobilenetv2_x0_5',
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
    trigger=dict(network=dict(arch='trigger', type='trigger', size=224),
                 optimizer=dict(type='Adam', lr=1e-2),
                 mask_clip_range=(0., 1.),
                 trigger_clip_range=(-1., 1.),
                 mask_penalty_norm=2),
    clean_train_dataloader=dict(dataset=dict(
        type=dataest_type,
        root='data',
        split='train',
        download=True,
        transform=[
            dict(type='RandomRotation', degrees=30),
            dict(type='RandomResizedCrop', size=224),
            dict(type='RandomHorizontalFlip'),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=dataset_mean, std=dataset_std)
        ]),
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                shuffle=True),
    clean_test_dataloader=dict(dataset=dict(type=dataest_type,
                                            root='data',
                                            split='test',
                                            download=True,
                                            transform=[
                                                dict(type='Resize',
                                                     size=(256, 256)),
                                                dict(type='CenterCrop',
                                                     size=224),
                                                dict(type='ToTensor'),
                                                dict(type='Normalize',
                                                     mean=dataset_mean,
                                                     std=dataset_std)
                                            ]),
                               batch_size=batch_size,
                               num_workers=num_workers,
                               pin_memory=pin_memory),
    poison_train_dataloader=dict(dataset=dict(
        type=poison_dataest_type,
        poison_label=1,
        root='data',
        split='train',
        download=True,
        transform=[
            dict(type='RandomRotation', degrees=30),
            dict(type='RandomResizedCrop', size=224),
            dict(type='RandomHorizontalFlip'),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=dataset_mean, std=dataset_std)
        ]),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 shuffle=True),
    poison_test_dataloader=dict(dataset=dict(type=poison_dataest_type,
                                             poison_label=1,
                                             root='data',
                                             split='test',
                                             download=True,
                                             transform=[
                                                 dict(type='Resize',
                                                      size=(256, 256)),
                                                 dict(type='CenterCrop',
                                                      size=224),
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
