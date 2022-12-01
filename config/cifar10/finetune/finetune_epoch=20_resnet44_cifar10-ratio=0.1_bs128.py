dataset_mean = (0.4914, 0.4822, 0.4465)
dataset_std = (0.2023, 0.1994, 0.2010)
max_epochs = 20
batch_size = 128
lr = 0.01 / 256 * batch_size
num_workers = 2
pin_memory = True

trainer = dict(type='NormalTrainer',
               network=dict(network=dict(arch='chenyaofo_cifar_models',
                                         type='cifar10_resnet44',
                                         pretrained=True,
                                         progress=True),
                            optimizer=dict(type='SGD',
                                           lr=lr,
                                           momentum=0.9,
                                           weight_decay=1e-5),
                            scheduler=dict(type='CosineAnnealingLR',
                                           T_max=max_epochs)),
               train_dataloader=dict(dataset=dict(
                   type='RatioCIFAR10',
                   ratio=0.1,
                   root='data',
                   train=True,
                   download=True,
                   transform=[
                       dict(type='RandomCrop', size=32, padding=4),
                       dict(type='RandomHorizontalFlip'),
                       dict(type='ToTensor'),
                       dict(type='Normalize',
                            mean=dataset_mean,
                            std=dataset_std)
                   ]),
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory,
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
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory),
               epochs=max_epochs,
               save_interval=1,
               epochs_per_validation=1,
               device='cuda')
