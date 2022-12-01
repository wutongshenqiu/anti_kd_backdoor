dataset_mean = (0.5071, 0.4867, 0.4408)
dataset_std = (0.2675, 0.2565, 0.2761)
max_epochs = 20
batch_size = 128
lr = 0.01 / 256 * batch_size
num_workers = 2
pin_memory = True

trainer = dict(type='NormalTrainer',
               network=dict(network=dict(arch='chenyaofo_cifar_models',
                                         type='cifar100_resnet56',
                                         pretrained=True,
                                         progress=True),
                            optimizer=dict(type='SGD',
                                           lr=lr,
                                           momentum=0.9,
                                           weight_decay=1e-5),
                            scheduler=dict(type='CosineAnnealingLR',
                                           T_max=max_epochs)),
               train_dataloader=dict(dataset=dict(
                   type='RatioCIFAR100',
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
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory),
               epochs=max_epochs,
               save_interval=1,
               epochs_per_validation=1,
               device='cuda')
