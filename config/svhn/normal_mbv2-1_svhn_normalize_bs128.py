num_classes = 10
dataest_type = 'SVHN'
dataset_mean = (0.4377, 0.4438, 0.4728)
dataset_std = (0.1980, 0.2010, 0.1970)
max_epochs = 100
batch_size = 128
num_workers = 4
pin_memory = True

trainer = dict(type='NormalTrainer',
               network=dict(network=dict(arch='cifar',
                                         type='mobilenetv2_x1_0',
                                         num_classes=num_classes),
                            optimizer=dict(type='SGD',
                                           lr=0.2,
                                           momentum=0.9,
                                           weight_decay=5e-4),
                            scheduler=dict(type='CosineAnnealingLR',
                                           T_max=max_epochs)),
               train_dataloader=dict(dataset=dict(type=dataest_type,
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
               test_dataloader=dict(dataset=dict(type=dataest_type,
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
               device='cuda')
