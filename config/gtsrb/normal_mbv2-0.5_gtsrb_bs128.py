num_classes = 43
dataest_type = 'GTSRB'
max_epochs = 50
batch_size = 128
num_workers = 4
pin_memory = True

trainer = dict(type='NormalTrainer',
               network=dict(network=dict(arch='cifar',
                                         type='mobilenetv2_x0_5',
                                         num_classes=num_classes),
                            optimizer=dict(type='SGD',
                                           lr=0.2,
                                           momentum=0.9,
                                           weight_decay=5e-4),
                            scheduler=dict(type='CosineAnnealingLR',
                                           T_max=max_epochs)),
               train_dataloader=dict(dataset=dict(type=dataest_type,
                                                  root='data',
                                                  split='train',
                                                  download=True,
                                                  transform=[
                                                      dict(type='Resize',
                                                           size=(32, 32)),
                                                      dict(type='ToTensor')
                                                  ]),
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory,
                                     shuffle=True),
               test_dataloader=dict(dataset=dict(type=dataest_type,
                                                 root='data',
                                                 split='test',
                                                 download=True,
                                                 transform=[
                                                     dict(type='Resize',
                                                          size=(32, 32)),
                                                     dict(type='ToTensor')
                                                 ]),
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory),
               epochs=max_epochs,
               save_interval=5,
               device='cuda')
