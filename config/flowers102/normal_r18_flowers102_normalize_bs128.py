num_classes = 102
dataest_type = 'Flowers102'
# below are not acctually mean/std of flowers102
# https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
dataset_mean = (0.485, 0.456, 0.406)
dataset_std = (0.229, 0.224, 0.225)
max_epochs = 50
batch_size = 4
lr = 0.2 / 128 * batch_size
num_workers = 4
pin_memory = False

trainer = dict(type='NormalTrainer',
               network=dict(network=dict(arch='cifar',
                                         type='resnet18',
                                         num_classes=num_classes),
                            optimizer=dict(type='SGD',
                                           lr=lr,
                                           momentum=0.9,
                                           weight_decay=5e-4),
                            scheduler=dict(type='CosineAnnealingLR',
                                           T_max=max_epochs)),
               train_dataloader=dict(dataset=dict(
                   type=dataest_type,
                   root='data',
                   split='train',
                   download=True,
                   transform=[
                       dict(type='RandomRotation', degrees=30),
                       dict(type='RandomResizedCrop', size=224),
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
               test_dataloader=dict(dataset=dict(type=dataest_type,
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
               device='cuda')
