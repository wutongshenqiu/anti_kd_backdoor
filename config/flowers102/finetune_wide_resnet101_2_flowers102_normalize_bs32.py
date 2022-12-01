dataest_type = 'Flowers102'
# below are not acctually mean/std of flowers102
# https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
dataset_mean = (0.485, 0.456, 0.406)
dataset_std = (0.229, 0.224, 0.225)
max_epochs = 20
batch_size = 32
lr = 0.1 / 256 * batch_size
num_workers = 4
pin_memory = False

trainer = dict(
    type='FinetuneTrainer',
    network=dict(
        network=dict(arch='torchvision',
                     type='wide_resnet101_2',
                     weights='DEFAULT',
                     progress=True),
        optimizer=dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-5),
        scheduler=dict(type='CosineAnnealingLR', T_max=max_epochs),
        finetune=dict(training=True, trainable_modules=['fc']),
        mapping=dict(
            fc=dict(type='Linear', in_features=2048, out_features=102))),
    train_dataloader=dict(dataset=dict(type=dataest_type,
                                       root='data',
                                       split='train',
                                       download=True,
                                       transform=[
                                           dict(type='RandomRotation',
                                                degrees=30),
                                           dict(type='RandomResizedCrop',
                                                size=224),
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
                                          dict(type='Resize', size=(256, 256)),
                                          dict(type='CenterCrop', size=224),
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
    device='cuda')
