import torch

def get_dataloaders(dataset, dataset_args, batch_size, num_workers):
    train_dataset_args = dataset_args.copy()
    val_dataset_args = dataset_args.copy()
    train_dataset_args['train'] = True
    val_dataset_args['train'] = False
    train_dataset = dataset(**train_dataset_args)
    val_dataset = dataset(**val_dataset_args)
    #train_dataset.test(1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    print('Data Size: train = {} val = {}'.format(len(train_dataset), len(val_dataset)))
    return dataloaders

