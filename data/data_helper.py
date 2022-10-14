from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import StandardDataset
from data.DataLoader import get_split_dataset_info, _dataset_info
from data.concat_dataset import ConcatDataset
from data.DataLoader import Dataset


vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
officehome_datasets = ["Art", "Clipart", "Product", "Real_world"]
terraincognita_datasets = ["100", "38", "43", "46"]
coloredmnist_datasets = ["train_1", "train_2", "test"]

available_datasets = officehome_datasets + pacs_datasets + vlcs_datasets + terraincognita_datasets

def get_train_dataloader(args):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    
    if args.dataset is 'ColoredMNIST':
        img_transformer = get_mnist_transformer(args)
        val_transformer = get_mnist_transformer(args)      
    else:
        img_transformer = get_train_transformer(args)
        val_transformer = get_val_transformer(args)
    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), f'txt_lists/{args.dataset}/{dname}/{dname}.txt'), 0.1)

        datasets.append(Dataset(name_train, labels_train, img_transformer=img_transformer, dataset=args.dataset))
        val_datasets.append(Dataset(name_val, labels_val, img_transformer=val_transformer, dataset=args.dataset))
        
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def get_test_dataloader(args):
    names, labels = _dataset_info(join(dirname(__file__), f'txt_lists/{args.dataset}/{args.target}/{args.target}.txt'))
    if args.dataset is 'ColoredMNIST':
        img_tr = get_mnist_transformer(args)
    else:
        img_tr = get_val_transformer(args)
    val_dataset = Dataset(names, labels, img_transformer=img_tr, dataset=args.dataset)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_train_transformer(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    img_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    img_tr.append(transforms.ToTensor())
    img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(img_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)


def get_mnist_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Lambda(lambda x: x/255.0)
              #transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
             ]
    return transforms.Compose(img_tr)
