from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.DataLoader import get_split_dataset_info, _dataset_info
from data.concat_dataset import ConcatDataset
from data.DataLoader import Dataset


def get_dataset_list(args):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    loaders = []
    val_datasets = []
    img_transformer = get_train_transformers(args)
    val_transformer = get_val_transformer(args)
    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), f'txt_lists/{args.dataset}/{dname}/{dname}.txt'), args.val_size)

        dataset = Dataset(name_train, labels_train, img_transformer=img_transformer,
                          dataset=args.dataset, path=args.data_path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=0, pin_memory=True, drop_last=True)
        loaders.append(loader)
        
        val_datasets.append(Dataset(name_val, labels_val, img_transformer=val_transformer,
                                    dataset=args.dataset, path=args.data_path))
    val_dataset = ConcatDataset(val_datasets)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=4, pin_memory=True, drop_last=False)
    return loaders, val_loader

def get_train_dataloader(args):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer = get_train_transformers(args)
    val_transformer = get_val_transformer(args)
    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), f'txt_lists/{args.dataset}/{dname}/{dname}.txt'), args.val_size)

        datasets.append(Dataset(name_train, labels_train, img_transformer=img_transformer, 
                                dataset=args.dataset, path=args.data_path))
        val_datasets.append(Dataset(name_val, labels_val, img_transformer=val_transformer,
                                    dataset=args.dataset, path=args.data_path))
        
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def get_test_dataloader(args):
    names, labels = _dataset_info(join(dirname(__file__), f'txt_lists/{args.dataset}/{args.target}/{args.target}.txt'))
    img_tr = get_val_transformer(args)
    val_dataset = Dataset(names, labels, img_transformer=img_tr, dataset=args.dataset, path=args.data_path)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_train_transformers(args):
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

def get_source_domains(args):
    if args.dataset == 'PACS':
        args.n_classes = 7
        if args.target == 'photo':
            args.source = ['art_painting', 'cartoon', 'sketch']
        elif args.target == 'sketch':
            args.source = ['art_painting', 'cartoon', 'photo']
        elif args.target == 'cartoon':
            args.source = ['art_painting', 'photo', 'sketch']
        elif args.target == 'art_painting':
            args.source = ['photo', 'cartoon', 'sketch']
        else:
            raise Exception(f"Unknown domain {args.target}")
 
    if args.dataset == 'VLCS':
        args.n_classes = 5
        if args.target == "CALTECH":
            args.source = ["LABELME", "PASCAL", "SUN"]
        elif args.target == "LABELME":
            args.source = ["CALTECH", "PASCAL", "SUN"]
        elif args.target == 'PASCAL':
            args.source = ["LABELME", "CALTECH", "SUN"]
        elif args.target == 'SUN':
            args.source = ["LABELME", "PASCAL", "CALTECH"]
        else:
            raise Exception(f"Unknown domain {args.target}")

    if args.dataset == 'OfficeHome':
        args.n_classes = 65
        if args.target == 'Real_World':
            args.source = ['Art', 'Clipart', 'Product']
        elif args.target == 'Art':
            args.source = ['Clipart', 'Real_World', 'Product']
        elif args.target == 'Clipart':
            args.source = ['Art', 'Product', 'Real_World']
        elif args.target == 'Product':
            args.source = ['Art', 'Clipart', 'Real_World']
        else:
            raise Exception(f"Unknown domain {args.target}")

    if args.dataset == 'TerraIncognita':
        args.n_classes = 16
        if args.target == '100':
            args.source = ['38', '43', '46']
        elif args.target == '38':
            args.source = ['43', '46', '100']
        elif args.target == '43':
            args.source = ['46', '100', '38']
        elif args.target == '46':
            args.source = ['100', '38', '43']
        else:
            raise Exception(f"Unknown domain {args.target}")
            
    if not args.dg:
        args.source = [args.target] 
        
    return args